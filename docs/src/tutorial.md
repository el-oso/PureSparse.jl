# Tutorial: choosing a decomposition

PureSparse, like [faer's sparse linear-solve guide](https://faer.veganb.tw/docs/sparse-linalg/linsolve/),
organizes its solvers by the **shape and symmetry of `A`**, not by algorithm name. Decide
which of the three cases below your matrix falls into, factor it, and solve. Every
decomposition shares one workflow — *analyze once, factorize many times* (the [Guide](guide.md)
covers it in full) — and one solve spelling: build a factor `F`, then `F \ b` (or the
in-place `solve!(x, F, b)`).

| `A` is… | use | factorization | solves |
|---|---|---|---|
| symmetric **positive definite** | `cholesky` | `A = L·Lᵀ` | `A·x = b` |
| symmetric **quasi-definite** (saddle-point / KKT) | `ldlt` | `A = L·D·Lᵀ`, `D` signed | `A·x = b`, indefinite |
| **rectangular** (over- or under-determined) | `qr` | `A·P = Q·R` | least-squares; minimum-norm (via `Aᵀ`) |

PureSparse is a *symmetric* solver plus a rectangular QR; a general **square unsymmetric**
`A` (faer's `sp_lu` case) is out of scope — use `SparseArrays.lu` / KLU for that, or solve the
normal equations `AᵀA·x = Aᵀb` via `cholesky` when `A` has full column rank (fastest, but
squares the condition number — `qr` is the robust choice, see below).

## `A` is symmetric positive definite

The default case: a symmetric `A` with all-positive pivots (stiffness matrices, graph
Laplacians plus a shift, `AᵀA` normal-equations systems). `cholesky` computes `A = L·Lᵀ`.
Only the **lower triangle** of `A` is read (`Symmetric(A, :L)` semantics), so a fully-stored
symmetric matrix works too.

```julia
using PureSparse, SparseArrays, LinearAlgebra

A = sprand(1000, 1000, 0.005); A = A + A' + 1000I   # symmetric, diagonally dominant → SPD
b = randn(1000)

F = PureSparse.cholesky(A)      # symbolic analysis + numeric factorization
PureSparse.issuccess(F)         # true iff every pivot was positive (never throws — query this)
x = F \ b                       # solve
```

`cholesky` never throws on a non-positive pivot: it records the failure (`issuccess(F) == false`,
failing column in `F.stats.fail_col`) so you can branch on it rather than catch an exception.
See the [Guide](guide.md) for split solves, custom orderings, and in-place refactorization.

## `A` is symmetric quasi-definite

Saddle-point / KKT systems — the workhorse of interior-point optimization — are symmetric but
**indefinite**. The *regularized* form has a `[H Aᵀ; A −D]` block structure with `H`, `D`
positive definite, which makes it **quasi-definite** (Vanderbei 1995): every symmetric
permutation admits an `L·D·Lᵀ` factorization with a **fixed pivot order** and a nonsingular
**signed** diagonal `D`, with no dynamic pivoting. (An *unregularized* equality-constrained KKT
system has `D = 0` and is *not* quasi-definite — `ldlt`'s signed regularization is exactly what
forces it into this class.) `ldlt` does this QDLDL/Clarabel-style; general Bunch–Kaufman 2×2
pivoting is deliberately out of scope.

```julia
using PureSparse, SparseArrays, LinearAlgebra

H = sparse(1.0I, 3, 3)                     # 3×3 SPD block
Ac = sprand(2, 3, 0.6)                      # 2×3 coupling
D = sparse(1.0I, 2, 2)                      # 2×2 SPD block
K = [H Ac'; Ac -D]                          # 5×5 symmetric quasi-definite (saddle-point)

# Tell ldlt the expected pivot signs: which diagonal entries are the +block vs the −block.
F = PureSparse.ldlt(K; signs = [1, 1, 1, -1, -1])
x = F \ randn(5)
F.stats                                     # reports the achieved inertia (n_pos, n_neg, n_zero)
```

Pass the block sizes as `ldlt(K; n_pos = 3, n_neg = 2)` instead of a full `signs` vector when
the `+` block precedes the `−` block. For the full interior-point workflow — refactorizing the
same KKT pattern every iteration, iterative refinement (`refine!`), and rank-`k` update/downdate
(`updowndate!`) — see the [Interior-Point Guide](ipm-guide.md).

## `A` is rectangular

For an over-determined `A` (`m > n`, more equations than unknowns) there is generally no exact
solution; `qr` + `F \ b` gives the **least-squares** minimizer `argmin‖b − A·x‖₂`. For an
under-determined `A` (`m < n`) there are infinitely many solutions; here `F \ b` returns a
*basic* solution (some unknowns pinned to zero) — to get the **minimum-norm** solution instead,
factor `Aᵀ` and call `solve_minnorm!` (shown in the [Sparse QR Guide](qr-guide.md)). PureSparse
factors `A·P = Q·R` with Householder reflectors and a fill-reducing column ordering (`COLAMD`,
which orders `A`'s columns without ever forming `AᵀA`).

```julia
using PureSparse, SparseArrays, LinearAlgebra

A = sprand(200, 50, 0.05) + sparse(1:50, 1:50, 1.0, 200, 50)   # 200×50, full column rank
b = randn(200)

F = PureSparse.qr(A; ordering = PureSparse.COLAMDOrdering())    # ordering is mandatory
x = F \ b                        # least-squares: argmin ‖b − A·x‖₂
norm(A' * (b - A * x))           # ≈ 1e-14 — normal-equations residual Aᵀ(b − Ax) ≈ 0
```

QR is also the **rank-detecting** choice: on an ill-conditioned or rank-deficient `A` it drops
numerically-dead columns and reports the discarded mass in `F.stats.dropped_norm` — an honest
error certificate (as good as an exact rank-revealing factorization when `dropped_norm` is
small) rather than a silently wrong answer. Minimum-norm solves, the singleton-peeling fast path
for LP-shaped matrices, and the refactor workflow are all in the [Sparse QR Guide](qr-guide.md).

## Under the hood: analyze once, factorize many

All three decompositions split the work the same way — the reason PureSparse suits solvers that
refactor the *same sparsity pattern* hundreds of times (interior-point optimizers especially):

- **Symbolic (analyze).** A *pattern-only* pass computes the fill-reducing permutation, the
  elimination tree, and the structure of the result. It depends on `A`'s sparsity **pattern**,
  not its values, so it runs **once** and is shared by reference (`Symbolic` / `QRSymbolic`).
  The fill-reducing ordering is the lever that makes sparse direct methods tractable: **AMD**
  (Approximate Minimum Degree) for the symmetric cases, **COLAMD** for QR's columns.
- **Numeric (factorize).** Fills the pre-sized factor with values. Refactorizing a new-values,
  same-pattern matrix (`cholesky!` / `ldlt!` / `qr!`) reuses the symbolic result and runs with
  **zero allocations** after warmup.

The numeric kernels differ by structure, in the spirit of faer's
[Cholesky variants writeup](https://faer.veganb.tw/docs/contributing/cholesky/):

- **Cholesky / LDLᵀ** are **supernodal** — columns with identical (or, after amalgamation,
  nearly identical) structure are grouped into dense panels so the heavy arithmetic is BLAS-3
  (`potrf!`/`trsm!`/`syrk!`), routed through the sibling
  [PureBLAS.jl](https://github.com/el-oso/PureBLAS.jl). LDLᵀ additionally has a
  simplicial (column-at-a-time) path for the rank-update case.
- **QR** is **left-looking** column-Householder by default (`method = :column`), with a
  **multifrontal** path (`method = :frontal`) that assembles dense frontal matrices for flop-rich
  problems — the same two-tier structure SuiteSparseQR uses. Pass `method = :auto` to let `qr`
  pick per matrix by predicted flop density (the QR guide's recommendation).

Whichever path runs, every dense kernel goes through PureBLAS.jl — PureSparse never calls
OpenBLAS/LAPACK directly. See [Provenance & Licensing](provenance.md) for the clean-room policy
(CHOLMOD/SuiteSparse source is never read; the design derives only from published papers and the
MIT-licensed faer).
