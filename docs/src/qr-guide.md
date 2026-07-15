# Sparse QR Guide

PureSparse's sparse QR (milestone M5) factors a rectangular `A::SparseMatrixCSC` as
`A·P = Q·R` with Householder reflectors, for least-squares solves (`m > n`),
minimum-norm solves (`m < n`), and rank-revealing robustness on ill-conditioned
problems. It is a clean-room reimplementation of the algorithm family SuiteSparseQR
implements [spqr2011](@cite) — see [Provenance & Licensing](provenance.md) for the
policy; the multifrontal numeric core additionally follows the MIT-licensed
[faer](@cite) (whose source, unlike SuiteSparse's, is freely readable).

The recommended ordering is `COLAMDOrdering()` — COLAMD [colamd2004](@cite)
[larimore1998](@cite) orders the columns of `A` directly, without ever forming
`AᵀA` (the symbolic phase then computes R's row counts on the same implicit pattern
[gnp1992](@cite)). `AMDOrdering()`/`NaturalOrdering()`/`GivenOrdering(perm)` from the
[Guide](guide.md) work too (AMD [amd1996](@cite) runs on the explicit pattern of
`AᵀA`). There is no default `ordering` yet — pass one explicitly.

## Least squares (m > n)

`qr` + `\` computes the least-squares solution `argmin ‖b − A·x‖₂`:

```julia
using PureSparse, SparseArrays, LinearAlgebra

A = sprand(200, 50, 0.05) + sparse(1:50, 1:50, 1.0, 200, 50)   # 200×50, full column rank
b = randn(200)

F = PureSparse.qr(A; ordering = PureSparse.COLAMDOrdering())
PureSparse.issuccess(F)            # true
(F.stats.rank, F.stats.n_dead)     # (50, 0) — full rank, no dropped columns

x = F \ b                          # least-squares solution; solve!(x, F, b) is the
                                   # in-place form, ldiv!(x, F, b) the stdlib spelling
norm(A' * (b - A * x))             # ≈ 1e-14 — normal-equations residual Aᵀ(b−Ax) ≈ 0
norm(x - Matrix(A) \ b) / norm(x)  # ≈ 1e-15 — matches the dense LS solution
```

Multi-RHS (`b::AbstractMatrix`) is supported. By default `qr` also peels *column
singletons* first (columns whose single surviving nonzero lets them be eliminated
without any Householder work — LP-shaped matrices are often almost entirely
singletons; the technique is SuiteSparseQR's [spqr2011](@cite), reimplemented from the
paper). `singletons = false` switches the peeling off — see the refactor section below
for why that matters.

## Minimum-norm solve (m < n)

For an underdetermined system, `F \ b` returns a *basic* solution (some unknowns
pinned to zero). To get the minimum-2-norm solution instead, factor **`Aᵀ`** (tall)
and call `solve_minnorm!`:

```julia
B = sprand(50, 200, 0.05) + sparse(1:50, 1:50, 1.0, 50, 200)   # 50×200, full row rank
c = randn(50)

Ft = PureSparse.qr(sparse(B'); ordering = PureSparse.COLAMDOrdering(), tol = 0)
x = zeros(200)
PureSparse.solve_minnorm!(x, Ft, c)

norm(B * x - c)                        # ≈ 5e-15 — exact solution of B·x = c
norm(x) - norm(pinv(Matrix(B)) * c)    # ≈ -9e-16 — same norm as the pseudoinverse solution
```

Two requirements, both enforced with a clear error rather than a wrong answer: `Ft`
must be the factorization of the **transpose**, and it must be **full rank**
(`Ft.stats.n_dead == 0` — the minimum-norm formula has no analogue of the basic
solution's dropped-column convention). `tol = 0` above disables rank detection for a
matrix you know is well-conditioned; leave it on (the default) otherwise and check
`n_dead` before solving.

## Choosing a method: `:column`, `:frontal`, `:auto`

`qr(A; method = ...)` selects the factorization architecture:

- **`:column`** (the default) — the left-looking column-Householder path (M5a).
  One sparse reflector per column, no dense blocks. Fine for small problems and for
  matrices whose `R` stays very sparse; generic over `T<:Real`; the only path that
  exploits singleton peeling.
- **`:frontal`** — the multifrontal path (M5b), architecture per SuiteSparseQR
  [spqr2011](@cite) with a numeric core ported from [faer](@cite): supernodes of the
  Cholesky factor of `AᵀA` become dense frontal matrices processed with BLAS-3
  kernels (PureBLAS's compact-WY apply). Wins decisively on flop-rich problems —
  on a 7000×4000 benchmark it is ~36–74× faster than `:column` (see
  [Benchmarking](benchmarking.md)). Float64-tuned; other element types currently
  fall back to `:column`.
- **`:auto`** — picks per matrix by the ratio `sym.flops / sym.nnzR` (predicted
  factorization flops per stored entry of R, both already computed by the symbolic
  phase, so the decision costs nothing extra). Above `QR_AUTO_METHOD_RATIO = 40.0`
  (a Preferences-overridable tunable in `tuning.jl`, threshold shared with
  [faer](@cite)'s analogous simplicial-vs-supernodal dispatch and re-verified on
  PureSparse's own gate set: every `:column`-winning matrix sat at ratio ≤ 7, every
  `:frontal`-winning one at ratio ≥ 863) it routes to `:frontal`.

```julia
S = PureSparse.symbolic_qr(A; ordering = PureSparse.COLAMDOrdering())
S.flops / S.nnzR       # 331.2 — flop-rich, well above the 40.0 threshold

Ff = PureSparse.qr(A; ordering = PureSparse.COLAMDOrdering(), method = :auto)
typeof(Ff)             # QRFrontFactor — :auto routed to the multifrontal path
xf = Ff \ b
norm(xf - x) / norm(x) # ≈ 1e-15 — same solution either way
```

Unless you know your problem is tiny or singleton-dominated, `method = :auto` is the
right call. `PureSparse.qr_frontal(A; ordering, tol)` is the direct one-shot entry to
the frontal path.

## Rank deficiency: `dropped_norm` and `issuccess`

PureSparse's QR does **not** do column pivoting (it would invalidate the symbolic
analysis and destroy sparsity — the standard trade-off for sparse QR, see
[sparsesurvey2016](@cite) §7.4). Instead it uses Heath's threshold test: at column
`k`, a pivot whose remaining column norm is `≤ τ` is declared *dead* and dropped,
following the Foster–Davis phase-1 strategy — and, deliberately, the factorization
**reports** what it dropped rather than pretending the answer is exact:

```julia
Ard = sparse([A[:, 1:48] A[:, 47] + A[:, 48] A[:, 47] - A[:, 48]])  # 2 dependent columns

Frd = PureSparse.qr(Ard; ordering = PureSparse.COLAMDOrdering())
PureSparse.issuccess(Frd)              # true — completing with dropped columns is
                                       # a *reported* outcome, not a failure
(Frd.stats.rank, Frd.stats.n_dead)     # (48, 2) — detected rank, dropped columns
Frd.stats.dropped_norm                 # 1.008 — ‖dropped mass‖_F, the error certificate

xrd = Frd \ b                          # basic solution: dead columns' unknowns are 0
count(iszero, xrd)                     # 2
```

`dropped_norm` is the honesty certificate: the Frobenius norm of everything the
dead-column drop discarded. When it is small relative to `‖A‖`, the basic solution is
as good as an exact rank-revealing one. When it is **not** small (as above — the
dependent columns carried real mass), the certificate tells you so, and the right
responses are (i) Tikhonov regularization — append `√γ·I` rows and refactor — or
(ii) the augmented-system `ldlt` route from the
[Interior-Point Guide](ipm-guide.md). This drop-and-report policy is the least
accurate of the published rank-handling strategies but the only one compatible with a
static pattern and zero-allocation refactoring; the exact alternatives (Heath's
Givens row-zeroing, second-phase null-space methods) are deliberate non-goals — see
`docs/design_qr.md` §5 for the full policy discussion.

`τ` defaults to `8 · max(m,n) · eps(T) · max_j ‖A[:,j]‖₂`; pass `tol` to override,
`tol ≤ 0` to disable numeric rank detection entirely.

## Analyze once, factorize many times

Like `cholesky!`/`ldlt!` in the [Guide](guide.md), `qr!` refactorizes in place when
only the *values* of `A` change and the sparsity pattern stays fixed — zero
allocations after warmup (gated in the test suite):

```julia
# a refactorable column factor must skip singleton peeling: a singleton set chosen
# for A's values is invalid for A2's, so qr! rejects factors built with it
F0 = PureSparse.qr(A; ordering = PureSparse.COLAMDOrdering(), singletons = false)

A2 = SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, A.nzval .* (1 .+ 0.01 .* randn(nnz(A))))
PureSparse.qr!(F0, A2)                 # same PATTERN, new values — in place
norm(A2' * (b - A2 * (F0 \ b)))        # ≈ 1e-14

# the frontal path refactors the same way (it never carries singletons):
Ff0 = PureSparse.qr_frontal(A; ordering = PureSparse.COLAMDOrdering())
PureSparse.qr!(Ff0, A2)
@allocated PureSparse.qr!(Ff0, A2)     # 0
```

`symbolic_qr(A; ordering)` exposes the analysis phase on its own (pattern-only:
ordering, column elimination tree, R/V row counts, workspace bounds) — useful for
inspecting `S.flops`/`S.nnzR` before committing to a factorization, as in the
`:auto` example above.

## Lower-level building blocks

The solve phase is exposed piecewise, mirroring the split-solve convention of the
Cholesky API: `apply_Qt!(y, F)` / `apply_Q!(y, F)` (apply `Qᵀ`/`Q` in the factor's
physical row space), `solve_R!(x, F, c)` / `solve_Rt!(x, F, c)` (back/forward
substitution with `R`, dead rows yielding zeros), and `solve_minnorm!` as shown
above. `F \ b`, `solve!` and `ldiv!` compose these; see the docstrings in the
[API Reference](api.md) for the exact spaces each operates in.

## When *not* to use QR

QR is the robust tool, not the fastest one for every least-squares problem —
PureSparse ships all three of the standard alternatives
([sparsesurvey2016](@cite) §7.5):

| Situation | Recommended PureSparse tool |
|---|---|
| Well-conditioned LS, no rank worries | normal equations: `cholesky(AᵀA)` — fastest, least memory |
| Moderately ill-conditioned, or dense rows in `A` | augmented system `[αI A; Aᵀ 0]` via `ldlt` + `refine!` (the choice of `α` is the caller's — see the [Interior-Point Guide](ipm-guide.md) for the `ldlt` workflow) |
| Ill-conditioned, rank-deficient, or robustness required | `qr` (this page) |
