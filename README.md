# PureSparse.jl

A pure-Julia supernodal sparse Cholesky/LDLᵀ solver, part of the **Pure Julia Ecosystem** —
pure-Julia replacements for Julia's non-Julia default libraries (siblings:
[PureBLAS.jl](https://github.com/el-oso/PureBLAS.jl),
[PureFFT.jl](https://github.com/el-oso/PureFFT.jl)). PureSparse replaces SuiteSparse's
CHOLMOD. It is a **clean-room reimplementation** — design and code are derived only from
published academic papers, never from CHOLMOD's source (see
[`docs/design.md` §11](docs/design.md) for the full provenance policy).

Dense per-supernode work (`potrf!`/`trsm!`/`syrk!`/`syr2k!`/`gemm!`) runs entirely through
[PureBLAS.jl](https://github.com/el-oso/PureBLAS.jl) — no OpenBLAS, no LAPACK.

**Status:** Milestone 1 in progress (AMD ordering + symbolic analysis + supernodal LLᵀ +
solve). See [`ROADMAP.md`](ROADMAP.md) for milestone status and
[`docs/design.md`](docs/design.md) for the full design document.

## Scope

- Fill-reducing ordering: Approximate Minimum Degree (AMD), pure Julia from scratch, behind
  an extensible ordering interface (nested dissection / METIS-style / user-supplied
  permutations can plug in later).
- Supernodal LLᵀ factorization for SPD systems.
- Supernodal + simplicial LDLᵀ for symmetric quasi-definite (SQD) systems — the primary
  downstream target is **interior-point optimizers**, which factor the same KKT sparsity
  pattern every iteration ("analyze once, factorize many times").
- Rank-k update/downdate (Davis–Hager) on the simplicial representation.
- GPU backend (CUDA.jl weak-dependency extension, in-package).

## Native API (sketch — M1 in progress)

```julia
using PureSparse

S = symbolic(A; ordering = AMDOrdering())    # analysis, once per sparsity pattern
F = cholesky(A; ordering)                     # symbolic + numeric
cholesky!(F, A2)                              # refactor, same pattern, zero allocations
x = F \ b
```

## Develop & test

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=test test/runtests.jl                      # full suite
```

Correctness oracle and performance gate: Julia's built-in `SparseArrays`/`SuiteSparse`
CHOLMOD wrapper (black-box output comparison only — never CHOLMOD's source). See
[`docs/design.md` §9](docs/design.md) for the full testing/benchmarking methodology and
[`CLAUDE.md`](CLAUDE.md) for the project's hard requirements.
