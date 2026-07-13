# PureSparse.jl

A pure-Julia supernodal sparse Cholesky/LDLᵀ solver, part of the **Pure Julia Ecosystem** —
pure-Julia replacements for Julia's non-Julia default libraries (siblings:
[PureBLAS.jl](https://github.com/el-oso/PureBLAS.jl),
[PureFFT.jl](https://github.com/el-oso/PureFFT.jl)). PureSparse replaces SuiteSparse's
CHOLMOD: fill-reducing AMD ordering, supernodal LLᵀ (SPD), supernodal + simplicial LDLᵀ
(symmetric quasi-definite, for interior-point KKT systems), rank-k update/downdate, and a
GPU backend — all dense per-supernode work goes through
[PureBLAS.jl](https://github.com/el-oso/PureBLAS.jl).

**Clean-room provenance.** CHOLMOD's Supernodal/Modify modules are GPL. PureSparse's
design and code derive only from published academic papers and independent reasoning —
CHOLMOD source is never read, in any form. See [Provenance & Licensing](provenance.md).

**Status: Milestone 1** (AMD + symbolic analysis + supernodal LLᵀ + solve) — core
factorization and solve are implemented and tested against dense `BigFloat` and CHOLMOD
oracles. The M1 wall-time gate (PureSparse+PureBLAS faster than CHOLMOD+OpenBLAS on at
least half a matrix set) is not yet met; see [Benchmarking](benchmarking.md) for the real
numbers and the diagnosed remaining gap. LDLᵀ/update-downdate (M2) and GPU (M3) are not
yet implemented.

```julia
using PureSparse, SparseArrays

A = sprand(1000, 1000, 0.005)
A = A + A' + 1000I   # symmetric, diagonally dominant -> SPD

F = PureSparse.cholesky(A)        # symbolic analysis + numeric factorization
x = F \ randn(1000)               # solve

# Refactorize the same sparsity pattern with new values (analyze once, factor many times —
# the workload interior-point optimizers need), zero allocations after warmup:
A2 = SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, A.nzval .+ 1e-3 .* randn(nnz(A)))
PureSparse.cholesky!(F, A2)
```

See the [Guide](guide.md) for the full workflow (custom orderings, split solves,
refactorization) and the [API Reference](api.md) for every exported symbol.
