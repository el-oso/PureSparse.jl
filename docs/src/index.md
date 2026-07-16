# PureSparse.jl

A pure-Julia sparse direct solver, part of the **Pure Julia Ecosystem** —
pure-Julia replacements for Julia's non-Julia default libraries (siblings:
[PureBLAS.jl](https://github.com/el-oso/PureBLAS.jl),
[PureFFT.jl](https://github.com/el-oso/PureFFT.jl)). PureSparse replaces SuiteSparse's
CHOLMOD **and** SuiteSparseQR: fill-reducing AMD/COLAMD ordering, supernodal LLᵀ (SPD),
supernodal + simplicial LDLᵀ (symmetric quasi-definite, for interior-point KKT systems),
rank-`k` update/downdate, and left-looking + multifrontal sparse QR (least-squares,
minimum-norm, rank-revealing) — all dense per-supernode/per-front work goes through
[PureBLAS.jl](https://github.com/el-oso/PureBLAS.jl). New here? Start with the
[Tutorial](tutorial.md).

**Clean-room provenance.** CHOLMOD's Supernodal/Modify modules are GPL. PureSparse's
design and code derive only from published academic papers and independent reasoning —
CHOLMOD source is never read, in any form. See [Provenance & Licensing](provenance.md).

**Status: M1 (Cholesky), M2 (LDLᵀ + update/downdate), M4 (drop-in), and M5 (sparse QR)
are closed** — each against its own non-negotiable wall-time gate (median PureSparse+PureBLAS
faster than the SuiteSparse+OpenBLAS baseline, own-ordering **and** under an identical
permutation), verified on clock-locked hardware and cross-checked against dense `BigFloat`
and SuiteSparse output oracles. See [Benchmarking](benchmarking.md) for the numbers. M6 (GPU,
CUDA weak-dep extension — renumbered from the original M3) is the remaining milestone.

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
