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

**Status:** Cholesky, LDLᵀ with update/downdate, the `LinearAlgebra` drop-in, sparse QR and
CUDA GPU factorization are complete. Each CPU milestone passed a wall-time gate: median
PureSparse+PureBLAS time below SuiteSparse+OpenBLAS, both with each library's own ordering
and with an identical permutation, measured on fixed-clock hardware. Results are
cross-checked against dense `BigFloat` factorizations and SuiteSparse's output. AMD ROCm
support is in progress. See [Benchmarking](benchmarking.md) for the numbers and `ROADMAP.md`
in the repository for open work.

PureSparse requires Julia 1.13 or later and is not yet registered:

```julia
using Pkg
Pkg.add(url = "https://github.com/el-oso/PureSparse.jl")
```

```julia
using PureSparse, SparseArrays, LinearAlgebra

A = sprand(1000, 1000, 0.005)
A = A + A' + 1000I                    # symmetric positive definite
b = randn(1000)

F = PureSparse.cholesky(A)            # ordering, symbolic analysis and factorization
x = F \ b

# Refactor the same sparsity pattern with new values (analyze once, factorize many times,
# the workload of interior-point optimizers). Neither call allocates.
A2 = A + 0.5I
PureSparse.cholesky!(F, A2)
solve!(x, F, b)
```

See the [Guide](guide.md) for the full workflow (custom orderings, split solves,
refactorization) and the [API Reference](api.md) for every exported symbol.
