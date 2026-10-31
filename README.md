# PureSparse.jl

[![CI](https://github.com/el-oso/PureSparse.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/el-oso/PureSparse.jl/actions/workflows/CI.yml)
[![Coverage](https://coveralls.io/repos/github/el-oso/PureSparse.jl/badge.svg?branch=master)](https://coveralls.io/github/el-oso/PureSparse.jl?branch=master)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://el-oso.github.io/PureSparse.jl/dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A pure-Julia sparse direct solver: sparse Cholesky, LDLᵀ and QR, replacing SuiteSparse's
CHOLMOD and SuiteSparseQR. It is part of the **Pure Julia Ecosystem**, pure-Julia
replacements for Julia's non-Julia default libraries (siblings:
[PureBLAS.jl](https://github.com/el-oso/PureBLAS.jl),
[PureFFT.jl](https://github.com/el-oso/PureFFT.jl)). All dense work runs through PureBLAS.jl.

PureSparse is a **clean-room implementation**: its design and code derive only from
published papers, never from SuiteSparse source. See
[Provenance](https://el-oso.github.io/PureSparse.jl/dev/provenance/) and
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Features

- AMD and COLAMD fill-reducing orderings, or a user-supplied permutation.
- Supernodal LLᵀ for symmetric positive definite matrices.
- Supernodal and simplicial LDLᵀ for symmetric quasi-definite matrices, such as
  interior-point KKT systems, with fixed pivots, signed regularization and inertia counts.
- Rank-k update and downdate of a simplicial factor.
- Sparse QR (left-looking and multifrontal) for least squares, minimum-norm and
  rank-deficient problems.
- An opt-in drop-in (`PureSparse.activate!()`, then restart Julia) that routes
  `LinearAlgebra.cholesky` and `ldlt` on sparse matrices to PureSparse instead of CHOLMOD.
- GPU factorization through package extensions: CUDA, with AMD ROCm in progress.
- Analyze once, factorize many times: refactoring a matrix with the same sparsity pattern
  reuses the symbolic analysis, and `cholesky!`, `ldlt!`, `qr!` and `solve!` allocate
  nothing.
- Generic over element types (`Float32`, ForwardDiff duals, ...) and compatible with
  `juliac --trim`.

Status by milestone, with open work: [`ROADMAP.md`](ROADMAP.md). Performance against
CHOLMOD and SuiteSparseQR:
[Benchmarking](https://el-oso.github.io/PureSparse.jl/dev/benchmarking/).

## Installation

PureSparse requires Julia 1.13 or later. It is not yet registered; install it from GitHub
(PureBLAS.jl is fetched automatically):

```julia
using Pkg
Pkg.add(url = "https://github.com/el-oso/PureSparse.jl")
```

## Example

```julia
using PureSparse, SparseArrays, LinearAlgebra

A = sprand(1000, 1000, 0.005)
A = A + A' + 1000I                    # symmetric positive definite
b = randn(1000)

F = PureSparse.cholesky(A)            # ordering, symbolic analysis and factorization
x = F \ b

A2 = A + 0.5I                         # new values, same sparsity pattern
PureSparse.cholesky!(F, A2)           # refactor without allocating
solve!(x, F, b)                       # solve in place without allocating
```

`PureSparse.ldlt` and `PureSparse.qr(A; ordering = COLAMDOrdering())` follow the same
pattern. The names are qualified because `LinearAlgebra` exports `cholesky`, `ldlt` and
`qr` too. See the [Guide](https://el-oso.github.io/PureSparse.jl/dev/guide/).

## Contributing

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) first: it lists the requirements every change
must meet, how to run the tests and how to benchmark.

```bash
git clone https://github.com/el-oso/PureSparse.jl
cd PureSparse.jl
julia --project=test -e 'using Pkg; Pkg.instantiate()'
julia --project=test --check-bounds=yes test/runtests.jl
```
