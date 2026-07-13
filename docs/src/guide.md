# Guide

## Input convention

`symbolic`/`cholesky` read `A::SparseMatrixCSC` via its **lower triangle only**
(`i >= j`, matching `Symmetric(A, :L)` semantics) — any stored upper-triangle entries are
ignored, not an error, so a fully-stored symmetric matrix works too (just redundantly).

## Analyze once, factorize many times

The core API is split into a symbolic analysis (`Symbolic`, pattern-only — permutation,
elimination tree, supernode partition) and a numeric factor (`SupernodalFactor`, values).
This split is the primary organizing principle (see `CLAUDE.md` requirement 7): an
interior-point optimizer factors the same sparsity pattern hundreds of times per solve,
and only the *numeric* factorization needs to repeat.

```julia
using PureSparse, SparseArrays, LinearAlgebra

A = sprand(1000, 1000, 0.005); A = A + A' + 1000I

S = PureSparse.symbolic(A)          # ordering + etree + supernodes; allocates once
F = PureSparse.cholesky(S, A)       # numeric factorization into a fresh factor
PureSparse.issuccess(F)             # true iff every pivot was SPD

A2 = SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, A.nzval .+ 1e-3 .* randn(nnz(A)))
PureSparse.cholesky!(F, A2)         # same sparsity PATTERN, new values — refactor in
                                     # place, zero allocations after warmup
```

`PureSparse.cholesky(A)` (without a pre-built `S`) is the one-shot convenience form:
symbolic analysis + numeric factorization together.

## Solving

```julia
b = randn(1000)
x = F \ b                 # allocating convenience wrapper
PureSparse.solve!(x, F, b)  # in-place

# Split solves (forward/backward triangular solves in FACTOR ordering, i.e. `y` is
# already permuted — see `solve!`'s source for the full permute/solve/unpermute sequence):
PureSparse.solve_L!(y, F)
PureSparse.solve_Lt!(y, F)
```

Multi-RHS (`b::AbstractMatrix`) is supported throughout.

## Non-SPD input

`cholesky!` never throws on a non-positive-definite pivot — it sets `F.ok = false` (query
via `issuccess(F)`) and records the failing column in `F.stats.fail_col`. `\`/`solve!`
still throw `PosDefException`-equivalent semantics are the caller's responsibility to
check via `issuccess` before using `F`'s solve.

## Orderings

```julia
abstract type AbstractOrdering end   # implement `order(alg, n, colptr, rowval)` for a new one
```

- `AMDOrdering(; dense_mult=10.0, aggressive=true)` — the default. Approximate Minimum
  Degree (Amestoy–Davis–Duff 1996); `aggressive` toggles aggressive element absorption.
- `NaturalOrdering()` — identity permutation, no reordering.
- `GivenOrdering(perm)` — use a caller-supplied permutation directly. The escape hatch for
  external orderings (nested dissection, `METIS.jl`) and how to feed an externally-computed
  permutation (e.g. for benchmarking against an identical permutation — see
  [Benchmarking](benchmarking.md)).

```julia
S = PureSparse.symbolic(A; ordering = PureSparse.GivenOrdering(my_perm))
```

## Testing

```bash
julia --project=test test/runtests.jl
# one item:
julia --project=test -e 'using ReTestItems, PureSparse; runtests(PureSparse; name="...")'
```
