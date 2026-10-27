# Contributing to PureSparse.jl

PureSparse is a pure-Julia sparse direct solver (AMD/COLAMD ordering, supernodal and
multifrontal Cholesky and LDLᵀ, sparse QR, GPU backends) and a member of the Pure Julia
ecosystem: pure-Julia replacements for Julia's non-Julia default libraries. Its siblings are
PureBLAS.jl (dense BLAS/LAPACK) and PureFFT.jl (FFT). PureSparse replaces the CHOLMOD and
SuiteSparseQR roles.

Read [`docs/design.md`](docs/design.md) for the architecture and [`ROADMAP.md`](ROADMAP.md)
for milestone status. The requirements below are numbered so that code and documentation
can cite them (for example "CONTRIBUTING.md req 5").

## Requirements

1. **Clean-room provenance.** CHOLMOD's Supernodal and Modify modules, and SuiteSparse in
   general (including SuiteSparseQR), are GPL. Design and code must derive only from
   published papers and independent reasoning. Never read CHOLMOD or SuiteSparse source
   code in any form: not on GitHub, not through search snippets, not from memory of source
   text, and not through a third-party port derived from that source. Do not reuse a
   SuiteSparse identifier, struct field name or numeric constant unless it is independently
   derivable. If a name or constant happens to match a CHOLMOD default, treat it as a
   defect and rename or re-derive it. Every name and constant must be traceable to a paper
   citation, a user-guide citation, or an in-document derivation. Comparing against
   CHOLMOD's *output* through `SparseArrays` (as a test oracle or benchmark baseline) is
   fine; only the source is off limits. See [`docs/design.md`](docs/design.md) §11 and
   [`docs/src/provenance.md`](docs/src/provenance.md).

2. **Performance gate: wall time, not GFlops.** The gate is
   `median_seconds(PureSparse+PureBLAS) < median_seconds(CHOLMOD+OpenBLAS)`, measured both
   with each library's own ordering and with an identical permutation (`GivenOrdering`).
   The same-permutation arm isolates factorization throughput from ordering quality and is
   part of the gate. GFlops is a secondary diagnostic only, because a worse ordering with
   higher fill inflates it (design.md §9.3 D2).

3. **Generic over `T<:Number` and `Ti<:Integer` on hot paths.** One implementation, no
   per-type duplication, so that automatic differentiation can trace through it. Float64 is
   the tuned path (PureBLAS's fast path); other element types are correct but generic.

4. **Trim-compatible.** `juliac --trim` must build a factor-and-solve entry point. No
   runtime `eval` or `invokelatest`, no `Vector{Any}` on hot paths, no runtime CPU-feature
   detection. Tuning constants (`src/tuning.jl`) are compile-time constants backed by
   Preferences.jl: overridable, not detected at runtime.

5. **Zero allocations after `symbolic()`.** `cholesky!`, `ldlt!`, `qr!` and `solve!` on an
   existing factor must not allocate. The authoritative gate is StrictModeTest's
   `@test_noalloc` (AllocCheck, every path); StrictMode's own `noalloc` verdicts are
   heuristics that only warn. `@allocated == 0` after warmup checks the executed path. The
   suite runs with StrictMode checks enabled (the default, required by StrictModeTest);
   disabling them only removes code. See design.md §9.1 point 7.

6. **Compile-time contracts and runtime checks are separate layers.** TypeContracts.jl
   interface contracts (`src/contracts.jl`) exist only at precompile time and are removed by
   the trimmer. Runtime pre- and postconditions live in the StrictMode.jl layer
   (`src/strict.jl`), gated behind `StrictMode.checks_enabled()`. Do not mix the two
   (design.md §9.1 D6).

7. **Analyze once, factorize many times.** This is the organizing principle of the API
   (design.md §1.2): the target users, interior-point optimizers, refactorize the same
   sparsity pattern hundreds of times. `Symbolic` is immutable and shared by reference;
   `cholesky!` and `ldlt!` on an existing factor never recompute the ordering, elimination
   tree or supernodes.

8. **LDLᵀ is fixed-pivot with signed regularization**, in the style of QDLDL and Clarabel,
   not dynamic Bunch–Kaufman pivoting (design.md §5.1). General indefinite systems with 2×2
   pivots are an explicit non-goal (§1.1). Inertia `(n_pos, n_neg, n_zero)` is tracked in
   `FactorStats` so that downstream solvers can run their own regularization loop.

9. **Dense work goes through PureBLAS.** Per-supernode and per-front dense operations
   (`potrf!`, `trsm!`, `syrk!`, `syr2k!`, `gemm!`) use PureBLAS.jl. Do not reimplement
   dense kernels and do not call OpenBLAS or LAPACK directly from `src/`. OpenBLAS appears
   only under `benchmark/`, as a comparison backend.

## Code and documentation conventions

- Comments and docstrings state what is true of the code now: invariants, constraints and
  measured facts. History belongs in the commit log. Do not reference plans, chunk IDs or
  work sessions.
- Use `isnothing(x)` and `!isnothing(x)`, not `=== nothing`.
- American spelling.
- Where an interface accepts `AbstractArray`, it must work for arbitrary axes and wrapper
  types, or declare `Base.require_one_based_indexing`. Do not add `@inbounds` without a
  measurement showing the bounds check matters and a test that runs under
  `--check-bounds=yes`.
- The design documents are a contract. If a change would skip, replace or reinterpret a
  requirement in `docs/design*.md`, raise it in the issue or pull request before
  implementing it.
- For the hardest published algorithms (AMD's approximate-degree bookkeeping, the
  left-looking scheduler, Davis–Hager update recurrences), implement from the paper's exact
  published form. Approximate reimplementation from memory risks both correctness bugs and
  requirement 1 violations.
- No Python in tooling, tests or benchmarks. Use a native library through `ccall` or a
  command-line tool as a subprocess.

## Testing

Tests use ReTestItems. Each `@testitem` is self-contained and can be run on its own.

```julia
using ReTestItems, PureSparse
runtests(PureSparse)                      # everything
runtests(PureSparse; name = "some item")  # one item, matched by name
```

Iterate with a single item or a name filter; run the whole suite before opening a pull
request or after changing shared core code.

Correctness oracles:

- Dense `BigFloat` factorization on small and medium matrices.
- Residual gates on the SuiteSparse Matrix Collection zoo.
- CHOLMOD and SuiteSparseQR *output*, through `SparseArrays`, as a black-box cross-check
  (never their source; requirement 1).

The matrix-zoo downloader uses a lock file and an atomic temporary-file-then-rename so that
parallel ReTestItems workers cannot corrupt the cache (design.md §9.1 D7). See design.md §9
for the full test-layer breakdown.

## Benchmarking

Methodology, shared with PureBLAS:

- Chairmarks.jl, single thread, on a host with a fixed CPU frequency (performance governor,
  turbo disabled).
- Call `@noinline` wrapper functions with concrete argument types, not closures, and repeat
  in-place reps.
- Report the **median**, not the minimum.
- Run on the benchmark host from a clean checkout of the commit under test.
- Write results to JSON first. Plots are regenerated from the saved JSON only, never from a
  live run.
- Benchmark plots are **violin plots**, not bar charts: they show the per-sample
  distribution from the saved JSON, not only a summary.
- PkgBenchmark.jl supplements these for commit-to-commit regression checks.

The comparison matrix has four arms (design.md §9.3):

| Arm | Status |
|---|---|
| PureSparse + PureBLAS | measured |
| PureSparse + OpenBLAS | measured |
| CHOLMOD + OpenBLAS | measured |
| CHOLMOD + PureBLAS | not available |

CHOLMOD + PureBLAS is not available because PureBLAS cannot forward the BLAS interface
(`lbt_forward`) from a live Julia process (design.md §9.3 D1).
