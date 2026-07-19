# Developer Guide

This page is for people working **on** PureSparse (contributors), not people using it. It covers
the repository layout, how to build and test, and the non-negotiable disciplines — clean-room
provenance, the wall-time performance gate, trim compatibility, and the StrictMode guarantee
gates — that every change must respect. The authoritative, always-current version of these rules
lives in the repository's `CLAUDE.md`; the full architecture is in `docs/design.md` (and
`docs/design_qr*.md`, `docs/design_gpu.md` for the QR and GPU milestones).

## Repository layout

| Path | What lives there |
|---|---|
| `src/ordering/` | Fill-reducing orderings: AMD (`amd.jl`), COLAMD (`colamd.jl`), the AᵀA/star-pattern builders |
| `src/symbolic/` | Elimination tree, postorder, column counts, supernode amalgamation — the analyze phase |
| `src/numeric/` | Supernodal LLᵀ (`llt.jl`), LDLᵀ (`ldlt.jl`), the panel solve (`solve.jl`) |
| `src/simplicial/` | Simplicial LDLᵀ + rank-1 update/downdate |
| `src/qr/` | Sparse QR: `symbolic.jl`/`numeric.jl` (`:column`, left-looking) and `frontal_*.jl` (`:frontal`, multifrontal-WY), plus `solve.jl`/`frontal_solve.jl` |
| `src/contracts.jl` | TypeContracts.jl compile-time interface contracts (eliminated by the trimmer) |
| `src/strict.jl` | The runtime StrictMode check layer (input-shape / finiteness guards), gated by `checks_enabled()` |
| `src/tuning.jl` | Compile-time, Preferences-backed tuning constants (no runtime CPU detection — trim rule) |
| `ext/gpu_shared.jl` + `ext/PureSparse{CUDA,AMDGPU}Ext.jl` | The GPU backend: one backend-generic KernelAbstractions engine + a thin per-vendor weak-dep ext (CUDA, ROCm) |
| `juliac/` | The `juliac --trim=safe` smoke build (`build.jl`, `entry.jl`) |
| `test/` | ReTestItems `@testitem`s, one file per module |
| `benchmark/` | Chairmarks harness + the GPU probes; results saved to `benchmark/results/*.json` (gitignored) |
| `docs/design*.md` | The design documents (in-repo, not published); every large change starts here |

Dense per-supernode / per-front kernels (`potrf!`, `trsm!`, `syrk!`, `gemm!`, `geqrf!`, WY apply)
are called **exclusively through [PureBLAS.jl](https://github.com/el-oso/PureBLAS.jl)** — never
reimplemented in `src/`, never routed to OpenBLAS/LAPACK directly.

## Building and testing

PureSparse uses **[ReTestItems.jl](https://github.com/JuliaTesting/ReTestItems.jl)**; each test is
a self-contained `@testitem` that can be run individually.

```julia
# full suite (the pre-merge / release gate) — from the package project:
julia --project=. -e 'using Pkg; Pkg.test()'

# a single item, while iterating (filtered by name):
julia --project=test -e 'using ReTestItems, PureSparse; runtests(PureSparse; name="qr! refactor")'
```

Run the whole suite only as a **gate** — iterate with a filtered subset (`name=`). The suite is
CPU-only; the GPU backend is exercised separately (see *GPU backend* below).

!!! warning "Use `Pkg.test()`, not a bare `--project=test` run"
    `test/Manifest.toml` is gitignored and can drift from the package's compat bounds. A direct
    `julia --project=test …` activates that (possibly stale) manifest and can fail with
    *"can not merge projects"*. `Pkg.test()` re-resolves a fresh environment and is the supported
    entry point; run `Pkg.update(...)` on `--project=test` if you want the local test manifest
    synced for filtered runs.

**Correctness oracles.** Every numeric path is checked three ways: a dense `BigFloat`
factorization on small/medium matrices (exact comparison), residual gates on the SuiteSparse
Matrix Collection zoo, and `SparseArrays`' own factorization (CHOLMOD / SuiteSparseQR) as a
**black-box** cross-check of the *output* (never its source — see *Clean-room provenance*).

**Zero-allocation gate.** `cholesky!`/`ldlt!`/`qr!`/`solve!` on an existing factor must be
`@allocated == 0`. This is tested with StrictMode checks **disabled** (the runtime check layer in
`src/strict.jl` allocates by design and must not be conflated with the gate — see below).

## Clean-room provenance (absolute)

**This is the rule most easily violated by accident.** CHOLMOD's Supernodal/Modify modules and
SuiteSparseQR are GPL. Design and all code derive **only** from published papers, official
user-guide documentation, and independent reasoning. Never read CHOLMOD/SuiteSparse source in any
form — not on GitHub, not from search snippets, not from model recall of source text, not via a
third-party port derived from it. Every identifier and numeric constant must survive *"where did
this come from?"* with a paper/user-guide citation or an in-repo derivation — a name or number
that "happens to match" CHOLMOD's is a defect (this caught two real coincidences during review).
Black-box comparison against CHOLMOD/SPQR **output** (via `SparseArrays`) is fine and used
throughout. `faer` (Rust, MIT) and `PureKLU.jl` (MIT) are readable references. See
[Provenance & Licensing](provenance.md) and `docs/design.md` §11.

## Performance discipline

- **The gate is wall-time, not GFlops.** `median_seconds(PureSparse+PureBLAS) <
  median_seconds(CHOLMOD+OpenBLAS)`, both under each solver's own ordering **and** under an
  identical `GivenOrdering` permutation (the latter isolates factorization throughput from
  ordering quality). GFlops is a secondary diagnostic only (it is gameable by a higher-fill
  ordering).
- **Analyze once, factorize many.** `Symbolic` is immutable and shared by reference; `cholesky!`/
  `ldlt!`/`qr!` on an existing factor never recompute ordering/etree/supernodes. This is the
  primary API principle (the target user — interior-point optimizers — refactorizes the same
  pattern hundreds of times per solve).
- **Generic over `T<:Number` / `Ti<:Integer` on hot paths** (one implementation, no per-type
  duplication, AD-traceable). `Float64` is the tuned path; other `T` are correct-but-generic.
- **Measure on a warm session.** Julia JIT-compiles per concrete-type specialization, so the first
  call pays a compile tax. Warm up, then benchmark with Chairmarks (single-thread pinned, median
  not min, locked clock). Results are saved to `benchmark/results/*.json` first; **plots (violins)
  regenerate from the saved JSON** — never re-run a benchmark to redraw a plot.

## Correctness & performance guarantees (StrictMode + TypeContracts)

Two **separate** mechanisms, easy to conflate:

- **TypeContracts.jl** (`src/contracts.jl`) — compile-time interface contracts, precompile-time
  only, eliminated by the trimmer. Never a runtime mechanism.
- **StrictMode.jl** (`src/strict.jl`) — a runtime pre/post-condition layer (shape, nnz, finiteness
  guards) gated behind `StrictMode.checks_enabled()`. These checks may allocate; they are for
  debugging, not the hot path, and are **off** in the perf/zero-alloc configuration.

On top of that, PureSparse runs **StrictMode `@assert_*` guarantee gates** in
`test/strictmode_guarantees_tests.jl`:

- `@assert_concurrency_safe cholesky/ldlt(sym, A)` — machine-proof that the allocating factor
  treats its `Symbolic` argument read-only, i.e. one immutable `Symbolic` is safe to share by
  reference across concurrent factorizations (the analyze-once thesis).
- `@assert_typestable solve!(...)` — concrete return type + no internal instability, all factor
  kinds.

These macros require checks **on** (`assert_enabled()` ⇒ `checks_enabled()`), which conflicts with
the checks-off zero-alloc gate — so they run in an **isolated subprocess** with its own
`[StrictMode] checks_enabled = true` preference plus `AllocCheck`/`JET`, and they target only
**check-free** functions (`solve!` and the constructors), so the analyzed body equals the shipped
one. When adding such a guarantee, keep this constraint in mind: asserting on a function that calls
`src/strict.jl` checks would analyze a checks-on body that is not what ships.

## Trim compatibility

PureSparse must build a factor-and-solve entry point under `juliac --trim=safe`: no runtime
`eval`/`invokelatest`, no `Vector{Any}` on hot paths, no runtime CPU-feature detection (tuning
constants are compile-time `Preferences`-backed consts). Two layers check this:

- The in-suite `TrimCheck.@validate` (`test/trim_tests.jl`) — a static reachability scan over
  **positional-argument** roots (`cholesky!`/`ldlt!`/`qr!`/`solve!`). It cannot express
  keyword-argument construction entries.
- The real end-to-end build:

  ```julia
  julia juliac/build.jl          # builds juliac/build/puresparse_smoke via --trim=safe
  ./juliac/build/puresparse_smoke # exit 0 + small residuals ⇒ pass
  ```

  Run this before a release — it is the only gate that exercises the actual smoke path in
  `juliac/entry.jl` (including keyword-argument entries the in-suite scan can't reach). Return
  types on the hot paths must be **concrete**; an abstract inferred return boxes the value and
  breaks trim resolution downstream.

## GPU backend

The GPU code is a set of **weak-dependency extensions** (`ext/`), one per vendor, sharing a single
backend-generic engine:

| File | Role |
|------|------|
| `ext/gpu_shared.jl` | The backend-generic engine — the pure KernelAbstractions kernels + `GPUSymbolic`/`gpu_symbolic` + the multifrontal numeric driver. Reaches the device only through a 2-function shim (`_dev_zeros`, `_dev_upload`) + `fill!`/`KA.synchronize`/`Array`. |
| `ext/PureSparseCUDAExt.jl` | NVIDIA: `using CUDA`; `_default_backend()=CUDABackend()`. Also includes the CUDA-only reference/vendor arms (`gpu_leftlooking_reference.jl`, `gpu_vendor_solve.jl` — cuSOLVER/cuBLAS). |
| `ext/PureSparseAMDGPUExt.jl` | AMD ROCm: `using AMDGPU`; `_default_backend()=ROCBackend()`. |

**One KA source, both vendors** — a per-backend ext supplies only `using CUDA`/`using AMDGPU` and its
`_default_backend()`; everything else is shared. Adding Intel oneAPI is a third ~15-line ext.

Status: **Cholesky + LDLᵀ run end-to-end on both CUDA and ROCm.** On CUDA/Float64 the pure kernels
match or beat cuSOLVER/cuBLAS on the crown fronts (see [Benchmarking](benchmarking.md); vendor libs are
reference arms only) and that path is perf-gated. The **ROCm path is correct but unoptimized** — FP64
matrix-core (MFMA) tuning for Instinct-class parts is deferred M8 work; a consumer iGPU is FP64-throttled
and used only as the portability/correctness canary. There is **no GPU QR** on any backend (the GPU-QR
milestone M7 was shelved — pure Householder can't beat vendor `geqrf`). GPU work is validated on a
machine with a device (`benchmark/gpu/gpu_mf_hybrid_test.jl` on CUDA, `amd_end2end_test.jl` on ROCm) —
the CPU test suite does not require one.

## Proposing a larger change

Non-trivial features follow the repository's design process, visible in `docs/design*.md`: write a
design document, subject it to **independent adversarial review** (a second model/reviewer, blind
where possible), fold every finding into a v2, and only then implement — task by task, each landing
with its oracle tests before the next. Milestones that turn on an unproven performance bet are
**measured first** with a cheap probe before any production code (the GPU-QR milestone, M7, was
shelved this way — its probes are in `benchmark/gpu/qr_*phase0.jl` and the verdict in
`docs/design_qr_gpu.md`). Delegate genuinely hard algorithmic pieces (AMD's degree bookkeeping,
Davis–Hager update recurrences) rather than approximating them from memory — approximation risks
both correctness bugs and clean-room violations.

## Cutting a release

PureSparse depends on other Pure-ecosystem packages (`PureBLAS`, `StrictMode`, `TypeContracts`),
so releases are **dependency-ordered**: those must be registered in General before PureSparse can
register. The typical flow: run the full suite and the `juliac` smoke build as gates, pin the
`[compat]` bounds to the versions you will register, tag `vX.Y.Z`, then register once the
dependency chain is in General.
