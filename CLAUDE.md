# PureSparse.jl — agent guidelines

Project-specific REQUIREMENTS for anyone (human or agent) working on PureSparse. This is
a **multi-session, long-horizon** project — preserve knowledge here and in `ROADMAP.md`
(the canonical status + next steps), not just in a chat transcript. Full design:
`docs/design.md`.

PureSparse is a member of the **Pure Julia Ecosystem** ("Pure"): pure-Julia replacements
for Julia's non-Julia default libraries. PureBLAS.jl (dense BLAS/LAPACK) and PureFFT.jl
(FFT) are siblings. PureSparse replaces SuiteSparse's CHOLMOD (supernodal sparse
Cholesky/LDLᵀ).

## What PureSparse is (architecture)

A pure-Julia, statically-compilable (juliac/trim-compatible), allocation-free-after-setup
sparse symmetric solver: AMD ordering, supernodal LLᵀ (SPD), supernodal + simplicial LDLᵀ
(symmetric quasi-definite, for interior-point KKT systems), rank-k update/downdate, GPU
backend. Dense per-supernode work goes exclusively through **PureBLAS.jl**
(`potrf!`/`trsm!`/`syrk!`/`syr2k!`/`gemm!`) — never reimplement dense kernels, never call
OpenBLAS/LAPACK directly from `src/`. See `docs/design.md` §1–§8 for the full
architecture; `ROADMAP.md` for milestone status.

## Hard requirements (MUST follow)

1. **Clean-room provenance, absolute — this is the constraint most likely to be violated
   by accident.** CHOLMOD's Supernodal/Modify modules are GPL. Design and ALL code must
   derive only from published academic papers (`refs/linear_algebra/`, gitignored —
   papers only, extracted from the user's reference archive) and independent reasoning.
   **Never read CHOLMOD/SuiteSparse source code, in any form** — not on GitHub, not via
   search snippets, not from training-data recall of source text, not via a third-party
   port derived from that source. Also never reuse a SuiteSparse identifier name, struct
   field name, or numeric constant unless it is independently derivable — if a name or
   number "just happens" to match CHOLMOD's actual defaults, that is a defect (this
   happened twice in the v1→v2 design review: `maxcsize`/`maxesize` field names and the
   `0.8/0.1/0.05` amalgamation thresholds both silently matched CHOLMOD internals and had
   to be renamed/re-derived — see `docs/design.md` §0 B1/B2). Every name and constant
   must survive "where did this come from?" with a paper citation, a user-guide citation,
   or an in-document independent derivation. Black-box comparison against CHOLMOD's
   *output* (via `SparseArrays`/stdlib, as a test oracle or benchmark baseline) is fine —
   only the source is off-limits.
2. **Performance gate: wall-time, not GFlops (non-negotiable).** `median_seconds(PureSparse
   +PureBLAS) < median_seconds(CHOLMOD+OpenBLAS)`, both own-ordering AND under an
   identical (`GivenOrdering`) permutation — the latter isolates factorization throughput
   from ordering quality and is part of the gate, not a supplementary extra. GFlops is a
   secondary diagnostic only (gameable by a worse/higher-fill ordering — design §9.3 D2).
3. **Generic over `T<:Number`/`Ti<:Integer`** on hot paths (mirrors PureBLAS's AD-
   traceability requirement — one implementation, no per-type duplication). Float64 is
   the tuned path (PureBLAS's faer fast path); other `T` are correct-but-generic.
4. **Trim-compatible** (juliac --trim must build a factor-and-solve entry point). No
   runtime `eval`/`invokelatest`, no `Vector{Any}` on hot paths, no runtime CPU-feature
   detection — tuning constants (`tuning.jl`) are compile-time consts backed by
   Preferences.jl, overridable but not runtime-detected.
5. **Zero allocations after `symbolic()`.** `cholesky!`/`ldlt!`/`solve!` on an existing
   factor must be `@allocated == 0` (gated in the StrictMode-checks-disabled test
   configuration — see design §9.1 point 7; StrictMode's own runtime checks may allocate
   and must not be conflated with this gate, design §9.1 D6).
6. **TypeContracts.jl for compile-time interface contracts** (`contracts.jl`) —
   precompile-time only, eliminated by the trimmer, never a runtime mechanism. Runtime
   pre/postconditions are a **separate** StrictMode.jl layer (`strict.jl`), gated behind
   `StrictMode.checks_enabled()`. Do not conflate the two (design §9.1 D6).
7. **"Analyze once, factorize many times" is the primary API organizing principle**
   (design §1.2) — the target user (interior-point optimizers) refactorizes the same
   sparsity pattern hundreds of times per solve. `Symbolic` is immutable and shared by
   reference; `cholesky!`/`ldlt!` on an existing factor never recomputes ordering/etree/
   supernodes.
8. **LDLᵀ is fixed-pivot + signed regularization (QDLDL/Clarabel-style), not dynamic
   Bunch–Kaufman pivoting** (design §5.1 — this is deliberate scope, not a missing
   feature; general indefinite systems with 2×2 pivots are an explicit non-goal, §1.1).
   Track inertia `(n_pos, n_neg, n_zero)` in `FactorStats` for downstream IPOPT-style
   consumers that want to run their own regularization loop on top.

## Testing (ReTestItems — self-contained, individually triggerable)

- `runtests(PureSparse)`; trigger one item via `runtests(PureSparse; name="...")`.
- Correctness oracles: dense `BigFloat` factorization on small/medium matrices, residual
  gates on the SuiteSparse Matrix Collection zoo, CHOLMOD's *output* via `SparseArrays`
  as a black-box cross-check (never its source — see requirement 1).
- Matrix-zoo downloader must use a lock file + atomic temp-then-rename (ReTestItems runs
  parallel processes — a race here corrupts the cache, design §9.1 D7).
- See `docs/design.md` §9 for the full test-layer breakdown and `ROADMAP.md` for
  milestone-specific gates.

## Benchmarking

Chairmarks.jl, PureBLAS methodology: single-thread pinned, `@noinline` concrete wrappers
(not closures), repeated in-place reps, **median** not min, locked CPU clock, results→JSON
first, plots regenerate from saved JSON only. PkgBenchmark.jl supplements for
commit-to-commit self-regression. 4-arm matrix (design §9.3): PureSparse+PureBLAS,
PureSparse+OpenBLAS, CHOLMOD+OpenBLAS, CHOLMOD+PureBLAS (this 4th arm is **N/A** — blocked
on PureBLAS's documented `lbt_forward`-from-live-Julia-process limitation; do not re-chase
it, see design §9.3 D1).

## Standing rules

- No Python anywhere (global rule). Native lib via `ccall` or CLI subprocess if external
  is needed.
- `isnothing(x)` / `!isnothing(x)`, never `=== nothing`.
- Commit author email: `15278831+el-oso@users.noreply.github.com` (never a real address).
- The approved design (`docs/design.md`) is a contract: do not skip/substitute a
  requirement without asking first. If you are tempted to deviate — even for a "better
  idea" — stop and ask.
- When stuck on a hard algorithmic piece (AMD's approximate-degree bookkeeping, the
  left-looking scheduler, Davis–Hager's update recurrences), delegate to a Fable-model
  agent rather than guessing — these algorithms have exact published forms; approximate
  reimplementation from memory risks both correctness bugs and clean-room violations
  (requirement 1).
