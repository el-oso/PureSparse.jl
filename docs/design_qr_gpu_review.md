# M7 GPU-QR design — independent adversarial review (Opus)

> Review of `design_qr_gpu.md` v1. Independent of the parallel Fable review
> (`design_qr_gpu_review_fable.md`). Kept as permanent audit trail (same convention as
> `design_qr_review.md`). Findings folded into `design_qr_gpu.md` v2.

## BLOCKER 1 — "we already win the trailing update" does not imply ≥1.0× vs `geqrf` (§Q2.1–Q2.3, §Q6.3)

cuSOLVER `geqrf` also spends its trailing time in cuBLAS `dlarfb`. If a front's cuSOLVER time
splits into trailing fraction `t` and panel fraction `p=1−t`, PureSparse relative time =
`t/1.14 + p·r`, where `r` = panel-time ratio (ours/theirs), **unmeasured**. For ≥1.0× at a
generous `t=0.8` you need `r ≤ 1.49`. The whole margin hangs on `r`, for which the doc gives no
evidence: dominance of a *shared* op caps the edge at the gemm ratio and shifts the verdict onto
the panel. The register-residency analogy fails exactly there: M6 held a small square ≤64×64
diagonal block in registers; QR's panel is `m_f × NB` with `m_f` the full tall front height, each
column's reflector needs `‖·‖` over the entire tall column then a `ger` over all `m_f` rows,
sequential across NB — a latency-bound chain `geqrf`'s dedicated panel kernel is tuned for.
Confirmed against faer `no_pivoting/factor.rs:25` (reflector `norm_l2` over the full tall column,
column-by-column, sequential across the panel). **Fix:** measure `r` Phase-0-style on real tall
front shapes *before* committing the design (as M6 measured the gemm flip before v2); if `r`
unfavorable, interim kernel = pure trailing gemm + `geqrf` on the tall panel (mirrors M6a).

## BLOCKER 2 — the per-panel CPU rank micro-step is the launch-bound hazard M6 paid to kill, and it can't coexist with the `dropped_norm` certificate (§Q2.4)

A crown front has `n_f/NB ≈ 40` panels; a CPU micro-step per panel is a D→H→D round-trip per
panel (~400 syncs over ten crown fronts), each a hard barrier gating the next panel's WY-apply.
M6's first solve was launch-bound at 555ms and only recovered via batching (21×); a per-panel sync
is strictly worse (a sync costs more than a launch and can't be batched — each decision gates the
next apply). Worse, the two goals are mutually exclusive: the `dropped_norm ≤ √(n_dead)·τ`
certificate is a frozen-residual argument (a skipped column is never touched again), so correctness
*forces* the host "column j is dead" to apply *before* the device WY-apply — i.e. forces the
serialization. **Fix:** commit to two paths — (a) pure all-device `τ<0`/full-rank fast path (no
branching, the headline), (b) rank-deficient inputs → CPU frontal fallback, not a per-panel
straddle; or an all-device single-workgroup panel kernel (reintroduces BLOCKER 1's tall-panel
problem; must be measured, not free).

## DEFECT — permanent V/T/R front storage the solve replays is unbudgeted (§Q1, §Q2.5, §Q3)

M6's bounded arena's value comes from *freeing* each U after its parent consumes it. QR's solve
replays every front's stored V/T and `solve_R` reads stored padded R — none can be freed; M5b
§A1.3 keeps permanent fronts `Σ_f m_f·n_f`. The arena reuse is valid only for the transient C
(the U analogue); the dominant device storage (permanent V `Σ_f mmax_f·n_f` + T slabs + padded R
`nnzRF`) is exactly what M6's compaction throws away and is never budgeted — for QR this can exceed
the whole Cholesky factor and be the OOM constraint. **Fix:** device-memory budget = bounded arena
for C (legit) **plus** permanent non-compactable residency `Σ_f mmax_f·n_f + nnzRF + ftau`; wire to
an M6-style loud CPU fallback; correct §Q1's "applies unchanged."

## DEFECT — build-order sequencing: rank path (step 6) lands after the gate (step 7) (§Q5, §Q4)

The stratum + oracle check `rank`/`nnz(R)`, meaningless unless it includes rank-deficient matrices
(real LS routinely are) — but rank is step 6, after the gate at step 7. Either the gate is
full-rank-only (cherry-picked, violates M6 §8.3), or step 7 needs step 6. **Fix:** move rank before
the gate, or split into a full-rank fast-path gate (early) + rank-deficient gate (after step 6);
state the stratum's rank composition.

## DEFECT — the gate conflates a dense kernel µ-bench with an end-to-end sparse baseline (§Q4)

`geqrf`/`ormqr` are dense-only; you cannot run them on the sparse LS problem. Two comparisons: (a)
per-front dense kernel vs `geqrf` (µ-bench), (b) end-to-end sparse factor+solve vs **SPQR** (CPU).
No GPU sparse-QR baseline exists (cusolverSp QR is square-system-only, cuDSS has no QR), so "≥1.0×
cuSOLVER at every size" has no end-to-end referent. **Fix:** separate the two clauses; end-to-end
bar is SPQR (analogous to M6's "beats CHOLMOD").

## DEFECT — rank/`nnz(R)` oracle unsound near τ; stratum underspecified (§Q4, §Q5)

`rank`/`nnz(R)` are not invariant across implementations near tolerance (SPQR does extra rank
refinement; GPU reduction order shifts `xnorm` by ulps, flipping boundary decisions). M5b §A9.2
already restricts rank agreement to "away from the τ boundary" and R-equality to "up to row signs";
§Q4 drops both. The BigFloat oracle needs identical τ and column order. Stratum names no sizes,
aspect ratios, rank fractions, or sources. **Fix:** gate `residual` + rank only away from the τ
band (carry §A9.2's caveat); pin the stratum before measuring; BigFloat oracle uses identical
τ + column order.

## DEFECT — `apply_Qᵀ` level-schedule validity asserted by analogy, not argued (§Q3)

M5b §A6's correctness is stated for postorder over a shared full-length `y`. Level batching is valid
here but for a *different* reason than the triangular solve: the batched fronts' gathered-row sets
must be disjoint (they are — A-rows partitioned by `leftcol`, children's C-rows from strictly-lower
disjoint subtrees). The doc borrows the Cholesky dependency argument, which is not this dependency.
Holds, so DEFECT not BLOCKER — but an unstated solve correctness argument is how races ship.
**Fix:** state the disjoint-row-set argument; add it as an executable invariant.

## NITs

- **N1 (§Q1, §Q2.5):** M6's arena sizes symmetric square `U_s`; QR C blocks are rectangular
  upper-trapezoidal with M5b §A3.2's own `crmax`/`mmax` recurrence — cite M5b §A3, not M6's sim.
- **N2 (§Q1):** M6's gfx1151 constraints were for Cholesky's scatter; QR's staircase counting-sort
  assembly + (if device) dead-pivot scans differ. State which pieces are device-side on AMD;
  "the whole path compiles + matches on ROCm" is only the numeric kernel + write-back.

## Verdict

M7 is **not yet a sound bet as written** — a plausible plan with its single decisive quantity
unmeasured. The §Q2 argument is a non-sequitur: dominance of a *shared* gemm caps the edge at the
gemm ratio and moves the verdict onto the panel, where the tall QR front breaks the small-square
register-residency that won M6. **Biggest risk: the tall-front panel factorization vs `geqrf`
(BLOCKER 1)** — measure it Phase-0-style *before* the design commits. Second: rank-revealing on GPU
(BLOCKER 2) forces the per-panel serialization M6 paid to eliminate; the honest resolution is a pure
full-rank fast path + CPU fallback for rank-deficient inputs, with gate and build order fixed to
match.
