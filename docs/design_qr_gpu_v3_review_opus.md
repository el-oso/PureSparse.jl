# M7 GPU-QR design v3 (Fable) — adversarial review (Opus)

> Review of `design_qr_gpu_v3_fable.md`. Independent, adversarial. All arithmetic recomputed from
> `benchmark/results/qr_panel_phase0.json` / `qr_front_project.json` directly (not the doc's
> numbers). Kept as permanent audit trail. Findings to fold into v4.

## Arithmetic verification (confirmed against raw JSON)
- **§R3.3 ρ table (γ=1.14 columns): CORRECT** to rounding (last cell 1.341 vs doc 1.36 — φ rounding, NIT-9).
- **§R5(c) hybrid floor 0.896–0.961: CORRECT given γ=1.14.**
- **"Σpure_panel alone exceeds geqrf front": REAL** (2048×512: 29.13 ms vs 12.05 ms front, 2.4×).
- **§R1.1 a(nb)/b(nb): reproduces only as a 2-point ENDPOINT fit**, not least-squares (NIT-7).

## BLOCKER-1 — the win rests on γ (batched-gemm ratio), which is unmeasured, likely ≤1.0, and the flattering half is hidden. §R3.2/§R3.3/§R5(c)/§R0
γ=1.14 is `[ASSUMED]` — M6 measured it on **single** `nt` gemms; it must now beat
`cublasDgemmStridedBatched`, cuBLAS's **strongest** regime (batched small-K already amortizes the
global round-trip M6's edge came from). §R3.3 shows only the γ=1.14 columns. The hidden γ=1.0 column:

| front | ρ @γ1.14 (shown) | ρ @γ1.0 (**hidden**) |
|---|---|---|
| 2048×512 | 1.029 | **0.967** |
| 2048×1024 | 1.067 | **0.924** |
| 4096×2048 | 1.126 | **0.856** |
| 8192×4096 | 1.341 | **0.611** |

At γ=1.0 the trailing is a net loss (φ/γ=1.07), the budget **inverts** (pure panels must be *faster*
than geqrf everywhere), and the "guaranteed floor (c)" goes to **exactly 1.000 at all six shapes —
no win**. (c) is mislabeled `[MEASURED projection]`; it's the γ=1.14 assumption on measured data.
**Fix:** measure γ (P0c) FIRST; add the γ=1.0 column; re-tag (c) `[PROJECTED, contingent on γ]`;
drop "guaranteed" from §R0.

## BLOCKER-2 — r_tsqr=0.3–0.75 (§R2.4) contradicts "0.9–1.1 central" (§R0/§R5). 
Front ratio = (k·g_p + T_r·φ/γ)/G, k = pure-panel/geqrf-panel. At φ/γ=0.938: k=1 → front 0.95–0.98
(this is the "central"); k=0.5 (§R2.4's headline) → front 0.64–0.87 (branch a). So "0.9–1.1 central"
silently assumes k≈1 (**no panel win**), contradicting §R2.4's own 0.3–0.75. Honest range is wider
both ways: k∈[0.5,1.7] → front ∈ [0.64, 1.45]. **Fix:** one honest range parameterized on (k, γ),
both unmeasured; stop calling 0.9–1.1 "central."

## DEFECT-3 — K1 latency floor mis-attributed; 30 µs low end unsupported. §R1.1/§R2.2
0.58 ms/512 steps = 2832 cyc/step ≈ 5 global round-trips, i.e. a multi-substep tree reduction +
barrier, not one round-trip. Shared-residency removes the global fraction but not the ~8-barrier
tree per column reduction (1 block/SM = 17% occupancy, no latency hiding). Realistic floor ≈ 512 ×
~8 × ~40 cyc ≈ **~60 µs, not 30**. The 0.3 ms cap survives; the 30 µs low end doesn't.

## DEFECT-4 — multi-column batching credited with a latency cut it doesn't deliver. §R2.2
Batching 8 trailing columns is a **throughput** win; the critical path is the sequential reflector
chain (reflector j gated by its column-norm reduction, can't start before j−1 applied) — ~O(nb·log rb)
barriers, unchanged by batching. Correct scheme, over-credited. **Fix:** reframe as occupancy aid.

## DEFECT-5 — the 0.3 ms P0b-1 kill threshold does NOT certify the small crown. §R6
Tightest budget (2048×512, ρ=1.03): avg geqrf panel 0.5112 ms → 3-launch TSQR panel must be
≤0.526 ms → **≤0.176 ms/launch**. At 0.3 ms/block the panel is 0.9 ms → k=1.76 → front ≈1.46,
blows ρ=1.03. 0.3 ms only certifies big crowns. **Fix:** split block vs merge-node thresholds; set
small-crown gate ≈0.18 ms/launch, or concede small crowns to the hybrid (per-shape dispatch).

## DEFECT-6 — the "guaranteed floor (c)" faces the same tall-K `tn` shape used to reject HR. §R2.5 vs §R5(c)
§R2.5 rejects Householder-reconstruction partly for "re-creating the tall-K `tn` trailing shape,"
but §R5(c)'s hybrid uses standard-V trailing = **that same tall-K `tn` split-K shape**. If tall-K is
solvable, HR becomes viable and **unifies the format** (standard V/trailing/solve-replay — no K5
tree-apply, no tree-replay family, no per-front format tag, no transient-T divergence); if it isn't,
(c) is no floor. **Fix:** treat tall-K `tn` as a first-class deliverable and re-evaluate HR on total
complexity (its "one extra panel pass" is plausibly cheaper than K5 + tree-replay + format tag), or
drop (c).

## MEDIUM — rank-certificate equivalence is SOUND (verified); residual risk is async flag-ordering. §R4
Checked against M5b §A5.3 (585–593): drop test is `xnorm=nrm2(Ff[k:stair,jj]); dead if xnorm≤τ`, and
`R_jj=−sign·xnorm` ⇒ **|R_jj|=xnorm exactly at detection**; R unique up to row sign, order- and
merge-independent. Contrapositive valid. **This is not a defect.** Caveats: (a) the D2H flag must
land *before the parent's assembly kernel launches* — show this fits existing level batching without
a per-front sync (else the M6 launch-bound stall / parent unwind); (b) K1 must carry M5b's B3
identity-reflector convention (the probe's unconditional `1/beta` is a latent zero-column div-by-0).

## MEDIUM — T-transient is fine for single-RHS; hazard is two-format surface. §R2.3/§R3.1
Single-RHS solve applies reflectors serially (`v·(vᵀx)`), never builds T → rebuild cost zero (doc
right). Risks: multi-RHS would rebuild T per solve (uncosted); two replay families + format tag
double the solve kernel surface (launch-shape fragmentation vs M6's "won by not being launch-bound").
**Fix:** confirm single-RHS is the only device-solve path; prefer format unification (DEFECT-6).

## NITs
- **N7:** §R1.1 is a 2-point endpoint fit ("fits cleanly" overstated); LS gives a(32)=0.482 ms not
  0.585; data visibly non-linear (m 2048→4096 ratio 1.99). Re-label or LS-fit.
- **N8:** "64 MB through one SM" is ~4× low; actual re-read ≈268 MB → ~9.7 GB/s (still supports
  1-SM-bound). Conclusion stands, number wrong.
- **N9:** last ρ cell 1.341 not 1.36 (φ rounding).

## Verdict
The TSQR bet is **not sound as prioritized**. The load-bearing number is not K1 block latency (that
projection, ~60 µs–0.3 ms, is defensible and self-capped) — it is **γ on batched shapes**: unmeasured,
facing cuBLAS's strongest regime, and if γ≈1.0 the budget inverts and *both* the pure case and the
"guaranteed" hybrid floor become losses (computed γ=1.0 column: 0.61–0.97 pure, 1.000 hybrid) — a
downside §R3.3 hides. Honest ceiling given the raw data is **(b) parity with a real (c)-or-worse
tail**, not the (a)/(b) the doc leads with. **Restructure: measure γ (P0c) FIRST** (cheapest,
both-branch-gating), then K1 latency; and build the vendor-hybrid before committing to TSQR's
two-format machinery — because if the tall-K `tn` kernel is solvable, Householder-reconstruction
unifies the format and moots most of v3's new machinery (K5, tree replay, format tag), and if it
isn't, (c) is no floor.
