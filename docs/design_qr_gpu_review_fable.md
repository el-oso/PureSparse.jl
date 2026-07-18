# M7 GPU-QR design — independent adversarial review (Fable)

> Review of `design_qr_gpu.md` v1. Independent of the parallel Opus review
> (`design_qr_gpu_review.md`), blind to its findings. Kept as permanent audit trail. Findings
> folded into `design_qr_gpu.md` v2. (Verified against the shipped M6 code: `ext/gpu_dense.jl`
> `_front_fused64_v3!`, `ext/gpu_solve.jl` `batched_solve!`, `ext/PureSparseCUDAExt.jl` kernel
> inventory.)

## BLOCKERS

**B1 — the §Q2.2 register-residency analogy is structurally false, and no replacement panel scheme
is named (§Q2.2, §Q6.1).** v3 works because the sequentially-factored object is the ≤64×64 diagonal
block held *redundantly in every workgroup's registers*, and the tall B rows are *forward-only
consumers* (read the published pivot column, never feed back). Householder QR inverts the second
property: forming reflector `j` needs `‖panel[k:stair, j]‖` (a global reduction over full panel
height) and the in-panel apply needs `vᵀ·(remaining cols)` (another tall reduction) *per column,
between every step*; the tall rows feed back into every pivot. Redundant-per-group residency is
impossible (each group would need the whole `m_f×NB` panel); a multi-group kernel needs grid-wide
sync per column, which KA doesn't portably provide. The honest options — (i) single-workgroup
latency-bound panel (defensible: `geqrf`'s panel is equally serial → parity reachable), (ii)
TSQR/CAQR reduction-tree, (iii) CholeskyQR2 — are never mentioned. **Fix:** v2 picks a panel scheme
explicitly and probes it vs `geqrf` on crown shapes day-1, before "the bet is credible" is asserted.

**B2 — "the part we already win" rides two kernels that don't exist (§Q2.1, §Q2.3).** The ext has
exactly one device gemm: `gpu_gemm_nt!` (`C=A·Bᵀ`, contracting the second dim of both). The WY apply
needs `W = Vᵀ·C` — a *tn* shape contracting the *first* dims with `K=m_f` (tall) and a tiny `bs×nt`
output (a completely different occupancy regime, likely split-K; the 4×4-tile kernel has almost no
output tiles to fill the GPU), and the M5b unpacked-C contract forbids packing `Cᵀ`. `C −= V·W` is
`nt`-expressible only via a small `Wᵀ` staging copy. The 1.14× was measured on the `nt`
trailing-update shape only. Same for §Q2.3's `VᵀV` syrk. So 100% of the claimed flop-dominant "part
we win" is unwritten, unmeasured kernels. **Fix:** add the tn-tall-K kernel (+ `Wᵀ` staging) as an
explicit build item with its own probe vs cuBLAS before §Q5 step 3.

**B3 — §Q2.5's arena reuse contradicts the M5b storage layout M7 claims to reuse verbatim (§Q1,
§Q2.5 vs M5b §A1.3/§A5.5).** M5b has *no* contribution-block stack: fronts are permanent, C *is* the
trailing rows of the child's stored front, pass-up writes no values. The M6 §M.3 arena exists
precisely because Cholesky's `U_s` is transient. QR's V/T must *persist* for the solve replay. So
"arena applies unchanged" means either (a) C duplicated into an arena (extra copy + memory for zero
benefit) or (b) fronts compressed/freed SPQR-style (breaks solve replay, abandons the "§A1.3
verbatim" claim). Neither stated. No device-memory feasibility bound (permanent fronts + ftau on
~11.5 GiB), no M6 §8.3 analogue. **Fix:** drop the arena claim; store permanent fronts
device-resident; add the memory criterion to stratum selection; if the stratum doesn't fit, design
compressed storage as a *documented* deviation from M5b.

## DEFECTS

**D1 — the rank decision is per-COLUMN, not per-panel; the split can't work as described (§Q2.4).**
A dead pivot changes the row cursor `k` for the *next column of the same panel* (§A5.3 `continue`
without advancing `k`); a per-panel micro-step after the sweep is too late (wrong reflectors already
formed), and a per-*column* CPU round-trip is the per-sync death that killed the left-looking M6
path. The cleaner scheme falls out of B1: with the panel as a single-workgroup kernel, the
dead-pivot test is a workgroup-uniform in-kernel branch (norm already in shared memory, cursor lives
in the kernel, `dropped_sq` in a device scalar) — no split, no launch hazard. The certificate
survives *only* under a single-source-norm rule (decision and reflector use the same `xnorm`; a CPU
decision on a D2H'd norm plus a device-recomputed norm can straddle τ and disagree).

**D2 — V compaction under dead pivots unaddressed (§Q2.4).** A dead column sits physically
interspersed holding frozen residual; the WY `V` is the live columns only (CPU gathers into
`ws.wyV`, §A4.4). The device path needs a gather-staging kernel + memory for the non-contiguous
live-column V; affects T-build and solve replay.

**D3 — §Q3 "reuse the batched-solve machinery" overstates: only the schedule transfers (§Q3).**
`batched_solve!`'s kernels walk Cholesky supernodal storage (`xv`, `d_rowind`, `d_super`). QR's
`apply_Qᵀ` needs `fval` fronts + `ftau` T-slabs + per-panel descriptors + numeric-time `frowind` +
`fpivotrow`; `solve_R` needs padded `rval/frptr/fcolind` — all new kernels + arrays; only the ~50
-line `SolveSchedule` reuses. (i) single-RHS apply is gemv-shaped (bandwidth-bound) → the gemm win
is irrelevant to solve perf, so "the block-apply is again the WY gemm" is wrong for the `b`/`x`
path; (ii) numeric metadata is value-dependent under rank deficiency (pass-up min-col rewrite
depends on `r_f_live`) → per-refactor H2D of variable metadata needs a budget term, muddying the "0
pattern H2D" discipline; (iii) level-batch correctness needs the row-disjointness argument stated +
asserted (it *is* sound — A-rows disjoint by `leftcol`, C-rows ascend a single chain — but the doc
never says so).

**D4 — the gate is underspecified on five axes (§Q4).** (a) "the vendor equivalent" is ambiguous —
there is no vendor sparse multifrontal QR; pin it as "our engine with per-front kernels swapped to
`geqrf`/`ormqr`" and decide whether cusolverSp csrqr is a context arm (building the vendor arm is
real work absent from §Q5). (b) No pinned stratum (M6 §8.3 pinned named matrices + memory bound +
counts before measuring). (c) τ for the perf arm unstated (`geqrf` isn't rank-revealing → pin the
τ<0 fast path). (d) "at EVERY size" — front sizes or matrix sizes? (e) SPQR thread config unstated.

**D5 — rank-agreement oracle flaky as written (§Q4).** Exact `rank`/`nnz(R)` agreement near τ is
ill-posed; GPU reduction order shifts `xnorm` by ulps → flips boundary decisions. M5b §A9.2 already
restricts rank agreement to "away from the τ boundary" and R-equality to "up to row signs." Use
constructed-rank fixtures; include the `dropped_norm ≤ √n_dead·τ` assertion in §Q4's oracle list
(currently only §Q5 step 6).

**D6 — §Q6.3's tall-front mitigation doesn't exist (§Q6.3).** "very tall fronts need the blocked
driver, exactly as `gpu_front!` blocks large `nscol`" is false equivalence: `gpu_front!` blocks the
*column* dim while the sequential factor stays in the ≤64 diag block; QR reflectors span all `m_f`
rows, and blocking the row dim changes the algorithm (that's TSQR, unproposed). B1 restated as a
claimed-but-nonexistent fallback.

## NITs

- **N1 (§Q6.3):** device NB is a new tunable; state how it's derived/measured, not inherited from
  the Cholesky 64.
- **N2 (§Q1):** `dropped_sq`/`n_dead`/rank counters must avoid atomic-rmw return values + Int64
  atomics (gfx1151); prescribe single-workgroup serial accumulation now.
- **N3 (§Q4):** `frontier_cutoff` is Cholesky-flop-calibrated; QR front cost scales with the
  staircase `m_f` — needs its own measured derivation.
- **N4 (§Q0):** "complex/BigFloat fronts stay CPU" — the CPU *frontal* path itself lacks them (they
  route to `:column`). Say ":column path."

## Verdict

M7 as drafted is **not** "the same bet M6 won twice": the Cholesky/LDL wins came from a structure —
locally-computable pivots plus forward-only tall consumption — that Householder QR mathematically
lacks, and the two headline supports (§Q2.1 "the part we already win," §Q2.2 "identical control
flow") are both false as stated, while the storage layer (§Q2.5) contradicts the M5b design it
claims to reuse verbatim. The re-scoped gate is probably still winnable — `geqrf`'s own panel is
serial, so parity is a fair fight — but only after v2 commits to an explicit panel scheme, writes
and probes the missing tn/nn gemm shapes, moves the rank branch inside the panel kernel, and
resolves the permanent-fronts-vs-arena memory story with a stratum feasibility bound.
