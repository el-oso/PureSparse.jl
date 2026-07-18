# PureSparse.jl — M7 Design: GPU Multifrontal Sparse QR

> **⛔ STATUS: SHELVED BY MEASUREMENT (2026-07-18). A pure-Julia GPU-QR front cannot beat
> cuSOLVER `geqrf`; the "pure beats vendor" milestone is not achievable for QR.** The full
> design process ran (v1 → 2 reviews → v2 → Fable v3 TSQR redesign → Opus review), then three
> Phase-0 probes on galen (RTX 4070) settled it before any production kernel:
> - **Panel** (`benchmark/gpu/qr_panel_phase0.jl`): pure single-workgroup Householder panel is
>   **3–10× slower** than `geqrf` standalone (1 SM of 46; occupancy-bound).
> - **Front projection** (`qr_front_project.jl`): best-NB front **1.49–2.70× slower**.
> - **Trailing WY-apply γ** (`qr_gamma_phase0.jl`, Opus's identified crux): **γ_WY ≈ 0.77** —
>   pure *loses* the trailing 1.3×. It wins the `nn` shape (γ_nn=1.15, as M6) but loses the
>   `tn` tall-K/tiny-output shape badly (γ_tn=0.575) — cuBLAS's most-tuned batched-split-K
>   regime. Even if pure reached `tn` parity, the trailing caps ~1.07× and the panel is
>   parity-at-best: **both places pure could win are losses or ties.**
>
> Root cause: M6's Cholesky win came from `potrf`/`syrk` fitting a fused register-resident pure
> kernel; QR's Householder tall-skinny panel + tall-K WY-apply is exactly what cuSOLVER/cuBLAS
> are most tuned for — the win does not transfer. JSONs in `benchmark/results/qr_*phase0*.json`,
> `qr_front_project.json`. Audit trail: `design_qr_gpu_v3_fable.md`, `design_qr_gpu_v3_review_opus.md`,
> `design_qr_gpu_review*.md`. **CPU sparse QR (M5) remains CLOSED and gate-passing** — this shelves
> only the *GPU* QR milestone. The v2 design below is retained as the record of what was designed.
>
> **Status: v2** (folds two independent adversarial reviews of v1 —
> `design_qr_gpu_review.md` (Opus) + `design_qr_gpu_review_fable.md` (Fable), which converged
> independently on the same three fatal flaws). Same v1→review→v2 arc as `design.md` (M1),
> `design_qr.md`/`design_qr_m5b.md` (M5), `design_gpu.md` (M6). Builds on M5b (the CPU
> multifrontal-WY QR, CLOSED) and M6 (§M of `design_gpu.md`, the GPU engine, CLOSED). §-numbers
> prefixed **Q**.

## §Q0 What M7 is — and the one thing v1 got wrong

Bring the M5b multifrontal-WY sparse QR onto the GPU, reusing the M6 engine (frontier split,
bounded arena for *transient* data, hybrid residency, level-scheduled solve, pure-KA kernels, AMD
portability) around a **new dense per-front kernel: a blocked Householder QR**.

**The correction that defines v2.** v1 claimed the M6 bet ("pure KA beats the CUDA vendor") transfers
to QR *by analogy* — trailing update is a `gemm` we win, panel is register-resident like the Cholesky
diagonal. Both reviews independently proved that false:

- **The Cholesky/LDL wins came from a structure QR mathematically lacks.** `_front_fused64_v3!` won by
  holding a *small square ≤64×64 diagonal block* redundantly in every workgroup's registers, with the
  tall below-rows as *forward-only consumers* (they read the published pivot column, never feed back).
  Householder QR inverts the second property: forming reflector `j` needs `‖panel[k:stair,j]‖` — a
  **global reduction over the full tall panel height** — and the in-panel apply is a tall reduction
  `vᵀ·(remaining cols)` *between every column*; the tall rows **feed back into every pivot**. You
  cannot hold an `m_f×NB` tall panel in registers. (Confirmed against faer
  `no_pivoting/factor.rs:25`: reflector `norm_l2` over the full tall column, sequential across the
  panel.)
- **"We already win the trailing update" is a non-sequitur.** cuSOLVER `geqrf` *also* runs its
  trailing update through cuBLAS. If a front's cuSOLVER time is trailing-fraction `t` + panel-fraction
  `p=1−t`, our relative time is `t/1.14 + p·r` where `r` = pure-panel-QR ÷ `geqrf`-panel. Dominance of
  the *shared* trailing `gemm` caps our edge at the gemm ratio and moves the **entire verdict onto the
  panel** — the one quantity v1 never measured. At a generous `t=0.8`, ≥1.0× needs `r ≤ 1.49`.

So v2 is **measurement-first**, exactly as M6 became credible only after measuring the pure-gemm flip
before its v2: the design commits to *measuring `r` on real tall crown fronts in Phase 0*, and
specifies two honest branches on the result (§Q2). The gate, storage, and rank story are all
re-derived to match.

**Scope (honest):** the perf headline is the **full-rank / `τ<0` fast path** on the GPU.
Rank-deficient least-squares inputs take a **CPU frontal fallback** unless the all-device
single-workgroup rank kernel (§Q2.3 option B) measures out. `method=:column` (M5a) stays CPU.
Complex/BigFloat fronts route to the CPU **`:column` path** (the CPU *frontal* path is Float64-only
too — a v1 error). Update/downdate is not a QR concept here.

## §Q1 Reuse map — corrected (what is actually free, what is NOT)

**From M5b, reused verbatim:** §A2 front tree + children, §A3 assembly simulation **including its own
`crmax`/`mmax` contribution-block recurrence** (this — not M6's symmetric-square arena sim — is the
correct sizer for QR's rectangular upper-trapezoidal C blocks), §A4 symbolic + `ftau`/`frptr` layout,
§A8 amalgamation. **Permanent front storage (§A1.3):** fronts persist; C *is* the trailing rows of the
child's stored front. M7 keeps this — it is load-bearing for the solve replay (§Q3).

**From M6, reused — but only the parts that are genuinely transient/schedule-shaped:**
- §5.2 frontier split on the QR front tree — but `frontier_cutoff` is Cholesky-flop-calibrated; QR
  front cost scales with the staircase `m_f`, so the cutoff needs its **own measured derivation** (a
  Phase-0 output, not an inherited constant).
- §M.3 bounded arena — applies **only to the transient contribution block C**, not to V/T/R. See
  §Q2.5; v1's "arena applies unchanged" was wrong.
- The `SolveSchedule` **level machinery** (~50 lines of `ext/gpu_solve.jl`) — reused. The solve
  *kernels* and device arrays are **new** (§Q3); `batched_solve!`'s bodies walk Cholesky supernodal
  storage and do not transfer.
- Pure-KA discipline (amendment C), Atomix bare-atomic-only + Int32-only + election-free write-back.

**AMD portability (§Q1, corrected):** device-side on AMD = the numeric front kernel + write-back +
(if it measures out) the single-workgroup rank kernel with **serial** counter accumulation (no
atomic-rmw return values, no Int64 atomics — gfx1151). Assembly (staircase counting-sort) and the rank
bookkeeping default to CPU. "The whole path matches on ROCm" means the numeric kernel + write-back,
not literally everything.

**NOT free (new work, each an explicit build item):** the panel-QR kernel (§Q2.2), the WY-apply
`gemm` shapes that don't exist yet (§Q2.1), the permanent-residency device-memory budget (§Q2.5), the
device apply_Qᵀ/solve_R kernels (§Q3), the vendor gate arm (§Q4).

## §Q2 The GPU front kernel — measurement-first

The CPU front (M5b §A5.3) is a staircase-blocked panel QR: per panel, a column-by-column reflector
loop (form `v`/`τ`, apply to the rest of the panel), then a compact-WY block apply
`C := (I − V T Vᵀ)ᵀ C`. Two sub-ops, two honestly-separated risk levels.

### Q2.1 The trailing WY-apply — a `gemm` we *can* win, but the shapes don't exist yet
The block apply is two `gemm`s: `W := Tᵀ(Vᵀ C)` then `C := C − V W`. Only the **second** matches the
shipped `gpu_gemm_nt!` (measured at 1.14× cuBLAS). `W := Vᵀ C` is a **`tn` shape contracting the first
dims with `K=m_f` (tall) and a tiny `bs×nt` output** — a different occupancy regime (likely split-K;
the 4×4-tile kernel has almost no output tiles), and the M5b unpacked-C contract forbids fixing it by
packing `Cᵀ`. `C −= V W` needs a small `Wᵀ` staging copy to be `nt`-expressible. The `VᵀV` for the
T-build (§A5.3) is a third new shape. **Build item:** write the tn-tall-K kernel (+ `Wᵀ` staging) and
**probe it vs cuBLAS** before the front kernel is trusted. This is the part where a win is *plausible*
(it's `gemm`); it is not yet *measured*.

### Q2.2 The panel factorization — THE decisive unmeasured quantity `r`, and the Phase-0 probe
The panel is a latency-bound chain: (tall reduction for `‖·‖` → tall `ger`) × NB columns, sequential.
The register-residency that won M6 **does not apply** (§Q0). But this is not hopeless: **`geqrf`'s own
panel is equally serial** — its `dlarfg`/`dlarf` critical path is the same tall-reduction chain — so
*parity is a fair fight*, and ≥1.0× overall needs only `r ≤ ~1.49` (§Q0). The design commits to
**measuring `r` first** (§Q5 Phase 0) on real tall crown-front shapes (`m_f × NB` for the actual
stratum fronts), with an explicit panel scheme:
- **Primary: single-workgroup latency-bound panel** — one workgroup owns the `m_f×NB` panel in shared
  memory, does the reflector reductions + in-panel applies serially, writes V/τ. Direct analogue of
  `geqrf`'s panel kernel; `r≈1` is the target. Device NB is its **own measured tunable** (not the
  Cholesky 64).
- **Fallback if `r` is unfavorable: TSQR/CAQR** reduction-tree panel (splits the tall panel into row
  blocks, factors independently, merges) — more parallelism at the cost of complexity; a documented
  option, not built unless the single-workgroup probe loses.
- **Interim (mirrors M6a):** if neither pure panel beats `geqrf` in Phase 0, ship **pure trailing
  WY-apply (owned) + cuSOLVER `geqrf` on the tall panel** and record it — a real, honest interim that
  still wins the trailing bulk while the panel catches up, exactly as M6a shipped pure-trailing +
  cuSOLVER-potrf before v3 took the diagonal.

**The fused front** (`gpu_qr_front!`) loops panels: (a) single-workgroup panel factor forming
V/τ/T, (b) WY-apply the trailing columns via the §Q2.1 kernels. Election-free group-1-to-scratch
write-back of R rows + stored V/T (portable).

### Q2.3 Rank-revealing dead pivots — full-rank fast path is the headline; two honest options
v1's per-panel CPU micro-step is **rejected** (both reviews): the rank decision is per-**column** (a
dead pivot changes the row cursor `k` for the *next column of the same panel*, M5b §A5.3), so a
per-panel post-hoc fix forms wrong reflectors, and a per-column CPU round-trip is the exact
per-sync serialization M6 paid 21× to kill — and the `dropped_norm ≤ √(n_dead)·τ` frozen-residual
certificate *forces* the host "column j is dead" to land before the device WY-apply, so async
throughput and the certificate are mutually exclusive. v2 commits to:
- **Full-rank / `τ<0` fast path (the perf headline):** no branching, pure all-device, the path the
  gate's perf arm exercises.
- **Rank-deficient inputs — option A (default): CPU frontal fallback.** Honest, no straddle; the GPU
  path declines rank-deficient fronts.
- **Rank-deficient inputs — option B (if it measures out): all-device single-workgroup rank kernel.**
  The dead-pivot test becomes a *workgroup-uniform in-kernel branch* — the reflector norm is already
  reduced in shared memory, the cursor lives in the kernel, `dropped_sq`/`n_dead` accrue in device
  scalars by **serial** accumulation (AMD-safe). No host round-trip. The certificate holds **only**
  under a single-source-norm rule (the dead-pivot decision and the reflector use the *same* computed
  `xnorm` — never a CPU decision on a D2H'd norm vs a device-recomputed one, which can straddle τ).
  Requires the §Q2.2 single-workgroup panel to exist, so it is gated on Phase 0.
- **V compaction:** a dead column sits interspersed holding frozen residual; the WY `V` is the live
  columns only. Both options need a gather-staging kernel (or the CPU gather) producing the contiguous
  live-column V for the T-build and the solve replay (M5b `ws.wyV`, §A4.4).

### Q2.5 Storage & device-memory budget — permanent fronts, NOT the arena
QR's V/T (and padded R) must **persist** for the solve replay (§Q3), so — unlike Cholesky's transient
`U_s` — they **cannot** go in the M6 bounded arena (whose whole value is *freeing* U after the parent
consumes it). The budget has two terms:
- **Bounded/compacted arena (M6 §M.3):** the *transient* contribution block C only. Sized by M5b
  §A3.2's `crmax`/`mmax` recurrence.
- **Permanent, non-compactable residency:** `Σ_f mmax_f·n_f` (V, staircase-zeros included) + `nnzRF`
  (padded R) + `ftau` (T slabs). For QR this can exceed the whole Cholesky factor and is typically the
  OOM constraint.

Total device budget = arena + permanent residency; wired to an **M6-style loud CPU fallback**
(`gpu_capacity_ok`) and made a **stratum feasibility criterion** (the M7 analogue of M6 §8.3's
`nnzL·8 + workspace ≤ ~9 GB` — pinned *before* measuring, so the stratum can't silently cherry-pick
matrices that fit).

## §Q3 Device solve — new kernels on the reused schedule

`x = R⁻¹(Qᵀ b)`, both halves on-device, reusing only M6's `SolveSchedule` **level machinery**:
- **`apply_Qᵀ`:** replay stored per-front V/T leaves→root; fronts at one elimination level batch into
  one launch. **Correctness argument (stated, per review):** same-level batching is safe because the
  batched fronts' **gathered-row sets are disjoint** — A-rows are partitioned by `leftcol`, and
  children's C-rows come from strictly-lower, disjoint subtrees. (This is a *different* argument from
  the Cholesky triangular solve's; v1 borrowed the wrong one.) Added as an **executable invariant**
  (assert batched `frowind` sets disjoint), M5b §10.2-style.
- **`solve_R`:** triangular substitution over the level-scheduled padded R (new kernels reading
  `rval/frptr/fcolind`).
- **Perf note (honest):** single-RHS apply is **gemv-shaped / bandwidth-bound** — the §Q2.1 gemm win
  is irrelevant to solve wall-time. The solve is won by *not being launch-bound* (the M6 batching
  lesson), not by the gemm ratio.
- New device arrays: `fval` fronts, `ftau` T-slabs, per-panel descriptors, numeric-time `frowind`,
  `fpivotrow`. Under rank deficiency some metadata is value-dependent (the pass-up min-col rewrite
  depends on `r_f_live`) → a per-refactor H2D term the budget must carry (it is not "0 pattern H2D").

## §Q4 The gate — two separate clauses (not one mis-posed one)

`geqrf`/`ormqr` are **dense-only**; there is no CUDA vendor *sparse* multifrontal QR (cusolverSp QR is
square-system-only; cuDSS has no QR). So v1's single "≥1.0× cuSOLVER at every size" had no end-to-end
referent. Two clauses:

1. **Per-front dense kernel µ-benchmark vs `geqrf`+`ormqr`:** on the real front shapes (`m_f×n_f`,
   tall + square), pure `gpu_qr_front!` ≥ 1.0× cuSOLVER `geqrf` at every shape — the τ<0 fast path
   pinned (`geqrf` isn't rank-revealing). This is where `r` (§Q2.2) is proven.
2. **End-to-end sparse factor+solve vs SuiteSparseQR:** beats SPQR (CPU, thread config **stated**) on
   the pinned stratum — the real bar, the QR analogue of M6's "beats CHOLMOD." Optional context arm:
   "our engine with per-front kernels swapped to `geqrf`/`ormqr`" (the vendor arm, a real build item).

**Stratum (pinned before measuring, M6 §8.3 discipline):** named least-squares/rectangular matrices,
an explicit `m/n` aspect-ratio spread, explicit rank composition (full-rank *and* rank-deficient
subsets), ≥N per stratum, and the §Q2.5 memory-feasibility criterion. Chairmarks medians, clock-locked
host, results→JSON, **violin** plots (project standing rule).

**Correctness oracle (rank-aware):** dense-BigFloat QR (identical τ + identical non-pivoted column
order) + `SparseArrays.qr` black-box. `residual` gated everywhere; **`rank`/`nnz(R)` agreement only
away from the τ band** (GPU reduction order shifts `xnorm` by ulps → flips boundary decisions; SPQR
does extra rank refinement — M5b §A9.2's caveat carried), R-equality up to row signs; constructed-rank
fixtures for the rank claim; the `dropped_norm ≤ √(n_dead)·τ` certificate asserted.

## §Q5 Build order — Phase 0 is a measurement, not code to trust

0. **Phase-0 probes (decide the design before building it):** (a) single-workgroup panel-QR vs
   `geqrf` → the ratio `r` on real tall crown fronts; (b) the tn-tall-K WY `gemm` vs cuBLAS; (c) the
   QR `frontier_cutoff` derivation; (d) device-memory budget vs the candidate stratum. Outputs pick
   the panel scheme (single-workgroup / TSQR / interim cuSOLVER-panel) and the rank option (A/B).
1. **Symbolic reuse layer** — `GPUQRSymbolic` = `QRFrontSymbolic` + frontier partition + M5b §A3
   sizing (pure, CPU-testable).
2. **CPU multifrontal-QR-on-permanent-fronts oracle** — factor on CPU, match `qr!` (validates
   assembly/maps/residency, zero GPU).
3. **`gpu_qr_front!`** (the Phase-0-chosen panel + the §Q2.1 WY-apply kernels, τ<0 fast path first) —
   oracle vs CPU front + gate clause 1.
4. **Hybrid frontier** (per-front dispatch + crossing uploads + `gpu_capacity_ok`) — oracle.
5. **Device solve** (`apply_Qᵀ` + `solve_R`, level-scheduled, disjoint-row invariant asserted) —
   residual oracle.
6. **Rank path** (§Q2.3 option A default; option B if Phase 0 backed it) — rank oracle
   (constructed-rank + `dropped_norm`). **Lands before the gate** (v1 had it after — a sequencing bug,
   since the stratum is rank-inclusive).
7. **Formal gate** — both clauses, pinned stratum; the full-rank fast-path arm and the rank-deficient
   arm reported separately.

## §Q6 The hard bets (v2, honest and re-derived)

1. **`r` (tall panel-QR ÷ `geqrf`-panel) is the whole ballgame** and is measured in Phase 0, not
   assumed. Best case `r≈1` (both panels serial) → clean ≥1.0× from the owned trailing update. Bad
   case → the interim cuSOLVER-panel hybrid (still an honest, shippable M7 that wins the trailing bulk
   and beats SPQR end-to-end). The Cholesky-analogy credibility claim is **withdrawn**.
2. **Rank-revealing on the GPU** has no M6 precedent; v2's answer is the full-rank fast path + CPU
   fallback (default), with the all-device single-workgroup kernel as a measured upgrade — *not* the
   per-panel straddle v1 proposed.
3. **Permanent front residency**, not the arena, is the device-memory constraint; it bounds the
   stratum and can force the CPU fallback on the largest problems — a stratum feasibility criterion,
   pinned before measuring.

## §Q7 Provenance

Clean-room, absolute (CLAUDE.md req 1): SPQR TOMS paper (Davis 2011) + **faer** (Rust, MIT, read this
review cycle — `no_pivoting/factor.rs`, `col_pivoting/factor.rs`, confirming the sequential tall-column
reflector loop and that PureSparse deliberately does *not* use faer's global column-pivoting) + M6's
own kernels (ours). Never CHOLMOD/SuiteSparse source; `SparseArrays.qr` (SuiteSparseQR) and
cuSOLVER/cuBLAS are black-box oracles/baselines only.
