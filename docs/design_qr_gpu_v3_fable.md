# PureSparse.jl — M7 Design v3: the GPU-QR front kernel, post-Phase-0

> **Authored by Fable** (design lead) after the Phase-0 probes killed the naive single-workgroup
> panel. Adversarially reviewed by Opus (`design_qr_gpu_v3_review_opus.md`). Amends
> `design_qr_gpu.md` v2 §Q2 (panel scheme, trailing, rank, storage, build order); v2's engine
> reuse (§Q1), solve (§Q3), and gate discipline (§Q4) stand unless amended here. Every number is
> tagged **[MEASURED]** (from `benchmark/results/qr_panel_phase0.json`, `qr_front_project.json`,
> RTX 4070, Float64, warm `CUDA.@elapsed` medians), **[PROJECTED]** (model from measured data —
> must be probed), or **[ASSUMED]** (inherited — must be re-measured on the new shapes).

## §R0 Honest verdict (three sentences)

A pure GPU-QR front **can plausibly reach parity with cuSOLVER `geqrf`, and a strict win is
possible but not yet claimable**: the TSQR panel scheme below removes the measured occupancy
bottleneck by construction (46 SMs instead of 1), but closing the last ~2× requires a
latency-optimized block kernel that is projected, not measured, and TSQR's own tree-apply flop
overhead (+7–19%) eats most of the 1.14× trailing-gemm edge, so the pure front's central estimate
is **0.9–1.1× `geqrf`**, shape-dependent. The guaranteed floor is the vendor-hybrid
(cuSOLVER-panel + owned trailing), whose front ratio projects to **0.90–0.96 [MEASURED
projection]** — and gate clause 2 (beat SPQR end-to-end) is winnable under every branch, so M7
ships regardless. The single next action is the Phase-0b TSQR probe (§R6), which settles the
pure-vs-hybrid branch with two numbers before any production code.

## §R1 Root-cause model of the Phase-0 failure — and its direct verification

### R1.1 The measured failure, decomposed

Probe 1 fits cleanly to `t_pure(m, nb) ≈ a(nb) + b(nb)·m` **[MEASURED, fitted]**:

| nb | a(nb) (latency floor) | b(nb) (per-row slope) | fit check |
|----|----------------------|----------------------|-----------|
| 32 | ≈ 0.58 ms | ≈ 0.80 µs/row | a, b both ∝ nb² |
| 48 | ≈ 1.27 ms | ≈ 1.76 µs/row | ✓ (×2.2 vs nb=32) |
| 64 | ≈ 2.22 ms | ≈ 3.10 µs/row | ✓ (×3.9 vs nb=32) |

Two separate bottlenecks, both consequences of **one workgroup = 1/46 of the GPU**:

- **`b·m` term = one SM's memory bandwidth.** The kernel streams ~`nb²/2` column passes over
  global memory through a single SM's load/store pipes. At 8192×64 that is ~64 MB of traffic
  through one SM (~15–25 GB/s effective) ≈ the observed ~24 ms tail. The board has 504 GB/s;
  45/46 of it is idle.
- **`a(nb)` term = a serial chain of ~`nb²/2` barrier-separated reduction steps**, each paying
  global-memory latency (~1.1 µs/step at nb=32: 512 steps ≈ 0.58 ms). This term does not shrink
  with `m` — it is the latency floor of *any* single-workgroup QR with this kernel structure,
  which is why the merge nodes of a naive TSQR would inherit it (§R2.4).

### R1.2 Verification probe (cheap, run first)

Per CLAUDE.md "don't guess — check": the 1-SM-bound diagnosis is inferred from scaling, not
counters. **P0b-0**: launch 8 independent single-WG panels concurrently (8 streams / one grid of
8 groups on 8 disjoint panels). Prediction: aggregate throughput ≈ 8×, wall ≈ 1×. If this fails,
the model in §R1.1 is wrong and §R2's parallelism argument needs re-derivation before proceeding.

## §R2 The panel scheme: staircase-aware TSQR (communication-avoiding QR)

Source: Demmel, Grigori, Hoemmen, Langou, *Communication-optimal parallel and sequential QR and
LU factorizations* (SIAM J. Sci. Comput., 2012) — published paper, clean-room compliant (§R8).
The panel `m_p × nb` (rows already staircase-trimmed per M5b §A5.3) is factored as:

1. **Level 0 — block QR.** Partition the panel's rows into `P = cld(m_p, rb)` blocks of `rb`
   rows. Each block is factored by **one workgroup, fully resident in shared memory**
   (`rb·nb·8 B ≤ 48 KB`), producing a local `nb×nb` R (top of the block) and local Householder
   V in place. All `P` blocks run concurrently — for crown panels (`m_p` = 2–8k, `rb` = 192,
   `P` = 11–43) this occupies most/all of the 46 SMs. Short panels (`m_p ≤ rb`, i.e. the lower
   staircase) degenerate to a single block with no merge — automatic, no special case.
2. **Merge tree — arity-q, triangular-packed.** Stack `q` block-R triangles (packed: only
   `q·nb(nb+1)/2` values, so `q ≤ 12288/(nb(nb+1))`; nb=32 → q up to 11, design center q=8) and
   QR the stack in one shared-resident workgroup per node. `P=43, q=8` → **2 levels** (43→6→1),
   i.e. the whole panel is a chain of **3 kernel launches** (launch boundaries are the portable
   global sync — no grid-sync, no atomics; AMD-safe by construction, §R2.6). The root's R is the
   panel's R.

The panel Q is now a **two-level tree of Householder factors**, not a single m-tall V. This is
the deliberate, load-bearing format change: it is what buys back 45 SMs, and §R3/§R5 account
honestly for what it costs on the trailing side.

### R2.1 Parameters (all swept in Phase-0b, centers stated)

| knob | center | constraint / rationale |
|------|--------|------------------------|
| `nb` (panel width) | **32** (sweep 16/32/48) | NB=32 was best at front level in probe 2 at every shape [MEASURED]; shared-mem fit; tree overhead ∝ nb/rb |
| `rb` (block rows) | **192** (sweep 128/192/256) | `rb·nb·8 ≤ 48 KB`; larger rb ⇒ fewer blocks (less parallelism) but lower tree overhead |
| `q` (merge arity) | **8** | packed-triangle shared fit; 2-level tree at P≤64 |
| WG size | 256 | matches probe; re-sweep with the optimized kernel |

`NB_gpu`/`rb`/`q` are M7 tunables in `tuning.jl` (compile-time consts, Preferences-backed —
CLAUDE.md req 4), derived from the Phase-0b sweep, not inherited from Cholesky.

### R2.2 The two kernels

**K1 `_tsqr_block_qr!`** (also used for merge nodes via packed-input variant K2): one workgroup,
block staged **into shared memory once**, factored there, written back once. The kernel is *not*
the probe kernel re-hosted in shared memory; it restructures the latency chain:

- Norm reductions and in-panel applies run on shared-resident data: per chain step, shared-mem
  latency (~30 cyc) + barrier, not a global round trip (~600 cyc).
- **Multi-column apply batching**: after forming reflector `v` (in shared), the
  `w = τ·(vᵀ·A_trail)` reductions for up to 8 trailing columns are computed per barrier phase
  (2-D thread layout: 32 threads/column × 8 columns), collapsing the chain from
  `O(nb²)` barrier steps to `O(nb²/8)`.
- Chain estimate at nb=32: ~600 barrier-separated shared-latency steps ≈ **30–60 µs/block
  [PROJECTED]** vs the probe's global-resident 0.58 ms floor. Even with 5× pessimism, ≤ 0.3 ms.
- Epilogue: while V is still in shared, optionally emit the block's T (see §R3.2 — T is
  transient, built at apply time; the epilogue is the cheap place if K3 wants it precomputed).
- Zero-column guard: `nrm == 0 ⇒ tau = 0`, skip scaling. (The Phase-0 probe kernel divides by
  `beta` unconditionally — a latent div-by-zero for exactly-zero columns that `randn` never
  exercises. K1 must carry the M5b B3 identity-reflector convention explicitly.)

**K2 `_tsqr_merge_qr!`**: identical structure, input = `q` packed R-triangles (from a
side slab), exploits the known triangular sparsity in the reduction/apply loops (halves flops
and traffic), output = one packed R-triangle + the node's packed V + per-node tau.

### R2.3 Storage: what persists, what is transient (amends §Q2.5)

Persist (needed by the §Q3 solve replay):
- **Level-0 V**: in place in the panel (below each block's local diagonal) — no extra memory
  beyond the front itself, exactly the M5b in-place discipline.
- **Merge V slabs**: packed-triangle format, per (panel, node); ≈ +5–8% of front V storage at
  the center config [derived from `Σ_nodes q·nb(nb+1)/2` vs `m_p·nb`].
- **tau**: per level-0 block and per merge node (generalizes `ftau`'s scalar-tau role).
- **Descriptors**: per-panel `P`, node offsets, `rb`, staircase row ranges — all `Int32`,
  computed **symbolically** (full-rank path: the staircase is symbolic, so the whole tree shape
  is known at analyze time; zero numeric-time H2D for pattern — the M6 "0 pattern H2D"
  discipline holds on the fast path).

Transient (M6 bounded arena — the arena's legitimate use per v2):
- **T factors** (level-0 `nb×nb` per block, per-node `nb×nb`): built at apply time (K3
  epilogue or K1 epilogue), consumed by the trailing apply, freed. **This is a deliberate
  deviation from M5b's "T's are stored" (§A5.3)** — on device, permanent T slabs would add
  ~25–30% to permanent front residency, and the solve replay (single-RHS, bandwidth-bound) can
  apply reflectors serially within a shared-resident block at no bandwidth cost (§Q3 note).
  Documented as a device-format divergence; the CPU frontal path keeps M5b's format.

The v2 §Q2.5 budget formula gains one term: permanent = `Σ_f mmax_f·n_f` (V, unchanged)
+ merge-V slabs (+5–8%) + tau + `nnzRF` (padded R); T moves from permanent to arena. Fronts
factored on GPU and fronts factored on CPU (below the frontier cutoff, or rank-refactored —
§R4) carry a **per-front format tag**; the device solve has two batched replay kernel families
(TSQR-tree replay; standard-WY replay for uploaded CPU-format fronts). Both are gemv-shaped and
batched per level — the M6 lesson that the solve is won by not being launch-bound, not by gemm
ratios, is unchanged.

### R2.4 Expected panel ratio r_tsqr — the honest arithmetic

With the **unoptimized** probe-kernel structure, TSQR already helps but does not close: panel
time ≈ `(1+L)·a(nb) + b(nb)·(rb + L·q·nb)`. At 8192×32 (`rb=192, q=8, L=2`):
`3×0.58 ms + 0.8 µs×704 ≈ 2.3 ms` vs geqrf 1.36 ms → **r ≈ 1.7 [PROJECTED from the measured
a/b fit]**. So TSQR *structure* alone recovers ~3× of the 5.2× gap; **the remaining ~2× must
come from K1's shared-residency + column-batching attack on `a(nb)`** (§R2.2, projected
0.03–0.3 ms per node vs 0.58 ms). Center projection: panel(8192×32) ≈ 3 launches ×
~0.1–0.3 ms + launch tax ≈ **0.4–1.0 ms → r_tsqr ≈ 0.3–0.75 [PROJECTED]**; pessimistic
(kernel optimization delivers only 2×): r_tsqr ≈ 1.0–1.3.

Why it can approach geqrf at all: geqrf's own panel is the same serial reflector chain
(confirmed structurally against faer, §Q0) — but executed with one panel's worth of parallelism.
TSQR beats that structure on tall panels by using the whole board for level 0; the only
irreducibly serial part left is a 3-deep kernel chain of shared-resident micro-QRs. cuSOLVER may
itself use a tall-skinny path internally (black box, unknowable) — which is why r_tsqr is a
measurement (P0b-2), not a claim.

### R2.5 Alternatives considered and rejected

- **CholeskyQR2**: fully gemm-shaped (syrk + chol + trsm, twice) — best possible occupancy, but
  (i) unstable for κ(panel) ≳ 10⁷ (Gram squaring; "twice" only rescues moderate κ), (ii) breaks
  down entirely on rank-deficient panels — incompatible with §R4's certificate scheme, and
  (iii) produces explicit-Q, not Householder form — breaks the WY trailing apply and the M5b
  solve-replay contract. Rejected on stability + format.
- **Householder reconstruction from TSQR** (Ballard, Demmel, Grigori, Knight, Lowery, Nguyen,
  IPDPS 2014): would restore a standard m-tall V (unifying formats, standard trailing apply) at
  the cost of ~one extra panel-height pass per panel *and* it re-creates the tall-K `tn` trailing
  shape that §R3 shows TSQR dissolves for free. Strictly dominated here. Cited, rejected.
- **Recursive / multi-block "hybrid" panel** (grid-synced multi-WG single-level): needs
  per-column grid-wide sync, which KA does not portably provide (Fable review B1). Rejected.

### R2.6 AMD portability (gfx1151, M6 constraints)

By construction: no atomics anywhere in the panel path (level boundaries = kernel launches; the
merge tree is the portable global sync); no atomic-rmw return values; no Int64 atomics; all
descriptors Int32; shared usage ≤ 48 KB fits the 64 KB LDS. The K1/K2 column-batching layout
uses only `@localmem` + `@synchronize` (no subgroup shuffles — not portably available in KA).
Same M6 posture: AMD = correctness arm, not a perf gate.

## §R3 The trailing WY-apply (amends §Q2.1)

### R3.1 TSQR dissolves the tall-K `tn` problem into batched small-K

v2's blocker was `W = Vᵀ·C` with `K = m_f` (tall) and a tiny `bs×nt` output — a split-K regime
none of our kernels cover. Under the TSQR format the trailing update per panel becomes:

1. **Level-0 batched WY apply** (the bulk): for each of P blocks independently,
   `C_b := (I − V_b T_b V_bᵀ)ᵀ C_b` with `V_b: rb×nb`, `C_b: rb×n_trail`. The contraction
   `W_b = V_bᵀ C_b` now has **K = rb ≈ 192 (small), batched over P** — total output
   `P·nb × n_trail` (e.g. 43·32 = 1376 rows × ~2000 cols), which fills the 4×4-tile kernel's
   grid without split-K. The tall-K problem is gone, *by construction*, not by a new kernel trick.
2. **Tree-level applies** (small): per merge node, apply the node's packed-triangle reflectors to
   the `q` corresponding nb-row slices of C — a fused gather-apply kernel (K5), batched over
   nodes per level, 2 levels.

Kernel inventory (new, each with its own probe before being trusted):
- **K3 `_wy_w_batched!`**: batched `W_bᵀ = (V_bᵀ C_b)ᵀ` — writes W **transposed**, so that
- **K4** the second gemm `C_b −= V_b·(T_bᵀ W_b)` is `nt`-expressible and **reuses the proven
  4×4-tile `gpu_gemm_nt!` structure with a batch index** (anchor on the proven-fastest path);
  the tiny `T_bᵀ W_b` trmm is fused into K3's epilogue (T built there from V/tau, transient).
- **K5 `_tsqr_apply_tree!`**: fused strided-slice apply for merge levels.
- Solve replay reuses K3/K4/K5 with `n_trail = 1`, batched across fronts per level (§Q3
  unchanged otherwise).

### R3.2 The honest cost: tree-apply flop overhead φ, and whether 1.14× survives

TSQR's trailing apply does the level-0 flops of a standard WY apply **plus** the tree applies:
overhead factor `φ ≈ 1 + (q/(q−1))·(nb/rb)·½` (½ from exploiting the packed triangles) ≈
**+7–10%** at the center config (nb=32, rb=192–256), +19% if triangularity is not exploited
[derived, arithmetic in probe P0c's script]. Meanwhile the 1.14× **[ASSUMED — measured in M6 on
single `nt` shapes only]** must be re-established on *batched* shapes against
`cublasDgemmStridedBatched`, a strong baseline. Net trailing speed vs the vendor's trailing:
`γ/φ` where γ = our batched-gemm ratio. At γ=1.14, φ=1.07: net ≈ **1.065×** — most of the M6
gemm edge is eaten by TSQR's extra flops. At γ=1.0 (cuBLAS parity only): net ≈ **0.93×** — the
trailing becomes a small *loss* and the panel must strictly beat geqrf's. This is the
make-or-break coupling, and it is why P0c is not optional.

### R3.3 The front-ratio arithmetic that the gate hangs on

From `qr_front_project.json` (NB=32 rows) **[MEASURED]**, with `G` = geqrf front, `g_p` =
Σ geqrf panels, `T_r = G − g_p` (trailing): pure front ≤ G requires
`Σ pure_panels ≤ g_p + T_r·(1 − φ/γ)`. Panel budget ratio `ρ = budget/g_p`:

| front (NB=32) | G (ms) | g_p (ms) | T_r (ms) | ρ at γ/φ=1.14/1.0 (v2's assumption) | ρ at γ/φ=1.14/1.07 (TSQR honest) |
|---|---|---|---|---|---|
| 2048×512 | 12.05 | 8.18 | 3.87 | 1.06 | 1.03 |
| 2048×1024 | 32.63 | 15.63 | 17.01 | 1.13 | 1.07 |
| 4096×1024 | 56.12 | 28.34 | 27.79 | 1.12 | 1.06 |
| 4096×2048 | 163.12 | 53.42 | 109.70 | 1.25 | 1.13 |
| 8192×2048 | 308.54 | 80.15 | 228.39 | 1.35 | 1.18 |
| 8192×4096 | 966.91 | 147.46 | 819.45 | 1.68 | 1.36 |

Read this table plainly: **for a pure win at every crown shape, r_tsqr must be ≈ 1.0–1.03 at the
small end** (2048×512 has almost no trailing slack) and ≤ 1.2–1.4 at the big end. The §R2.4
center projection (0.3–0.75) clears this; the pessimistic projection (1.0–1.3) wins big crowns
and loses small ones. That spread is exactly what P0b-2 measures.

## §R4 Rank-revealing: post-scan certificate + per-front CPU refactor (amends §Q2.3)

TSQR makes v2's option B (in-kernel per-column dead-pivot branch) impossible for GPU fronts: the
transformed-column norm exists only after the merge, and the M5b cursor semantics (`k` does not
advance on a dead pivot) cannot be replicated mid-tree. But TSQR enables something better than
v2's option A (decline all rank-deficient inputs):

**Scheme.** Run the fast non-revealing TSQR factorization of every GPU front unconditionally.
A tiny device kernel (K6) scans the front's **pivotal** R diagonals and sets a per-front `Int32`
flag if any `|R_jj| ≤ τ·(1+ε_m)` (ε_m a small safety margin). Flags are downloaded at the
engine's existing batch granularity (crown fronts are few — tens — so this is not the per-panel
sync BLOCKER 2 killed; the flag must land before the *parent's* assembly, which the postorder
batching already orders). A flagged front is **re-assembled and refactored on the CPU** with the
full M5b §A5.3/§A5.4 rank-revealing loop (re-assembly is possible because children's C rows are
the trailing rows of their *permanent* stored fronts — download, plus host A rows), and its
outputs cross back via the M6 frontier-crossing machinery.

**Equivalence argument (recorded for the reviewer).** If the scan finds no flagged pivotal
diagonal, the fast-path output *is* the rank-revealing output: suppose the revealing run would
first drop at column c; before c no drops occurred, so both runs' transformed states are
identical, and the revealing run's test quantity `xnorm(rows k:stair, col c)` equals the fast
run's `|R_cc|` (the reflector maps the transformed column onto its diagonal); `xnorm ≤ τ` would
have flagged. Contrapositive: no flag ⇒ no drops ⇒ identical outputs. The margin ε_m makes
ulp-level reduction-order drift (the §Q4/§A9.2 τ-band caveat) fail *safe*: a spurious flag costs
one redundant CPU refactor, never a wrong result — the CPU refactor is authoritative either way.
The `dropped_norm ≤ √n_dead·τ` certificate is then computed entirely by the CPU path with M5b's
single-source norms — the v2 mutual-exclusivity problem (host decision vs device apply) never
arises because the device never makes a rank decision.

**Honest perf statement.** τ>0 inputs now run on the GPU with only genuinely-deficient fronts
paying CPU (bounded waste: ≤ 1 discarded GPU factor per deficient front). Heavily
rank-deficient matrices degrade toward CPU-frontal speed; the gate reports the full-rank and
rank-deficient arms separately (v2 §Q4 discipline, unchanged). `τ<0` skips K6 entirely.

## §R5 The honest ceiling (deliverable 3, stated plainly)

- **(a) Pure wins outright** — requires *both* r_tsqr ≲ 1.0 at small crowns **and** γ ≥ ~1.1 on
  batched shapes. [PROJECTED, center-case; probability honest-uncertain until P0b/P0c.]
- **(b) Pure parity ±10%** — the **central estimate** given §R2.4's projection spread and
  §R3.2's φ tax. A pure front at 0.95–1.1× geqrf across shapes is a defensible M7 headline
  ("pure KA within noise of the vendor on dense QR fronts") but is *not* the M6-style strict win.
- **(c) Vendor-hybrid floor** — cuSOLVER geqrf panels + owned trailing:
  **0.90–0.96× [MEASURED projection]** (per-shape: 0.961, 0.936, 0.939, 0.917, 0.909, 0.896),
  contingent only on the trailing kernels actually reaching γ=1.14 with φ≈1 (standard-V format —
  note the hybrid then *does* face the tall-K `tn` shape; if that kernel only reaches cuBLAS
  parity the hybrid floor degrades to ≈1.0×, i.e. no win at all). This is the M6a-style interim
  and the shipping fallback.
- **Gate restructure consequence:** clause 1 (≥1.0× geqrf per shape) is the *headline bet*,
  decided by Phase-0b/0c — it may honestly land at (b) and the docs will say so. Clause 2
  (beat SuiteSparseQR end-to-end on the pinned stratum) is the *must-win* and is winnable under
  all three branches — even a 1.5× -vs-geqrf front kernel dwarfs CPU SPQR on crown-dominated
  problems (M6 precedent: ≤51× vs CHOLMOD). M7 does not become "a non-goal" in any branch; only
  the *purity of the panel* is at risk, and it fails toward a recorded hybrid, exactly as M6a did.

Nothing in this section is oversold: (a) and (b) rest on two projections named as such; (c) rests
on one assumption (γ on new shapes) explicitly scheduled for measurement.

## §R6 Build + probe order (deliverable 4)

**Phase-0b — the next measurement, before any production code** (script
`benchmark/gpu/qr_tsqr_phase0.jl`, same methodology as the Phase-0 probes: warm `CUDA.@elapsed`
medians, JSON to `benchmark/results/`, run on galen):

- **P0b-0**: 8 concurrent single-WG panels — verifies §R1's occupancy-bound mechanism (cheap,
  minutes).
- **P0b-1**: K1 optimized shared-resident block QR standalone; sweep `rb ∈ {128,192,256}`,
  `nb ∈ {16,32,48}`. **Kill/branch threshold**: block latency ≤ 0.3 ms at 192×32 keeps the
  §R2.4 chain inside every ρ budget; > 0.45 ms means the 3-launch chain cannot beat the small
  crowns' ρ=1.03 and the pure panel is dead again.
- **P0b-2**: full TSQR panel (K1 + K2 chain) vs geqrf on the probe-1 shape grid → **r_tsqr**.
  Branch: r_tsqr ≤ ~1.0 → pure panel confirmed; 1.0–1.4 → pure on big crowns only, hybrid on
  small (per-shape dispatch, recorded); > 1.4 → vendor-hybrid panel, clause 1 rescoped to (c).
- **P0c**: K3/K4/K5 batched trailing chain vs (`cublasDgemmStridedBatched` composition) and vs
  the vendor's own trailing (geqrf_front − panels, the probe-2 methodology) → measured **γ** and
  realized **φ**. Also probes the standard-V tall-K `tn` shape (needed by the hybrid floor and
  the CPU-format solve replay).
- **P0d**: re-run the front projection (`qr_front_project.jl` methodology, same conservatisms)
  with measured TSQR panels + measured γ/φ instead of the 1.14 assumption → the go/no-go for
  gate clause 1, and the final (nb, rb, q) pick.

**Then** the v2 §Q5 build order 1–7 proceeds unchanged except: step 3's panel scheme is fixed by
P0b (TSQR or hybrid), step 6 is replaced by §R4's scan-and-refactor (which also moves *before*
the gate, satisfying the v2 sequencing fix), and the §R2.3 format tag + two solve replay
families are added to step 5. Every probe result is committed to `benchmark/results/*.json`
before design text claims it (standing rule: regenerate from saved data, never re-run).

## §R7 Amended memory budget (delta to §Q2.5)

Permanent: V in place (unchanged `Σ_f mmax_f·n_f`) **+ packed merge-V slabs (+5–8%)** + per-node
tau + `nnzRF`. **Removed from permanent: T slabs** (now arena-transient, §R2.3) — net change
roughly neutral-to-negative. Arena: transient C (unchanged) + per-panel T working set. The
`gpu_capacity_ok` loud fallback and the stratum feasibility criterion carry over verbatim.

## §R8 Provenance (clean-room, absolute)

TSQR/CAQR: Demmel–Grigori–Hoemmen–Langou (SIAM JSC 2012) — published. Householder
reconstruction: Ballard et al. (IPDPS 2014) — published, cited-and-rejected. Staircase panel
mechanics, dead-pivot semantics, dropped-mass bound: M5b (own design, SPQR *paper* Davis 2011 +
faer MIT). Block-kernel structure, arena, frontier, batching, AMD constraints: M6 (own kernels).
cuSOLVER/cuBLAS: black-box baselines only; no vendor or SuiteSparse source read. Every constant
in this document (rb, q, ε_m, thresholds) is derived in-document from our own measurements or
shared-memory arithmetic — none matches a vendor default by construction.
