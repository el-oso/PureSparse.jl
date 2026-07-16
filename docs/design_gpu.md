# PureSparse.jl — GPU offload (M6) design

**Status: v2 (2026-07-17).** v1 → two independent blind adversarial reviews (Opus-family:
3 BLOCKER/9 DEFECT/3 NIT; Fable: 3 BLOCKER/8 DEFECT/7 NIT, non-overlapping except two shared
BLOCKERs) → this v2. Supersedes design.md §8 and the stale ROADMAP `### M3` (which
contradicted each other). This is the single live GPU design; it takes the same v1→review→v2
path design.md and design_qr.md took. v2 additionally folds a **verified Phase-0 result that
flips the kernel strategy** (§3).

Inputs: Fable's M6 architecture review; the Phase-0 measurement pass on galen; two user
decisions (scope = Cholesky+LDLᵀ; kernel = pure-primary, Option 1); the two v1 reviews.

---

## §0 Changelog v1 → v2 (review anchors; each fix traced by finding)

**Kernel-strategy flip (verified Phase-0 result, not a review finding).** v1's central risk
("can pure Julia beat cuBLAS?") is **resolved: yes.** A pure-Julia KernelAbstractions FP64
gemm (`C=A·Bᵀ`, the supernode trailing-update shape) beats cuBLAS by **1.07–1.19×**
(1.14× at K≥128: 344–350 GF vs cuBLAS 300–308), **bitwise-identical** (relerr 0), and
KA == raw CUDA.jl (1.00× — vendor-portable for free). Root cause of v1's 0.48× number:
Julia is IEEE-strict and does not contract `a*b+acc` into an FMA without `muladd`; both v1
kernels issued DMUL+DADD separately (2× FP64 instructions → the identical 148 GF plateau).
`muladd` → 349 GF = 99% of FP64 peak at galen's locked 1920 MHz. Reproduced independently.
`benchmark/gpu/ka_gemm.jl`, `benchmark/results/gpu_kernel_ka_final_galen.json`.
**Consequence:** the pure KA kernel is the SHIPPED hot-path kernel (§3), not future R&D.
This **dissolves** both reviewers' "milestone closes on a vendor binary, against the Pure
ethos" objection and simplifies the whole story: dense work is *ours* on device too, exactly
as PureBLAS is on CPU.

**BLOCKERs fixed:**
- **[both reviews] M6b LDLᵀ was mis-scoped as a "cheap delta."** `ldlt.jl` has no separable
  diagonal factorization to keep on CPU — steps 3+4 are one fused right-looking column loop
  over the full panel height (signed regularization + full-height `inv(dj)` scale + `ger!`
  rank-1 update to rows `(j+1):nsrow`; no `trsm` panel-solve). Rewritten as its own real
  slice with a **blocked device-LDL** formulation (§6), its own kernels and oracle. "Cheap"
  deleted.
- **[both] gate dropped req 2's mandatory `GivenOrdering` same-permutation arm.** Restored
  in all relevant clauses (§8.1).
- **[Opus] "race-free by construction" is false under concurrency.** The frontier property
  (no GPU→CPU update edge) is real and executable-checkable (§5.2, verified by Fable against
  the §3.4 superset invariant), but concurrent CPU/GPU execution + async uploads needs an
  **explicit stream/event dependency barrier** (§5.4). "by construction" deleted.
- **[Fable] pivot-failure detection undesigned + is a hidden per-supernode device→host
  sync.** Added failure-semantics §4.3 (deferred batched `devinfo`, amended `fail_col`).
- **[Fable] gate timed unequal end states** (GPU factor not solve-ready without a D2H that
  recurs per IPM refactor; the 170 vs 308 ms figures were inconsistent). Gate timed region
  redefined to **refactor + make-solve-ready** (§8.1); D2H cost reworked per-refactor with
  reconciled measured bandwidth (§7).

**DEFECTs fixed:** capacity check uses `max_extend_rows²` not the abandoned
`max_update_size`, adds an uploaded-CPU-panel budget term, reworded "exact"→"queried-free-mem
+ margin, loud fallback" (§5.3); generic-`T` story made explicit — Float64 gated, Float32
works, other `T`→CPU fallback (§1, §3.3); 2× margin marked provisional-pending-stratum and
derived from *achieved* not peak (§8.1); the "0.48× not a ceiling" claim is now the measured
result (§3); zero-alloc amendment given a concrete host-byte bound + pre-allocated cuSOLVER
workspace/`devinfo` via `bufferSize` (§9.A); vendor-ethos concern resolved by the flip;
"small supernodes stay on CPU" softened for forced near-root ancestors (§5.2); scatter
attribution corrected (`ir`/`rs`, not `relmap`) and the run-structure scan located on host
(pattern-only, refactor-invariant) (§4.2); added the **third** contract amendment
(on-device dense-kernel path, §9.C); added missing sections — user-facing API (§2.2), type
sketches (§2.3), executable frontier invariants (§5.2/§10.2), gate stratum spec (§8.3),
stream/event design (§5.4).

**NITs:** `unsafe_wrap` is a CPU-compile workaround, irrelevant on device (device uses
strided `CuArray` views) — F3 wording fixed; memory figures cite `CUDA.available_memory()`
on galen; oracle phrasing made normwise (design.md §9.2); the CHOLMOD-GPU skip is justified
by "not what Julia ships," not clean-room.

---

## §1 Scope

- **M6a — supernodal LLᵀ (`cholesky!`) GPU offload.** Float64 gated; kernels generic-`T`.
- **M6b — supernodal LDLᵀ (`ldlt!`) GPU offload** (§6). Real slice (blocked device-LDL), not
  a delta — but it reuses M6a's frontier, scheduler, uploads, and gemm/syrk kernels; the new
  work is the device diagonal-block LDL + D-scaled panel update + `dvec` residency.
- **Element types:** Float64 is the gated path. Float32 works (pure kernels are generic-`T`;
  cuSOLVER `potrf`/`trsm` support F32). Any other `T` (Duals, BigFloat, complex) **falls back
  to the CPU path** via dispatch — cuSOLVER can't run them and the interim `potrf`/`trsm` are
  vendor-typed; the pure gemm/syrk are generic but the diagonal factorization is not, until
  §6's pure device-LDL/potrf lands. Stated so the generic-`T` promise isn't overclaimed.
- **OUT of M6:** sparse QR (different multifrontal-WY arch; gate already closed vs SPQR).
  Simplicial update/downdate (latency-bound; stays CPU). Device solves (§7 — solves stay CPU
  in M6; the gate accounts for the factor D2H).

Primary gate number: **warm refactor + make-solve-ready** (§8.1) — the IPM-relevant path.

## §2 Extension architecture

### §2.1 Weak-dep extension
`ext/PureSparseCUDAExt` (mirrors `ext/PureSparseForwardDiffExt.jl`). `[weakdeps]`:
`CUDA`, `KernelAbstractions`. `[extensions]`: `PureSparseCUDAExt = ["CUDA","KernelAbstractions"]`.
**Zero hooks in `src/`**: the ext defines `GPUSymbolic`/`GPUSupernodalFactor` and adds
methods to the existing generic entry points; dispatch is type-driven (GKH rule — ownership
resolved at compile time by argument type, no runtime registry). Core stays trim-compatible;
the trimmed CPU build never loads the ext (§11). A CI job proves the package loads and the
full CPU suite passes with CUDA absent.

### §2.2 User-facing API (how to opt in)
GPU offload is **opt-in via a backend-typed symbolic**, not an implicit rewrite of
`cholesky(A)`:
```julia
using PureSparse, CUDA
S  = PureSparse.symbolic(A; ordering=AMDOrdering(), backend=CUDABackend())  # → GPUSymbolic
F  = PureSparse.cholesky(S, A)          # → GPUSupernodalFactor (device-resident)
PureSparse.cholesky!(F, A2)             # warm refactor on device
x  = F \ b                              # triggers make-solve-ready D2H (§7), CPU solve
```
`backend=CUDABackend()` (a `KernelAbstractions.Backend`) is the single switch; absent it,
`symbolic` returns the CPU `Symbolic` unchanged. This makes §8.1 clause 2 ("no regression on
the existing set") **well-posed**: plain `cholesky(A)` never touches ext code, so the
existing gate is untouched by construction; the GPU path is measured only on `GPUSymbolic`.
The `backend` argument also carries the vendor (CUDABackend/ROCBackend/oneAPIBackend) for the
portable kernels — only CUDA is verified in M6 (galen), others compile but are untested and
gated off with a `@warn`.

### §2.3 Type sketches (host vs device residency)
```
GPUSymbolic{Ti}                         # immutable, shared by reference
  cpu::Symbolic{Ti}                     #   the full CPU symbolic (etree, supernodes, relmap…)
  backend::KA.Backend
  on_gpu::Vector{Bool}    (host)        #   frontier membership per supernode (§5.2)
  gpu_order::Vector{Ti}   (host)        #   GPU supernodes in ascending-finalize order
  d_rowind, d_rowptr, d_super, d_snode_of, d_relmap   (device)  # pattern arrays, uploaded once
  d_amap                  (device)      #   assembly map A-values → panels
  workspace_bytes::Int    (host)        #   pre-sized device workspace need (§5.3)

GPUSupernodalFactor{T,Ti}
  sym::GPUSymbolic{Ti}
  d_nzval::CuVector{T}    (device)      #   the factor L (device-resident)
  d_cbuf::CuMatrix{T}     (device)      #   scatter workspace, max_extend_rows² (§5.3)
  d_potrf_ws::CuVector{T} (device)      #   pre-allocated cuSOLVER potrf workspace (§9.A)
  d_devinfo::CuVector{Cint} (device)   #   pivot-failure flags, one per GPU supernode (§4.3)
  d_uploadbuf::CuVector{T} (device)    #   bounded ring for CPU-descendant panel uploads (§5.3)
  host_mirror::SupernodalFactor{T,Ti}  #   filled on make-solve-ready (§7); solves run here
  ok::Bool, fail_col::Ti  (host)       #   resolved post-hoc from d_devinfo (§4.3)
  streams::NTuple{2,CuStream}          #   compute + upload streams (§5.4)
```

## §3 Kernel strategy (Option 1: pure-primary)

Dense per-supernode work on device goes through a small interface —
`gpu_gemm!`/`gpu_syrk!`/`gpu_trsm!`/`gpu_potrf!` — with a selectable backend, defaulting to
**pure**:

- **Pure KA kernels (default, shipped hot path):** the verified 4×4-register-blocked
  `muladd` kernel (§0) for the **trailing update** (gemm/syrk — the flop-dominant step, where
  the 1.14× win is), generic over `T`, portable (AMD/Intel via KA). A **syrk-shaped variant**
  (compute only the lower/upper block for the symmetric self-update) and the `alpha=-1,beta=1`
  epilogue (the actual `C -= A·Bᵀ` update) are the two productization deltas — both already
  supported by the kernel's epilogue args.
- **cuSOLVER `potrf`/`trsm` (interim) for the small diagonal blocks:** the diagonal
  factorization and off-diagonal solve are low-flop (the diagonal block is `nscol×nscol`,
  typically ≪ the trailing update). Pure device `potrf`/`trsm` are a **follow-up** (§6 needs a
  blocked device factorization anyway; the pure potrf falls out of it). Float32/64 only —
  another reason other `T` falls back to CPU (§1).
- **cuBLAS/cuSOLVER kept as a wired, selectable backend** for two jobs, **not** the default
  hot path: (a) an **in-loop correctness reference** — run a GPU supernode's update on both
  backends and assert bitwise-identical factors, inside the real sparse loop; (b) the
  **benchmark baseline / context arm** (pure-vs-cuBLAS attribution, §8.2).

Backend selection is a Preferences-baked const, so swapping backends is a flag, not a loop
rewrite. **Caveat (Fable D5):** because `potrf`/`trsm` are cuSOLVER-only in M6a, the
"backend flip" is really *per-op* (pure gemm/syrk + vendor potrf/trsm) until §6's device
factorization lands; the interface reflects this (four independent op hooks, not one switch).

### §3.3 generic-`T` reality
gemm/syrk pure kernels: generic-`T` (verified Float32/Float64, relerr 0). potrf/trsm: F32/F64
(cuSOLVER) in M6a. So a `GPUSymbolic` is only constructed for `T ∈ {Float32,Float64}`; other
`T` returns a CPU `Symbolic` with a `@warn` (dispatch-level fallback). Full generic-`T` GPU
factorization is unlocked only when §6's pure device-LDL/potrf replaces cuSOLVER — tracked,
not promised for M6a.

## §4 Offloaded LLᵀ loop (derived from current `llt.jl`)

The left-looking driver stays on CPU. For each **GPU-side** supernode `s` (§5.2), its
descendant updates and its own factorization run on device.

### §4.1 Trailing update — preserve the contiguity fast path
`llt.jl`'s common case: a descendant `d`'s remaining rows form a contiguous run of `s`'s row
list → syrk/gemm with `β=1` **straight into the ancestor panel**, no staging. Preserved on
device: the panel is already resident (§5.1), so this is `gpu_syrk!`/`gpu_gemm!` with `β=1`
into a `CuArray` sub-view (strided view; the `unsafe_wrap` CPU-compile workaround is
irrelevant on device — NIT fix). The non-contiguous case scatters through `d_cbuf` + a device
scatter kernel.

### §4.2 Scatter + run-structure (attribution fix)
`_scatter_update!` consumes `ir`/`rs` (the run structure), **not** `relmap` (v1 error). The
`ir`/`rs` run-detection scan and the k1-split/contiguity decision are **pattern-only and
refactor-invariant**, so they are computed **once on the host** at `GPUSymbolic` build time
and stored (host keeps the pattern → **zero pattern H2D per refactor**, satisfying §9.A's
"0 pattern H2D" gate). The device scatter kernel consumes device copies (`d_relmap` for the
final scatter target map; `ir`/`rs` uploaded once as device arrays). The scatter kernel itself
is a simple indexed add (pure Julia — it is the sparse part, and it is genuinely a trivial
kernel *given* the precomputed run structure; the non-trivial scan is host-side and one-time).

### §4.3 Failure semantics (pivot detection) — Fable BLOCKER
`cholesky!` sets `F.ok=false`, records `fail_col`, and returns early on a non-SPD pivot.
cuSOLVER `potrf` reports failure via a **device** `devinfo`. To avoid a per-supernode D2H sync
(which would serialize the async pipeline), M6 uses **deferred batched detection**: each GPU
supernode writes its `potrf` result into its own slot of `d_devinfo` (pre-allocated, one Cint
per GPU supernode); the whole array is D2H'd **once** at the end of `cholesky!`. `F.ok` and
`fail_col` are resolved post-hoc: `ok = all(devinfo .== 0)`; `fail_col` = the column offset of
the lowest-index failed supernode. **Amended semantics (user-visible, → §9):** on the GPU
path a failed pivot does **not** early-return — later supernodes still compute (possibly on
NaNs). This matches "basic solution / query `issuccess` before use" discipline but differs
from the CPU early-return; `check_finite` is the backstop. Recorded on the §9 sign-off list.

## §5 Memory model + frontier + concurrency

### §5.1 Device-resident factor
`nnzL` exact at `symbolic` time. The factor L lives on device (`d_nzval`); no per-supernode
staging because descendant panels are already resident.

### §5.2 Upward-closed etree frontier (replaces `gpu_flop_threshold=2e9`)
At `symbolic` time: mark each supernode with (update+factor) flops ≥ `frontier_cutoff`, then
take the **upward closure** in the supernodal etree. **Fable verified** the key property: every
update target of supernode `d` is `snode_of[r]` for `r ∈ rowind(d)`, which by the §3.4 superset
invariant is an *ancestor* of `d` — so an upward-closed GPU set **never emits a device→host
update edge**. Traffic is therefore one-way (CPU→GPU only) and once-only (§5.4). The single
tunable is `frontier_cutoff`; its default comes from the Phase-0 measured CPU-vs-GPU per-shape
crossover on galen (§8.3), with a derivation comment. **Honesty (Opus DEFECT):** upward closure
can pull a small, low-flop supernode onto the GPU if it sits above a heavy subtree; "small
stays on CPU" holds only for nodes *not* on a GPU-ancestor path. Near-root supernodes are
typically dense so this is usually negligible; the frontier builder may *puncture* the closure
for a trivially-small near-root node by keeping it on CPU and uploading its (small) panel —
measured, not assumed.

### §5.3 Capacity (reworded — not "exact")
Budget against **queried** free memory (`CUDA.available_memory()` on galen: ~11.57 GiB after
context) at device-buffer-allocation time, with a stated safety margin: `d_nzval` (nnzL·8) +
`d_cbuf` (`max_extend_rows²`·8 — note: `ws.c` in `types.jl` is sized `max_extend_rows²`;
`max_update_size` is a *diagnostic*, not a bound — v1 error fixed) + `d_potrf_ws` (cuSOLVER
`bufferSize` over supernode shapes, §9.A) + `d_uploadbuf` (bounded ring for CPU-descendant
panels, sized `max_extend_rows × max_gpu_panel_cols`, freed-after-consume — so uploaded panels
are NOT a growing term; "upload once" means once *per refactor into the ring*, consumed then
reused) + A-values pinned staging (nnz(A)·8). If the sum exceeds free-memory−margin, **fall
back to the CPU path loudly** (`@warn` + return a CPU factor) — the *loudness* is the
guarantee, never an OOM at factor time.

### §5.4 Stream/event concurrency (Opus BLOCKER — the barrier)
Two streams: `compute` (GPU supernode work) and `upload` (async H2D of finalized CPU-side
descendant panels). Left-looking finalizes each panel before any consumer reads it *on a
single processor*; with concurrency that ordering must be **enforced**, not assumed: (1) when a
CPU subtree finalizes a panel that a GPU ancestor consumes, the host enqueues its H2D on
`upload` and records a `CuEvent`; (2) a GPU supernode `s`'s first update kernel is enqueued on
`compute` only after `compute` waits on the events of all its CPU-descendant uploads
(`CUDA.wait(event, compute)`). This is a DAG of stream dependencies mirroring the etree edges
crossing the frontier. Cost: the events are cheap; the real constraint is that a GPU supernode
cannot start until its cross-frontier descendants are uploaded — which the frontier's
upward-closure already batches (all cross-edges enter the GPU set from below, once). "race-free
by construction" is replaced by "race-free under this explicit event DAG," and §10.2 adds an
executable upload-once + ordering check.

## §6 M6b — LDLᵀ device slice (real work, not a delta)

`ldlt.jl` steps 3+4 are one fused right-looking column loop over the full panel height:
per-column signed pivot `dj` (regularization + inertia), full-height scale `panel[i,j]*=inv(dj)`
for `i∈(j+1):nsrow`, and `ger!` rank-1 trailing update — **no `trsm` panel-solve to keep on
CPU**. So "diagonal LDL on CPU, offload the rest" is impossible. The honest device formulation:

- **Blocked device-LDL.** Factor the `nscol×nscol` **diagonal block** into `L11·D·L11ᵀ`
  (small; either a device block-LDL kernel or a diag-block-only D2H→CPU→H2D — the block is
  small, so the round-trip is bounded and does *not* touch the flop-dominant tall panel).
- **Device D-scaled panel solve.** The below-diagonal panel `L21` = `A21·L11⁻ᵀ·D⁻¹` via a
  device unit-`trsm` (against `L11`) + a `D⁻¹` column-scale kernel (pure, generic-`T`).
- **Device `L·D` trailing update.** The descendant update uses `L21·D·L21ᵀ` — one extra
  column-scale (`W = L21·D`, a `dvec`-slice broadcast) before the pure gemm/syrk. `dvec`
  lives on device (`d_dvec`); D-slices are pushed H2D once per refactor with the pattern.
- **Inertia:** `(n_pos,n_neg,n_zero)` reduced from `d_dvec` signs on device, D2H'd once
  (like `d_devinfo`).

This keeps §5's one-way/once-only traffic for `ldlt!` (only the small diag block round-trips,
bounded; the tall panel stays device-resident). It is **new numeric code with its own
BigFloat/CPU-`ldlt` oracle and its own zero-alloc gate** — M6b, sequenced after M6a's gate.

## §7 Solves + the D2H (Fable BLOCKER)

Factor is device-resident; solves run on CPU after a **make-solve-ready** step that D2H's the
GPU-slice of the factor into `host_mirror`. Reconciled cost: at galen's measured **pinned**
D2H bandwidth (Phase-0: 13.4 GB/s; v1's "170 ms for 4 GB" was wrong — 4 GB ÷ 13.4 GB/s ≈
300 ms), a device factor of `g` GB costs `g/13.4` s to make solve-ready. **This recurs per
warm refactor** in the IPM loop (each refactor invalidates the mirror). Therefore:
- The gate's timed region **includes** make-solve-ready (§8.1) — the GPU arm is not credited a
  free D2H.
- An **IPM-cycle context arm** (§8.2) reports refactor+D2H+solve×k both ways, so the per-cycle
  economics are visible, not hidden.
- Device solves (eliminating the recurring D2H for repeated solves on one factor) are the
  obvious next increment but are **out of M6** — flagged, with the cost stated so the decision
  is data-driven.

## §8 Gate

### §8.1 Definition (req 2, corrected)
On the §8.3 large-matrix stratum, all median wall-time, **single-thread CPU methodology**
(stated so it's not mistaken for threaded-CPU), timed region = **warm refactor +
make-solve-ready**:
1. GPU-enabled `cholesky!`+D2H ≥ **2×** faster than our own single-thread CPU PureSparse.
   *(2× is **provisional pending §8.3 measurement**; it will be derived from the *achieved*
   crossover — cuBLAS-class 305 GF / our 349 GF pure vs measured single-thread DGEMM ~55–65
   GF ≈ 5–6× dense headroom, discounted by assembly/scatter/launch/D2H Amdahl — not from the
   455 GF peak.)*
2. **No regression** on the existing M1/M2 gate set: with the auto frontier, small/medium
   matrices stay on CPU (or the ext isn't even constructed, §2.2) → regression ≤ the harness's
   established locked-clock run-to-run band (a concrete number from galen/wintermute, not
   "noise").
3. GPU-enabled PureSparse **still beats CHOLMOD+OpenBLAS** on the stratum, **both own-ordering
   AND under an identical `GivenOrdering(p)` permutation** (req 2's mandatory same-perm arm —
   restored; v1 dropped it).

### §8.2 Context arms (reported, not gated)
cuDSS (NVIDIA sparse, black-box, like faer for QR); pure-vs-cuBLAS kernel attribution (§3);
the §7 IPM-cycle cost both ways.

### §8.3 Stratum spec (pin criteria BEFORE measuring — anti-cherry-pick)
Selected now, not after GPU numbers exist: SPD + SQD problems, `nnzL·8 + workspace ≤ ~9 GB`,
from the SuiteSparse collection FEM/KKT/Laplacian classes (candidate: Fault_639, Emilia_923,
Serena, audikw_1 — exact `nnzL` checked at selection) + synthetic large 3-D-grid Laplacians and
random-KKT (design.md §9.4 permits synthetics). ≥6 matrices, ≥2 SQD. The stratum + CPU
baselines are the remaining Phase-0 item, run BEFORE the frontier cutoff is fixed.

## §9 Contract amendments (require explicit user sign-off)

**A — req 5 (zero-alloc) on GPU.** Verbatim `@allocated==0` is impossible (kernel launches
allocate host bytes; cuSOLVER queries workspace). Proposed: *"warm `cholesky!`/`ldlt!` on a
GPU factor: **0 device-pool allocations after setup** (cuSOLVER workspace + `devinfo` +
`cbuf` + upload ring pre-allocated at factor construction via `bufferSize`; low-level cuSOLVER
API, not the auto-workspace wrappers) AND **0 pattern H2D** (§4.2); host bytes per warm
refactor **≤ a constant independent of n and nsuper** (launch bookkeeping only — measured on
galen, gated as a ceiling, not merely reported)."* The host-side floor is measured first
(remaining Phase-0), then gated.

**B — req 2 (gate baseline) on GPU.** The §8.1 three-clause gate, including the restored
`GivenOrdering` same-perm arm and the refactor+D2H timed region.

**C — dense-kernel exclusivity on device.** CLAUDE.md says CPU dense work goes *exclusively*
through PureBLAS. Proposed: *"dense per-supernode work goes through PureBLAS on CPU and through
the §3 pure-KA device-kernel interface on device; cuSOLVER `potrf`/`trsm` are an explicit
interim for the small diagonal blocks (Float32/64), replaced by pure device kernels when §6's
device factorization lands; cuBLAS is a reference/baseline backend, never the default hot
path."* (The pure-primary flip makes this largely self-satisfying — the shipped hot-path
kernel is ours.)

## §10 Correctness + invariants

### §10.1 Numeric oracle
Same `A` → CPU factor vs GPU factor, **normwise** `‖L_gpu−L_cpu‖ ≤ c·n·eps(T)·‖A‖^{1/2}`
(design.md §9.2 methodology; device reduction order differs → tolerance-based, calibrated on
dense potrf first). In-loop pure-vs-cuBLAS bitwise check (§3). Full stratum + winnable-zoo
sweep; `--check-bounds=yes` device run; StrictMode preconditions on the device path.

### §10.2 Executable frontier invariants (H-analogues, design.md §9.1 layer 3)
Cheap symbolic-time assertions turning §5's prose into gates:
- **Upward-closure:** `∀ s ∈ GPU-set, ∀ r ∈ rowind(s): snode_of[r] ∈ GPU-set` (no
  device→host update edge). O(nnz-pattern).
- **Upload-once:** a per-refactor counter asserts each CPU-descendant panel consumed by a GPU
  ancestor is H2D'd exactly once (§5.4).
- **Solve-ready:** after make-solve-ready, `host_mirror` equals a full-CPU factor within §10.1.

## §11 Trim + zero-alloc

Trim gate extended **early**: the `juliac --trim` smoke runs against the weakdep-bearing
`Project.toml`, proving the ext's existence doesn't perturb the trimmed CPU build (the ext is
not in the trimmed image). Alloc/transfer discipline per §9.A, tested in the
StrictMode-checks-disabled configuration like the CPU gate.

## §12 Task list

**Phase 0 (DONE):** CUDA.jl functional on galen; **pure KA FP64 gemm beats cuBLAS 1.14×,
portable, generic-`T`, relerr 0** (`benchmark/gpu/ka_gemm.jl`); launch latency 5.5 µs; PCIe
pinned 13.4 GB/s. Remaining Phase-0: §8.3 stratum selection + CPU baselines + host-alloc floor
(feeds §8.1's 2×, the frontier cutoff, and §9.A).

**Phase 1:** this v2 → two independent adversarial reviews (Fable + Opus) → v3. Carries the
three §9 amendments for user sign-off.

**Phase 2 (each step lands with tests; GPU items run on galen, rsync+verify first):**
1. Ext scaffolding (`CUDA`/`KernelAbstractions` weakdeps; loads-with-CUDA-absent CI job).
2. Trim gate vs the weakdep `Project.toml` — first, not last (§11).
3. Pure device dense-kernel module: gemm/syrk (the proven kernel) + syrk-shape + α=−1/β=1
   epilogue; unit-oracle vs cuBLAS + CPU PureBLAS, generic-`T` (F32/F64).
4. `GPUSymbolic`: frontier partition + §10.2 invariants; one-time pattern upload; upload-once
   test.
5. `GPUSupernodalFactor`: device buffers + pre-allocated cuSOLVER workspace/`devinfo` (§9.A) +
   capacity check + loud CPU fallback (§5.3).
6. Device assembly + scatter kernels (§4.2), unit-oracled.
7. Hybrid `cholesky!` loop (§4): CPU subtrees untouched; stream/event DAG (§5.4); GPU
   supernodes factor on device (pure gemm/syrk + cuSOLVER potrf interim); deferred `devinfo`
   (§4.3).
8. make-solve-ready D2H + CPU solve (§7); numeric oracle (§10.1) full-zoo + `--check-bounds`.
9. Alloc/transfer discipline (§9.A) in checks-disabled config.
10. Gate on galen (§8) + §8.3 stratum + context arms; frontier cutoff calibrated + documented.
11. **M6b:** blocked device-LDL (§6) — its own kernels, oracle, zero-alloc gate; then re-gate.

**Track 2 (parallel, mostly landed):** pure gemm/syrk **done and winning**; remaining is pure
device `potrf`/`trsm` (falls out of §6) to retire the cuSOLVER interim and unlock full
generic-`T` + full vendor-portability. Not blocking M6a's gate.

## §13 Clean-room (restated for GPU)

CHOLMOD's GPU module is GPL — never read; concept papers only (Rennich et al. 2016).
cuBLAS/cuSOLVER/cuDSS are closed NVIDIA binaries: used black-box (reference/baseline) or
reported baselines; never disassembled/source-inspected. JuliaGPU packages (CUDA.jl,
KernelAbstractions.jl, GemmKernels.jl) are MIT, freely readable — unrelated to the SuiteSparse
prohibition. The CHOLMOD-GPU baseline is skipped because it is *not what Julia ships* (stdlib
SuiteSparse_jll has no CUDA), not for clean-room reasons (we already benchmark GPL CHOLMOD's
CPU build black-box). Every constant (frontier cutoff, memory cap+margin, the 2× gate margin)
carries a measurement citation or a derivation.
