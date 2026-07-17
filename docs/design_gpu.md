# PureSparse.jl — GPU offload (M6) design

**Status: v3 (2026-07-17).** v1 → 2 blind reviews → v2 → 2 more blind reviews (Opus + Fable,
each verified against source) → this v3. The v2 reviews **converged**: both independently found
the same two BLOCKERs (the cross-frontier upload/concurrency model; the M6b inertia mechanism),
neither a wording fix. v3 reworks those two areas and folds the defects. Supersedes design.md §8
and the stale ROADMAP `### M3`. Same v1→review process design.md and design_qr.md took, one round
deeper (the GPU failure domain — async execution, device memory, remote-only verification —
earned it).

Inputs: Fable's M6 architecture review; the Phase-0 measurement pass on galen (incl. the verified
pure-kernel win, §0); user decisions (scope = Cholesky+LDLᵀ; kernel = pure-primary, Option 1);
four adversarial reviews across two rounds.

## §0′ Changelog v2 → v3 (both reviewers converged — fixes are well-determined)

**BLOCKER 1 [both] — upload/concurrency model was self-contradictory.** A cross-frontier CPU
panel is consumed by *every* GPU ancestor on its etree chain (k≥2 is the common case, since
upward closure makes all of a boundary node's ancestors GPU), at separated schedule points — so
v2's "upload once + freed-after-consume ring" could not hold, and the per-*supernode* upload
barrier risked deadlock/OOM. **v3 fix (§5.3/§5.4):** the granularity is per-*descendant* — each
descendant-update kernel waits on that descendant's own upload event — and cross-frontier boundary
panels are **persisted device-resident for the whole refactor** (uploaded genuinely once, freed at
end), with a Σ-over-boundary-panels **budget term** (symbolic-time computable) replacing the ring.
No barrier, no ring, no deadlock; "upload once" becomes literally true.

**BLOCKER 2 [both] — M6b inertia mechanism was non-equivalent and destroyed the req-8 report.**
`ldlt.jl` classifies inertia on the **pre-perturbation** pivot (with a running-`dmax` zero test),
then overwrites `dvec` with the regularized value — so reducing inertia from the final `d_dvec`
gives tautological *forced* signs, defeating the whole point (IPOPT consumers read *observed*
inertia). And the running-global-`dmax` is a sequential cross-supernode dependency concurrency
can't reproduce. **v3 fix (§6):** the device block-LDL runs the signed-regularization column loop
itself and emits per-supernode pre-perturbation stats `(n_pos,n_neg,n_zero,n_perturbed,max_pert,
dmin,dmax)` into a device stats array reduced once at end (like `d_devinfo`); the zero-test is
redefined **order-free** (per-supernode-local `dmax`, delta-anchored) and recorded as amendment E;
`ascale` computed host-side during the A-staging pass.

**DEFECTs folded:** kernel β=0 must **overwrite** (BLAS semantics; `0*NaN=NaN` corrupts the
uninitialized `d_cbuf`) — stated as a kernel requirement + fixed in the ext (§3/§4.1); the
syrk-shape is **not** an epilogue-arg freebie (triangular masking is real kernel work — but the
full-gemm-overwrites-unused-upper approach is legitimate since the diagonal block's strict-upper
is never read, so M6a uses that, §3); the pure flip "dissolves the vendor-binary objection" claim
**scoped** to the flop-dominant trailing update (M6a still uses cuSOLVER for the pivoted diagonal
factorization, §0/§3); D2H/H2D priced at **pageable** bandwidth unless host storage is pinned/
registered (§7, budgeted §5.3); §9.A host bound restated `≤ c·(#GPU launches)` not constant;
hybrid **CPU-side** pivot failure must sync both streams before early-return (§4.3) + new amendment
D; §4.2 scatter uses per-pair `ir`/`rs` (pattern-only, host-precomputed, uploaded once), `relmap`
dropped (it's a transient, not a pattern array), bytes budgeted (§5.3); §5.2 "puncture" option
**deleted** (it would create the device→host edge the invariant forbids); §8 adds a
**multi-threaded CHOLMOD** context arm (single-thread gate else reads as GPU-vs-one-core), and a
**multi-device** confirmation note (galen + a coming neuromancer eGPU → the two-host gate bar; an
AMD eGPU would also verify the KA portability claim). **NITs:** median not min for the gate (§8);
complex excluded by scope not cuSOLVER capability (§1); `§5.2` cites the etree row-subtree property
(not §3.4 superset); workspace stored as element counts (§2.3); added M6b `GPULDLFactor` sketch +
`W=L21·D` staging buffer term (§2.3/§6); hybrid-driver pseudocode (§5.5).

Inputs to v2 (unchanged, for the record): Fable's M6 architecture review; the Phase-0 measurement
pass on galen; two user decisions; the two v1 reviews.

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
  cuSOLVER `potrf`/`trsm` support F32). Other `T` (Duals, BigFloat) **fall back to the CPU
  path** via dispatch — the interim cuSOLVER `potrf`/`trsm` don't run them and the pure diagonal
  factorization isn't built until §6. Complex is excluded by **scope**, not capability (cuSOLVER
  *does* support `Complex{F32,F64}` potrf) — SPD/SQD Hermitian GPU factorization is a later
  increment. Stated so the generic-`T` promise isn't overclaimed.
- **Device solves are IN (v4, user-directed 2026-07-17):** the gate path is symbolic(CPU, once)
  → numerical factor(GPU) → solve(GPU); the factor stays device-resident, no full-factor D2H
  (§7). Supernodal device triangular solves are new kernels (§12).
- **OUT of M6:** sparse QR (different multifrontal-WY arch; gate already closed vs SPQR).
  Simplicial update/downdate (latency-bound; stays CPU).

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
  boundary::Vector{Ti}    (host)        #   CPU supernodes with a GPU ancestor (§5.3 persist set)
  d_rowind, d_rowptr, d_super, d_snode_of  (device)  # pattern arrays, uploaded once
  d_irrs                  (device)      #   per-cross-edge ir/rs scatter structure (§4.2)
  d_amap                  (device)      #   assembly map A-values → panels
  workspace_elts::NTuple  (host)        #   pre-sized device workspace ELEMENT counts (×sizeof(T)
                                        #   at factor construction — NIT: Ti-param, not bytes)

GPUSupernodalFactor{T,Ti}               # (M6a, LLᵀ)
  sym::GPUSymbolic{Ti}
  d_nzval::CuVector{T}    (device)      #   the factor L (device-resident)
  d_cbuf::CuMatrix{T}     (device)      #   scatter workspace, max_extend_rows² (§5.3)
  d_boundbuf::CuVector{T} (device)      #   persisted boundary panels, Σ_boundary bytes (§5.3)
  d_potrf_ws::CuVector{T} (device)      #   pre-allocated cuSOLVER potrf workspace (§9.A)
  d_devinfo::CuVector{Cint} (device)   #   pivot-failure flags, one per GPU supernode (§4.3)
  d_b, d_x::CuVector{T}   (device)     #   solve RHS/solution device buffers (§7 device solves;
                                       #     no host_mirror — factor stays device-resident)
  ok::Bool, fail_col::Ti  (host)       #   resolved post-hoc from d_devinfo (§4.3)
  streams::NTuple{2,CuStream}          #   compute + upload streams (§5.4)

GPULDLFactor{T,Ti}                      # (M6b, LDLᵀ) — adds to the LLᵀ layout:
  d_dvec::CuVector{T}     (device)      #   the diagonal D (device-resident)
  d_wd::CuMatrix{T}       (device)      #   W = L21·D staging, ws.cd analogue, col-chunked (§6)
  d_stats::CuVector{...}  (device)      #   per-supernode pre-perturbation inertia stats (§6/§9.E)
  # inertia/n_perturbed/max_pert resolved post-hoc from d_stats (NOT from d_dvec — §9.E)
```

## §3 Kernel strategy (Option 1: pure-primary)

Dense per-supernode work on device goes through a small interface —
`gpu_gemm!`/`gpu_syrk!`/`gpu_trsm!`/`gpu_potrf!` — with a selectable backend, defaulting to
**pure**:

- **Pure KA kernels (default, shipped hot path):** the verified 4×4-register-blocked
  `muladd` kernel (§0) for the **trailing update** (gemm/syrk — the flop-dominant step, where
  the 1.14× win is), generic over `T`, portable (AMD/Intel via KA). The `alpha=-1,beta=1`
  epilogue (`C -= A·Bᵀ`) **is** an epilogue-arg (verified, relerr 0 on galen), with the
  hard requirement that **β=0 OVERWRITES** (not `0*C`, which is `NaN` on an uninitialized
  `d_cbuf` — design.md §4.3 relies on this; fixed in the ext). The **syrk symmetric self-update
  is done by the full gemm kernel** (`B=A`) which harmlessly overwrites the never-read strict-
  upper of the diagonal block (legitimate — `ldlt.jl`/`llt.jl` never store/read it — at ~2× flops
  on the *small* diagonal block only); a triangular-masked syrk variant is a **real kernel change**
  (group-index masking, *not* an epilogue arg — v3, Opus/Fable DEFECT), deferred as an
  optimization, not claimed as free.
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
"0 pattern H2D" gate). The resolved target rows `ir` + run structure `rs` (per cross-edge) are
uploaded once as `d_irrs` (§2.3/§5.3 — budgeted). **v3 (Fable DEFECT):** `relmap` is dropped —
it's a per-supernode *transient* (`relmap[row]=k`, refilled every supernode), not a pattern
array, so "uploaded once" was meaningless; the precomputed `ir` already *is* the resolved scatter
targets. The scatter kernel is a simple indexed add (pure Julia — the sparse part, trivial
*given* the precomputed run structure; the non-trivial scan is host-side and one-time).

### §4.3 Failure semantics (pivot detection) — Fable BLOCKER
`cholesky!` sets `F.ok=false`, records `fail_col`, and returns early on a non-SPD pivot.
cuSOLVER `potrf` reports failure via a **device** `devinfo`. To avoid a per-supernode D2H sync
(which would serialize the async pipeline), M6 uses **deferred batched detection**: each GPU
supernode writes its `potrf` result into its own slot of `d_devinfo` (pre-allocated, one Cint
per GPU supernode); the whole array is D2H'd **once** at the end of `cholesky!`. `F.ok` and
`fail_col` are resolved post-hoc: `ok = all(devinfo .== 0)`; `fail_col` = the column offset of
the lowest-index failed **GPU** supernode. **Amended semantics (amendment D):** on the GPU path
a failed pivot does **not** early-return — later supernodes still compute. **`check_finite` is
NOT a valid backstop (v3, both reviewers):** cuSOLVER `potrf` on a non-SPD block reports `info>0`
but can leave *finite-but-wrong* values (not NaN), which propagate through trailing updates and
pass a finiteness check — so `d_devinfo` is the **sole** failure signal (and StrictMode's
`check_finite` is off in the gate config anyway). **Hybrid CPU-side failure:** a CPU supernode
that fails (`llt.jl` early-returns) must **synchronize both streams first** (in-flight GPU kernels
are writing into `F`'s buffers), then set `fail_col` reconciled against the GPU set (min of the
CPU failure column and the lowest failed GPU supernode). Recorded as amendment D.

## §5 Memory model + frontier + concurrency

### §5.1 Device-resident factor
`nnzL` exact at `symbolic` time. The factor L lives on device (`d_nzval`); no per-supernode
staging because descendant panels are already resident.

### §5.2 Upward-closed etree frontier (replaces `gpu_flop_threshold=2e9`)
At `symbolic` time: mark each supernode with (update+factor) flops ≥ `frontier_cutoff`, then
take the **upward closure** in the supernodal etree. The key property (both reviewers confirmed
sound): every update target of supernode `d` is `snode_of[r]` for a below-diagonal row `r ∈
rowind(d)`, which by the **etree row-subtree property** (a below-diagonal row index of column
`j` is an etree *ancestor* of `j` — the standard sparse-Cholesky fact, not §3.4's pattern-
superset invariant) is an ancestor of `d`. So an upward-closed GPU set **never emits a
device→host update edge**: all cross-frontier edges point CPU→GPU. This is what makes the
traffic one-way. The single tunable is `frontier_cutoff`; its default comes from the Phase-0
measured CPU-vs-GPU per-shape crossover on galen (§8.3), with a derivation comment.
**Honesty (Opus):** upward closure can pull a small, low-flop supernode onto the GPU if it sits
above a heavy subtree; "small stays on CPU" holds only for nodes *not* on a GPU-ancestor path.
Near-root supernodes are typically dense so this is usually negligible. **v3: the v2 "puncture"
escape hatch is deleted** — keeping a near-root node on CPU while its descendants are GPU would
create exactly the device→host update edge the invariant (and §10.2's executable check) forbids;
if that optimization is ever wanted it must be the fully-specified "GPU-assembled, CPU-factored"
node (panel stays device-resident, GPU descendants update it there, D2H once, factor on CPU),
out of scope for M6.

### §5.3 Capacity + the persisted boundary-panel budget (v3 — replaces the ring)
The cross-frontier upload model is **persist, not ring** (BLOCKER 1 fix). A **boundary** CPU
supernode is one with ≥1 below-diagonal row updating a GPU supernode; by §5.2 all of its
ancestors are then GPU, so its whole factored panel is read by several GPU ancestors at
separated schedule points. v3: each boundary panel is uploaded **once** (genuinely) when the CPU
finalizes it, kept **device-resident** for the rest of the refactor (`d_boundbuf`, a
symbolic-time-sized arena), and freed at refactor end. This makes "upload once" literally true,
needs no slot-reuse protocol, and cannot deadlock. The boundary set and its total bytes
`Σ_boundary (nsrow·ncol·sizeof(T))` are **computable at symbolic time** (pattern-only).

Budget against **queried** free memory (`CUDA.available_memory()` on galen ≈ 11.57 GiB after
context) at device-buffer-allocation time, with a safety margin:
`d_nzval` (nnzL·`sizeof(T)`) + `d_cbuf` (`max_extend_rows²`·`sizeof(T)` — verified: `ws.c` in
`types.jl` is `(max_extend_rows,max_extend_rows)`; `max_update_size` is a diagnostic, not a
bound) + `d_boundbuf` (the Σ_boundary term above) + `d_irrs` (per-cross-edge `ir`/`rs` scatter
structure, §4.2, pattern-only, uploaded once; Σ-over-cross-edges bytes, symbolic-computable) +
`d_potrf_ws` (cuSOLVER `bufferSize` over supernode shapes, §9.A) + M6b: `d_wd`
(`W=L21·D` staging, the device analogue of `ws.cd`, `max_extend_rows²` **column-chunked** since
`ncol_d` is unbounded — §6) + pinned staging for A-values H2D and the `host_mirror` D2H (§7). If
the sum exceeds free-memory−margin, **fall back to the CPU path loudly** (`@warn` + CPU factor)
— loudness is the guarantee, never an OOM at factor time.

### §5.4 Stream/event concurrency — per-descendant dependency (v3, BLOCKER 1 fix)
Two streams: `compute` (GPU supernode work) and `upload` (async H2D of finalized boundary
panels). The dependency is **per-descendant, not per-supernode** (v2's per-supernode barrier —
"wait on *all* of `s`'s uploads before `s`'s first kernel" — forced every boundary panel of `s`
resident at once and risked deadlock). v3: (1) when the CPU finalizes a boundary panel `d`, the
host enqueues its H2D on `upload` and records a `CuEvent e_d`; (2) each individual
descendant-update kernel `s -= d`-contribution on `compute` is preceded by `CUDA.wait(e_d,
compute)` — it waits only on **that** descendant's upload. The single `compute` stream serializes
GPU supernodes in ascending finalize order (so GPU→GPU descendant ordering is free); the
per-`e_d` waits enforce CPU→GPU ordering. The DAG is acyclic (waits only go `compute ← upload`)
→ **deadlock-free**. "race-free by construction" is replaced by "race-free under this explicit
per-descendant event DAG"; §10.2 adds the executable upload-once + ordering check. Host-side
merge for solves: CPU subtrees factor directly into `host_mirror.nzval`; only the GPU-slice is
D2H'd at make-solve-ready (§7).

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
  column-scale (`W = L21·D`, a `dvec`-slice broadcast into `d_wd`, the `ws.cd` analogue,
  column-chunked per §5.3) before the pure gemm/syrk. `d_dvec` lives on device.
- **Signed regularization + inertia (v3, BLOCKER 2 fix).** Regularization is **not** separable
  from the diagonal-block factorization: in `ldlt.jl` the regularized `dⱼ` feeds the Schur
  complement of later columns *in the same block*, and inertia is classified on the
  **pre-perturbation** pivot. So the device block-LDL must run the signed-regularization column
  loop itself, and for each pivot emit — **before** perturbing it — its sign/magnitude classified
  against the zero test into a per-supernode device stats record
  `(n_pos, n_neg, n_zero, n_perturbed, max_pert, dmin, dmax)`. These are reduced once at the end
  (like `d_devinfo`) into `FactorStats`. Deriving inertia from the final `d_dvec` (post-
  regularization) is **wrong** — it returns the tautological *forced* signs and voids req 8's
  observed-inertia contract that IPOPT-style consumers depend on. Two concurrency-forced
  semantics changes, recorded as **amendment E**: (i) the CPU path's zero test uses a
  **running-global** `dmax` across the whole factorization — a sequential cross-supernode
  dependency concurrency can't reproduce — so the device test is redefined **order-free**
  (per-supernode-local `dmax`, delta-anchored: `adⱼ ≤ ζ·max(dmaxₗₒ𝒸ₐₗ, δ)`); (ii) `δ =
  LDLT_DELTA·ascale` where `ascale = max|assembled A|` is computed **host-side** during the
  A-value staging pass (host has `A.nzval`; O(nnz), free).

This keeps §5's one-way traffic for `ldlt!` (only the small `nscol×nscol` diag block round-trips
if the device block-LDL kernel isn't used; the tall panel stays device-resident). It is **new
numeric code with its own `GPULDLFactor` type (§2.3), BigFloat/CPU-`ldlt` oracle, inertia-match
test, and zero-alloc gate** — M6b, sequenced after M6a's gate.

## §7 Solves — ON DEVICE (v4, user-directed 2026-07-17)

**Solves run on device** (`solve_L!`/`solve_Lt!` as supernodal triangular-solve kernels against
the device-resident factor). The factor **never** does a full make-solve-ready D2H — it stays on
device across factor→solve and across refactors. Per solve, only the **RHS `b`** is H2D'd and the
**solution `x`** D2H'd (length-`n` vectors, `n·sizeof(T)` — negligible vs the factor). This is the
change that made amendment B's clause 1 clean: the whole `symbolic(CPU, once) → numerical
factor(GPU) → solve(GPU)` pipeline is device-resident after the one-time pattern upload, with no
per-refactor factor-sized transfer.

Consequences vs the v3 "solves-on-CPU + make-solve-ready-D2H" design (now retired):
- The v3 recurring per-refactor **factor D2H** is **gone** (was the dominant IPM-cycle tax). The
  only per-refactor value transfer is the A-value **H2D** for assembly (`nnz(A)·sizeof(T)`, pinned;
  §9.A gates "0 *pattern* H2D", not value H2D) and the tiny per-solve `b`/`x` vectors.
- `host_mirror` (v3) is dropped; there is no host factor copy. A user who wants the factor on the
  host asks for it explicitly (an opt-in D2H), outside the gated path.
- Repeated solves on one factor are cheap (no re-transfer) — the natural IPM shape.
- The device triangular solves are new kernels (supernodal forward/back substitution) with their
  own CPU-oracle test and their own zero-alloc discipline (pre-allocated, amendment A).

## §8 Gate

### §8.1 Definition (req 2, corrected; v4 amendments user-approved 2026-07-17)
On the §8.3 large-matrix stratum, all median wall-time, **single-thread CPU methodology**
(stated so it's not mistaken for threaded-CPU). Both arms share the CPU `symbolic` (analyze
once); the timed region is the **warm numerical factor + solve** — on the GPU arm both run on
device (§7, no full-factor D2H; only `b`/`x` vectors transfer), on the CPU arm both run on CPU.
1. GPU-enabled `cholesky!`+solve ≥ **3×** faster than our own single-thread CPU PureSparse
   (`cholesky!`+solve). *(3× — user-set target, believed achievable on this GPU (2026-07-17);
   **provisional pending §8.3 measurement**, to be confirmed from the *achieved* crossover —
   our 349 GF pure kernel vs measured single-thread DGEMM ~55–65 GF ≈ 5–6× dense headroom,
   discounted by assembly/scatter/launch Amdahl — not from the 455 GF peak. Held under the
   no-fudge rule: if the hybrid cannot reach 3× on the stratum we investigate or report the
   miss, we do not silently lower it.)*
2. **No regression** on the existing M1/M2 gate set: with the auto frontier, small/medium
   matrices stay on CPU (or the ext isn't even constructed, §2.2) → regression ≤ the harness's
   established locked-clock run-to-run band (a concrete number from galen/wintermute, not
   "noise").
3. GPU-enabled PureSparse **still beats CHOLMOD+OpenBLAS** on the stratum, **both own-ordering
   AND under an identical `GivenOrdering(p)` permutation** (req 2's mandatory same-perm arm —
   restored; v1 dropped it).

All gate numbers are **medians** (CLAUDE.md benchmarking; the `min`-based Phase-0 kernel numbers
were fine for the wide-margin flip decision but the gate must not inherit them).

### §8.2 Context arms (reported, not gated)
cuDSS (NVIDIA sparse, black-box, like faer for QR); pure-vs-cuBLAS kernel attribution (§3);
the IPM-cycle cost both ways (factor + solve×k — cheap on the GPU arm now that device solves
avoid re-transfer, §7); **multi-threaded CHOLMOD+OpenBLAS** (Opus DEFECT — the gate is
single-thread-CPU per req 2, which is the correct *contractual* comparison but reads as
GPU-vs-one-core; this arm shows GPU-vs-best-CPU so the headline isn't misread).

### §8.2a Multi-device confirmation (the two-host gate bar)
Every prior PureSparse gate verdict required **two clock-locked hosts** ([[reference_benchmark_machines]]).
For M6 the second GPU host is a **neuromancer eGPU** (planned). Two cases: an **NVIDIA** eGPU gives
a second CUDA data point + the two-host confirmation; an **AMD** eGPU additionally **verifies the
KernelAbstractions portability claim** (the pure kernel running on ROCm — currently theoretical,
KA==CUDA measured only on NVIDIA). Until the second host exists, galen is the sole gate host and
that limitation is stated in the verdict, not hidden.

### §8.3 Stratum spec (pin criteria BEFORE measuring — anti-cherry-pick)
Selected now, not after GPU numbers exist: SPD + SQD problems, `nnzL·8 + workspace ≤ ~9 GB`,
from the SuiteSparse collection FEM/KKT/Laplacian classes (candidate: Fault_639, Emilia_923,
Serena, audikw_1 — exact `nnzL` checked at selection) + synthetic large 3-D-grid Laplacians and
random-KKT (design.md §9.4 permits synthetics). ≥6 matrices, ≥2 SQD. The stratum + CPU
baselines are the remaining Phase-0 item, run BEFORE the frontier cutoff is fixed.

## §9 Contract amendments (require explicit user sign-off)

**A — req 5 (zero-alloc) on GPU. ✅ APPROVED (user, 2026-07-17).** The 0-device-pool and
0-pattern-H2D parts are hard gates; the host-byte ceiling is set from the Phase-0 measured
per-launch floor (gate to 0 if a host-alloc-free launch path is demonstrated first).
Verbatim `@allocated==0` is impossible (kernel launches
allocate host bytes; cuSOLVER queries workspace). Proposed: *"warm `cholesky!`/`ldlt!` on a
GPU factor: **0 device-pool allocations after setup** (cuSOLVER workspace + `devinfo` + `cbuf`
+ boundary arena + stats arrays pre-allocated at factor construction via `bufferSize`; low-level
cuSOLVER API, not the auto-workspace wrappers) AND **0 pattern H2D** (§4.2); host bytes per warm
refactor **≤ `c_launch · (#GPU kernel launches)`** with `c_launch` a measured per-launch
constant (v3, Opus/Fable DEFECT — the count scales with #GPU-supernodes, so a bound *independent
of nsuper* is the wrong shape; if a zero-host-alloc launch path is demonstrated first, gate that
instead)."* The per-launch floor is measured first (remaining Phase-0), then gated as a ceiling.

**B — req 2 (gate baseline) on GPU. ✅ APPROVED WITH CHANGES (user, 2026-07-17).** The §8.1
three-clause gate, with the user-directed changes: (i) clause 1 margin **3×** (was 2×), (ii)
timed region = **numerical factor + solve, both on device** (§7 device solves; no full-factor
D2H — only `b`/`x` vectors move). Retained: the `GivenOrdering` same-perm arm and clause 3 (GPU
PureSparse still beats **CPU** CHOLMOD+OpenBLAS, both ordering arms — this carries the original
non-negotiable req 2 forward; confirmed with the user that the opening "no GPU-CHOLMOD
comparison" agreed with the *rewording rationale*, not dropping clause 3).

**C — dense-kernel exclusivity on device. ✅ APPROVED WITH RESOLUTION (user, 2026-07-17).**
The CPU rule's pure-*exclusivity* relaxes to a **best-measured-kernel-per-op** policy on device.
**Framing (recorded correctly, per the user discussion):** writing pure KA kernels *fulfills*
the Pure ethos — they are the device analogue of PureBLAS, the pure replacement the ecosystem
exists to build; *calling cuSOLVER/cuBLAS is the deviation* (cuBLAS is the GPU analogue of the
forbidden OpenBLAS), NOT the compliant path. **Policy:** dense work uses **pure KA kernels where
they win or where portability requires them** (gemm/syrk/triangular-solves — ours; gemm/syrk beat
cuBLAS); **cuSOLVER/cuBLAS are permitted right now for pragmatism (ship M6a) + as the gating
oracle** (in-loop correctness reference + benchmark baseline) on the small diagonal
`potrf`/`trsm`. **COMMITTED follow-up (user: "obviously write pure for portability"): pure device
`potrf`/`trsm` WILL be written — REQUIRED for full AMD/Intel (ROCm/oneAPI) portability, since
cuSOLVER is NVIDIA-only** (optional for NVIDIA-only performance, but the portability pitch is only
partial until they exist). **Scope honesty (Opus):** M6a's gate closes with cuSOLVER on the
low-flop pivoted diagonal — so "M6a closes on a pure kernel" is true only for the flop-dominant
trailing update + solves; a fully vendor-free, fully portable factorization lands with the pure
`potrf`/`trsm` follow-up.

**D — GPU failure semantics. ✅ APPROVED (user, 2026-07-17).** (v3, referenced by §4.3.) *"On a GPU factor, a non-SPD pivot does
**not** early-return (deferred batched `d_devinfo`, one D2H at end; `ok`/`fail_col` resolved
post-hoc as the lowest-index failed supernode). `check_finite` is **not** a backstop (cuSOLVER
leaves finite-but-wrong values, and StrictMode checks are off in the gate config) — `d_devinfo`
is the sole failure signal. In the hybrid loop a **CPU-side** failure early-returns only after
synchronizing both streams, and `fail_col` reconciles the CPU failure column against the GPU
set."*

**E — LDLᵀ inertia + order-free zero test. ✅ APPROVED (user, 2026-07-17; M6b-only).** (v3, referenced by §6.) *"Device inertia is emitted
per-supernode from the **pre-perturbation** pivot into a stats array reduced once (not derived
from the regularized `d_dvec`); the zero-pivot test is redefined order-free (per-supernode-local
`dmax`, delta-anchored) since the CPU path's running-global `dmax` is a sequential dependency
concurrency can't reproduce. `n_perturbed`/`max_perturbation` tracked the same way; `ascale` is a
host-side O(nnz) pass during A-staging (using the **`amap≠0` filter**, matching `ldlt.jl`'s
assembly-loop max, not raw `A.nzval`)."* **Scope of the change (v3 focused review — corrected):**
the order-free test feeds **only the inertia counts** — regularization (`delta`/`target`/`newd`)
is provably `dmax`-independent, so the **factor L and D are bit-identical** and §10.1's normwise
oracle is fully preserved. Only `n_pos`/`n_neg`/`n_zero` can diverge, and only in a narrow band
(`ζ·max(dmaxₗₒ𝒸ₐₗ,δ) < ad_j ≤ ζ·dmax_global`) that opens on heterogeneous-scale KKT blocks. Hence
the **inertia-match oracle (§10) must run the CPU reference with the same order-free local-`dmax`
test** (or tolerance the band) — a stock-`ldlt!` inertia comparison would spuriously fail there.
A user-visible *inertia-report* change (not a factor change), so an explicit amendment.

**F — multifrontal supersedes the left-looking transfer model. ✅ APPROVED (user, 2026-07-17).**
Measured motivation: the left-looking GPU path is launch-bound (a separate gemm+scatter per
descendant; a near-root front has thousands of descendants → best hybrid 0.72–0.95× vs CPU,
unchanged by removing allocs/D2H). The GPU Cholesky path becomes **multifrontal** (per-front
CPU/GPU dispatch, §M). This **replaces** §5.3's boundary-panel-persist arena and §5.4's
per-descendant event DAG with: a symbolic-time-sized **update-matrix arena** (the device arena
also holds crossing CPU subtrees' U matrices); **one U + one panel-slab H2D per crossing CPU
subtree** (a whole CPU subtree's contribution to the GPU crown is a single U matrix; no U ever
downloads, by upward closure); a **synchronous v1 schedule** (streams only if the gate misses
3×); and **`d_emap`** (one ascending per-child extend-add map, Σ ≤ |rowind|) replacing `d_irrs`.
`d_cbuf`/`d_boundbuf` are retired. Left-looking `cholesky!` stays the CPU product path + oracle
arm; `gpu_cholesky_sync!`/`_hybrid!` stay as in-loop reference arms until the gate closes.

## §M Multifrontal engine (Path B — amendment F; Fable-advised 2026-07-17)

Replaces the left-looking per-descendant GPU updates (launch-bound) with front assembly: each
front does **one** potrf + trsm + syrk + a scatter per child (≈ `4 + nchildren` launches),
instead of thousands of per-descendant launches. Ref: Liu, *The Multifrontal Method* (SIAM Rev
1992) — clean-room-safe canonical.

**§M.1 Formulation.** Front = supernode; front tree = `sparent`. **Split the front, don't
materialize it:** the *panel* region (`nsrow×nscol`) is the existing `d_nzval` panel **in
place** (factor layout stays bit-compatible → oracle/assembly/solve unchanged); the *update*
region `U_s` (`(nsrow−nscol)²`) lives in a separate arena. Per front:
1. **Extend-add** each child's `U_c` into the panel + `U_s` regions (one scatter per child).
2. `potrf` on `panel[1:nscol,1:nscol]`; 3. `trsm` on `panel[nscol+1:nsrow,1:nscol]`;
4. **`U_s = (extend-added trailing block) − L21·L21ᵀ`** — `gpu_syrk_nt!(U_s, L21, −1, 1)`, **β=1**
into the already-assembled trailing block. **CRITICAL (Fable pitfall #1): `U_s` MUST include the
extend-added trailing block** (the multifrontal *relay* of generation-skipping contributions) —
`−L21·L21ᵀ` alone silently drops them (passes toy tests, fails real matrices). Lower-triangle
discipline throughout (`a ≥ b`).

**§M.2 Extend-add maps (symbolic, pattern-only).** By symmetry `U_c`'s rows = cols = c's
below-diagonal rows, so ONE ascending map per child: `emap_c[i] = relmap_parent[rowind(c)[nscol_c+i]]`
(containment `rowind(c)\cols(c) ⊆ rowind(sparent(c))` guaranteed — assert it). Because both
patterns are sorted, `emap_c` is **strictly ascending** → the panel/U split is a **prefix**:
first `k1_c` entries (`emap ≤ nscol_s`) → panel columns; rest → `U_s` at `emap−nscol_s`.
Storage: concatenated `emap` + `emap_ptr` (nsuper+1), Σ ≤ |rowind|, uploaded once as `d_emap`
(0-pattern-H2D gate). Only `nsuper−1` edges — the launch-count collapse.

**§M.3 Arena, not a stack.** One postorder simulation at symbolic time emits per-front arena
offset `uoff[s]` + exact host/device peak occupancy (single source of truth for order+offsets+
peak — divergence is pitfall #3). No runtime stack. Zero-alloc; sizing IS the allocation. Feeds
`gpu_device_bytes` + `gpu_capacity_ok` loud fallback. `d_cbuf`/`d_boundbuf` retired.

*Bounded layout (IMPLEMENTED).* `arena[1 : max_usize]` = a **work slot**; `arena[max_usize+1 …]`
= a **bounded stack** of the live U's. Each front builds its U in the work slot (extend-add reads
children from the stack *above* — disjoint, no aliasing), then **compact-copies** it into
`arena[uoff[s]:]`, reusing the freed space where its children sat. The compaction is
non-overlapping by construction (`uoff[s] ≥ max_usize+1 > usize[s]`, so work-slot source and
stack dest never overlap — no `memmove` hazard). The postorder stack simulation computes `uoff`
(each front lands at its deepest child's offset = `cbase`) and the exact peak = `max_usize +
max-live-stack`, **5.9× smaller than the monotonic Σ-all-U's at grid3d_44** and the ratio grows
with size — the difference between OOM and fit for the large SQD/KKT gate stratum. Hybrid: two
physical arenas (host + device), each with its own work slot at offset 1; a crossing CPU front
compacts to its host stack slot then H2D-uploads that slot to the same device stack offset.

**§M.4 Hybrid residency.** Device arena holds GPU-front U's **plus crossing CPU U's** (a CPU
child of a GPU parent: U computed on host, H2D to its device slot). No U downloads (upward
closure). **Crossing set = CPU fronts whose `sparent` is GPU** (maximal-CPU-subtree roots) —
smaller than the left-looking boundary set; each such subtree is a **contiguous `px` range** →
its factored panels upload as **one slab**. Per crossing subtree: 2 H2D (panel slab + root U).
Whole GPU path is multifrontal, per-front `on_gpu[s]` dispatch (CPU fronts: PureBLAS + host
arena; GPU fronts: device). Left-looking `cholesky!` stays the CPU product.

**§M.5 Pitfalls (ranked):** (1) the relay omission (§M.1); (2) region split + lower-triangle
(`a<b` into U corrupts it); (3) sizing-vs-execution order divergence → arena aliasing; (4) **stale
U slots across refactors** — zero a front's U slot before its first child scatter *every*
refactor (β=0-overwrite family); (5) fragmented supernodes → front-count blowup (amalgamation is
a later measured knob; v1 = existing supernodes verbatim); (6) mid-tree pivot failure → amendment
D's deferred `d_devinfo` carries over. Non-goal: bitwise CPU match (summation order differs;
normwise §10.1 oracle is the approved one).

**§M.6 Build order (each lands with its oracle):** (1) symbolic layer (children-CSC + emap/k1 +
arena simulation; pure, CPU-testable); (2) **CPU multifrontal numeric** (PureBLAS, host arena)
oracle vs `cholesky!` — validates formulation+maps+arena with zero GPU; (3) extend-add device
kernel, unit-oracled; (4) all-GPU multifrontal, oracle + predicted launch-count assert + first
perf; (5) hybrid (per-front dispatch + crossing uploads), oracle + gate; (6) streams only if (5)
misses 3×. Write **fresh** (~350 lines); steal only the QR engine's children-CSC idiom +
sizing-as-simulation pattern.

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
- **Upload-once:** a per-refactor counter asserts each **boundary** CPU panel (§5.3) is H2D'd
  **exactly once** and stays resident until refactor end (the persist model, §5.3) — not
  re-uploaded per (descendant, ancestor) edge.
- **Boundary budget:** the Σ-over-boundary-panels bytes computed at symbolic time equals the
  actual peak `d_boundbuf` occupancy (no growth).
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
