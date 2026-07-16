# PureSparse.jl — GPU offload (M6) design

**Status: v1 draft (2026-07-16), awaiting two independent adversarial reviews → v2.**
Supersedes design.md §8 (Cholesky-only, predates M2/M5, per-supernode vendor-BLAS staging)
and the stale ROADMAP `### M3` section (KA-kernels/level-set/reported-not-gated) — those
two contradicted each other (Fable M6 review, F1). This document is the single live GPU
design. It goes through the same process design.md and design_qr.md did: v1 → two blind
adversarial reviews → v2, before any implementation.

Produced from: Fable's M6 architecture review (2026-07-16, all 3 factual findings verified
against source), the Phase-0 measurement pass on galen (below), and two explicit user
decisions (scope; dual kernel strategy).

---

## §0 What changed from design.md §8, and why (review anchors)

- **F1 (fixed here):** §8 and ROADMAP `### M3` were two contradictory designs. This doc
  replaces both.
- **F2 (fixed §5):** `gpu_flop_threshold = 2e9` is underived and never fires on any
  existing gate matrix (they top out at mega-to-low-gigaflop *total*). Replaced by a
  symbolic-time upward-closed etree frontier (§5) with a Phase-0-calibrated cutoff.
- **F3 (constraint on §4):** the GPU offload must be derived from `src/numeric/llt.jl` **as
  it is today** — including its contiguity β=1 fast path (accumulate straight into the
  ancestor panel, no staging) and `unsafe_wrap` panel wrappers — NOT from §4.3's older
  staged-scatter description, which the CPU code already measured and abandoned for the hot
  case.
- **F5 (→ contract amendment, §7.3):** zero-alloc-after-symbolic (design.md req 5) is
  unachievable verbatim on GPU (kernel launches / CUBLAS calls allocate host bytes). Needs
  an explicit user-approved wording amendment.
- **F7 (framing, §7):** RTX 4070 FP64 is 1:64 (~455 GF peak). The gate compares against the
  project's established **single-thread** CPU methodology — stated out loud so the result
  isn't mistaken for a threaded-CPU comparison later.

---

## §1 Scope (user decision: Cholesky + LDLᵀ together)

- **M6a — supernodal LLᵀ (`cholesky!`) GPU offload.** Float64 first; the shipped dense
  kernels are cuBLAS/cuSOLVER (§3), correctness-fallback kernels generic over `T`.
- **M6b — supernodal LDLᵀ (`ldlt!`) GPU offload.** Shares M6a's scheduler and frontier
  wholesale; the only deltas are the `L·D` column-scale staging and keeping the diagonal
  LDL block factorization on CPU. Cheap once M6a lands.
- **OUT of M6:** sparse QR (different multifrontal-WY architecture; its gate already closed
  vs SPQR — a separate milestone if ever wanted). Simplicial update/downdate (latency-
  bound, wrong shape for a GPU — stays CPU forever). Device solves (M6a/b keep solves on
  CPU with the measured D2H cost, §6; device solves are a later increment).

The primary gate number is the **warm refactor** (`cholesky!`/`ldlt!` on an existing
factor) — the IPM-relevant path and the project's established gate slice.

## §2 Extension architecture

`ext/PureSparseCUDAExt` — a weak-dep extension (mirrors `ext/PureSparseForwardDiffExt.jl`).
`CUDA` (and `KernelAbstractions`, §3) go in `[weakdeps]`/`[extensions]`; **zero hooks in
`src/`**. The ext defines a new factor type (`GPUSupernodalFactor`) and a `GPUSymbolic`, and
adds methods to the existing generic entry points — dispatch-driven, no runtime registry
(GKH rule: ownership resolved by type at compile time, not a runtime `Dict`). The core stays
dependency-free and trim-compatible; the trimmed CPU build never loads the ext (§7.4).

## §3 Kernel strategy (user requirement: dual track)

The dense per-supernode kernels (gemm/syrk trailing update, trsm off-diagonal, potrf/LDL
diagonal) are the leaf of the sparse algorithm. PureSparse's value and "purity" live in the
**sparse orchestration** (frontier scheduling, assembly, scatter-add), which is pure Julia
on GPU regardless of the leaf. Two tracks for the leaf:

**Track 1 — cuBLAS/cuSOLVER on the hot path (ships M6).** `CUDA.CUBLAS.gemm!/syrk!/trsm!`
and `CUDA.CUSOLVER` `potrf!` on the device panels. This is the proven-fast device dense
kernel and the baseline the pure track must beat. Clean-room-fine: a closed vendor binary
used black-box, exactly as OpenBLAS/CHOLMOD outputs are used on CPU — we never read its
source (there is none to read). **Explicitly NOT** cuDSS / cusolverSp (NVIDIA's *sparse*
direct solvers): those stay black-box benchmark baselines like CHOLMOD; the sparse
factorization is ours.

**Track 2 — pure-Julia, vendor-portable kernels tuned to BEAT cuBLAS (parallel R&D, then
replaces Track 1 on the hot path once it wins).** Rationale: the sibling PureBLAS.jl already
beats OpenBLAS *and* MKL on CPU, so beating cuBLAS FP64 is a legitimate target; and portable
pure kernels (KernelAbstractions.jl → AMD ROCm / Intel oneAPI) are a strategic capability
cuBLAS cannot provide, plus they keep generic-`T`. Phase-0 status: a naive and a 4×4
register-blocked pure kernel both sit at 0.48× cuBLAS with **no register spills**, i.e. an
un-profiled occupancy/FP64-ILP bottleneck, not a ceiling (cuBLAS itself is only at 67% of
the FP64 peak). This track is NCU-profiled and iterated on galen; its results feed the
`design_gpu.md` v2 and decide when it takes over the hot path. **The gate does not wait on
Track 2** — M6 ships on Track 1; Track 2 is an unlock, measured against Track 1.

Design consequence: the numeric loop calls a small dense-kernel interface
(`gpu_gemm!`/`gpu_syrk!`/`gpu_trsm!`/`gpu_potrf!`) with two backends (cuBLAS; pure-KA),
selected by a Preferences-baked const — so swapping Track 2 in is a backend flip, not a
loop rewrite.

## §4 The offloaded numeric loop (derived from current `llt.jl`)

The left-looking driver stays on CPU. For each **GPU-side** supernode `s` (frontier, §5):
its descendant updates and its own factorization run on device. Derived from `llt.jl` as it
is (F3):

- **Assembly:** `s`'s panel is assembled on device from A's values via the precomputed
  `amap` (an assembly kernel, pure Julia — it is the sparse part).
- **Trailing update (the flop-dominant step):** for each descendant `d` of `s`, apply
  `d`'s contribution. `llt.jl`'s **contiguity fast path** (descendant's remaining rows form
  a contiguous run of `s`'s row list → syrk/gemm with β=1 straight into the panel, no
  staging) is the common case and is preserved on device: the panel is already resident, so
  the update is a device gemm/syrk with β=1 into the panel sub-view. The non-contiguous
  case scatters through a device `cbuf` + a scatter kernel using an on-device `relmap`
  (mirrors `_scatter_update!`).
- **Diagonal factorization:** potrf (LLᵀ) / signed LDL (LDLᵀ) on the diagonal block.
  M6a slice: via cuSOLVER `potrf!` on device (LDL diagonal block on CPU with a per-block
  round-trip — measured, revisited in M6b/later).
- **Finalize:** the factored panel stays device-resident (§5 — no per-supernode D2H).

CPU-side supernodes (below the frontier) run the existing `cholesky!`/`ldlt!` body
**untouched** — the closed M1/M2 gate must not be destabilized; shared helpers are factored
out only where provably identical.

## §5 Memory model + the frontier split (replaces the scalar threshold)

**Device-resident factor.** `nnzL` is known exactly at `symbolic` time. Budget on the 12 GB
card: ~12.3 GB − CUDA context (~0.5 GB) − device workspace (`cbuf` at `max_update_size`,
pattern arrays, A-values) → cap the factor at ~8–9 GB ≈ 1.0–1.1e9 nnzL (Float64). Large
SuiteSparse-collection SPD/SQD problems fit; the decision is an **exact precomputed capacity
check** at symbolic time that falls back to the CPU path **loudly** (never an OOM at factor
time). Because the factor is resident, there is no per-supernode PCIe staging cost to weigh.

**Upward-closed etree frontier (replaces `gpu_flop_threshold=2e9`).** At `symbolic` time,
mark each supernode whose (update+factor) flops ≥ a cutoff, then take the **upward closure**
in the supernodal elimination tree → the GPU set is a top slice, the CPU set the small
subtrees below. Consequences: (a) small/latency-bound supernodes stay on CPU (launch latency
~5.5 µs each — a 100k-supernode matrix would burn seconds in launches alone); (b) traffic is
**one-way and once-only** — a finalized CPU-side panel that any GPU ancestor consumes is
uploaded exactly once, asynchronously, overlapped with CPU work (left-looking finalizes each
panel before any consumer reads it → race-free by construction). The single tunable is the
frontier flop cutoff; its default comes from Phase-0's measured CPU-vs-GPU per-shape
crossover on galen (with a derivation comment — F2's discipline), not a guessed constant.

## §6 Solves (CPU in M6a/b)

Factor is device-resident; solves run on CPU after a one-time factor D2H (measured: a 4 GB
factor ≈ 170 ms at ~13 GB/s PCIe — affordable at gate scale where CPU factor times are
seconds, but stated + measured, not assumed). Device solves are a later increment (they kill
the D2H for repeated solves but are not needed for the warm-refactor gate).

## §7 Gate, baseline, and contract amendments

### §7.1 Gate (Fable Q3)
On a **new large-matrix stratum** (flop-rich SPD/SQD, factor fits ~9 GB), three clauses, all
median wall-time, single-thread CPU methodology (F7):
1. **GPU-enabled `cholesky!`/`ldlt!` warm refactor ≥ 2× faster than our own CPU PureSparse**
   on the stratum. (2× justified by the ~8× FP64 flop headroom over single-thread CPU DGEMM;
   a design-review parameter.)
2. **No regression** on the existing M1/M2 gate set: with the auto frontier, small/medium
   matrices stay on CPU and regress by ≤ noise.
3. GPU-enabled PureSparse **still beats CHOLMOD+OpenBLAS** on the stratum (preserves req 2
   through the new path).
Context arms (reported, not gated): cuDSS (NVIDIA sparse, black-box, like faer was for QR);
"PureSparse-GPU + cuBLAS" vs "PureSparse-GPU + pure-kernel" (the Track-1-vs-Track-2
attribution, mirroring §9.3's OpenBLAS attribution arm).

### §7.2 Why beat our own CPU, not CHOLMOD-GPU
CHOLMOD's GPU build is GPL (clean-room: never read) and is not what Julia ships (stdlib
SuiteSparse_jll has no CUDA), so it's nobody's baseline. The honest gate for an *offload
feature* is against the thing it accelerates: our own CPU factorization.

### §7.3 Contract amendment A — req-5 zero-alloc on GPU (NEEDS USER SIGN-OFF)
design.md req 5 ("`@allocated cholesky!(F,A2) == 0`") is unachievable verbatim: CUDA kernel
launches and CUBLAS calls allocate host bytes. Proposed amended wording: **"warm
`cholesky!`/`ldlt!` on a GPU factor allocates 0 device-pool bytes after setup and 0 pattern
H2D transfers; host `@allocated` is measured and bounded (target 0, actual reported)."** The
zero-device-alloc + upload-once parts are hard gates; the host-byte part is measured, not
zero. **Not yet approved.**

### §7.4 Contract amendment B — req-2 GPU gate baseline (NEEDS USER SIGN-OFF)
req 2 is "median_seconds(PureSparse+PureBLAS) < median_seconds(CHOLMOD+OpenBLAS)". The GPU
baseline definition is §7.1's three-clause gate. **Not yet approved.**

## §8 Correctness oracle

Same `A` → CPU factor vs GPU factor, elementwise `‖L_gpu − L_cpu‖ ≤ c·n·eps(T)·scale`
(design.md §9.2 methodology; device reduction order differs, so tolerance-based, calibrated
on dense potrf first). Full winnable-zoo sweep; `--check-bounds=yes` device run; StrictMode
preconditions on the device path. GPU test items skip cleanly when no device is present and
run on galen via the existing remote-gate workflow (rsync+verify before every run).

## §9 Trim + zero-alloc

Trim gate extended **early** (not last): the `juliac --trim` smoke (`juliac/entry.jl`,
`test/trim_tests.jl`) runs against the weakdep-bearing `Project.toml`, proving the ext's mere
existence doesn't perturb the trimmed CPU build. Alloc/transfer discipline per §7.3, tested
in the StrictMode-checks-disabled configuration like the CPU gate.

## §10 Task list

**Phase 0 — galen probes (DONE, 2026-07-16).** CUDA.jl functional; cuBLAS FP64 ~305 GF;
pure kernels 0.48×, no spills; launch latency 5.5 µs; PCIe ~13 GB/s. `benchmark/gpu/
phase0_probe.jl`, `kernel_diag.jl`. Remaining Phase-0 item: the large-matrix stratum
selection + CPU baselines (feeds §7.1 and the frontier cutoff).

**Phase 1 — this doc → two independent adversarial reviews → v2.** Includes the two contract
amendments (§7.3/§7.4) for explicit user approval, which the review cycle carries.

**Phase 2 — implementation (each step lands with tests; GPU items run on galen):**
1. Ext scaffolding (`CUDA`/`KernelAbstractions` weakdeps; loads-with-CUDA-absent CI job).
2. Trim gate against the weakdep Project.toml (§9) — first, not last.
3. `GPUSymbolic`: one-time pattern upload; upload-once test (2nd refactor = 0 pattern H2D).
4. `GPUSupernodalFactor`: device buffers + capacity check + loud CPU fallback (§5).
5. Dense-kernel interface (`gpu_gemm!/syrk!/trsm!/potrf!`) — cuBLAS/cuSOLVER backend
   (Track 1); each unit-tested elementwise vs CPU PureBLAS on random blocks.
6. Device assembly + scatter kernels (pure Julia — the sparse part), unit-oracled.
7. Hybrid numeric loop (§4): CPU subtrees untouched; frontier panels upload once/async; GPU
   supernodes factor on device.
8. Correctness oracle (§8) full-zoo + `--check-bounds=yes`.
9. Alloc/transfer discipline (§7.3).
10. Gate on galen (§7.1) + stratum + cuDSS/attribution context arms; frontier cutoff
    calibrated + documented.
11. **M6b:** LDLᵀ variant (reuse scheduler/frontier; `L·D` staging + CPU diagonal LDL).

**Track 2 (parallel R&D, then hot-path swap):** pure-Julia portable FP64 gemm/syrk beating
cuBLAS (NCU-profiled on galen); KA-vs-CUDA.jl perf delta measured; swaps into the §3 backend
interface when it wins. Does not block Phase 2.

## §11 Clean-room (unchanged, restated for GPU)

CHOLMOD's GPU module is GPL — never read, concept-level papers only (Rennich et al. 2016).
cuBLAS/cuSOLVER/cuDSS are closed NVIDIA binaries: used black-box (Track 1) or as reported
baselines (cuDSS); never disassembled or source-inspected. JuliaGPU packages (CUDA.jl,
KernelAbstractions.jl, GemmKernels.jl) are MIT and freely readable — unrelated to the
SuiteSparse prohibition. Every constant (frontier cutoff, memory cap, the 2× gate margin)
carries a derivation or a measurement citation.
