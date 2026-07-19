# PureSparse.jl — M8 Design: Multi-backend GPU (AMD + Intel oneAPI), FP64

> **Status: v1 REVIEWED → PROBE-GATED (measurement-first, 2026-07-19).** Two independent adversarial
> reviews (`design_gpu_multibackend_review_fable.md`, `_opus.md`) **converged on the same
> BLOCKER**: the M6 "pure ≥ vendor" win is very likely **not portable** to AMD/Intel — and for a
> *structural* reason, not a tuning gap. M6's pure kernels beat cuBLAS only because **Ada has no
> FP64 matrix path** (cuBLAS DGEMM on Ada is itself vector-FMA-bound, so the pure `muladd` kernel
> tied the vendor's own pipeline at ~99% of the *shared* vector peak and won on fusion). **AMD CDNA
> (MFMA-f64) and Intel Max (XMX/DPAS-f64) DO have FP64 matrix cores** that rocBLAS/oneMKL DGEMM ride
> at ~2× the vector rate; a pure KernelAbstractions `muladd` kernel **cannot emit them**. So clause 2
> (pure ≥ vendor) is likely **foreclosed by construction** on the flop-dominant gemm — the same
> instruction-class lesson that shelved M7.
>
> **Decision (user, measurement-first like M7's Phase 0): a ~$5 day-0 probe settles the premise
> before v2.** Run `benchmark/gpu/dgemm_vendor_probe.jl` on a rented AMD **CDNA** part (MI210/300):
> pure-KA FP64 gemm vs rocBLAS DGEMM at crown shapes. γ = vendor/pure:
>  - **γ ≈ 0.5** → rocBLAS uses MFMA the pure kernel can't → clause-2 FORECLOSED → **v2 re-scopes M8
>    to a portability + beats-(multithreaded)-CPU milestone** (clause 3 the gate; clause 2 a recorded
>    probe with expected-loss stated), OR the hybrid best-per-op option (pure front + vendor DGEMM
>    trailing). §B0.2 must NOT claim "mirrors M6".
>  - **γ ≈ 1.0** → rocBLAS isn't using MFMA at these shapes → clause 2 alive → proceed toward the
>    original gate.
> The probe also runs on NVIDIA (should reproduce M6's ~1.1–1.14×) as a methodology sanity check.
>
> **Other findings folded into v2 (both reviews):** the CUDA "residue" is ~40 sites (a backend-shim
> refactor, not 5 swaps — §B1); clause 3 must specify **multithreaded** CHOLMOD (else "beats one
> core"); the rented-cloud gate can't be clock-pinned/two-host-confirmed (amend the protocol or mark
> verdicts provisional); "bit-identical" over-promises (use M6's *normwise* oracle); the vendor arm
> must be the **full** vendor dense stack incl. trailing gemm (not pure-with-vendor-potrf); FP64
> `@atomic` add + `reshape(view(dx))` need day-0 smokes on each backend; dual-backend load + per-
> backend Preferences keys. **v2 is written only after the probe number lands.**
>
> Same v1 → 2 reviews → v2 arc as `design.md` (M1), `design_qr*.md` (M5), `design_gpu.md` (M6),
> `design_qr_gpu.md` (M7). Builds on M6 (§M of `design_gpu.md`, the CUDA multifrontal engine, CLOSED).
> §-numbers prefixed **B**. The v1 body below is retained as the record of what was designed; its
> §B0.2 gate premise and §B3 "tuning" framing are SUPERSEDED by the probe outcome above.

## §B0 What M8 is — and the decisions locked before this draft

Extend PureSparse's GPU backend beyond CUDA to **AMD (ROCm / AMDGPU.jl)** and **Intel
(oneAPI.jl)**, at **Float64**. The shipped multifrontal path is already pure KernelAbstractions
(amendment C), so the kernels are portable in principle; M8 is the work of making the *engine*
backend-agnostic, standing up per-backend extensions, and — the real milestone — **retuning the
pure kernels to clear the "pure ≥ vendor" bar on each new vendor**.

Four decisions were made with the user before this draft (they are premises, not open questions):

1. **Precision: FP64-first.** Target AMD **CDNA** (MI210/MI250/MI300, full-rate FP64) and Intel
   **Data Center Max / Ponte Vecchio** (full-rate FP64). **Apple Metal is explicitly out of scope**
   — Apple GPUs have *no* FP64 at all, so a Metal backend is a separate Float32 / iterative-
   refinement milestone, deferred (§B11). Consumer AMD RDNA (incl. the gfx1151 iGPU) and Intel Arc
   throttle/emulate FP64 → correctness-only there, never a perf target.
2. **Gate: pure ≥ vendor, mirroring M6.** Per backend, pure-KA factor+solve ≥ the vendor GPU
   library (rocSOLVER/rocBLAS on AMD, oneMKL on Intel) at every size, **and** beats CHOLMOD+OpenBLAS
   on CPU. The vendor libraries are **diagnostic comparison arms only**; the shipped path is pure KA.
3. **Architecture: shared engine files in `ext/`, included by each per-backend ext; KA stays a weak
   dependency.** No new hard dep on the CPU core (preserves the lean/trim philosophy).
4. **Rollout: shared de-CUDA refactor first (re-verified bit-identical on CUDA/galen), then AMD,
   then Intel.**

Non-goals (carried from M6 + the above): Metal / any FP32-only backend; consumer-GPU FP64 perf;
complex/BigFloat GPU fronts (Float64-tuned path); the `:column` QR path and update/downdate (CPU).

## §B1 Architecture — de-CUDA the shared engine

Today the CUDA extension `ext/PureSparseCUDAExt.jl` includes the engine files
(`frontier.jl`, `multifrontal.jl`, `gpu_dense.jl`, `gpu_ldlt_dense.jl`, `gpu_solve.jl`,
`gpu_numeric.jl`, and now `gpu_vendor_solve.jl`). The shipped multifrontal engine is already pure
KA; the CUDA-specific residue is small and enumerable:

| CUDA-specific today | Backend-generic replacement |
|---|---|
| `CuArray(x)`, `CUDA.zeros(T, …)` (device allocation in the drivers) | `KernelAbstractions.zeros(backend, T, …)` / `KA.allocate(backend, …)` |
| `CUDA.synchronize()` | `KA.synchronize(backend)` |
| `get_backend(dx)` on a `CuArray` | unchanged — already backend-generic |
| cuSOLVER `potrf!`/`geqrf!`, cuBLAS `trsm!`/`trsv!`/`gemv!` (vendor arms + `gpu_vendor_solve.jl`) | **move out of the shared engine** into per-backend vendor-arm files (§B2) |
| `PosDefException`, `LAPACK`/`BLAS` (CPU-front fallbacks in `multifrontal.jl`) | pure Julia / PureBLAS — already CPU-side, backend-neutral |

Concretely: gather the KA-generic engine into an `ext/gpu_engine/` group (or keep the filenames,
purged of the above). Each per-backend ext `include`s that group. The device-allocation call sites
are the only non-mechanical part — they must thread the `backend` (obtained from an
input device array, or a `KA.Backend` the entry point holds) into every `KA.zeros`/`allocate`. The
`_extend_add_cpu!` / `_cpu_ldl_front!` CPU-front helpers are already backend-neutral (they run on
host arrays for CPU fronts).

**No-regression bar for §B1:** the CUDA path must stay **bit-identical** on galen after the refactor
(the M6 oracle + a gate-timing A/B) before any new backend is added — the de-CUDA-ing is a pure
refactor, verified by the existing `gpu_mf_hybrid_test.jl` / `gpu_ldlt_e2e_test.jl` oracles.

## §B2 Per-backend extensions

Three thin weakdep extensions, each `include`-ing the shared engine and providing only:
- the backend's **device array type + allocation** entry (so the driver's `KA.zeros` resolves to
  `CuArray`/`ROCArray`/`oneArray`);
- the **vendor comparison arms** (diagnostic, never shipped): cuSOLVER/cuBLAS (exists),
  rocSOLVER/rocBLAS (AMDGPU), oneMKL (oneAPI);
- the public entry (`gpu_symbolic`, `gpu_cholesky`/`ldlt`/`solve` wrappers).

```
ext/gpu_engine/*.jl            # KA-generic: kernels + multifrontal driver + device solve
ext/PureSparseCUDAExt.jl       # include engine + cuSOLVER/cuBLAS arms   (exists, slimmed)
ext/PureSparseAMDGPUExt.jl     # include engine + rocSOLVER/rocBLAS arms (new)
ext/PureSparseoneAPIExt.jl     # include engine + oneMKL arms            (new)
```

`Project.toml` gains `AMDGPU`, `oneAPI` as `[weakdeps]` + the two `[extensions]` triggers. KA stays
a weakdep (each GPU package pulls it transitively). The `Base.get_extension(PureSparse, :…Ext)`
test access generalizes to the loaded backend.

## §B3 The crux — per-backend kernel tuning (this IS the milestone)

The port (§B1–B2) is a few days. The **gate (§B0.2) is the milestone**, and it is *not* free:

- The pure kernels beat cuBLAS FP64 on Ada (M6) because they were **tuned for NVIDIA**: 4×4
  register tiles, 64×8 shared-memory staging, 256-thread groups, the `muladd`/IEEE-strict trick,
  and the fused-front register-residency (`_front_fused64_v3!`), all validated against the *Ada*
  occupancy/register model.
- **AMD CDNA** (gfx90a/gfx942) has 64-wide wavefronts (vs NVIDIA's 32), different LDS/register
  budgets, and a different FP64 pipeline; **Intel Xe** (oneAPI) has sub-groups of 8/16/32 and its
  own SLM model. The Ada-tuned tile/group sizes will almost certainly be *sub-optimal* on both, and
  **rocBLAS / oneMKL are strong FP64 libraries**. Clearing "pure ≥ vendor" on each backend is a
  **per-backend tuning campaign** — sweep tile sizes / group widths / fusion thresholds (Preferences-
  backed, extending the existing `tuning.jl` consts like `FUSE_M_MAX`, `LDL_FUSE_M_MAX`), possibly
  add per-backend kernel variants (a 64-wide-wavefront front kernel for CDNA), and measure.
- **Honest, measurement-first stance (per M7):** if pure cannot beat the vendor lib on a backend
  after a reasonable tuning effort, we say so and it ships there as **"pure, portable, beats-CPU"**
  (the §B0.2 clause-3 CPU bar, which is a GPU-throughput win independent of the vendor) rather than
  faking a "beats-vendor" claim. The gate has two clauses; clause-2 (≥ vendor) is the headline bet,
  clause-3 (beats CPU) is the must-win and is very likely on FP64-capable hardware.

## §B4 Backend-specific gotchas (to probe, not assume)

- **AMD atomics.** The gfx1151 (RDNA3.5 iGPU) needed workarounds: Atomix bare-atomic-add only (no
  atomic-rmw *return* values — they segfault GPUCompiler's `check_ir!`), Int32-only (Int64 atomics
  crash even bare), election-free group-1-to-scratch write-back. **CDNA (gfx90a/gfx942) has far more
  mature ROCm/LLVM support** — these may not apply, or may fail differently. **Re-probe atomics on
  CDNA first** (a day-0 smoke), don't port the iGPU hacks blindly. The shipped multifrontal engine
  already avoids atomic-rmw-returns via the election-free scatter, which is portable and should be
  kept regardless.
- **Intel oneAPI FP64.** Requires an actual FP64-capable Intel GPU (Max/Ponte Vecchio) + the FP64-
  enabled level-zero runtime; Arc will silently be slow/emulated. oneMKL-on-GPU is the vendor arm —
  confirm the AD-traceable generic-`T` path still holds (oneAPI's array ops).
- **rocSOLVER/rocBLAS FP64** are full-featured; the comparison arm is straightforward. `geqrf` is
  out of scope (QR-GPU shelved, M7).
- **KA backend maturity.** AMDGPU.jl + oneAPI.jl KernelAbstractions backends are supported but less
  battle-tested than CUDA.jl for FP64 kernels — expect to file/​work-around a compiler quirk or two
  per backend (budget for it).

## §B5 The gate (per backend, mirrors M6 §8 re-scoped)

Per backend, on a **pinned stratum** (named SPD + SQD/KKT matrices, sizes, ≥N, memory-feasibility
bound — the M6 §8.3 discipline, anti-cherry-pick):
1. **Correctness (must pass):** pure-KA hybrid factor matches the CPU factor at every frontier
   cutoff, `relerr < 1e-10` (bit-identical to the M6 oracle path); LDLᵀ inertia + solve residual
   likewise.
2. **Clause 2 — pure ≥ vendor (headline bet):** pure factor+solve ≥ rocSOLVER/rocBLAS (AMD) /
   oneMKL (Intel) at every size. Measurement-first (§B3) — may land at parity/loss and be recorded.
3. **Clause 3 — beats CPU (must-win):** median(pure-KA on the GPU) < median(CHOLMOD+OpenBLAS on
   CPU), both own-ordering and same-permutation (CLAUDE.md req 2).

Chairmarks medians, clock-locked host, results→JSON, **violin** plots (project convention).

## §B6 Verification & hardware

No AMD/Intel FP64 hardware on hand (the local box is an AMD *iGPU*, FP64-throttled). Use **short
cloud rentals**:
- **AMD CDNA:** MI210 (cheapest full-FP64) or MI300X via RunPod / Vultr / TensorWave / AMD Developer
  Cloud (~$2–5/hr; a 2–3 h gate session ≈ $10–40).
- **Intel Max:** Intel **Tiber Developer Cloud** (Max 1100/1550 access), or a cloud MI-equivalent.

Driven with the **ephemeral-remote workflow already proven on galen**: rsync the repo → run the
backend-parameterized oracle (correctness) + perf (vendor A/B) → pull JSON → tear down. The remote
is stateless (rsync tree, not a git checkout), so the "remote-sync-before-benchmarking" rule applies
(verify the synced tree before each gate run).

## §B7 Testing

Backend-parameterize the existing GPU scripts: `gpu_mf_hybrid_test.jl`, `gpu_ldlt_e2e_test.jl`,
`gpu_mf_hybrid_perf.jl`, `gpu_ldlt_perf.jl` take a `backend` + array constructor (default CUDA), so
one script runs on any vendor via `--backend=amdgpu|oneapi|cuda`. CPU-side unit tests
(`test/gpu_*_tests.jl` — pure, CPU-only: frontier/symbolic/arena) are unchanged and stay in the
suite. GPU gates remain **manual/remote** (CI runners have no GPU — the CUDA gate is already manual;
AMD/Intel are the same).

## §B8 Build order

0. **Shared de-CUDA refactor** (§B1) → galen bit-identical re-verify (no-regression gate). Ships CUDA
   unchanged; unblocks all backends.
1. **`PureSparseAMDGPUExt`** (§B2) + CDNA atomics re-probe (§B4) → correctness oracle on rented MI210
   → rocSOLVER/rocBLAS vendor arm → per-backend tuning (§B3) → the §B5 gate → record verdict.
2. **`PureSparseoneAPIExt`** → correctness on rented Intel Max → oneMKL arm → tuning → gate → verdict.
3. Publish per-backend numbers to `docs/src/benchmarking.md` (violins) with honest per-backend
   clause-2 outcomes.

Each backend is its own sub-milestone with its own gate; M8 "closes" when the shared refactor +
≥1 new backend clear correctness + clause-3, with clause-2 recorded per backend.

## §B9 Hard bets (honest)

1. **Clause 2 (pure ≥ vendor) may not hold on every backend.** Ada-tuned kernels vs strong
   rocBLAS/oneMKL on differently-shaped hardware (64-wide wavefronts, 8–32-wide sub-groups) is a
   real fight; parity or a small loss is a plausible outcome after tuning. M8 still succeeds via
   clause 3 (beats CPU) + portability; clause 2 is the stretch, measured not assumed.
2. **KA backend maturity** on AMDGPU/oneAPI for FP64 kernels — expect compiler quirks; budget
   probing time (the CDNA atomics question is the first).
3. **Hardware access friction** — cloud FP64 AMD/Intel availability shifts; the gate depends on
   securing a rental slot.

## §B10 Provenance & clean-room

Clean-room (CLAUDE.md req 1) unchanged: engine + kernels are ours (KA); rocSOLVER/rocBLAS, oneMKL,
cuSOLVER/cuBLAS are **black-box comparison baselines only** — never their source. KernelAbstractions,
AMDGPU.jl, oneAPI.jl are the portability layer (MIT/permissive, standard deps). No CHOLMOD/SuiteSparse
source, in any form.

## §B11 Deferred: Metal (FP32 + iterative refinement)

Apple Metal has no FP64, so a Metal backend is a *different* project: Float32 GPU factor + solve, then
CPU **iterative refinement** (residual in FP64, correct) to recover FP64 accuracy for the ill-
conditioned IPM/KKT target. That is a numerical-methods design in its own right (convergence,
when-to-stop, FP32 pivot stability) — a separate future milestone, not part of M8.
