# M8 multi-backend GPU design — adversarial review (Fable)

> Review of `design_gpu_multibackend.md` v1. Independent of the parallel Opus review. Permanent
> audit trail. Findings to fold into v2.

## BLOCKER 1 — Clause 2 on CDNA is a hardware instruction-class fight (FP64 matrix cores), not a tuning problem (§B3/§B5/§B9.1)
§B3 frames the AMD gap as occupancy/tile retuning. The real gap: CDNA (gfx90a/gfx942) has **FP64
matrix cores (MFMA)** — published FP64 matrix rate ~2× the vector rate (MI210 ~22.6 TF vector /
45.3 TF matrix; MI300X ~81.7 / 163.4), and rocBLAS DGEMM emits MFMA-f64. A KernelAbstractions
kernel built from scalar `muladd` **cannot reach MFMA** (KA has no matrix-op abstraction; LLVM won't
auto-fuse scalar FMAs to MFMA). So on the flop-dominant crown-front gemm/syrk, the pure kernel's
ceiling is ~½ rocBLAS's, regardless of tiles. **M6's 1.14× win existed because Ada has NO FP64
matrix path — cuBLAS and pure shared the same weak FFMA pipeline, and pure won on fusion/launch at
99% of the *shared* peak. That mechanism inverts on CDNA.** (Asymmetry the doc misses: Intel PVC has
no FP64 in XMX → oneMKL DGEMM is vector-FP64 → clause 2 is a fair fight on Intel but a structural
loss on AMD at large K.) **Fix:** (a) §B9.1 must say clause 2 at large fronts is *expected to lose*
~2× on published rates; gate per-shape not "every size". (b) decide now which escape is in scope
(AMD-specific MFMA kernel = weeks, non-portable; or amendment-C best-per-op letting rocBLAS take
large-front gemm = abandons "shipped path is pure KA"). (c) day-0 CDNA probe: one rocBLAS-DGEMM-vs-
pure-KA measurement at a crown shape BEFORE funding a tuning campaign (~$5 of rental).

## BLOCKER 2 — The gate cannot be authoritative under the project's own benchmarking law (§B5/§B6)
Project law: perf verdicts need **clock-locked** hosts + **two-host** confirmation (M6 flagged
galen-only as a limitation). M8 runs on **short cloud rentals**: shared/virtualized instances where
clock pinning needs root/bare-metal (unavailable in containers), the CPU differs per rental (clause-3
baseline drifts), and it's one host/session (two-host bar silently dropped). §B5 says "clock-locked
host" as if given. **Fix:** amend the protocol (user sign-off): require bare-metal + record
deterministic clocks in JSON (fail loudly if unpinnable); two independent rental sessions agreeing;
pin clause-3 CPU baseline to one named instance type; else publish verdicts as *provisional*.

## DEFECT 3 — Vendor arms are braided INTO the shared drivers; the §B1 residue table is incomplete
`gpu_front!` (gpu_dense.jl:714–721) calls `CUDA.CUSOLVER.potrf!`/`CUDA.CUBLAS.trsm!` inline under
`mode==:vendor`; `gpu_multifrontal_ldlt_hybrid!` (gpu_numeric.jl:561–578) has a `:vendor` branch. The
shared engine does NOT compile in a non-CUDA ext as-is. Table also misses `CUDA.@allowscalar`,
`CUDA.fill!`, 5-arg offset `copyto!`, `Array(ldlws.stats)`, per-driver `CUDA.zeros` defaults — ~40
`CUDA.`-prefixed sites by grep, not 5. **Fix:** define a vendor-injection seam (`vendor_front!(::Backend,…)`
hook per backend); regenerate the residue table from `grep -n 'CUDA\.'`; state the fate of the
left-looking reference arms (they import CUDA.CUSOLVER + LAPACK/BLAS directly).

## DEFECT 4 — Vendor-arm composition unpinned → clause 2 risks a strawman
In current code `:vendor` swaps only potrf/trsm; the flop-dominant trailing `gpu_syrk_nt!` stays PURE
in every mode. If the AMD vendor arm is built the same way, clause 2 compares pure vs
pure-with-vendor-potrf — the MFMA advantage (finding 1) never appears and "pure ≥ vendor" is won by
construction. **Fix:** vendor arm = full vendor dense stack per front (rocSOLVER potrf + rocBLAS trsm
+ **rocBLAS gemm/syrk for trailing**; ditto oneMKL), stated before measuring.

## DEFECT 5 — Register-residency of the fused kernels is a fresh per-compiler bet with no verify step
`_front_fused64_v3!` holds two 4×4 FP64 `@private` tiles with `@nexprs` literal indexing; the file
records a runtime-index demote-to-local = measured **3×** slowdown on NVIDIA. Whether AMD LLVM / Intel
IGC keep those in registers is exactly the M7 register-residency bet. `FUSE_M_MAX=6000` etc. are galen
constants baked into `:auto` dispatch → on a backend where v3 spills, `:auto` silently picks a 3×-slow
kernel. **Fix:** per-backend register/spill audit (AMDGPU `@device_code` VGPR+scratch; Level-Zero
stats) BEFORE the sweep; re-measured crossovers per backend; budget "v3 doesn't hold → fall back to
v2/v1 structure" (per-backend kernel *selection*, not tuning).

## DEFECT 6 — The solve's FP64 global atomic add is an unprobed oneAPI dependency
`gpu_solve.jl:106` `@atomic y[…] += -acc` — FP64 global atomic add on every forward-solve level, in
the timed factor+solve path. CDNA has native `global_atomic_add_f64` (fine); **oneAPI FP64 `@atomic`
lowering is unverified** and §B4's Intel gotchas mention only FP64-hardware + oneMKL. If it doesn't
lower, the solve (and clauses 2/3) is dead on Intel. **Fix:** day-0 "FP64 `@atomic` add compiles +
correct" smoke on BOTH new backends; name the non-atomic two-pass fallback.

## DEFECT 7 — §B0.2 vs §B8 contradiction: the "gate" is not a gate
§B0.2 "Gate: pure ≥ vendor, mirroring M6." §B8 "M8 closes when refactor + ≥1 backend clear
correctness + clause-3, clause-2 recorded." The milestone closes even when the headline gate fails —
M6's gate was pass/fail; M8's clause 2 is a measurement. Defensible scope (clause-3-only is real: no
GPU sparse Cholesky exists for AMD/Intel Julia users today) but "mirroring M6" while defining a weaker
bar is the face-saving-reframe the process catches. **Fix:** clause 3 is THE gate; clause 2 is a
recorded stretch with expected per-backend outcomes stated up front. Don't call it "mirroring M6".

## DEFECT 8 — Dual-backend load semantics unspecified
Each ext including the engine works (relative `include`), duplicate defs harmless in separate ext
modules — BUT `CUDAExt.GPUSymbolic ≠ AMDGPUExt.GPUSymbolic`, and "get_extension generalizes to THE
loaded backend" is undefined with CUDA+AMDGPU both loaded (AMD iGPU + NVIDIA dGPU = the user's own
boxes). If each ext adds an identical-signature method to a shared `PureSparse.symbolic`, the second
load clobbers the first (kwargs don't dispatch). **Fix:** entry methods dispatch on positional
`KA.Backend`/device-array type, one method per ext, never same-signature; both-loaded smoke in §B7.

## DEFECT 9 — §B6 prices the gate session, not the campaign
§B3 says the tuning campaign IS the milestone; §B6 budgets one 2–3h session. Each stateless rsync
session re-pays depot + AMDGPU/oneAPI-vs-system-ROCm setup (version roulette) + precompile + zoo
download + per-size compile tax ≈ 30–60 min before the first datapoint; multi-day loops lose sweep
state. **Fix:** persistent volume / prebuilt pinned container; checkpoint JSON continuously; budget
O(10) sessions.

## DEFECT 10 — oneMKL potrf/trsm via oneAPI.jl unverified
Intel clause-2 arm assumes callable oneMKL potrf/trsm/gemm from oneAPI.jl. BLAS wrappers exist; GPU
LAPACK (`potrf`) coverage is unverified. If absent → new ccall wrapper work, unbudgeted. **Fix:**
verify before committing the Intel sub-milestone; else declare Intel clause-2 gemm-only.

## NITs
- **11 (§B3):** per-backend Preferences keys (`fuse_m_max_cuda/_amdgpu/_oneapi`) — single key clobbers CUDA tuning for a dual-GPU user + galen re-verify.
- **12 (§B5):** "relerr < 1e-10 (bit-identical)" is self-contradictory + the solve's atomic scatter isn't even run-to-run bit-stable. Normwise per §10.1; drop "bit-identical" except the §B1 CUDA A/B.
- **13 (§B5 clause 3):** HIP launch latency > CUDA's 5.5µs; multifrontal ≈ 4+nchildren launches/front → small-front strata compress the clause-3 margin; re-derive `frontier_cutoff` per backend.

## Verdict
M8 is sound **as a portability milestone gated on clause 3** (the de-CUDA refactor is mostly
mechanical though under-enumerated; beats-CPU on full-FP64 AMD/Intel is very probably winnable). It is
**NOT** sound on its headline: clause 2 on CDNA is an MFMA instruction-class fight tile-tuning cannot
win, the "tuning campaign" framing + cost estimate hide that, and the rented gate can't authoritatively
adjudicate the near-parity outcomes it predicts. Single biggest risk: spending the milestone's budget
tuning toward a clause-2 outcome on AMD that a $5 day-0 rocBLAS-vs-pure probe (and a spec sheet) settles
first.
