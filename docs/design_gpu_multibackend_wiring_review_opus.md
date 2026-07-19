# Adversarial review — commit 9140195 (M8 backend-generic GPU driver)

Reviewer: Opus (independent audit-trail). Scope: the de-CUDA shim refactor that makes the
shipped multifrontal driver run on ROCm. Goal: break it — find correctness bugs, regressions,
unsound assumptions the passing CUDA/AMD e2e tests don't cover.

Environment caveat (read first): this review was performed on a **CPU-only host** — no CUDA and
no ROCm device was available. Every claim about *device* runtime behavior (scalar-readback
allocation parity, `fill!` on device views, extension load) is reasoned from the source + Julia
semantics, not measured. Claims marked **[needs HW]** must be confirmed on galen/gfx before being
trusted. Claims about source structure, symbol resolution, and precompile reachability are verified
against the files and are not hardware-dependent.

Verdict summary: **no correctness BLOCKER found on the shipped `:auto` path.** One real DEFECT
(vendor-arm footgun on AMD), and several coverage/robustness NITs. The shim swaps are
byte-for-byte equivalent on CUDA as far as static analysis can show.

---

## DEFECT-1 — `frontmode=:vendor` on the AMD backend throws `UndefVarError`, not a clean error

Files/lines:
- `ext/gpu_numeric.jl:334` — `trsm!('R','L','T','U', …)` (bare, cuBLAS) in the LDLᵀ hybrid `:vendor` branch.
- `ext/gpu_dense.jl:716,718` — `CUDA.CUSOLVER.potrf!` / `CUDA.CUBLAS.trsm!` in `gpu_front!`'s `mode==:vendor` branch, reached from the Cholesky hybrid via `gpu_front!(panel, nscol, ws; mode = frontmode)` (`ext/gpu_numeric.jl:145`).

Facts established:
- `trsm!` is imported *only* in `ext/gpu_leftlooking_reference.jl:17` (`using CUDA.CUBLAS: trsm!`), which is included **only** by `PureSparseCUDAExt` (`ext/PureSparseCUDAExt.jl:16`), not by `PureSparseAMDGPUExt`.
- PureSparse imports `trsm!` from PureBLAS (`src/PureSparse.jl:12`) but does **not** export it (`src/PureSparse.jl:43-52`), so `using PureSparse` in the AMD ext does **not** bring a `trsm!` into scope. The name is therefore genuinely undefined in `PureSparseAMDGPUExt` — *not* silently resolved to PureBLAS's CPU `trsm!` (which would have been a worse, silent-wrong-result bug). This is the good news.
- `CUDA` is likewise undefined in the AMD ext, so `CUDA.CUSOLVER.potrf!` is a `getproperty` on an undefined global.

Failure scenario: a user (or a future benchmark) calls
`gpu_multifrontal_hybrid!(…; frontmode=:vendor)` or `gpu_multifrontal_ldlt_hybrid!(…; frontmode=:vendor)`
on ROCm. Precompile and load succeed (the branch is guarded by a runtime `if`, and Julia resolves
GlobalRefs lazily — this is why the AMD ext loads fine). At the first GPU front the call throws a
bare `UndefVarError: trsm! not defined` / `UndefVarError: CUDA not defined`, deep in the driver,
with no hint that the vendor arm is CUDA-only.

Not a BLOCKER: the shipped path defaults to `frontmode=:auto`, which never selects `:vendor`; the
`:auto` front picks `:fused3`/`:fused`. So the tested end-to-end path is unaffected.

Suggested fix: one guard at the top of both hybrid drivers (or in `gpu_front!`):
`frontmode === :vendor && !(backend isa CUDABackend) && error("frontmode=:vendor is CUDA-only (cuSOLVER/cuBLAS); use :auto on this backend")`.
`CUDABackend` isn't in scope in the AMD ext either, so gate on `_default_backend()` type or a
`const HAS_VENDOR = true/false` per ext instead. Cheapest correct form: a `_vendor_supported()`
predicate defined per-ext (`true` in CUDA, `false` in AMD) checked before dispatch.

Verdict: **DEFECT** — cryptic runtime error instead of a clean "CUDA-only" rejection on a
user-reachable (non-default) kwarg. Fix is one guard; does not touch the shipped path.

---

## DEFECT-2 (marginal) — the AMD e2e tests never exercise the preallocated (`d_*`/`ws`) fast path

Files: `benchmark/gpu/amd_end2end_test.jl:27`, `benchmark/gpu/amd_ldlt_e2e_test.jl:23` both call the
drivers **without** the `d_emap`/`d_dummy`/`d_Anz`/`ws`/`ldlws`/`d_W`/`d_signs` kwargs. So on AMD every
factorization takes the `isnothing(…)` allocation arm: `_dev_upload(backend, A.nzval)`,
`_dev_upload(backend, Msym.emap)`, `_dev_zeros(...)`, `FrontWS(...)`/`LDLFrontWS(...)` are rebuilt each
call. The "analyze once, factorize many, zero-device-pool-alloc" preallocated path (req 5 /
amendment A) is verified **only on CUDA** (`benchmark/gpu/gpu_zeroalloc_probe.jl`), and that probe is
**LDLᵀ-only** — the Cholesky preallocated path has no zero-alloc probe on either backend.

Consequence: the shim's behavior *on the preallocated fast path specifically* (that
`_dev_upload`/`_dev_zeros` are correctly skipped, and no stray device alloc slips in) is unverified
on ROCm. Static reading says it's fine — every `_dev_upload`/`_dev_zeros`/`FrontWS`/`LDLFrontWS` in the
shipped functions is behind an `isnothing(...)` guard, mirroring the old `CuArray`-behind-`isnothing`
structure exactly (see the diff: guards unchanged, only the RHS constructor swapped). But it is a
coverage gap, not a proof.

Suggested fix: add an AMD analogue of `gpu_zeroalloc_probe.jl` (device-pool alloc = 0 on warm
refactor with `d_*` passed), and a Cholesky zero-alloc probe on at least one backend.

Verdict: **DEFECT (coverage)** — the req-5-relevant path is untested on the newly-wired backend.
No evidence of an actual regression; the guards are structurally identical to the pre-refactor code.

---

## NIT-1 — the "zero device-pool alloc" probe cannot see the per-call host readback allocations

Files: `ext/gpu_numeric.jl:102,177` (`Int(Array(ws.info)[1])`), `:274,377` (`Array(ldlws.stats)`);
gate = `benchmark/gpu/gpu_zeroalloc_probe.jl:28` (`CUDA.@allocated`).

`CUDA.@allocated` measures **device-pool** bytes only. `Array(ws.info)` and `Array(ldlws.stats)`
allocate **host** `Vector`s (one per driver call) and are structurally invisible to that gate. So the
probe's "ZERO device-pool alloc ✓" says nothing about host allocation on the factorize-many path.

Is this a *regression*? Reading the diff: the `ws.info` readback changed
`CUDA.@allowscalar ws.info[1]` → `Array(ws.info)[1]`. GPUArrays' scalar `getindex` under
`@allowscalar` itself allocates a 1-element host `Array` internally, so the host-alloc count is ≈
unchanged (both allocate one tiny host array per call). The `ldlws.stats` readback was **already**
`Array(ldlws.stats)` before the commit (the diff shows no `-`/`+` on that line), so it is not a new
allocation. **Conclusion: no new host allocation introduced by the shim** — but the GPU driver path
was never host-zero-alloc, and req 5's `@allocated == 0` gate is a CPU-path property that does not
cover these drivers. **[needs HW]** to confirm the `@allowscalar`-vs-`Array` host-alloc parity on the
CUDA version actually shipped.

Suggested (optional) fix if host-zero-alloc on the GPU path is ever desired: keep a preallocated
1-element pinned host buffer in `FrontWS`/`LDLFrontWS` and `copyto!(host_buf, ws.info)` instead of
`Array(...)`. Not required by any current gate.

Verdict: **NIT** — gate blind spot, not a regression. Document that the GPU driver path is out of
scope of the CPU zero-alloc gate.

---

## NIT-2 — Float32 GPU path is generic-but-untested; inertia stats degrade to Float32

Files: `ext/gpu_numeric.jl:237,245,275-276,294,377-378`; `ext/gpu_ldlt_dense.jl:361-362`.

`ldlws = LDLFrontWS(backend, T)` makes `stats::Vector{T}`. For `T=Float32`, inertia counts
(`n_pos/n_neg/n_zero/n_perturbed`) accumulate in Float32 and are read back as `Int(st[k])`. Float32
represents integers exactly only to 2^24 ≈ 16.7M; a matrix with >16.7M supernode columns would
mis-count inertia. Not a practical concern at any realistic size, and `max_pert` in Float32 loses
mantissa vs the Float64 CPU-front accumulator it is `max`'d against — a diagnostic-only imprecision.
Every e2e test (CUDA + AMD) is Float64; the Float32 path is exercised by no test on either backend.

Verdict: **NIT** — correct-but-generic path, untested. Consider forcing the stats/`max_pert`
accumulator to `Float64` regardless of `T` (they are counts + a scalar norm, not part of the factor),
and add one Float32 smoke test.

---

## Cleared (checked, found sound)

- **Bit-identity on CUDA.** `CUDA.fill!`→`fill!` (identical; CUDA re-exports `Base.fill!`),
  `CUDA.zeros`→`KA.zeros(CUDABackend,…)` (same allocate+zero), `CuArray(host)`→`_dev_upload`
  (`KA.allocate`+`copyto!`, same bytes), `CUDA.synchronize()`→`KA.synchronize(CUDABackend())`
  (the latter calls the former). No reduction/order change in any kernel (kernels untouched by this
  commit). CUDA factor is byte-identical to pre-refactor. **[needs HW]** to confirm empirically, but
  no code path differs.
- **`_dev_upload` argument types.** All call sites upload plain `Vector`s (`cpu.rowind`,
  `rowind_ptr`, `super`, `snode_of`, `amap`, `A.nzval`, `Msym.emap`, `signs::Vector{Int8}`) —
  `gpu_shared.jl:155-159`, `gpu_numeric.jl:69,70,121,122,238,241,244,293,300,302`. **No `SubArray`/`view`
  is ever passed to `_dev_upload`.** (The one `CuArray(view(...))` still living in the code is
  `gpu_leftlooking_reference.jl:93`/`:187` region — CUDA-only reference arm, not touched, not shipped.)
  `size(host)...` for a `Vector` is `(n,)`; `KA.allocate`+full `copyto!` overwrites all n elements.
  Equivalent to `CuArray(host)`.
- **`GPUSymbolic{Ti,VI}` typing unchanged on CUDA.** `_dev_upload(CUDABackend(), ::Vector{Ti})`
  returns `CuArray{Ti,1}`, same concrete type as the old `CuArray(v)` → `VI` type param is stable, no
  inferred-type widening.
- **Extension triggers.** `Project.toml:38-39` gives both exts trigger `["<vendor>", "KernelAbstractions"]`;
  `AMDGPU` is correctly in `[weakdeps]` (line 32), **not** `[deps]`; `KernelAbstractions` is a weakdep
  (line 35). `using AMDGPU` alone activates the ext because AMDGPU depends on KernelAbstractions (KA is
  loaded transitively, satisfying the second trigger) — identical mechanism to the already-working CUDA
  ext, whose trigger has the same shape. Well-formed.
- **`_default_backend()` default kwarg.** Defined per-ext (`PureSparseCUDAExt.jl:13`,
  `PureSparseAMDGPUExt.jl:16`) **before** `include("gpu_shared.jl")`, so the `backend = _default_backend()`
  default in `gpu_symbolic` (`gpu_shared.jl:142`) resolves to the enclosing ext's method at call time.
  Defined at module top-level before any call → no world-age hazard.
- **Precompile safety of the AMD ext.** The dangling `trsm!` / `CUDA.*` GlobalRefs (DEFECT-1) sit in
  runtime-guarded branches; Julia infers undefined globals as `Any` (a deferred `getglobal` throw), not
  a precompile error. `_ldl_block!` — the other symbol used in the `:vendor` LDLᵀ branch — is defined in
  `ext/multifrontal.jl` (included by `gpu_shared.jl`), so it *is* available in the AMD ext; only
  `trsm!`/`CUDA` dangle. Consistent with the reported "AMD ext precompiles + tests pass."
- **CUDA `:vendor` `trsm!` resolution.** In `PureSparseCUDAExt`, `using CUDA.CUBLAS: trsm!`
  (`gpu_leftlooking_reference.jl:17`) is an explicit import that binds `trsm!` in the ext module; it
  wins over the implicit (and here absent) `using PureSparse` export regardless of include order, so the
  vendor branch resolves to cuBLAS `trsm!` at runtime. Include order (numeric before leftlooking) is not
  a hazard because the binding is module-scoped, not lexical.
- **Crossing-U on AMD.** Both AMD tests do run genuine hybrids (Cholesky test sweeps
  `all-GPU/25%/75%/all-CPU`; LDLᵀ test frontiers at the median), so the CPU-front-with-GPU-parent
  `copyto!(device_arena, uo, host_arena, uo, us)` crossing path *is* covered on ROCm.
- **`copyto!(d_Anz, A.nzval)` return value.** `copyto!` returns its destination, so the ternary
  `isnothing(d_Anz) ? _dev_upload(...) : copyto!(d_Anz, A.nzval)` yields the device array in both arms,
  same as the old `CuArray(A.nzval)`/`copyto!` ternary. Correct.
- **Inertia reset across refactors.** `fill!(ldlws.stats, zero(T))` runs at the head of every
  ldlt driver (`gpu_numeric.jl:245,294`), so buffer reuse doesn't accumulate stale inertia.

---

## One-line verdicts
- DEFECT-1: `:vendor` on AMD → cryptic `UndefVarError`; guard it. Not on shipped `:auto` path.
- DEFECT-2: preallocated / zero-alloc fast path untested on AMD; Cholesky zero-alloc untested anywhere.
- NIT-1: device-pool alloc probe is blind to the per-call host readback; not a regression (parity holds).
- NIT-2: Float32 GPU path + Float32 inertia stats untested; consider Float64 stats accumulator.
- Everything else audited: sound; CUDA path is byte-identical modulo unmeasured HW confirmation.

---

## Resolution (Opus, applied — commit follows this file)

Both reviews converged: no BLOCKER; the shipped `:auto` path is a faithful, semantically-equivalent
port on both backends; CUDA no-regression holds. Applied:

- **DEFECT-1 (`:vendor` on AMD → `UndefVarError`; false "no CUDA calls" comments) — FIXED.** Added a
  `_vendor_available()` guard to both vendor branches (`gpu_dense.jl` gpu_front!, `gpu_numeric.jl`
  ldlt_hybrid): on a non-CUDA backend `frontmode=:vendor` now throws a clear
  "CUDA-only reference arm" error. Verified on gfx1152: `VENDOR-GUARD: CLEAN`. Corrected the
  misleading header comments in `gpu_shared.jl`/`gpu_numeric.jl` to name the `:vendor` exception.
- **Coverage (both) — CLOSED.** `amd_end2end_test.jl` now runs the Cholesky **device solve**
  (`gpu_solve!`, res 4.5e-16 on gfx1152) and a **Float32** end-to-end case (relerr 3e-8, solve 2e-7),
  in addition to the factor oracle.
- **A NEW bug the re-gate caught (not in either review):** defining `_vendor_available()` in
  `gpu_shared.jl` and overriding it in the CUDA-only file = *method overwriting during precompilation*,
  which **Julia 1.12 rejects** (the CUDA ext only loaded via non-precompiled fallback). Fixed by
  defining `_vendor_available()` **exactly once per ext** (CUDA→true, AMD→false); `gpu_shared.jl` does
  not define it. Re-verified on galen: CUDA ext precompiles clean, `:auto` oracle PASSES, `:vendor` arm
  WORKS (ok=true, relerr=7.9e-17).
- **NOT changed (accepted as noted):** Float32 inertia-stats precision (NIT — realistic sizes fine);
  AMD test-env reproducibility (manual hardware-gated scripts, same status as the CUDA gpu tests).
