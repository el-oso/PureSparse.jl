# Adversarial review — commit 9140195 "M8 wiring: end-to-end Cholesky + LDLᵀ on AMD ROCm"

Reviewer: Fable-model agent, 2026-07-19. Method: full diff read (`9140195^` vs `9140195`),
line-by-line audit of every shim substitution and every file the AMD ext includes, plus
**empirical probes run on this box's gfx1152 iGPU in a fresh throwaway env** (scratchpad
`amd_review_probe.jl`, offline `Pkg.develop` of PureSparse + dev PureBLAS + `Pkg.add(AMDGPU)`).
Dynamic claims below marked **[verified by run]** were executed, not inferred.

**Verdict: no BLOCKER — the shipped `:auto` path is a faithful, semantically equivalent
port on both backends — but the "backend-generic, no CUDA-API calls" claim is false in two
shared files (`frontmode=:vendor` throws raw `UndefVarError` on AMD, verified by run), and
the new extension silently fails to activate in any pre-commit-resolved environment
(including the standing `amd_probe` env on this box, which additionally cannot re-resolve).**

---

## DEFECTS

### D1. `frontmode=:vendor` on AMD → raw `UndefVarError`; two CUDA-qualified call sites remain in AMD-included files

- `ext/gpu_dense.jl:714-719` — `gpu_front!`'s `mode == :vendor` branch calls
  `CUDA.CUSOLVER.potrf!` / `CUDA.CUBLAS.trsm!` **fully qualified**. `gpu_dense.jl` is
  included by `gpu_shared.jl:109`, i.e. by `PureSparseAMDGPUExt`, where `CUDA` is an
  undefined global.
- `ext/gpu_numeric.jl:334` — `gpu_multifrontal_ldlt_hybrid!`'s `:vendor` branch calls
  unqualified `trsm!`, whose only import (`using CUDA.CUBLAS: trsm!`,
  `gpu_leftlooking_reference.jl:17`) exists only in the CUDA ext.

**[verified by run]** on gfx1152:

```
:vendor Cholesky on AMD threw: UndefVarError: `CUDA` not defined in `PureSparseAMDGPUExt`
:vendor LDLT     on AMD threw: UndefVarError: `trsm!` not defined in `PureSparseAMDGPUExt`
```

Concrete failure scenario: any user (or future benchmark script copied from the CUDA gate
harness, e.g. `benchmark/gpu/gpu_gate_breakdown.jl`-style arm-4 comparisons) passing
`frontmode=:vendor` to `gpu_multifrontal_hybrid!`/`gpu_multifrontal_ldlt_hybrid!` on ROCm
gets an unexplained `UndefVarError` instead of a diagnostic. No silent corruption — it
throws before touching data — and `:auto`/`:fused*`/`:split_*` never reach these lines
(checked: `gpu_dense.jl:714` tests `mode == :vendor` before the `md` computation at
`:734`; `gpu_numeric.jl:323` tests `frontmode == :vendor` explicitly; a typo'd mode
symbol falls into the pure branches, not the vendor one).

The load/precompile-hazard half of this attack does NOT materialize: undefined globals in
method bodies are legal at lowering/precompile time in Julia 1.12, and the AMD ext
demonstrably precompiles and loads (**[verified by run]** — fresh-env build produced
`~/.julia/compiled/v1.12/PureSparseAMDGPUExt/*.so` and the `:auto` hybrid ran green).
Likewise the CUDA ext's include order (`gpu_numeric.jl` referencing `trsm!` before
`gpu_leftlooking_reference.jl` imports it) is safe: the import executes at module load,
before any call can resolve the binding; galen oracles re-ran green.

**But the header claims are now false:** `gpu_shared.jl:5` ("everything here … compiles +
runs on any KA backend"), `gpu_numeric.jl:2-4` ("No CUDA-API calls here"), and
`gpu_numeric.jl:13-14` ("VENDOR … split out to gpu_leftlooking_reference.jl … included
ONLY by the CUDA ext") — while the vendor *branches* still live in the shared files at
`gpu_dense.jl:714` and `gpu_numeric.jl:323`.

Suggested fix (pick one):
1. Cheapest: in both vendor branches, guard with
   `isdefined(@__MODULE__, :trsm!) || error("frontmode=:vendor requires the CUDA extension")`
   — or simply `error(...)` unconditionally in a stub the CUDA ext overrides; and fix the
   three header comments to say "vendor branch is CUDA-only and errors elsewhere".
2. Cleaner: hoist the two vendor branches into `_vendor_front!`/`_vendor_ldl_front!`
   defined in `gpu_leftlooking_reference.jl` (CUDA ext) with an `error(...)` fallback
   method in `gpu_shared.jl`. That makes the "no CUDA API in shared files" claim true.

### D2. Extension activation is Manifest-gated: every pre-commit-resolved env silently gets `ext === nothing`; the standing `amd_probe` env is dead and cannot re-resolve

Julia records a package's extension list in the consuming environment's **Manifest** at
resolve time. Any environment resolved before this commit has, for the path-dev'd
PureSparse, only `PureSparseCUDAExt`/`PureSparseForwardDiffExt` recorded — and
`PureSparseAMDGPUExt` will **never load** there, no matter what is `using`'d.

**[verified by run]** on this box:
- `/home/el_oso/Documents/claude/amd_probe` (the env the AMD probe scripts were built
  around; Manifest mtime 2026-07-17, pre-commit): `Base.get_extension(PureSparse,
  :PureSparseAMDGPUExt) === nothing` both without and **with** an explicit
  `using KernelAbstractions`.
- Worse, `Pkg.resolve()` in that env **fails outright**: `Unsatisfiable requirements …
  StrictMode … restricted to versions 0.3.9 - 0.3 by PureSparse … restricted to versions
  0.3.8 by an explicit requirement — no versions left`. So the documented run instruction
  in `benchmark/gpu/amd_end2end_test.jl:5` ("Run: julia --project=<env-with-PureSparse+
  AMDGPU>") is not satisfiable by the only standing AMD env on the machine that verified
  this commit. (The commit's verification run was real — the ext `.so` cache is
  timestamped 12:41 today and the repo-local gitignored Manifest was re-resolved at 12:45
  with the ext recorded — but that env is not reconstructible from anything committed.)
- A **fresh** env (`Pkg.develop` PureSparse + `Pkg.add AMDGPU`, offline) works: the ext
  activates with only `using PureSparse, AMDGPU` — no explicit
  `using KernelAbstractions` needed (KA loads transitively as AMDGPU's hard dep and the
  trigger fires) **[verified by run]**, consistent with the CUDA ext's behavior.

Suggested fix: (a) repair or delete `amd_probe` (the StrictMode-0.3.8 conflict predates
this commit but now blocks the only named AMD env); (b) one line in
`amd_end2end_test.jl`/`amd_ldlt_e2e_test.jl` headers and `docs/src/developer.md`: "env
must be (re)resolved after PureSparse gained the AMDGPU extension — a stale Manifest
silently yields `get_extension(...) === nothing`". The scripts' `@assert ext !== nothing`
already catches it loudly at run time, which is why this is a DEFECT, not a BLOCKER.

---

## NITs / verified-clean attack surfaces (kept for the audit trail)

### N1. Shim semantic equivalence — audited call-by-call, clean

- `Array(ws.info)[1]` vs `CUDA.@allowscalar ws.info[1]` (`gpu_numeric.jl:102,177`):
  `ws.info` is length-1 `Int32` (`gpu_dense.jl:684-686`), so both forms are a 1-element
  D2H copy returning `Int32`; `Array(...)` needs no prior explicit sync for correctness
  (stream-ordered after the loop's kernels) and the explicit
  `KernelAbstractions.synchronize(backend)` precedes it anyway. Equivalent. Same for
  `Array(ldlws.stats)` (`gpu_numeric.jl:274,377`) — 6-element buffer, slots 1-5 read,
  slot 6 is the documented dmax carry (`gpu_ldlt_dense.jl:348-353`), and this line is
  **unchanged** from the pre-commit code.
- `_dev_upload` call sites — **all eleven** (`gpu_shared.jl:155-159`;
  `gpu_numeric.jl:69,70,121,122,238,241,244,293,300,302` + `signs` uploads) take plain
  `Vector`s (`Ti`, `Float`, `Int8`). No `SubArray`/`view` argument anywhere; the one
  historical `CuArray(view(ir, 1:ctot))` lives in `gpu_leftlooking_reference.jl:93`
  (CUDA-only, unchanged). `KA.allocate + copyto!` preserves eltype and length exactly as
  `CuArray(host)` did.
- `CUDA.fill! → fill!`: `CUDA.fill!` *is* `Base.fill!` (CUDA.jl extends Base's generic
  function), so this is a no-op rename; green oracles on both backends confirm the
  `fill!(view(d_arena, 1:us), 0)` work-slot zeroing works (wrong zeroing would corrupt
  every extend-add and fail the 1e-10 gates).
- `CUDA.zeros → KA.zeros`, `CUDA.synchronize() → KA.synchronize(backend)`: equivalent on
  the task-local stream; correctness of the subsequent D2H reads is guaranteed by stream
  ordering even independent of the explicit sync.

### N2. `_default_backend()` as kwarg default — no definition-order or world-age issue

Kwarg defaults evaluate at call time in the defining module's scope; both exts define
`_default_backend()` before `include("gpu_shared.jl")` anyway
(`PureSparseCUDAExt.jl:13/15`, `PureSparseAMDGPUExt.jl:16/18`). **[verified by run]** —
the AMD probe called `gpu_symbolic` without `backend` and factored on ROCm.

### N3. Zero-alloc requirement (CLAUDE.md req 5) — no regression, but the GPU drivers were never under that gate

Req 5 gates `cholesky!`/`ldlt!`/`solve!` — the CPU entry points. The GPU drivers are
module-local ext functions reached via `Base.get_extension` (core `src/` has zero GPU
hooks), so they are outside the `@allocated == 0` gate, before and after this commit.
Within the GPU refactor loop, every changed line is alloc-neutral: the old
`@allowscalar` scalar read was itself a 1-element host copy; the `_dev_upload`/
`LDLFrontWS`/`FrontWS` allocations happen only when the caller omits the preallocated
kwargs (same `isnothing` structure as before). `benchmark/gpu/gpu_zeroalloc_probe.jl`
measures `CUDA.@allocated` (device pool) and prints rather than asserts zero — unchanged
by this commit. Note in passing: the probe passes `d_emap/d_W/d_dummy/d_Anz` but not
`d_signs`/`ldlws`, so its device-pool number includes a per-call `signs` upload +
workspace — pre-existing, not this commit.

### N4. Coverage the green oracles do NOT provide (what could still be broken without any test noticing)

- **Float32** end-to-end is exercised on *no* backend (all oracles are Float64), despite
  design §1 admitting `T ∈ {Float32,Float64}` to the GPU path. The shim is generic and
  structurally fine, but "correct at Float32 tolerances" is unmeasured.
- **Cholesky device solve on AMD**: `amd_end2end_test.jl` checks the factor only;
  `gpu_solve!`'s non-unit-diagonal `batched_solve!` arm (`unitdiag=false`) never runs on
  ROCm (the LDLᵀ test covers only the unit-diag + D-scale arm).
- **Standalone all-GPU drivers** `gpu_multifrontal_cholesky!`/`gpu_multifrontal_ldlt!`
  never run on AMD (the tests use the hybrids; cutoff-0.0 covers the same kernels but
  not those two functions' own driver lines).
- **Warm multi-refactor reuse on AMD** (preallocated `d_Anz`/`d_emap`/`ws`/`ldlws`
  across repeated calls): only the CUDA zeroalloc probe exercises it.
- None of the AMD scripts are `@testitem`s; ROCm coverage exists only as manual
  `benchmark/gpu/` scripts (consistent with the CUDA precedent, but worth stating).

### N5. Checked and clean (attacks that did not land)

- Arena compact copy `copyto!(d_arena, uo, d_arena, 1, us)` cannot overlap: the stack
  starts at `max_us + 1` (`multifrontal.jl:83-92`), so `uo > us` always.
- Upward closure (asserted in `gpu_symbolic`, `gpu_shared.jl:148`) guarantees CPU fronts
  have only CPU children, so the hybrid's host-side reads never race in-flight device
  work; crossing-U H2D and panel D2H are stream-ordered.
- `frontier.jl` double-inclusion (test suite includes it directly,
  `test/gpu_frontier_tests.jl:7`; gpu_shared includes it per-ext-module): separate module
  scopes, no clash.
- Both exts loaded simultaneously: each module owns its own `GPUSymbolic`/kernels/
  `_default_backend`; no shared generic function is pirated.
- `Project.toml` compat: installed AMDGPU 2.7.0 declares KA 0.9.2 — consistent with the
  existing `KernelAbstractions = "0.9"` bound; `AMDGPU = "2"` matches what the commit was
  verified against.
- Empty/degenerate fronts (`below_s == 0` → `d_dummy`; `us == 0` skips fill; empty
  `emap`): unchanged logic, shim-neutral.

---

## Suggested follow-ups in priority order

1. D1: stub-or-guard the two vendor branches; correct the three "no CUDA here" headers.
2. D2: fix or remove `amd_probe`; add the one-line stale-Manifest warning to the AMD
   script headers / developer.md.
3. N4: add a Float32 case and a Cholesky `gpu_solve!` call to `amd_end2end_test.jl`
   (both are ~5 lines inside the existing loop).

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
