# M8 multi-backend GPU design — adversarial review (Opus)

> Review of `design_gpu_multibackend.md` v1. Independent of the parallel Fable review (verified
> against `ext/gpu_dense.jl`, `gpu_numeric.jl` (40 CUDA-API sites), `gpu_solve.jl`, `multifrontal.jl`,
> `PureSparseCUDAExt.jl`). Permanent audit trail; findings fold into v2.

## BLOCKER — §B3/§B9: clause-2 loss is structural (FP64 matrix cores), not a tuning gap
The kernels **run correctly** on CDNA/Xe (only `@localmem`/`@synchronize`/`@private`/`muladd`/
`@nexprs` — no warp shuffle/ballot; workgroup barriers correct at any wavefront width). But that's
not the bet — clause 2 (pure ≥ vendor) is, and §B3's diagnosis is wrong in KIND:
- **Why M6 won on Ada:** Ada has no FP64 matrix path cuBLAS DGEMM can use → cuBLAS DGEMM on Ada is
  itself vector-FMA-bound. A vector-`muladd` kernel ties the vendor's own path; "99% of FP64 peak" is
  99% of the *vector* peak both share. A fair fight between two vector kernels.
- **Why it doesn't transfer:** CDNA2 (gfx90a/gfx942) has FP64 **MFMA** matrix cores; rocBLAS DGEMM
  dispatches to them (vector FP64 ≈22 TF vs matrix ≈45 TF on MI210-class). Intel Xe-HPC (PVC/Max) has
  FP64-capable systolic XMX/DPAS that oneMKL targets. A pure KA scalar-`muladd` kernel **cannot emit
  MFMA/XMX** (GPUCompiler exposes no intrinsic; using them = restructuring around 16×16×4 matrix ops =
  a rewrite, not a tile Preference). So pure is structurally capped at ~½ the vendor FP64 ceiling
  before occupancy is even discussed.
- **Exactly the M7 pattern:** M7 shelved because "the vendor runs a hardware path the pure kernel
  can't reach." §B3 frames the same failure as "sweep tiles / maybe a wavefront-64 kernel." A tile
  sweep cannot close an instruction-class gap. Honest cost for clause 2 ≠ "few days + tuning"; it's
  "re-derive a matrix-core kernel per backend," which the pure-KA constraint forbids → **clause 2 is
  very likely UNWINNABLE on AMD and Intel by construction.**
**Fix:** rewrite §B3 — clause 2 is gated on the *vendor lacking* an FP64 matrix path; pure's ceiling
is the vector-FMA rate, not the vendor rate. Make the sweep a bounded ≤1-session probe, not a campaign.

## BLOCKER — §B1: "small, enumerable CUDA residue" is false; the shipped driver is CUDA-coupled at 40 sites
§B1's table lists 3 substitutions + calls device-alloc "the only non-mechanical part." Actual
`gpu_numeric.jl`: 13× `CuArray(...)`, 9× `CUDA.zeros`, 8× `CUDA.fill!`, 6× `CUDA.synchronize`, 2×
`CUDA.@allowscalar`, + 5× potrf!/6× trsm! (cuSOLVER/cuBLAS **inside the shipped hybrid path**, not
only the vendor arm). `CuArray(host)` has NO single KA-generic equivalent (`KA.zeros(backend,…)`
allocates but doesn't upload → `allocate + copyto!`/`adapt` at all 13 sites). The "few-days pure
refactor" is 40 edits in one file + a semantic-equivalence audit, and the bit-identical no-regression
bar fails on the first missed `synchronize`/scalar-read. **Fix:** re-scope §B1 as "rewrite the
multifrontal driver against a small backend shim (`dev_zeros`, `dev_upload`, `dev_sync`, `allowscalar`),
~40 sites in gpu_numeric.jl"; budget honestly.

## DEFECT — §B5 clause 1: "bit-identical, relerr < 1e-10" over-promises + contradicts M6
M6 §10.1 is **normwise** (`‖L_gpu−L_cpu‖ ≤ c·n·eps·√‖A‖`), tolerance-based *because device reduction
order differs*, and M6 §M.5 lists "Non-goal: bitwise CPU match." A 64-wide wavefront changes the
4×4/64-tile accumulation grouping vs Ada's 32-wide; extend-add/scatter order differs → the AMD/Intel
factor is provably NOT bit-identical to CPU or to the CUDA factor (IEEE `fma` matches per-op, not
across reordered reductions). A *flat* 1e-10 can also spuriously fail large/ill-conditioned KKT
members the `c·n·eps·√‖A‖` bound passes. **Fix:** strike "bit-identical"; use M6's normwise bound;
calibrate tolerance on a dense potrf per backend first.

## DEFECT — §B5 clause 3: robust only under an UNSTATED CPU thread count
Given the BLOCKER, M8's realistic outcome is the §B9 fallback "pure, portable, beats CPU." Throughput
math (MI210): pure at a first-port 40–60% of vector peak ≈ 9–13 TF; multithreaded CHOLMOD+OpenBLAS on
a fat node ≈ 1–2 TF → GPU wins clause 3 on **big fronts** even untuned. The risk is **Amdahl on
small/medium fronts** (assembly, extend-add, launch latency, per-refactor H2D) — and `frontier_cutoff`
/ `FUSE_M_MAX=Ref(6000)` deciding which fronts go to GPU are **Ada-measured** → on CDNA they're
meaningless and could keep the wrong fronts on-GPU, sinking clause 3. Load-bearing ambiguity: **M6
§8.1 gates clause 3 vs *single-thread* CPU**; multithreaded CHOLMOD is a §8.2 context arm (reported,
not gated). M8 §B5 says "beats CHOLMOD+OpenBLAS" with **no thread count**. Single-thread = beats one
core = weak claim for a $2–5/hr datacenter GPU in 2026; multithreaded = uncertain + unmeasured.
**Fix:** pin clause 3 to **multithreaded** CHOLMOD (state cores) — the only defensible 2026 bar — and
re-derive `frontier_cutoff`/`FUSE_M_MAX` per backend as an explicit build item.

## DEFECT — §B6/§B5: one-shot rental cost model is wrong for an iterative tuning campaign
§B3 calls clause 2 "a per-backend tuning campaign"; §B6 budgets "$10–40 per gate session" one-shot.
Each variant recompiles under Julia's compile tax; stateless rsync→run→teardown re-pays full precompile
(no warm daemon like galen); MI210 slots intermittent, MI300X queued/pricier. **Fix:** separate a
correctness+clause-3 gate (one short session — the real deliverable) from clause-2 exploration
(bounded, explicitly optional, O(sessions)); keep a persistent box for the loop.

## DEFECT — §B0.3/§B2: "engine included per ext" sound only under an unstated constraint
Including `ext/gpu_engine/*.jl` into both exts is fine for module-local funcs + methods on
device-array-parameterized ext types (`GPUSymbolic{Ti,VI}` differs by VI). It **breaks** the instant
the engine adds a method to a PureSparse generic on a shared, non-device-typed signature (e.g.
`PureSparse.foo(::SparseMatrixCSC)`) — CUDA-ext + AMD-ext define identical signatures, the second load
redefines/invalidates the first. Also unmentioned: N backends → engine precompiles N× (N× method
tables, N× load). **Fix:** rule — every engine method on a `PureSparse.*` generic MUST dispatch on a
device-array-typed arg; none may key on a host/`SparseMatrixCSC`-only signature; note the N× precompile.

## NITs
- **`unsafe_wrap`/`pointer` worry is a non-issue** (all on HOST arrays, backend-neutral). The real
  device surface is `reshape(view(dx,rng))` (`_dpanel`/`_dslab`) — contiguous-view reshape returning a
  kernel-indexable device array is GPUArrays-provided but untested on ROCArray/oneArray → day-0 smoke.
- **oneAPI ≫ AMD risk** — gate Intel behind an explicit "does FP64 KA compile+run on Max at all" day-0
  go/no-go; permit dropping Intel from M8 without it counting as a miss.
- **`FUSE_M_MAX`/`:auto` mode thresholds are Ada-measured + perf-load-bearing** — on CDNA the
  register-resident v3 is the M7 register-residency failure shape; "is v3 even the right variant on
  CDNA" is an open question, not a sweep parameter.

## Verdict
**Not a sound bet as framed.** The kernels port for *correctness* (no warp-width primitives), but the
clause-2 "pure ≥ vendor" milestone that makes it "mirror M6" is very likely **structurally unwinnable**
on both new backends for the identical reason M7 was shelved: rocBLAS/oneMKL DGEMM ride FP64 matrix
cores (MFMA/XMX) a pure scalar-`muladd` KA kernel cannot emit — a tile sweep can't close an
instruction-class gap, and M6 only won because Ada's cuBLAS shares the vector-FMA path. The honest
ceiling is the §B9 fallback ("portable + beats CPU"), real but far weaker than the framing and hinging
entirely on the **unstated CPU thread count** in clause 3 (single-thread = trivial-but-weak;
multithreaded = uncertain). **Single biggest risk:** spending scarce rental budget "tuning" clause 2
toward a bar hardware forecloses — decide before renting that clause 2 is an expected loss on AMD/Intel,
ship clause-3-vs-multithreaded-CHOLMOD as the deliverable, and rewrite §B1's "small residue / few-days
port" to match the 40-site CUDA-coupled driver that actually exists.
