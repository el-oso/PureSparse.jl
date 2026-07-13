# PureSparse.jl — Roadmap & Status

Canonical status + next steps for this multi-session project. Update this file as
milestones land. Full design: [`docs/design.md`](docs/design.md). Design produced by
Fable (v1) → adversarially reviewed by Opus (2 BLOCKERs, 7 DEFECTs found, all fixed) →
corrected by Fable (v2, current). Clean-room policy: `docs/design.md` §11 — CHOLMOD
source must never be read, only published papers.

## CURRENT FOCUS — M1 core complete, M1 hardening/benchmark/docs remain

M1 tasks 1–6 are done and tested (12220/12220 tests passing): scaffold, AMD ordering,
etree/postorder/column-counts, fundamental-supernode detection + relaxed amalgamation,
row-structure/workspace-bound computation, the `symbolic()` driver, and the numeric
supernodal LLᵀ factorization + solve (`cholesky`/`cholesky!`/`solve!`/`solve_L!`/
`solve_Lt!`/`\`).

**Unofficial preview benchmark** (informal `@elapsed`, no CPU pinning, no median-of-many —
NOT the rigorous §9.3 gate methodology, just a sanity check): random SPD matrices,
`cholesky!` refactor time vs `LinearAlgebra.cholesky!` (CHOLMOD) refactor time —

| n | nnz(A) | nnzL (PureSparse AMD) | nnzL (CHOLMOD) | PureSparse | CHOLMOD | CHOLMOD/PureSparse |
|---|---|---|---|---|---|---|
| 200 | 379 | 490 | 488 | 0.078ms | 2.369ms | 30.2x |
| 500 | 1730 | 13923 | 24822 | 0.41ms | 0.441ms | 1.08x |
| 1000 | 6042 | 146289 | 192797 | 3.95ms | 3.77ms | 0.95x |

Encouraging given zero performance tuning has happened yet: AMD is already finding
noticeably less fill than CHOLMOD's own ordering (18-40% less at n=500/1000), and
PureSparse is within 5% of CHOLMOD's wall-time at n=1000 despite that fill advantage not
yet translating into a wall-time win — meaning there's real headroom once tasks 7-8 land
(zero-alloc + the calibrated amalgamation thresholds). Re-run properly (locked clock,
Chairmarks, median) once task 8's harness exists — do not treat this table as the gate.

Remaining M1 tasks: (7) allocation hardening for `cholesky!` (currently
correct but not zero-alloc — see "known follow-up" below) and `solve!` (deliberately
deferred, allocates per call); (8) Chairmarks/PkgBenchmark harness + amalgamation
threshold calibration; (9) DocumenterVitepress docs pages.

**Dependency note:** PureBLAS.jl's `Project.toml` had its `TypeContracts` compat bumped
from `"0.13.1"` to `"0.13.1, 0.14"` and its TypeContracts dependency switched to
`Pkg.develop`-track the local `TypeContracts` repo (was a frozen 0.13.1 snapshot), so
both PureBLAS and PureSparse can share the current local TypeContracts (0.14.0). PureBLAS's
own test suite re-verified green after the bump (see PureBLAS.jl git history for the
commit, if the user wants to review/commit that change). That bump also surfaced a real
regression (TypeContracts 0.14's `_seal_verified!` needs `TypeContracts` imported into the
calling module for `@verify_strict`), fixed with a one-line import in PureBLAS's
strictmode_tests.jl (also committed there).

**Lesson learned — PureBLAS kernel calls on `reshape(view(...))` types (compile tax):**
calling PureBLAS's kernels (`potrf!`/`trsm!`/`syrk!`/`gemm!`) on a
`Base.ReshapedArray{T,2,SubArray{...}}` (the natural type from `reshape(view(x, range),
nrow, ncol)` for a supernode panel) triggers a catastrophic first-call LLVM compile —
measured directly, `potrf!` alone took **93 seconds** on that type vs **1.3 seconds** on
an `unsafe_wrap(Array, pointer(x, off), (nrow,ncol))`-constructed plain `Matrix{T}`
sharing the same memory (a ~70x difference). `src/numeric/llt.jl`'s `_panelview` helper
uses `unsafe_wrap` (safe here: the buffer is always kept alive by the caller's
`GC.@preserve` for the duration of the call) specifically to avoid this. **Any new code
calling PureBLAS kernels on a supernode panel must use `_panelview`, never
`reshape(view(...))`, or it will silently reintroduce a many-second-per-call compile
tax.** This cost significant debugging time before being correctly diagnosed (it looked
exactly like an infinite loop from the outside — steady CPU, plateaued memory — until a
backtrace during a kill caught it mid-`jl_compile_codeinst_now`/LLVM `SelectionDAGISel`).

**Lesson learned — test-helper bug, not a real one:** the first `L*L' ≈ P·A·Pᵀ`
reconstruction test failures (4/28 random cases, all n≥30 with heavy supernode
amalgamation) traced back to `test/llt_tests.jl`'s `dense_L` helper reading the
strictly-upper triangle of a supernode's own diagonal block — `potrf!` (like LAPACK)
never writes there, leaving stale/undefined data, and `cholesky!`/`solve!` never read it
either (`trsm!` with `uplo='L'` only references the lower triangle) — but the TEST helper
naively copied the whole panel, corrupting its own reconstruction. Verified the real
factor was correct throughout via a full dense LAPACK oracle on the captured pre-`potrf!`
block. Fixed by skipping the diagonal block's strict-upper positions in `dense_L`.

**Known follow-up (M1 task 7):** `cholesky!` is correct and mostly allocation-light
(`_panelview`'s `unsafe_wrap` allocates a small `Array` *header*, not the underlying
data, per panel view — not yet the zero-alloc target) but not literally zero-alloc yet;
`solve!` allocates a permuted-RHS scratch buffer per call (documented in
`src/numeric/solve.jl`'s header). True zero-alloc needs panel-view objects pre-cached on
the `Workspace`/factor and reused across calls instead of re-`unsafe_wrap`ping every
time — scoped as M1 task 7, deliberately not done as part of getting the numerics
correct first.

## Milestones (design §10)

### M1 — AMD + Symbolic + Supernodal LLᵀ + Solve
**Deliverables:** `tuning.jl`, `types.jl`, `contracts.jl`, `ordering/interface.jl`,
`ordering/amd.jl`, `symbolic/etree.jl`, `symbolic/counts.jl`, `symbolic/supernodes.jl`,
`numeric/llt.jl`, `numeric/solve.jl`; full test files for these; Chairmarks + PkgBenchmark
harness (design §9.3, 4-arm with quadrant 4 marked N/A); docs skeleton
(DocumenterVitepress).

**Gate:** full zoo correctness (dense `BigFloat` oracle + CHOLMOD black-box cross-check);
zero-allocation gate (`@allocated cholesky!(F, A2) == 0`, StrictMode-checks-disabled
config); wall-time gate — `median_seconds(PureSparse+PureBLAS) < median_seconds(CHOLMOD+
OpenBLAS)` on the M1 KKT/FEM set, both own-ordering and same-permutation arms, strictly
faster on at least half the set; `juliac --trim` smoke build succeeds; AMD fill ≤ 1.15×
CHOLMOD-AMD fill on the zoo.

**Task list:**
1. Scaffold `Project.toml`/module/`tuning.jl`/`types.jl`/`contracts.jl`. *(in progress)*
2. Elimination tree + postorder + column counts, brute-force-oracle tests.
3. AMD (longest single task — budget accordingly). Paper §-by-§: quotient graph storage →
   pivot loop → approximate degree scan → supervariable detection/mass elimination →
   aggressive absorption → dense rows → garbage compaction.
4. Fundamental supernode detection + relaxed amalgamation.
5. Symbolic driver (rowind/px/assembly-map/workspace-bound sizing).
6. Supernodal LLᵀ numeric (load → linked-list update loop → potrf/trsm) + solve.
7. Refactorize/allocation hardening + StrictMode guards.
8. Benchmark harness + gate run + amalgamation threshold calibration.
9. Docs pages (Home/Tutorial/Benchmarking via DocumenterVitepress).

### M2 — LDLᵀ/SQD + Update/Downdate
**Deliverables:** `numeric/ldlt.jl` (incl. block LDLᵀ base case, signed regularization,
inertia stats), `simplicial/updown.jl` (simplicial storage + Davis–Hager update/downdate),
split solves for all three factor types, IPM guide docs, SQD benchmark additions.

**Gate:** SQD zoo (synthetic IPM iterate sequences) factor without failure; inertia
matches construction; update/downdate round-trip ≤ 100·eps·n; zero-alloc `ldlt!`.

**Tasks:**
1. `ldlt_block!` base case + dense unit tests vs `bunchkaufman`.
2. LDL descendant updates (syr2k-with-D path) + panel solve.
3. Signed regularization + inertia stats + `signs` plumbing.
4. Simplicial storage + conversion (`simplicial(F)`).
5. Rank-1 update/downdate (Davis–Hager Method C) incl. pattern growth.
6. Rank-k (successive single-rank first, then multiple-rank optimization).
7. Refinement helpers + simplicial split solves.
8. IPM guide docs.

### M3 — GPU (CUDA weakdep extension, in-package)
**Deliverables:** `ext/PureSparseCUDAExt/*`; level-set scheduler (host-side); device
factor/solve; GPU testitems (skipped when no device); GPU benchmark config (reported, not
gated against CPU).

**Gate:** bitwise-tolerance agreement with CPU factors on the performance set;
upload-once verified (second `cholesky!` on device performs zero host→device pattern
transfers); batched-small-supernode kernel beats naive per-supernode launches by ≥3×.

**Tasks:**
1. Level-set construction + pattern-array upload plan.
2. KA device kernels (gemm/syrk/trsm/scatter).
3. Batched small-supernode kernel.
4. Device driver + LDL variant.
5. Device solves.
6. Tests/benchmarks.

### M4 — Drop-in
**Deliverables:** `dropin.jl` + `activate!`/`deactivate!` (Preferences-gated); stdlib
surface parity (`logdet`, `det`, `diag`, `issuccess`, `check=`, `shift=`, `perm=`,
`Symmetric` wrappers, Int32 indices, `SparseMatrixCSC` extraction of `F.L`/`F.U`/`F.p`);
`dropin_tests.jl` running captured stdlib cholesky test expectations against our factors.

**Gate:** with dropin active, a downstream SparseArrays-dependent smoke test suite passes
unmodified; M1 perf gate still holds through the dropin entry point.

## Standing rules

- **Clean-room, absolute:** never read CHOLMOD/SuiteSparse source, in any form. Only
  published papers (`refs/linear_algebra/`, gitignored) and independent reasoning. See
  `docs/design.md` §11.
- **Dense kernels exclusively via PureBLAS.jl** (`potrf!`/`trsm!`/`syrk!`/`syr2k!`/
  `gemm!`) — never reimplement, never call OpenBLAS/LAPACK directly in `src/`.
- **Performance gate is wall-time**, not GFlops (GFlops is gameable by ordering quality —
  design §9.3 D2). Primary comparison: PureSparse+PureBLAS vs CHOLMOD+OpenBLAS, both
  own-ordering and same-permutation.
- Generic over `T<:Number`/`Ti<:Integer` on hot paths (AD-traceable, PureBLAS
  convention); Float64 is the tuned path, others correct-but-generic.
- Trim-compatible: no runtime eval/invokelatest, no `Vector{Any}` on hot paths, no
  runtime CPU detection — tuning constants are compile-time Preferences-backed consts.
- Commit author email: `15278831+el-oso@users.noreply.github.com`.
- The approved plan (`docs/design.md`) is a contract: do not skip/substitute a
  requirement without asking first.
