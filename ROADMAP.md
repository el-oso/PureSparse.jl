# PureSparse.jl — Roadmap & Status

Canonical status + next steps for this multi-session project. Update this file as
milestones land. Full design: [`docs/design.md`](docs/design.md). Design produced by
Fable (v1) → adversarially reviewed by Opus (2 BLOCKERs, 7 DEFECTs found, all fixed) →
corrected by Fable (v2, current). Clean-room policy: `docs/design.md` §11 — CHOLMOD
source must never be read, only published papers.

## CURRENT FOCUS — M1 scaffolding

Repo just created. In progress: M1 task list item (1) scaffold — `Project.toml`,
`tuning.jl`, `types.jl`, `contracts.jl`, `ordering/interface.jl` skeleton. Next: item (2)
elimination tree + postorder + column counts (`symbolic/etree.jl`, `symbolic/counts.jl`)
with brute-force-oracle tests.

**Dependency note:** PureBLAS.jl's `Project.toml` had its `TypeContracts` compat bumped
from `"0.13.1"` to `"0.13.1, 0.14"` and its TypeContracts dependency switched to
`Pkg.develop`-track the local `TypeContracts` repo (was a frozen 0.13.1 snapshot), so
both PureBLAS and PureSparse can share the current local TypeContracts (0.14.0). PureBLAS's
own test suite re-verified green after the bump (see PureBLAS.jl git history for the
commit, if the user wants to review/commit that change).

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
