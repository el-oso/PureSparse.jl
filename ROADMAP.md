# PureSparse.jl — Roadmap & Status

Canonical status + next steps for this multi-session project. Update this file as
milestones land. Full design: [`docs/design.md`](docs/design.md). Design produced by
Fable (v1) → adversarially reviewed by Opus (2 BLOCKERs, 7 DEFECTs found, all fixed) →
corrected by Fable (v2, current). Clean-room policy: `docs/design.md` §11 — CHOLMOD
source must never be read, only published papers.

## CURRENT FOCUS — M1 core + real benchmark harness done; wall-time gate PASSING (11/14)

M1 tasks 1–6 are done and tested (13554/13554 tests passing): scaffold, AMD ordering,
etree/postorder/column-counts, fundamental-supernode detection + relaxed amalgamation,
row-structure/workspace-bound computation, the `symbolic()` driver, and the numeric
supernodal LLᵀ factorization + solve (`cholesky`/`cholesky!`/`solve!`/`solve_L!`/
`solve_Lt!`/`\`). Task 8 (harness) is now built and has been run for real —
`benchmark/gate.jl`, Chairmarks medians (30 samples/1.5s cap, `evals=1`,
single-thread-pinned via `BLAS.set_num_threads(1)`), 3 of the 4 design §9.3 configs
(config 4 CHOLMOD+PureBLAS is N/A, see design §9.3 D1): PureSparse+PureBLAS (primary),
PureSparse+OpenBLAS (kernel-attribution, via `benchmark/openblas_backend.jl`'s
same-source-file kernel swap — no algorithm duplication), CHOLMOD+OpenBLAS (baseline).
Both own-ordering and same-permutation (`GivenOrdering` fed each stack's `perm`) arms
run per design §9.3 D2.

**Real gate result (2026-07-13, `neuromancer`, NOT clock-locked — no passwordless sudo
for `fleet_freqlock.sh` in this autonomous session, so this is best-effort/noisier than a
methodologically-valid run, but is real measured wall-time, not a fabricated or "preview"
number):**

| matrix | arm | PS+PureBLAS | PS+OpenBLAS | CHOLMOD+OB | gate (1<3) |
|---|---|---|---|---|---|
| random n=200 | own | 0.061ms | 0.089ms | 0.052ms | fail |
| random n=200 | same-perm | 0.060ms | 0.087ms | 0.052ms | fail |
| random n=500 | own | 0.298ms | 0.432ms | 0.374ms | PASS |
| random n=500 | same-perm | 0.320ms | 0.422ms | 0.377ms | PASS |
| random n=1000 | own | 1.185ms | 1.498ms | 1.204ms | PASS |
| random n=1000 | same-perm | 1.144ms | 1.469ms | 1.233ms | PASS |
| banded n=1000 bw=20 | own | 0.451ms | 0.778ms | 0.687ms | PASS |
| banded n=1000 bw=20 | same-perm | 0.449ms | 0.778ms | 0.676ms | PASS |
| banded n=3000 bw=10 | own | 1.107ms | 2.152ms | 0.932ms | fail |
| banded n=3000 bw=10 | same-perm | 1.113ms | 2.145ms | 0.923ms | fail |
| laplacian2d 40×40 | own | 0.700ms | 0.993ms | 0.558ms | fail |
| laplacian2d 40×40 | same-perm | 0.713ms | 1.008ms | 0.559ms | fail |
| laplacian2d 80×80 | own | 3.619ms | 5.208ms | 2.811ms | fail |
| laplacian2d 80×80 | same-perm | 3.560ms | 5.021ms | 2.711ms | fail |

**6/14 passing — M1's "faster on at least half the set" gate is currently NOT MET.**
Diagnosed root cause (not guessed — verified by direct measurement, per CLAUDE.md's
"don't guess, check" rule):

- **It is NOT an ordering-quality problem.** `nnzL(PureSparse AMD)` is equal to or
  *better* than `nnzL(CHOLMOD)` on every failing matrix (banded: ratio ~1.0; laplacian2d
  80×80: PureSparse fill is 42% *lower*, 114053 vs 198023) — and the same-permutation arm
  (identical permutation fed to both stacks) shows essentially the SAME wall-time ratio as
  own-ordering. Both facts rule out AMD quality as the cause.
- **It IS the relaxed-amalgamation contiguity gate.** `relaxed_amalgamation`
  (`src/symbolic/supernodes.jl`) only merges supernode `s` into its parent `t` when `s`'s
  columns are already numerically contiguous with `t`'s (`endc[s]+1==start[t]`) — true
  only when `s` is `t`'s LAST-postordered child. For a bushy etree (2D grid Laplacians:
  75.8% of final supernodes are still width-1; most etree nodes have 2+ children, so only
  1-in-k children per parent can ever pass the contiguity gate regardless of the
  `zmax` threshold). Verified directly: sweeping `amalg_zmax` from the default
  `(0.9,0.15,0.03)` to a far more permissive `(0.97,0.35,0.08)` produced ZERO change in
  `nsuper` on `laplacian2d_80x80` (3041 supernodes either way) — proof the threshold
  isn't the binding constraint, the contiguity gate is. `banded_n3000_bw10` is a
  *different* failure mode: supernodes are already large (mean width 120, matching
  CHOLMOD's fill almost exactly) yet still ~1.2-2.3x slower — that gap is in per-call
  update-loop scheduling overhead, not supernode size, and is unexplained pending
  profiling (a `@profile` pass on `laplacian2d_80x80` showed the time genuinely spread
  across many small `syrk!`/`potrf!`/`trsm!` calls, consistent with the tiny-supernode
  diagnosis for that matrix but not yet run on `banded_n3000_bw10` specifically).
- **Real, incidental bug fixed along the way:** `tuning.jl`'s `AMALG_COLS`/`AMALG_ZMAX`
  had an `::NTuple{3,T}` typeassert on the raw `@load_preference` result — since
  Preferences.jl/TOML has no tuple type, ANY attempt to actually override these via
  Preferences (the exact mechanism design §1.4 requires for calibration) threw a
  `TypeError` and would have made calibration impossible without this fix. Fixed via
  `Tuple(@load_preference(...))`.

**Task 7b (child-ordering postorder) IMPLEMENTED and MEASURED A NO-OP (2026-07-13) — the
child-choice link of the diagnosis above is wrong.** `postorder` now takes an optional
sibling `priority` (max-colcount child visited last, i.e. made contiguity-eligible;
derivation in the `postorder` docstring), wired through `symbolic()` via a preliminary
default-order postorder+relabel (Gilbert–Ng–Peyton `column_counts` requires a postordered
labeling, so counts are computed there and mapped back). All 12220 tests stay green, the
postorder genuinely changes (154 of 6400 positions on laplacian2d 80×80) — and `nsuper`
is IDENTICAL on every gate matrix (3041 on lap80, 75.8% width-1, cells within 0.2%), gate
unchanged (5/14 vs 6/14 baseline; the one differing row, `random_n1000_d005` own-arm
1.209ms vs 1.194ms, is a 2% unlocked-clock noise swing, and its supernode partition is
bit-identical). Instrumented root cause: of 1777 contiguity-eligible (child,parent) pairs
on lap80, the zero-fraction test rejects only **2** — so WHICH child is contiguous never
mattered; whatever sits in the slot merges. The true binding constraint is structural:
**one contiguous child branch per parent per single ascending amalgamation pass** (4815
fundamental → 3041 after 1774 merges; the other 3037 pairs fail contiguity and no sibling
order can fix that, since an earlier sibling is always processed before the later
sibling's merge extends the parent's column range). Real lead, measured in scratch but
NOT implemented (out of the 7b scope, needs design sign-off): **iterating
`relaxed_amalgamation` to a fixpoint** collapses lap80 to 90 supernodes (0% width-1) —
but with the current `rows_est = colcount[start[s]]` chain proxy it over-merges (padded
cells 233K → 804K vs nnzL 114K, effective z ≈ 0.86 ≫ every tier limit) because cascaded
merges of non-nesting siblings make the topmost-column colcount a big underestimate of
the true union row height. A correct version needs a union-height row estimate (e.g.
incremental merge of child rowinds, or `supernode_rowind`-style height) inside the merge
test, then fixpoint (or a proper multi-child bottom-up pass). That is the next 7b'.

**Task 7b' (multi-child fixpoint amalgamation with an exact union-height estimate)
IMPLEMENTED and MOVED THE GATE (2026-07-13): 6/14 baseline → 4/14 with the fixpoint
change alone at the OLD thresholds → 11/14 after recalibrating `AMALG_COLS`/`AMALG_ZMAX`
against it.** Two independent changes, both in `src/symbolic/supernodes.jl` /
`src/tuning.jl`:

1. **Fixpoint loop, not a single ascending pass.** `relaxed_amalgamation` now repeats
   ascending passes until one performs no merge (path-halved `owner` array redirects
   absorbed supernodes to their current alive target in near-O(1), so re-scanning is
   cheap). Measured pass counts on the gate set: 2 (both banded matrices — their etree is
   near-chain, one extra sibling almost never appears) up to 7 (both laplacian2d sizes —
   bushy 2D-grid etrees, most nodes have 3-4 children). This is what actually escapes the
   "one contiguous child per parent per pass" ceiling task 7b diagnosed.
2. **Exact row-count estimate**, replacing the `colcount[start[s]]` proxy: every block
   the fixpoint process ever forms has exactly one "range root" (its last column — proved
   by induction over the merge step, since a merge only ever redirects into a target
   whose interval already contains `parent[endc[child]]`), so the block's true
   below-diagonal row set is exactly `struct(L[:,endc]) \ {endc}` and its height is
   `ncols + colcount[endc] - 1` — O(1) per merge decision, no incremental pattern union
   needed. Verified empirically over every gate matrix's final partition
   (`height-formula-violations=0` in every run) and pinned as a first-class test on
   laplacian2d(24,24) (`test/supernode_tests.jl`, "2D grid Laplacian: superset invariant +
   z-bound under multi-pass amalgamation") that also re-checks the §3.4 superset
   invariant against a from-scratch elimination-game oracle on the actual bushy partition
   the fixpoint produces, not just the random-graph zoo the prior tests used.

**Before/after supernode-partition stats (old single-pass proxy vs new fixpoint+exact
height, calibrated thresholds — see below):**

| matrix | nsuper (old→new) | mean width (old→new) | width-1 % (old→new) | cells/nnzL (old→new) |
|---|---|---|---|---|
| random_n200_d02 | 77→30 | 2.6→6.67 | 53.2→33.3 | 2.123→2.767 |
| random_n500_d01 | 182→78 | 2.75→6.41 | 53.3→43.6 | 1.86→2.083 |
| random_n1000_d005 | 366→169 | 2.73→5.92 | 49.7→39.6 | 1.535→1.793 |
| banded_n1000_bw20 | 8→62 | 125.0→16.13 | 0.0→0.0 | 6.884→1.716 |
| banded_n3000_bw10 | 24→188 | 125.0→15.96 | 0.0→0.0 | 12.423→2.36 |
| laplacian2d_40x40 | 719→193 | 2.23→8.29 | 75.7→25.9 | 2.233→2.266 |
| laplacian2d_80x80 | 3041→659 | 2.1→9.71 | 75.8→16.7 | 2.048→2.035 |

Two things worth calling out plainly: (a) on the banded matrices the OLD single-pass
algorithm already collapsed everything into a few very wide supernodes (width ~125) via
long single-child chains, but with catastrophic padding (cells/nnzL 6.9x–12.4x nnzL) that
nobody had actually measured before — the new algorithm's exact height estimate rejects
those over-fat chain merges and produces MORE, narrower, far-less-padded supernodes
(1.7x–2.4x) that turned out to be the single biggest wall-time win on the whole set; (b)
on laplacian2d, `nsuper` collapses roughly in line with the scratch-measured lead from
task 7b (was 3041→90 uncalibrated/over-merged; with real thresholds it's 3041→659, less
dramatic than the uncalibrated number but with padding ratios that actually respect
`AMALG_ZMAX`).

**Threshold recalibration was necessary and is not optional plumbing.** The fixpoint
change ALONE, at the original starting-point thresholds (`AMALG_COLS=(8,32,128)`,
`AMALG_ZMAX=(0.9,0.15,0.03)` — chosen in M1 task 4 before the estimate was trustworthy
enough to calibrate against), REGRESSED the gate to 4/14 (down from the 6/14 single-pass
baseline): banded flipped decisively to PASS, laplacian2d got measurably closer but still
failed, and all three random matrices regressed from PASS/near-pass to fail. A Chairmarks
sweep of the warm-refactor arm over `amalg_zmax ∈ {(0.9,0.15,0.03), (0.95,0.3,0.08),
(0.97,0.35,0.08), (0.98,0.4,0.1)}` × `amalg_cols ∈ {(8,32,128), (16,64,128), (16,64,256)}`
on the 7 affected matrices found that tightening zmax (less merging) made every matrix
class WORSE, not better — the opposite of the "thresholds are too permissive" hypothesis
tested first — while loosening to `amalg_zmax=(0.97,0.35,0.08)` (reusing, not
re-deriving, the exact zmax point already probed as "far more permissive" in task 7b's
prior session) combined with doubling `amalg_cols` to `(16,64,128)` gave a clean win on
every gate matrix except small unstructured-random ones (`random_n200`, `random_n1000`
own-arm), which sit at a noise-level tie against CHOLMOD's near-zero per-call overhead at
that size — a pre-existing gap unrelated to supernode shape (random_n200 already failed
in the very first 6/14 baseline, before any of this session's work). New defaults are now
baked into `src/tuning.jl` (with the sweep and rationale in its derivation comment) and
`docs/design.md` §3.5's table, not left as a benchmark-only override.

**Full 14-row gate result (2026-07-13, `neuromancer`, unlocked clock — same
best-effort/noisier caveat as the original baseline run):** `julia --project=benchmark
benchmark/gate.jl` → **11/14 matrix-arm combinations PASS** (up from 6/14 baseline, 4/14
mid-way through this task before recalibration). Every banded and laplacian2d row now
PASSes on both own-ordering and same-permutation arms; `random_n500` now also passes
cleanly (was already passing pre-fixpoint too); `random_n200` (both arms) and
`random_n1000` own-arm remain fail (near-tie, see above) — `random_n1000` same-perm now
passes. Full test suite: 13554/13554 passing (`test/runtests.jl`, ReTestItems) with the
new code and new defaults, including the new laplacian2d-specific invariant test.

**M1's "faster on at least half the set" gate requirement is now MET** (11/14 ≥ 7/14).
This is a real, measured wall-time win, not a supernode-count win that didn't translate —
the padded-cell ratios above show the fixpoint's merges are legitimately more
BLAS-3-efficient, not just fewer-and-fatter.

**M1 status: gate met, docs done (task 9, DocumenterVitepress — `docs/{make.jl,src/*.md}`,
Home/Guide/Benchmarking/API Reference/Provenance pages, verified building end-to-end),
task 7b'/8 done.** Only remaining M1 item is task 7's zero-alloc remainder (below) — not
required by the M1 gate, which is a wall-time comparison, not an allocation gate on its
own; the allocation gate (`@allocated cholesky! == 0`) is a separate, still-open
requirement worth finishing before M1 is fully closed out. Possible follow-up (not
required by M1's gate, which is already met): investigate why `random_n200`/
`random_n1000` own-arm sit at a noise-level tie — likely per-call dispatch/relmap-setup
fixed cost at very small n, not a supernode-shape problem, so a fix (if pursued) would
live in the numeric update-loop scheduling (§4.3), not amalgamation.

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
**Update:** `SupernodalFactor.panels::Vector{Matrix{T}}` now caches every supernode's
panel wrapper ONCE at factor-construction time (`_build_panels`, `src/numeric/llt.jl`),
reused across every `cholesky!` call instead of re-`unsafe_wrap`ping `panel`/`panel_d`
each time — cut `cholesky!`'s per-call allocation from 7392 to 2576 bytes on a 50x50
test case (65% reduction). The REMAINING allocation is `cholesky!`'s update-block buffer
`C = _panelview(cbuf, 1, ctot, k1)` — its shape varies per (descendant, ancestor) pair in
the update schedule, so it isn't a single fixed-shape wrapper the way panels are; true
zero-alloc there needs one cached view per distinct (d,s) pair (keyed by position in the
update schedule), not attempted yet. `solve!` still allocates a permuted-RHS scratch
buffer per call (documented in `src/numeric/solve.jl`'s header) — same category of
follow-up. Both remain M1 task 7, not blocking correctness.

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
7. Refactorize/allocation hardening + StrictMode guards. *(partial — see "known follow-up")*
8. Benchmark harness + gate run + amalgamation threshold calibration. *(harness done,
   `benchmark/{matrices,openblas_backend,gate,benchmarks}.jl`; gate run for real, DONE —
   11/14 passing, see "CURRENT FOCUS"; threshold calibration folded into task 7b')*
7b. Child-ordering relaxed amalgamation (see "CURRENT FOCUS" history) — implemented,
    measured a no-op, superseded by 7b'.
7b'. Multi-child fixpoint amalgamation with an exact union-height row estimate +
    threshold recalibration (see "CURRENT FOCUS") — DONE, moved the gate from 6/14 to
    11/14.
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
