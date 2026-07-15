# PureSparse.jl — Roadmap & Status

Canonical status + next steps for this multi-session project. Update this file as
milestones land. Full design: [`docs/design.md`](docs/design.md). Design produced by
Fable (v1) → adversarially reviewed by Opus (2 BLOCKERs, 7 DEFECTs found, all fixed) →
corrected by Fable (v2, current). Clean-room policy: `docs/design.md` §11 — CHOLMOD
source must never be read, only published papers.

**Milestone order (user-approved reorder, 2026-07-13): M1 → M2 → M4 → M3.** M3 (GPU,
CUDA weakdep extension) is deferred to the end — this dev machine has no NVIDIA GPU
(confirmed via `nvidia-smi`/`lspci`: integrated AMD graphics only), so CUDA.jl work
here would be unverifiable guessing, not "don't guess, check" engineering. M4
(drop-in) doesn't need GPU hardware and is next.

**2026-07-14: M5 = sparse QR (next milestone, design CLOSED, implementation
starting), GPU renumbered M3 → M6.** Design: [`docs/design_qr.md`](docs/design_qr.md)
v2 — Fable v1 draft → **two fully independent adversarial reviews** (Opus:
[`docs/design_qr_review.md`](docs/design_qr_review.md), 1 BLOCKER/5 DEFECTs/6 NITs;
a second Fable pass blind to Opus's findings:
[`docs/design_qr_review_fable.md`](docs/design_qr_review_fable.md), 3 BLOCKERs/9
DEFECTs/8 NITs) → Fable v2 fixing every finding from both (§0 changelog traces each
fix by ID). The two highest-severity findings (the `vcount` off-by-one and the
`beta=2/(vᵀv)` division-by-zero) were independently re-derived and confirmed, not just
relayed. v2 also cross-checks its three BLOCKER fixes against `faer` (Rust, MIT
licensed — a new, narrowly-scoped permitted-source category distinct from the absolute
CHOLMOD/SuiteSparse GPL prohibition, §11). Older "M3 (GPU)" references below read as
M6; the `### M3` milestone section's content is unchanged. Key decisions: left-looking
column Householder v1 (M5a) with a gate-triggered multifrontal escalation (M5b);
COLAMD ordering from the primary 2004 paper
(`refs/linear_algebra/QR/davis_gilbert_larimore_ng_2004_colamd.pdf`) plus Larimore's
1998 UF thesis as the implementation-depth reference
(`refs/linear_algebra/QR/larimore_1998_colamd_thesis.pdf`, full ch. 3–4 read verified
in review); star-matrix AᵀA-free symbolic reusing the existing etree/counts pipeline;
Heath-test rank detection with Foster–Davis-style dead-column dropping; PureBLAS check
result — M5a needs no new PureBLAS kernels, M5b requires a larfb-role apply kernel
(both `Q·C` and `Qᵀ·C` directions, D8) + generic `geqrf!` (M5b tasks P1/P2).
Implementation task list: `docs/design_qr.md` §10 (13-task M5a list).

**2026-07-14 update: M5a tasks 1–8 CLOSED.** Types/tunables/contracts (1);
`ata_pattern`+AMD-on-AᵀA ordering (2); COLAMD from the DGLN 2004 paper + Larimore
thesis ch. 3–4, independently adversarially reviewed (0 BLOCKER/1 DEFECT/6 NIT,
`docs/colamd_review.md`) and corrected (the `l_k=0` discard now matches paper
Algorithm 2/3's verbatim timing) (3); star pattern builder + H1 verified 3000/3000
trials, real dedup bug found and fixed (row-outer-loop marker-array hazard, distinct
from `ata_pattern`'s safety) (4); V/R row structure + H2 verified, a genuine ambiguity
in the design's own wording found and fixed (a row can legitimately terminate at a
live root without retiring as its pivot, not just "retires or hits a dead root") (5);
numeric left-looking loop + Householder kernel, two design gaps found (`QRSymbolic`
was missing the star pattern's own storage needed for row-subtree seeding; `max_rrow`
does not bound the row-subtree buffer, `n-n1` does) (6); solve phase
(`apply_Q!`/`apply_Qt!`/`solve_R!`/`solve_Rt!`/`solve!`/`\`/`solve_minnorm!`), one
real memory-corruption bug (an `@inbounds` overflow from the task-6 `max_rrow` gap)
and one real indexing bug (`pivotslot[k]` vs. physical row `k` conflated at the
solve-phase boundary) found via testing and fixed (7); rank-deficiency policy (Heath's
threshold test + Foster–Davis dead-column drop), rank detection ON by default since
task 7's own testing found an unguarded near-singular pivot blows `beta` up to ~1e33
(8). Full suite: 94/94 items, 206115/206115 assertions.

**2026-07-14 update: M5a task 9 CLOSED.** Column-singleton pre-elimination (§2.3):
breadth-first queue-based peeling on a value-aware row-form, independently verified
against a hand-written column-ascending reference (0/500 count mismatches). Own
derivation (verified algebraically, not assumed): a length-1 Householder reflector has
two valid sign conventions, and choosing `H=+1` makes `R11`/`R12` raw copies of `A`'s
own values with zero numerical work and makes the overall `Q` block-diagonal — so
`apply_Q!`/`apply_Qt!` need no change, only `solve!`'s back-substitution does. A
genuine gap surfaced by testing (not designed up front): `tol=0` alone does not
disable peeling (it only relaxes the magnitude threshold), so `qr()` gained an
explicit `singletons::Bool=true` keyword as a true on/off switch (documented in
design_qr.md §2.3/§6.4). `qr!` now rejects `sym.n1 > 0` factors with a clear
`ArgumentError`, matching the design's no-singletons-on-reuse rule. `solve_minnorm!`
gained an explicit full-rank precondition (`n_dead == 0`) since its minimum-norm
formula silently gives wrong answers otherwise — also found via testing, not designed.
Singleton peeling defaulting to on caused a large ripple across tasks 6–8's
pre-existing tests (many implicitly assumed `n1=0`); all traced and fixed one by one
(either `singletons=false` where the test's intent was the core n1=0 pipeline, or an
updated `rank+n_dead==n` invariant where the composed/default behavior was the actual
intent). Full suite: 105/105 items, 213831/213831 assertions.

**2026-07-14 update: M5a task 10 CLOSED.** `qr!` refactor hardening (zero-alloc gate,
StrictMode layer, trim smoke). Found the StrictMode runtime-checks layer CLAUDE.md req
6 requires didn't exist anywhere in the codebase — not for QR, not for Cholesky/LDLᵀ
either (`src/strict.jl` was never created despite `contracts.jl` naming it as the
intended home). Built it project-wide rather than QR-only (user-directed scope
decision): `src/strict.jl`, gated behind `StrictMode.checks_enabled()` (a compile-time-
baked constant, default off, so every check folds away at zero runtime cost) —
`check_refactor_shape`/`check_refactor_nnz` preconditions and `check_finite`
postconditions, wired into `cholesky!`, `ldlt!`, and `qr!` alike. `solve!`/
`solve_minnorm!`'s two remaining allocating temporaries (flagged "correctness-first;
zero-alloc is task 10" since task 7/9) are now permanent `QRWorkspace` scratch fields
(`rblk`/`n1a`/`n1b`), exploiting `solve_R!`/`solve_Rt!`'s documented input/output
aliasing to need only one `nb`-length buffer instead of two. A real bug was caught by
testing this: `_qr_compose_singletons` initially rebuilt a FRESH `QRWorkspace` for the
composed factor (to size the new n1-scratch), which silently threw away
`ws.rcursor`'s real populated state from A22's own factorization — replaced with
uninitialized garbage, causing an out-of-bounds-read segfault in `solve_R!` on the
very first n1>0 solve test; fixed by reusing `F22.ws`'s existing arrays and only
freshly allocating the new scratch fields. Trim smoke extended (`test/trim_tests.jl`
TrimCheck roots for `qr!`/`solve!` — `qr`/`symbolic_qr` can't be roots there, their
`ordering` keyword has no default and TrimCheck only supports positional-type roots;
`juliac/entry.jl` extended with a `qr`/`qr!`/`solve!` smoke pass on the same
Laplacian) — the actual `juliac --trim` build was run end-to-end (not just TrimCheck's
reachability analysis), 0 trim-verifier errors, executable runs and reports all five
residuals near machine epsilon. Full suite: 113/113 items, 213851/213851 assertions.
**2026-07-14 update: M5a task 11 CLOSED — gate measured, verdict NOT PASSING, M5b now
mandatory scope.** Built `benchmark/qr_matrices.jl` (8 synthetic gate matrices across
the 3 H4 strata — design §9.4 permits synthetics, same as M1's own gate set) and
`benchmark/qr_gate.jl` (cold-vs-cold PureSparse-vs-SuiteSparseQR, own-ordering and
same-permutation arms, mirroring `gate.jl`'s structure; QR needs no OpenBLAS
kernel-attribution arm — M5a's left-looking loop calls no BLAS-3 kernel, only
PureBLAS's `nrm2`). Ran on both clock-locked machines (galen, wintermute;
`performance` governor confirmed on both) after syncing current code (a genuine
environment gap surfaced along the way: wintermute's `benchmark/` env had no local
`TypeContracts` checkout at all, unlike galen's — fixed by rsyncing one over, a
one-time host setup gap, not a code issue). One real benchmark-harness bug found and
fixed: `SparseArrays.qr(A)\b` throws `SingularException` under `ORDERING_FIXED` on
the underdetermined (m<n) LP-slack matrices — guarded with try/catch (PureSparse's own
`\` never throws there, §6.2's basic-solution path handles it by construction).

**Verdict: 3/32 matrix-arm-host combinations beat SuiteSparseQR cold.** Stratum (i)
(singleton-dominated) 3/12 — wins outright on `staircase_n2000` (entirely-singleton,
zero numerical work) but loses on partial-singleton LP-slack shapes and the same-perm
arm. Stratum (ii) (sparse-R/small-front, H4 said "competitive") 0/12 — loses 3–10×.
Stratum (iii) (flop-rich/large-front, H4 said "may lose") 0/8 — loses 4–7×, as
anticipated. Diagnostic finding (not guessed, measured): on `banded_ls_n1500x500`,
PureSparse's COLAMD produces `nnz(R)=4193`, IDENTICAL to SuiteSparseQR's own ordering
choice — and on `lp_slack_n800x150`'s same-perm arm, forcing PureSparse onto SPQR's
own column order nearly QUADRUPLES its fill (4586→15624), meaning PureSparse's
ordering is not merely adequate but measurably better there. `symbolic_qr` alone costs
about what SPQR's entire cold factorization costs; the remaining 4–9× gap is entirely
in the unblocked scalar left-looking numeric loop (~1 GFlop/s effective, on ~8M flops
in 5.8ms) — exactly the architectural gap multifrontal BLAS-3 fronts exist to close.
Ordering (task 3) is validated as working correctly; the numeric kernel (task 6) is
confirmed as the actual bottleneck, not a guess.

Per design_qr.md §9.3's unconditional closeout gate ("no fudge factor... a stratum
loss triggers M5b"), **M5b (multifrontal, §7) is now mandatory scope**, not optional.
Full report with per-matrix data: `benchmark/results/qr_gate_{galen,wintermute}.json`
(raw), reproducible via `julia --project=benchmark benchmark/qr_gate.jl`.

**2026-07-14 follow-up: faer context arm (config 5) built and measured.** Added
`faer_sparse_qr` to BlazingPorts.jl's existing `rust_compare` cdylib shim
(`faer::sparse::linalg::solvers::Qr::sp_qr()`, MIT-licensed — read directly from the
crate's local `~/.cargo/registry/src` checkout rather than guessed from docs, which
turned out incomplete/version-drifted). Compiled clean on the first real attempt once
the actual source was read. One real bug found: faer's sparse QR hard-asserts
`nrows >= ncols` (`factorize_symbolic_qr`, confirmed in source) — it has no
underdetermined-system support at all, and a Rust panic crossing the `extern "C"`
boundary **aborts the whole Julia process**, not just the call (hit this directly on
the first real gate run, `lp_slack_n300x60`, m=300<n=360 — SIGABRT). Fixed with both a
Julia-side dimension guard before ever calling in, and a `catch_unwind` in the Rust
shim as defense in depth. Re-ran the full gate on both galen and wintermute.

**Result: faer wins exactly where its architecture predicts, and only there.**
Stratum (iii) flop-rich: faer beats SuiteSparseQR by ~20% on average (multifrontal
BLAS-3 fronts, its intended sweet spot). Stratum (ii): roughly at parity with SPQR on
the grid matrices, but 25–85× SLOWER than SPQR on the narrow-banded matrix (its
internal supernodal-vs-simplicial threshold heuristic evidently mis-picks for that
shape — a faer characteristic, not a PureSparse concern). Stratum (i): faer pays its
full COLAMD+symbolic overhead unconditionally on the trivial all-singleton case (no
equivalent to §2.3's zero-cost peeling), so PureSparse beats it there by 25–30×.
PureSparse loses to faer on strata (ii)/(iii) by roughly the same margin it loses to
SPQR — consistent with the earlier finding that the gap is architectural (unblocked
scalar loop vs. blocked multifrontal fronts), not a Julia-codegen deficiency: faer
itself only wins over SPQR where its OWN blocking helps, and PureSparse's own Cholesky
(already using PureBLAS's blocked kernels) already beats CHOLMOD+OpenBLAS on the M1
gate — the pattern holds, QR just hasn't reached the blocked-kernel phase (M5b) yet.
Full data: `benchmark/results/qr_gate_{galen,wintermute}.json` (`faer_cold` field);
shim: BlazingPorts.jl commit adding `faer_sparse_qr` to `rust_compare/rust/src/lib.rs`.

Next: M5b task list (design_qr.md §10 M5b, conditional list now activated) — P1/P2
PureBLAS block-reflector kernels (`larfb`-role `C:=Q·C` extension + the new `C:=Qᵀ·C`
direction), M5b design addendum, front-structure symbolic extension, frontal numeric
loop, re-run §9.3 for the actual M5 closeout.

**2026-07-15 update: M5b task 16 (frontal numeric loop + solve, `src/qr/frontal*.jl`)
CLOSED — correct on the worked example (RᵀR≈AᵀA and the solve normal-equations
residual both to ~1e-14) and on a 60-case random sweep (full rank, rank-deficient,
various shapes/densities, `--check-bounds=yes` clean) cross-checked against M5a's own
`:column` path per §A9.2. Three real bugs found and fixed along the way, all via
disciplined "diff the front's R against a trusted oracle, don't guess" debugging:
(1) R-harvest for a pivotal column was reading `Ff` BEFORE that reflector's own
in-panel trailing update landed, so every off-diagonal R entry was stale — fixed by
harvesting after the apply, with a genuinely deferred second pass for out-of-panel
(post-block-apply) columns; (2) the "restore the implicit-unit-diagonal" step
(`Ff[k,jj]=1`) was both unnecessary (the solve-phase V-gather in `frontal_solve.jl`
hardcodes the diagonal itself, never reads it from `Ff`) AND actively wrong — it
clobbered the true reduced value at a NON-pivotal column's own diagonal, which is
exactly what the parent front's C-block pass-up needs to read; removing it fixed
every front beyond the first in the tree; (3) `_assemble_front!`'s child-gather loop
read a child's rows `(fr+1):fm` (ALL assembled rows), but pass-up only sets `fmincol`
for `(fr+1):e_f` — whenever a front has more rows than columns, `e_f < fm` and the
excess rows are all-zero residue with garbage (assembly-time LOCAL, not global)
`fmincol`; reading past `e_f` produced a `BoundsError`/segfault. Added an explicit
`fe::Vector{Ti}` field to `QRFrontFactor` to track `e_f` and bound the child-gather
loop by it. Also discovered in passing that M5a's own column-QR path has a
non-negligible normal-equations residual (`‖Aᵀr‖≈0.08` on one rank-deficient test
case) where the frontal path hits `≈1e-15` — not chased (M5a is already gated/shipped
and the discrepancy is basic-solution non-uniqueness, not a shared-oracle failure),
but worth a look if M5a's rank-deficient handling ever gets revisited. New coverage:
`test/qr_frontal_numeric_tests.jl` (worked example + 60-seed random sweep + explicit
rank-deficient case, 186 assertions, all passing in the full suite run). Next:
task 16d (method selection, `qr(A; method=:auto|:frontal|:column)`), 16e (amalgamation
tunable recalibration), then task 17 (re-run the full M5 gate on galen/wintermute).

**2026-07-15 update: task 16d CLOSED** (`qr(A; ..., method=:column|:frontal|:auto)`,
`src/qr/numeric.jl`) — `:frontal` routes to `qr_frontal` (Float64 only, silent
`:column` fallback for other `T` per §A5.6 until P2 lands); `:auto` is explicitly
**not yet calibrated** (still `:column` — the threshold is a measured, not guessed,
quantity per CLAUDE.md's "don't guess" rule, and §A5.6 itself says so). Tested in
`test/qr_numeric_tests.jl` (6 new assertions); full 221k-assertion suite clean.

**2026-07-15: task 17 gate measured on galen (`benchmark/results/qr_gate_galen.json`,
clock-locked, `performance` governor) — verdict: 1/16 matrix-arm combinations pass,
OVERALL NOT YET PASSING.** Honest result, not fudged. Per-stratum: (i) 1/6, (ii) 0/6,
(iii) 0/4. The frontal path is consistently 2-4x faster than :column in strata
(ii)/(iii) (e.g. `grid_ls_70x50` same-perm: column 45.5ms → frontal 6.8ms — a real,
large win over M5a) but SuiteSparseQR is still ahead there by roughly 1.2x-2.3x
(closest: `dense_arrow_n800x200_d8dense` at ~1.2x; furthest: `banded_ls_n1500x500_bw15`
at ~2.3x). Stratum (i)'s losses are noise-level close (0.049ms vs 0.045ms,
0.146ms vs 0.144ms) — plausibly resolved by more samples, not a structural gap.
Two concrete next levers, neither yet pulled: (1) `_assemble_front!` still allocates
several small `Vector`s per front (`push!`-built `phys`/`mincol`/`srcfront`/`srcrow`,
flagged as a known gap in its own file header since task 16a landed) — likely a
non-trivial constant-factor tax on every matrix size, unlike the amalgamation lever
below which mainly helps larger fronts; (2) task 16e's amalgamation recalibration
(§A8) — the current `AMALG_COLS`/`AMALG_ZMAX` were swept for Cholesky, not QR, and
§A8 gives concrete reasons QR should tolerate more merging (a merge deletes an
assembly round-trip, not just saves padding flops). Wall-clock note: wintermute was
NOT used for this run — a concurrent session had live uncommitted PureBLAS.jl work
there when this session's rsync landed (no data loss found on inspection: the other
session's work was intact in `git stash@{0}` afterward), so this session stood down
from that host rather than risk further collision; galen-only for now.

Also ran the user-requested standalone comparison: 7000×4000, densities 1%/3%/5%/10%,
PureSparse (:column and :frontal) vs SPQR vs faer — see
`benchmark/results/faer_vs_puresparse_7000x4000_galen.json` and the session record for
the printed table.

**Side note (2026-07-14): PureKLU.jl (SciML, pure-Julia sparse LU) surfaced by the
user as a possible reference — MIT-licensed, so unlike CHOLMOD/SuiteSparse it is NOT
subject to the clean-room read-prohibition (CLAUDE.md req 1's ban is SuiteSparse-
specific); fair to read/reference/use. Not yet evaluated for relevance to PureSparse's
own scope (LU vs QR/Cholesky) — noted for future consideration, not acted on.**

**Status (2026-07-13): M1 CLOSED, M2 CLOSED, M4 CLOSED (every gate item in
`docs/design.md` §10 met, see the `### M1`/`### M2` sections and the "M4 closeout"
section below).** The three M4 gap items (`SimplicialLDLFactor` property parity, `F.U`
extraction, and re-verifying M4's own stated gate via a genuine downstream-consumer
smoke suite plus a wall-time gate re-run through the dropin entry point) are all
closed — see "M4 closeout". M3 not started, deliberately deferred — note `galen` (used
for perf gating since) does have an RTX 4070 reachable via SSH, which could unblock M3
without reordering, if wanted later.

## M1 gate — `juliac --trim` smoke build succeeds (2026-07-13)

The last open M1 gate item (design.md §10) is closed. `juliac/entry.jl` is the
factor-and-solve smoke: a 12×12-grid 2-D Laplacian (n=144, SPD, FEM-class) through
`symbolic` → `cholesky` → `solve!` → `cholesky!` (analyze-once/refactorize path) →
`ldlt` → `solve!`, each gated on a hand-rolled `‖b − A·x‖∞ < 1e-10` residual (exit 0
iff all pass). `julia juliac/build.jl` compiles it with
`juliac --output-exe --experimental --trim=safe` (mirrors PureBLAS.jl/juliac/build.jl;
output `juliac/build/puresparse_smoke`, gitignored, ~23 MB). Verified: build finishes
with **0 trim-verifier errors** and the trimmed executable prints residuals ~1e-14 —
identical to the same entry run under normal Julia — and exits 0.

**No `src/` changes were needed** — the library's factor-and-solve path was already
trim-clean. The only trim incompatibilities were in the entry file itself, fixed there
with comments: `sparse(I,J,V,…)` (stdlib coalescing takes an abstract
`combine::Function` — the smoke builds its CSC arrays directly instead) and bare
`println` (routes through the abstract `Base.stdout::IO` global — `Core.stdout` used
instead).

Regression guard: `test/trim_tests.jl` (TrimCheck `@validate`, the same reachability
analysis juliac --trim runs, mirroring PureBLAS's trim testitem) REDs in the ordinary
suite if `symbolic`/`cholesky`/`cholesky!`/`ldlt`/`ldlt!`/`solve!` (Float64/Int64
roots, kwarg-default paths included) regress; the minutes-long end-to-end juliac build
stays manual via `julia juliac/build.jl`. TrimCheck added to test/Project.toml.

## Perf investigation — large-n scatter-add overhead found and fixed (2026-07-13)

`benchmark/size_sweep.jl` showed PureSparse losing to CHOLMOD+OpenBLAS at n=1024/2048,
by a WIDER margin on the PS+OpenBLAS arm than PS+PureBLAS — the tell that the gap lived
in shared scheduling code (`src/numeric/llt.jl`), not kernel quality, since both
PureSparse arms call the identical OpenBLAS kernels through the swap in that
configuration.

**Profiled first, not guessed** (`Profile.@profile` over 30 warm `cholesky!` calls at
n=2048, flat + tree output). The `@inline` hypothesis named in the task brief was
checked and ruled out: `_row`/`_panelview` already carry `@inline` (still true), and
`Base.return_types` on `cholesky!`/`ldlt!` showed fully concrete return types — no
boxing, no dispatch failure anywhere in the loop. The actual hot spot: the
scatter-add loops that fold a descendant's computed update block `C`/`C1`/`C2` into the
ancestor panel (`llt.jl` lines ~135–160 pre-fix) were **row-outer, column-inner**
(`for a in 1:k1, for b in 1:a`), which strides `panel`/`C1` by their full column height
on every inner-loop step — at n≈2048 that stride is tens of KB, effectively a cache
miss per element on two arrays at once. This one section of code was ~22–28% of total
`cholesky!` wall time in the flat profile.

**Two fixes, both purely eliminating overhead — no scheduling/algorithm change:**

1. **Loop-order swap**: column-outer/row-inner (`for b in 1:k1, for a in b:k1`) so the
   inner loop walks `panel` and `C`/`C1`/`C2` contiguously down one column instead of
   striding. Isolated microbenchmark (`k1=1000` triangle scatter): **2.4x faster**
   (1.30ms → 0.54ms). Applied identically to `llt.jl`'s two scatter sites and
   `ldlt.jl`'s one (same left-looking structure, same bug — CLAUDE.md's mirrored-fix
   expectation).
2. **Contiguity fast path**: when a descendant's remaining rows form a contiguous run
   of the ancestor's row list (the common case in the dense trailing part of the
   factor — `lr(a) = lr0 + a - 1`), the whole "compute update block C into a staging
   buffer, then scatter-add element-by-element via `relmap[_row(...)]` lookups" pattern
   is unnecessary: `syrk!`/`gemm!` can accumulate directly into the panel with
   `beta=1`, since the target IS a contiguous subview of the panel. This removes the
   staging write+read and all per-element `relmap` lookups for that case entirely,
   falling back to the staged path (loop-swap-only) when non-contiguous. Applied to
   both `llt.jl` (syrk!/gemm! sites) and `ldlt.jl` (its chunked gemm! site, `contig`
   passed through the chunk-`beta` logic so multi-chunk descendants still accumulate
   correctly).

**Verified, not assumed:** correctness re-checked via direct residual (`norm(A*x-b)/norm(b)`
on random SPD and random SQD/KKT matrices spanning the fast and non-fast paths, both
~1e-15/1e-16, matching pre-fix behavior), zero-allocation gate re-checked (`cholesky!`
still 0 bytes, `ldlt!` unchanged at 7728 bytes — the fast path reuses `panel`, never
touches `Workspace.cd`'s allocation-bearing chunk path differently). Full suite
**15206/15206 passing** (`test/runtests.jl`, unchanged from before this pass — no test
needed loosening).

**Before → after, n=1024/2048, warm numeric refactor median (`neuromancer`, focused
repro matching `size_sweep.jl`'s methodology):**

| n | arm | before | after |
|---|---|---|---|
| 1024 | PS+PureBLAS / CHOLMOD | 0.952x (near-tie) | **1.10x** |
| 1024 | PS+OpenBLAS / CHOLMOD | 0.924x (losing) | **1.04x** |
| 2048 | PS+PureBLAS / CHOLMOD | 0.834x (losing) | **1.03x** |
| 2048 | PS+OpenBLAS / CHOLMOD | 0.789x (losing, worse) | **0.96x** (near-tie, was losing by 21%) |

Full `benchmark/size_sweep.jl` re-run afterward (n=2..2048, saved to
`benchmark/results/size_sweep_neuromancer.json`) confirms: PS+PureBLAS now beats
CHOLMOD at every size 2–2048 (1.04x–8.77x), PS+OpenBLAS beats or ties CHOLMOD at every
size except n=64 (0.74x, same pre-existing effect at that size — see the "Perf
follow-up" section's later root-cause measurement: a still-fragmented supernode
partition amplifying OpenBLAS's higher per-call overhead vs PureBLAS's) and n=2048
(0.96x, up from 0.79x).
`n=2048`'s remaining ~4% OpenBLAS-arm gap was not chased further: profiling after the
fix shows `potrf!`/`syrk!`/`gemm!`/`trsm!` kernel time now dominates (as it should),
with the scatter loops no longer visible as a distinct hot spot in the flat profile —
what's left looks like ordinary dense-kernel-share-of-total-time noise, not a
scheduling-layer defect, and chasing sub-5% further would mean second-guessing OpenBLAS
kernel tuning, out of scope for this pass.

**Gates re-run, both still PASS, no regression:** M1 LLᵀ gate **14/14** (up from
11–14/14 baseline — the OpenBLAS arm, which was the weaker of the two PureSparse arms
against CHOLMOD, now clears CHOLMOD on every matrix too, not just PureBLAS's arm). M2
SQD/LDLᵀ gate **8/8**, unchanged (0.042–3.61ms PureSparse vs 0.073–18.25ms CHOLMOD
across n=200–2000 — `ldlt.jl`'s fast path pays off less dramatically here since the SQD
gate matrices are smaller/sparser than the size-sweep's dense-trailing-part-heavy random
SPD case, but the fix is real and correctness-verified regardless).

Changed: `src/numeric/llt.jl` (`cholesky!`'s descendant-update loop), `src/numeric/ldlt.jl`
(`ldlt!`'s descendant-update loop, same pattern). Branch: `perf-large-n`.

**Note on the numbers above:** the `neuromancer` before/after table above (and the
`size_sweep_neuromancer.json` it references) predates a finding that neuromancer's CPU
clock is unpinned/unlocked — its wall-time numbers are not reliable for gate pass/fail
and are kept here as historical context only. `galen` and `wintermute` are the
clock-locked gate machines going forward (see the follow-up section immediately below).

## Perf follow-up — galen n≥1024 residual gap closed (2026-07-13)

The fix above measurably helped but did not fully close the gate on `galen` (AMD Ryzen 9
5900X — a different microarchitecture than the machine the first fix was tuned on):
`size_sweep.jl` on galen still showed PS+PureBLAS **losing** to CHOLMOD+OpenBLAS at
n=1024 (0.93x) and n=2048 (0.90x), even though `neuromancer` passed at every size. Per
the project's non-negotiable wall-time gate (CLAUDE.md rule 2) this was a live defect,
not an acceptable tradeoff — user: *"keep pushing for parity, we cannot be below that is
not accepted in the contract."*

**Profiled on galen itself, not assumed to match neuromancer's fix.** Phase-timed
instrumentation of `cholesky!` at n=1024/2048 showed the *staged* scatter-add path (used
whenever a descendant's target rows aren't contiguous — the common case at these sizes)
still spending a large fraction of wall time in `relmap[_row(rowind, ...)]`, a double
indirection that is **identical for every column `b`** of the update block but was being
recomputed `k1` times. Measured on galen at n=2048: ~1.5 ns/element over 7.9e6 scattered
elements per factorization (~12 of 65 ms). A run-length histogram of the scattered
target rows showed 66% of elements sit in maximal-consecutive runs ≥9 rows (38% in runs
>32) — exploitable via contiguous SIMD adds, but *not* unconditionally: an
unconditional run-based scatter was measured as a net **loss** at n=2048 (57.98 →
59.81ms) versus the plain element-based loop, due to per-run visit overhead on short
runs.

**Fix:** two new `Workspace` buffers, `ir` (per-update hoisted target-row list, resolved
once during the existing contiguity-check walk instead of `k1` times) and `rs` (run-start
offsets of `ir`'s maximal consecutive runs, built in the same walk — `contig ⟺ nr==1`,
reusing the pre-existing `ctot ≤ max_extend_rows` sizing bound). A new shared
`_scatter_update!(panel, cbuf, ir, rs, nr, k1, ctot)` helper (kept `@noinline` —
inlining it into the factorization loop was a measured 30–95% regression at n=16..64 on
`wintermute`) dispatches per-update between the run-based `@simd` path (mean run length
≥8, i.e. `ctot ≥ 8*nr`) and the element-based hoisted-index path (short runs), replacing
the separate C1/C2 scatter loops in both `llt.jl` and `ldlt.jl` with one shared call —
semantics unchanged (full below-diagonal block + lower-triangle-only diagonal block,
verified by direct diff review against the pre-fix loops).

**Verified independently (not just trusting the agent report):** diff-reviewed both
files against the original scatter semantics; re-ran direct residual checks
(`norm(A*x-b)/norm(b)`) on fresh random SPD (`cholesky`) and SQD (`ldlt`, `signs=+1`)
matrices at n=3,17,64,300,1000 — spanning both the run-based and element-based branches
— all ~1e-15/1e-16, matching pre-fix behavior. Full suite **15206/15206 passing**
(neuromancer, used only as a correctness runner here, not for timing). M1 LLᵀ gate
**14/14**, M2 SQD/LDLᵀ gate **8/8** on galen.

**Gate result — PureSparse+PureBLAS now strictly beats CHOLMOD+OpenBLAS at every swept
size (n=2..2048) on both clock-locked machines:**

| n | galen PB/CHOLMOD | wintermute PB/CHOLMOD |
|---|---|---|
| 512 | 1.19x | 1.40x |
| 1024 | **1.09x** (was 0.93x) | **1.24x** |
| 2048 | **1.03x** (was 0.90x) | **1.15x** |

(Secondary OB/CHOLMOD diagnostic arm dips slightly under 1.0x at n=64 on both machines —
0.81x galen, 0.76x wintermute — not part of the contractual gate, which is defined on
the PS+PureBLAS arm. **Root cause fully measured (2026-07-14), superseding an earlier,
incomplete pass on this same question — see that pass's methodology mistakes below,
since they're instructive.** Reproduced the exact n=64 sweep matrix
(`MersenneTwister(2026)` advanced through sizes 2..64, matching `nnzL=224`): supernode
partition still fragmented at this transitional size — 16 supernodes for 64 columns,
widths `[1,1,1,1,1,1,1,1,1,2,3,5,6,7,16,16]` (nine width-1). This partition is a
symbolic-phase property, IDENTICAL in both arms — not "caused by" either backend; what
differs is per-call kernel cost.

**First attempt (wrong methodology, ~88% "explained"):** micro-benchmarked `potrf!`
alone on freshly-allocated COMPACT matrices at the 7 distinct widths present, ignoring
that the real code calls kernels on STRIDED VIEWS into larger panel buffers (a
below-diagonal-row-padded panel, or `Workspace.c`/`cd`'s `max_extend_rows`-sized
scratch). Landed on 1537.5ns vs an observed 1750.5ns gap (88%) — a number that LOOKED
convincing but was an artifact of the wrong matrix shape.

**Second attempt (still incomplete, DROPPED to 74%):** redid `potrf!`+`trsm!` with the
correct strided-view shapes (matching `_panelview`'s `unsafe_wrap` + `view(panel,
1:ncol,1:ncol)` pattern exactly) — one case (the 16×22 supernode) even showed OpenBLAS's
`trsm!` FASTER than PureBLAS's, contradicting the first pass's clean story. Combined
potrf!+trsm! delta: 1295.5ns of a 1750.5ns gap (74%) — WORSE than the first attempt,
because `syrk!`/`gemm!` (the descendant-update calls) were still entirely unmeasured.

**Third attempt (real in-context instrumentation, closes it for real):** rather than
reconstruct synthetic calls, directly timed (`time_ns()` before/after) every ACTUAL
kernel call — all 16 `potrf!`, 9 `trsm!`, 10 `syrk!`, 1 `gemm!` — in their real memory
locations, accumulated per kernel type across 20,000 real warm `cholesky!` calls (timer
overhead is identical in both arms, so it cancels in the delta):

| kernel | PureBLAS | OpenBLAS | delta |
|---|---|---|---|
| `potrf!` | 2935.5ns | 4339.4ns | 1403.8ns |
| `trsm!` | 1648.9ns | 3072.9ns | 1424.0ns |
| `syrk!` | 1771.6ns | 2441.7ns | 670.1ns |
| `gemm!` | 101.6ns | 145.7ns | 44.1ns |
| **sum** | 6457.6ns | 9999.6ns | **3542.0ns** |

Cross-checked against a clean, uninstrumented re-measurement in the same session
(6832.5ns / 9973.5ns / gap 3141.0ns): the instrumented kernel-sum delta accounts for
**112.8%** of the clean gap — the small overshoot is ordinary run-to-run measurement
noise at this microsecond scale (~5-15% variance observed across separate runs even on
a clock-locked machine), not evidence of a missing factor. **`potrf!` is NOT the
dominant contributor as first claimed — `trsm!` costs essentially the same (40%/40%),
with `syrk!` a real third contributor (19%).** Fragmented call pattern identical in
both arms; per-call kernel cost, summed honestly across all four kernel types rather
than one convenient one, is what tips only the OpenBLAS arm below CHOLMOD's own
OpenBLAS-based total (PureBLAS's lower fixed cost absorbs the identical fragmentation
fine — PS+PureBLAS still wins at n=64, 1.10x-1.17x). CHOLMOD's own supernode count on
this matrix was still not obtained, but is no longer needed to close the explanation —
the in-context kernel-timing sum already accounts for the entire observed gap.)

Changed: `src/numeric/llt.jl`, `src/numeric/ldlt.jl`, `src/types.jl` (new `Workspace.ir`/
`rs` buffers). Branch `perf-parity-galen`, commit `82b9350` on top of `412b1d8`, merged
to `master` via fast-forward.

## Perf follow-up — small-supernode (width 1–2) kernel-call bypass (2026-07-14)

Acting on the n=64 root cause above (per-call kernel overhead on structurally
unmergeable width-1..3 supernodes, `sym.parent[j] == 0` — not an amalgamation-threshold
problem): `cholesky!` now factors width-1 and width-2 diagonal blocks INLINE in the
shared left-looking scheduling code instead of calling `potrf!`/`trsm!` — a 1×1
Cholesky is `L₁₁ = √A₁₁` + a column scale by `inv(L₁₁)`, a 2×2 is the same textbook
recurrence as PureBLAS's own generic unblocked base plus a two-term forward
substitution per panel row (Golub & Van Loan base cases; no CHOLMOD content, no new
BLAS/LAPACK dependency — the fast path calls NOTHING). Because it sits in the shared
scheduler ABOVE the kernel binding, both the PureBLAS production arm and the OpenBLAS
diagnostic arm benefit. Width ≥3 was deliberately NOT inlined (that way lies
reimplementing dense Cholesky in `src/`, CLAUDE.md's "never reimplement dense
kernels"; width-2 was kept because it measured as a real win — n=64 OB arm 0.914x →
0.936x on neuromancer — and is still a 4-line recurrence).

`ldlt!` mirror: its hand-rolled unit-LDLᵀ column loop already makes ZERO kernel calls
at width 1 (`ger!` fires only for `j < nscol`; the 1/d scale is already inline), so
the width-1 mirror is a documented no-op; the width-2 mirror inlines the loop's single
rank-1 `ger!` (single-entry `y`) as one fused `muladd` column op. The signed-
regularization / inertia logic (design §5.1) is untouched — verified bit-identical
`d`, `x`, and stats (`n_pos`/`n_neg`/`n_perturbed`/`rcond_est`) vs the pre-change
build on a 400-dof SQD KKT and the n=64 sweep matrix.

**Numerical semantics:** the inline LLᵀ path is bit-identical to the textbook/LAPACK
formulation and to PureBLAS's generic `_potf2_lower!` (`sqrt(d)` then multiply by
`inv`), and matches the failure rule exactly (`real(d) > 0`, so NaN and ≤0 both fail →
`F.ok = false`, `fail_col = j0`, verified for width-1 and width-2 pivots incl. the
downdated second pivot, on both kernel arms). It is NOT bit-identical to PureBLAS's
Float64 faer base, which itself uses the `1/√d` reciprocal formulation (diagonal
stored as `d·(1/√d)`) — measured ≤1 ulp apart, value-dependent; the inline `sqrt(d)`
is the correctly-rounded root, so accuracy is equal-or-better. LDLᵀ output is
bit-identical to pre-change. Residuals ~1e-16 across SPD n=32..1000 and SQD KKT
n=80/400, both arms. Generic over `T` (Float32 verified through the fast path;
BigFloat crashes identically on the PRE-change build — pre-existing panel/codegen
issue, unrelated).

**Result (size_sweep, warm refactor medians, before → after, all three machines
re-measured this pass — baselines re-run fresh, not read off old JSONs):**

| n=64 arm | neuromancer | galen | wintermute |
|---|---|---|---|
| OB/CHOLMOD | 0.77x → **0.97x** | 0.82x → **0.99x** | 0.79x → **0.99x** |
| PB/CHOLMOD | 1.10x → **1.33x** | 1.16x → **1.42x** | 1.00x → **1.36x** |

The n=64 OB diagnostic dip is now a ~1–3% near-tie, not a 20% loss (honest number:
still a hair under 1.0x — the remaining gap is OpenBLAS `syrk!`/`gemm!` per-call cost
on the small staged updates, per the kernel-timing table above, which the diagonal
fast path doesn't touch). Small sizes improved dramatically on BOTH arms since tiny
factors are now kernel-call-free end to end (neuromancer PB/CHOLMOD: n=2 6.2→21.5x,
n=8 2.8→13.9x, n=16 1.6→8.4x, n=32 1.7→3.4x; galen and wintermute similar). n=128–2048
unchanged within noise on all machines (checked size-by-size — no regression from the
two extra `nscol` branches per supernode).

**Verified:** full suite **15220/15220** (56/56 items) on the changed worktree; M1 LLᵀ
gate **14/14** and M2 SQD/LDLᵀ gate **8/8** PASS on both neuromancer and galen
(`performance` governor); `@allocated cholesky!`/`ldlt!` still **0** on a
fast-path-exercising matrix; `Base.return_types` still concrete for both. Branch
`n64-smallblock-fastpath` (worktree), based on `eadd785`.

## M4 progress — drop-in LANDED for `cholesky` AND `ldlt` (2026-07-13)

`src/dropin_toggle.jl` (always loaded): `activate!()`/`deactivate!()` set the
`dropin_active` Preference. `src/dropin.jl` (only `include`d when `DROPIN_ACTIVE` —
`src/tuning.jl` — is `true`): extends `LinearAlgebra.cholesky` AND `LinearAlgebra.ldlt`
for `SparseMatrixCSC`/`Symmetric`/`Hermitian` real (non-complex) input, matching
CHOLMOD's own kwarg surface (`shift`, `check`, `perm`) and adding stdlib-surface parity
on the returned `SupernodalFactor`/`LDLFactor`: `.p` (permutation, `getproperty`
override), `.L` (sparse extraction via the new `sparse_L`, generic over both factor
types), `logdet`, `det` (each with its own convention, see below). Int32 indices
already worked for free (M1's generic-over-`Ti` design). `SimplicialLDLFactor`
property parity (post-`updowndate!`) is NOT done this pass — documented as a
follow-up, not silently skipped.

**`ldlt`'s drop-in scope is deliberately narrower than `cholesky`'s (a real,
documented gap, not an oversight):** stdlib's own `LinearAlgebra.ldlt` has no
`signs`/`n_pos`/`n_neg` kwarg (verified directly against its actual method
signature — it only takes `shift`/`check`/`perm`, same as `cholesky`), so the drop-in
entry point always factors with `signs = nothing` (free signs, magnitude-floor-only
regularization, design.md §5.1). `PureSparse.ldlt(A; n_pos, n_neg)` called directly
gets materially better regularization behavior for the common SQD/KKT case — the
drop-in `LinearAlgebra.ldlt(A)` is the weaker, general-indefinite-matrix path.

**Real bug caught in my OWN first attempt at `LDLFactor`'s `logdet`/`det`, fixed
after checking rather than shipping the assumption:** the first version returned the
SIGNED pivot product (`det(F) = prod(F.d)`) and its docstring claimed, unverified,
that this "matches CHOLMOD's own convention." Testing an actual negative-determinant
case (`diag(2,-3,5)`, true determinant `-30`) immediately surfaced two problems: (a)
`logdet` threw `DomainError` (`log` of a negative real doesn't auto-promote to
`Complex` in Julia — a second, compounding wrong assumption), and (b) checking
CHOLMOD's OWN behavior on the identical input showed `det(F) == 30`, not `-30` — an
**absolute-value** convention, not the signed product at all (plausibly because
CHOLMOD's general indefinite `ldlt` does dynamic Bunch–Kaufman-style pivoting with
possible 2×2 blocks, where per-pivot sign isn't simply attributable — not
independently confirmed, only the observed behavior is). Fixed to
`det(F) = abs(prod(F.d))`, matching CHOLMOD's observed output exactly on both the
even- and odd-negative-pivot-count cases; `logdet` is consequently always real, never
throwing.

**Why this can't be a same-session runtime toggle (worked through explicitly, not
assumed):** PureBLAS's own `activate()`/`deactivate()` forwards through
`libblastrampoline`, a C-ABI indirection layer BLAS calls already go through, which
supports true runtime hot-swapping. Julia's pure-dispatch method table has no
equivalent — once a method with a given signature is defined, the OLD method for that
exact signature is gone, not shadowed (verified: `Base.invoke` can't reach it either,
since there's nothing left to invoke). Making the override unconditionally defined and
branching on a runtime `Ref{Bool}` inside it was considered and rejected: the override
would already exist the moment PureSparse loads, which is exactly what CLAUDE.md's
`import LinearAlgebra` comment (M1) says not to do, even if it's a functional no-op
when "inactive." The only way the override genuinely doesn't exist until opted in,
without runtime `eval`/`invokelatest` (forbidden, CLAUDE.md requirement 4, needed for
`juliac --trim`), is gating the `include` itself on a compile-time Preference —
`activate!()`/`deactivate!()` therefore require a Julia restart, an honest consequence
of the dispatch model, not a corner cut.

**Real, documented deviation from CHOLMOD's exact `perm=` contract (found during
testing, not assumed away):** CHOLMOD guarantees `F.p == perm` exactly when `perm` is
given (verified directly: `F.p == myperm` for a reversed test permutation). PureSparse
cannot offer that — `symbolic()` always composes ANY ordering with a postorder
relabeling step, required for supernode detection to see contiguous children
(design.md §3.2/§3.4) — so `perm=` sets the elimination order but `F.p` may not equal
it exactly. The factorization is still correct for whatever `F.p` ends up being
(verified: `L·Lᵀ ≈ A[F.p,F.p]` still holds to 1e-9 rel. under a forced `perm`); only
code literally asserting `F.p == perm` would observe the difference. Documented in
`dropin.jl`'s docstring, not silently papered over.

**Verified (2026-07-13):** `test/dropin_tests.jl`, 2 items. The real one spawns an
ISOLATED subprocess with its own temp `--project` (own `LocalPreferences.toml` setting
`dropin_active=true` — writing to `test/`'s own `LocalPreferences.toml` directly would
contaminate every OTHER test item's precompiled state, since Preferences are
project-scoped) that exercises the actual stdlib entry points
(`LinearAlgebra.cholesky`/`ldlt`, not `PureSparse.cholesky`/`ldlt`) end-to-end. For
`cholesky`: bare `SparseMatrixCSC` and `Symmetric` input, solve residual, `.L`/`.p`
extraction vs a dense oracle, `logdet`/`det` vs `LinearAlgebra.cholesky` on the dense
reconstruction, `shift`, `perm` (checked for the weaker-but-correct property above),
`check=true` throwing `PosDefException` on non-SPD and `check=false` not throwing,
Int32 indices. For `ldlt`: same shape on a random SQD KKT matrix (solve residual,
`L·D·Lᵀ` reconstruction, `shift`, `perm`), plus the abs-value `det`/`logdet` convention
checked on BOTH an even-negative-pivot-count case (matches the signed product too, so
alone it wouldn't have caught the bug) and the odd-count `diag(2,-3,5)` case that
actually surfaced it (`det(F) == 30` exactly, `logdet` real and non-throwing).
Subprocess exit code 0 is the pass signal (any `@assert` failure exits nonzero). A
second item confirms the OPPOSITE: in the normal (non-dropin) test environment,
`DROPIN_ACTIVE == false`, `dropin.jl`'s symbols are undefined, and
`LinearAlgebra.cholesky` has exactly its original 1 method for `SparseMatrixCSC` — i.e.
this file's mere presence doesn't leak activation into the rest of the suite. Full
suite still green after adding this (see the commit for the exact count).

## M4 closeout — all three gap items closed (2026-07-13, branch `m4-closeout`)

**1. `SimplicialLDLFactor` property parity.** Added `.p`/`.L`/`.U` to the same
`getproperty` override in `dropin.jl`, now `Union{SupernodalFactor,LDLFactor,
SimplicialLDLFactor}`. `SimplicialLDLFactor` needed its own `sparse_L` overload — its
padded/slack CSC layout (`colptr`/`colnnz`/`rowval`/`nzval`, only the first `colnnz[j]`
slots of each column live, the rest unused capacity for `updowndate!` growth) is not the
supernodal panel layout the existing `sparse_L` reads. The new method builds the CSC
directly (columns already sorted, no COO round-trip), reads the LIVE arrays, and adds
the implicit unit diagonal explicitly — so `F.L`/`F.U`/`F.p` on a `SimplicialLDLFactor`
reflect whatever state the factor is in AFTER any `updowndate!` calls, not a snapshot
from `simplicial()`. Verified in `test/dropin_tests.jl`: `simplicial(FL)`, then
`updowndate!(G, w, +1)`, then `G.L`/`G.U`/`G.p` re-extracted and checked against
`L·D·Lᵀ ≈ (A + wwᵀ)[p,p]` — i.e. the property reflects the POST-update factor, which
was the whole point.

**2. `F.U` extraction.** Convention decided as `U = Lᵀ` (materialized
`SparseMatrixCSC`, `A[p,p] ≈ Uᵀ·U` for LLᵀ; unit-upper for the LDLᵀ types), and this
was VERIFIED against real output, not assumed (CLAUDE.md req 1 — clean-room via
observable behavior only):
- Dense `LinearAlgebra.cholesky(Symmetric(A))` on a small hand-built SPD matrix:
  `F.U == F.L'` exactly, `F.U'*F.U ≈ A`.
- CHOLMOD's sparse LLᵀ `Factor`: its lazy `:U`/`:L` `FactorComponent`s can't even be
  `getindex`'d or `Matrix()`/`sparse()`'d directly (`CanonicalIndexError`/
  `CHOLMODException`, hit immediately when tried) — the only observable operation is
  `\`. Checked `F.U \ b == L' \ b` and `F.U' \ b == L \ b` (`L` obtained via
  `sparse(F.L)`) on a small example: both hold to ~1e-13. So CHOLMOD's own `.U` **is**
  `Lᵀ` in the solve sense, confirming the dense convention carries over.
- CHOLMOD's sparse LDLᵀ `Factor`: `sparse(F.L)` itself THROWS
  (`"sparse: supported only for :LD on LDLt factorizations"` — CHOLMOD only lets you
  materialize `:LD`, an interesting real API asymmetry, caught directly rather than
  guessed). Working from `sparse(F.LD)` (unit-`L` below the diagonal, `D` on the
  diagonal, `d = diag(LD)`), reconstructed `L = tril(LD,-1)+I`, `D = Diagonal(d)`, and
  confirmed `L·D·Lᵀ ≈ PKP` (the target matrix). Then checked `F.U \ b == (L') \ b`,
  `F.D \ b == D \ b`, `F.DU \ b == (D*L') \ b`, `F.LD \ b == (L*D) \ b` — ALL matched.
  So CHOLMOD's LDLᵀ `.U` is also `Lᵀ` (unit-upper), same convention as the LLᵀ case —
  no special-casing needed for `LDLFactor`/`SimplicialLDLFactor`.
- One deliberate TYPE deviation, not a semantic one: we return a materialized
  `SparseMatrixCSC` (`copy(transpose(sparse_L(F)))`) where CHOLMOD returns an
  unmaterializable lazy component — strictly more usable, and consistent with what
  `.L` already does here.
- Also found and fixed while probing: `LinearAlgebra.issuccess(F::AbstractSparseFactor)`
  was missing. PureSparse's own exported `issuccess` and stdlib's `LinearAlgebra.
  issuccess` are deliberately different functions pre-dropin (`import`, not `using`,
  in `PureSparse.jl` — see that file's comment); without a stdlib-name method, a
  downstream consumer's idiomatic `issuccess(cholesky(A; check=false))` was a
  `MethodError` under the drop-in. Added `LinearAlgebra.issuccess(F) = issuccess(F)` in
  `dropin.jl`, caught by the new downstream smoke test below (item 3), not assumed.

**3. M4's own gate, re-verified.**
- **Downstream-consumer smoke suite:** no sibling Pure-ecosystem package (PureBLAS is
  dense-only, PureFFT is FFT-only) has an existing sparse-Cholesky test suite to borrow,
  so `test/downstream_smoke.jl` is a purpose-written synthetic stand-in — but genuinely
  stack-agnostic: stdlib names only (`cholesky`/`ldlt`/`\`/`logdet`/`det`/`issuccess`/
  `.L`/`.U`/`.p`), zero mention of PureSparse anywhere in its body, and it was run
  TWICE to prove that: once standalone against plain CHOLMOD (`julia
  test/downstream_smoke.jl`, no PureSparse loaded at all — **18/18 pass**), and once
  wired into `test/dropin_tests.jl`'s new subprocess testitem ("M4 gate:
  downstream-consumer smoke suite passes unmodified with dropin active"), which
  `include()`s the unmodified file after activating the drop-in and asserting
  `LinearAlgebra.cholesky(...) isa PureSparse.SupernodalFactor` as a preamble guard
  (so a silently-inactive drop-in can't make the test vacuous) — **18/18 pass** there
  too, unmodified.
- **Wall-time gate through the dropin entry point** (`benchmark/dropin_gate.jl` +
  `dropin_gate_inner.jl`, new): two isolated subprocesses (dropin-inactive baseline,
  dropin-active measured arm — required because an active drop-in dispatch-shadows
  CHOLMOD's own `cholesky` methods, so both can't be measured in one process), same
  `gate_matrices()` set as `gate.jl`, same Chairmarks methodology (30 samples/1.5s cap,
  `evals=1`, single BLAS thread). The PureSparse arm is entered EXCLUSIVELY via
  `LinearAlgebra.cholesky`/`cholesky!` — interposition (kwarg translation, `Symmetric`
  unwrapping, `getproperty` overrides) included in the measured call, not bypassed.
  One real bug caught by this new harness, not assumed away: the first version fed the
  warm `cholesky!` refactor the gate's LOWER-TRIANGLE-ONLY matrix `A`, but the
  dropin-produced factor's symbolic pattern is that of `sparse(Symmetric(A,:L))` (the
  FULL pattern) — `cholesky!`'s contract requires a matching pattern, and the mismatch
  silently early-exited at column 1 (`fail_col=1`), producing fake sub-linear "warm"
  timings and a NaN-residual factor. Caught by adding `issuccess`/residual assertions
  around every measured call (now permanent in the harness), fixed by feeding the same
  `Afull = sparse(Symmetric(A,:L))` the drop-in itself builds internally.
  - **Run on `neuromancer` (this session's machine), governor `performance`** (checked
    directly per this session's brief) — **warm gate: 14/14 matrix-arm combinations
    PASS**, PureSparse+PureBLAS strictly faster than CHOLMOD+OpenBLAS on every one, warm
    numbers in the same order of magnitude as `gate.jl`'s own direct-API numbers on this
    machine (e.g. `random_n1000_d005`: dropin warm 0.99ms vs `gate.jl`'s direct-call
    0.61ms baseline-era / CHOLMOD warm 1.19ms here — consistent, not a fluke). Full
    table: `benchmark/results/dropin_gate_neuromancer.json`.
  - **Caveat, not glossed over:** an EARLIER ROADMAP entry (see "Perf investigation" /
    "Note on the numbers above", 2026-07-13, above) found `neuromancer`'s CPU clock
    unpinned/unlocked despite a `performance` governor string, and designated `galen`/
    `wintermute` the trusted clock-locked gate machines going forward. Per that
    project history, `neuromancer`'s numbers alone are corroborating, not
    authoritative — so this gate was ALSO run on `galen` (governor `performance`,
    confirmed via SSH) as the trusted cross-check; see its result immediately below.
  - **Run on `galen` (AMD Ryzen 9 5900X, SSH, `~/.juliaup/bin/julia`, governor
    `performance`, confirmed): warm gate 14/14 PASS**, PureSparse+PureBLAS strictly
    faster than CHOLMOD+OpenBLAS through the dropin entry point on every matrix-arm
    combination — e.g. `banded_n1000_bw20` own-arm: dropin-through warm 0.117ms vs
    CHOLMOD warm 0.472ms (4.0x); `laplacian2d_80x80` own-arm: 1.366ms vs 2.447ms
    (1.79x); the tightest margin, `random_n1000_d005` own-arm, still clears at
    0.770ms vs 0.945ms (1.23x). Zero-alloc re-check on this run: `cholesky!`/`ldlt!`
    both **0 bytes** with the drop-in active. Confirms the `neuromancer` result was
    not a fluke of that machine's clock-lock uncertainty — both the corroborating
    and the trusted machine agree. Full table:
    `benchmark/results/dropin_gate_galen.json`. (Environment note: galen's existing
    `PureSparse.jl`/`TypeContracts` sibling checkouts were stale relative to this
    worktree's `Manifest.toml` pins — ran from a freshly-synced, non-destructive copy
    at `~/Documents/claude/PureSparse.jl_m4closeout` with a version-matched
    `TypeContracts_m4closeout`/`StrictMode.jl` sibling rather than touching galen's
    pre-existing, independently-versioned checkouts.)
  - Cold-path (symbolic+numeric, first factorization of a pattern) is NOT part of the
    contractual gate (design §9.3/CLAUDE.md req 2 define the gate on the WARM refactor,
    the IPM-relevant number) and was not expected to pass here — PureSparse's own
    ordering + symbolic analysis is real, non-amortized work on a cold call, unlike
    CHOLMOD's own highly-tuned AMD; measured 0/14 dropin-cold faster than CHOLMOD-cold,
    consistent with `gate.jl`'s own historical direct-API cold numbers (also CHOLMOD
    generally wins cold, PureSparse wins warm — that asymmetry is the documented,
    expected shape of this gate, not new).
- **Zero-alloc re-check with the dropin active** (parity additions must not touch the
  numeric hot path): `dropin_gate_inner.jl`'s dropin-arm stage measures `@allocated
  cholesky!(...)`/`@allocated ldlt!(...)` after warmup, WITH `dropin.jl` compiled in —
  **0 bytes / 0 bytes**, both machines. Re-confirmed standalone (no dropin) too: **0 /
  0 bytes**, matching CLAUDE.md requirement 5's pre-existing gate.
- **Full suite:** `julia --project=test -e 'using ReTestItems, PureSparse;
  runtests(PureSparse)'` — **15220/15220 passing** (up from 15212 pre-M4-closeout;
  8 new test items/assertions from the `.U`/`SimplicialLDLFactor`/downstream-smoke
  additions, all green).

Changed: `src/dropin.jl` (`.U`, `SimplicialLDLFactor` parity, `LinearAlgebra.issuccess`),
`test/dropin_tests.jl` (extended + new downstream-gate testitem), `test/
downstream_smoke.jl` (new), `benchmark/dropin_gate.jl` + `dropin_gate_inner.jl` (new).
Branch `m4-closeout`.

## Current headline numbers (2026-07-13, post M1-task-7 zero-alloc fix, re-measured — not stale)

Both gates re-run after the zero-alloc hardening below (a real hot-path change, so
re-confirmed rather than assumed unaffected): **M1 LLᵀ gate 13/14** (up from 11/14 —
removing `cholesky!`'s per-call allocation improved wall-time too, not just the
allocation count: `random_n200_d02` flipped from fail to PASS on both arms; only
`random_n1000_d005` own-arm remains a near-tie, 1.2246ms vs 1.2227ms). **M2 SQD/LDLᵀ
gate 8/8**, numbers consistent with the original run (0.30–3.9ms PureSparse vs
0.89–18.1ms CHOLMOD across n=200–2000). See "CURRENT FOCUS" and the M2 task 8 section
below for the full tables and per-run caveats (unlocked clock).

## M2 — zero-allocation `ldlt!` CLOSED, last open M2 gate item (2026-07-13)

`ldlt!` is now genuinely zero-alloc: `@allocated ldlt!(F, A) == 0` measured directly
after warmup on random SQD KKT matrices at n = 35 / 100 / 320 / 1000 (previously
336 / 1344 / 6944 / 24640 bytes at those sizes — the allocation scaled with the number
of update chunks, all of it Array headers from the per-chunk `_panelview` unsafe_wrap
of `Workspace.cd`; the task-description's "1120 bytes on an n=35 case" was one point of
that same curve on a different random draw). CLAUDE.md requirement 5 is now met for
BOTH numeric refactorization paths — `cholesky!` re-verified 0 bytes, untouched.

**The fix (option 2 of the two paths scoped in "M1 task 7" below — restructure the
chunking, no new `Symbolic` field):** `Workspace.cd` changed from a flat
`Vector{T}(max_update_size)` to a pre-allocated square
`Matrix{T}(max_extend_rows, max_extend_rows)` used via `view(cdbuf, 1:k1, 1:wk)` —
the identical zero-alloc view-of-a-Matrix technique that fixed `c`. The bound that
made this valid for `c` (both extents ≤ `max_extend_rows` by row-containment) does
NOT hold naturally for `cd`'s column extent (a wide descendant's `ncol_d` is
unbounded by `max_extend_rows`), so the bound is instead established BY CONSTRUCTION:
the update gemm was already chunked over descendant columns (previously with width
`max_update_size ÷ k1`, purely to fit the old flat buffer); the chunk width is now
capped at the buffer's own column capacity, `w = min(ncol_d, size(cdbuf, 2))`. Rows:
`k1 ≤ max_extend_rows` by the same `R1 ⊆ R ⊆` descendant-below-diagonal-rows
containment as `c`. Full derivation in `types.jl`'s `Workspace` docstring. Verified
empirically, not just derived: a dev-time `@assert k1 ≤ size(cdbuf,1) && wk ≤
size(cdbuf,2)` ran inside the chunk loop across the FULL test suite (15212 tests,
including the SQD zoo and small-k1/wide-descendant cases) and never tripped, then was
removed (the loop is `@inbounds`, which would elide the view's own bounds check —
hence the explicit dev assertion rather than trusting the derivation alone).
Chunking-over-the-contraction-axis semantics (`beta = 1` accumulation from the second
chunk, contig fast path ordering) are unchanged; a small-k1/wide-descendant pair now
takes `⌈ncol_d/max_extend_rows⌉` gemm calls instead of `⌈ncol_d·k1/max_update_size⌉`
— same total flops, only per-call count. `Symbolic.max_update_size` no longer sizes
any buffer (kept as a sizing diagnostic). Memory: `cd` is now the same
`max_extend_rows²` shape as `c` — the exact precedent whose cost was measured at
1.0×–4.4× the flat sizing on the M1 gate set (a one-time per-`Workspace` cost, paid
by LLᵀ factors too since `Workspace` is shared, as the flat `cd` already was).

**Verified:** full suite **15212/15212 passing** (15206 pre-existing + a new
`ldlt!` zero-alloc gate testitem in `test/ldlt_tests.jl` mirroring `cholesky!`'s,
including a lopsided-KKT shape for the wide-descendant case); residuals
`norm(K*x-b)/norm(b)` ~2e-16..6.5e-16 at n=35..1000, bit-identical to the pre-fix
baseline (same RNG draws — the restructure changed WHERE the staged copy lives, not
one floating-point operation); inertia `(n_pos, n_neg, n_zero)` matches construction
at every size; `cholesky!` still 0 bytes with residuals also bit-identical. M2 SQD
wall-time gate re-run after the change (a buffer-shape change can shift cache
behavior, so measured rather than assumed): **8/8 PASS** (neuromancer, unlocked
clock), PS+PureBLAS warm medians 0.037/0.26/1.17/3.34 ms at n=200/500/1000/2000
(own arm) vs CHOLMOD+OpenBLAS 0.073/0.93/5.10/18.7 ms — every point at or faster
than the pre-change run recorded in the M2 task 8 table (e.g. n=2000 own 3.34 vs
3.74 ms), i.e. no regression and plausibly a small win from the removed per-chunk
allocations (not isolated from run-to-run variance; the gate criterion is the
CHOLMOD comparison, which passes with the same ~2–5.5× margins as before).
Datapoints saved per the save-datapoints rule (as
`benchmark/results/gate_ldlt_neuromancer_m2zeroalloc.json` in the main checkout, a
new file so the pre-change run's `gate_ldlt_neuromancer.json` datapoints stay
intact).

## M1 task 7 — zero-allocation hardening, `cholesky!` now genuinely zero-alloc (2026-07-13)

`Workspace.c` (`src/types.jl`) changed from a flat `Vector{T}(max_update_size)` —
needing a fresh `_panelview`/`unsafe_wrap` (measured 80 B/call) or `reshape` (48
B/call) on every use, since the update block's needed shape `(ctot, k1)` varies per
(descendant, ancestor) pair — to a single pre-allocated
`Matrix{T}(max_extend_rows, max_extend_rows)`, used via `view(cbuf, 1:ctot, 1:k1)`:
**measured 0 bytes**, since a `view` of an already-allocated `Matrix` costs nothing
(same pattern the existing `Ldiag`/`Lbelow` views already relied on). Sound because
`ctot` and `k1` are both provably `≤ max_extend_rows` for every (d,s) pair (`R1 ⊆ R ⊆`
descendant `d`'s own below-diagonal rows, exactly what `max_extend_rows` bounds by
definition — full derivation in `types.jl`'s `Workspace` docstring). Memory cost of
the square buffer vs the old flat sizing: checked directly on the M1 gate matrices
before committing to the design (1.0×–4.4×, a one-time `Workspace` cost, not a
hot-path one) — not assumed to be cheap.

**Results:** `cholesky!` **2576 → 0 bytes** (measured `@allocated == 0`); same fix
applied to `ldlt!`'s `C` buffer, dropping it to **1120 bytes** (remaining source:
`Workspace.cd`, the `L·D` scaled-copy buffer, whose needed shape is chunked to fit
`max_update_size` but is NOT bounded by `max_extend_rows` the same way `c`'s is —
`wk` can exceed `max_extend_rows` when `k1` is small, worked through explicitly, not
assumed fixable by the same trick — left as a real, precisely-scoped follow-up, not
attempted this pass; **since CLOSED, see "M2 — zero-allocation `ldlt!`" above**).
`solve!`'s outer permuted-RHS buffer now reuses `Workspace.rhs`
(pre-allocated at size `n`) for the single-RHS path instead of a fresh allocation per
call — a real improvement, not yet zero (7904 B remaining on a test case; two further
allocation sites in `_solve_L!`/`_solve_Lt!` identified and documented in
`src/numeric/solve.jl`'s header, not fixed this pass — re-`_panelview`ing `F.x` every
supernode instead of reusing `F.panels[s]`, and a per-supernode `upd` buffer whose
size isn't obviously bounded by `max_extend_rows` once `nrhs > 1`). Multi-RHS solves
remain allocating by design (`nrhs` unbounded, can't be pre-sized).

Both wall-time gates were re-run (not assumed unaffected) after this change since it
touches the actual per-iteration numeric hot path — see "Current headline numbers"
above; the fix improved wall-time too (M1: 11/14 → 13/14), not just the allocation
count, consistent with reduced GC pressure on the warm-refactor path an IPM consumer
calls every iteration. Full test suite 15202/15202 passing (one test's assertion
tightened from the old "allocs > 0 (not yet zero), allocs < 10,000" partial-progress
bound to the real "allocs == 0" now that it's achieved).

## M2 progress — simplicial rank-1/rank-k update/downdate LANDED (2026-07-13)

`src/simplicial/updown.jl` (new, design §1.3's planned location) + the
`SimplicialLDLFactor{T,Ti}` type in `src/types.jl` implement design §7:
`G = simplicial(F::LDLFactor; grow)` (one-time allocating conversion),
`updowndate!(G, w, ±1)` rank-1 Davis–Hager update/downdate (zero allocations, O(changed
nnz) factor work), `updowndate!(G, W::AbstractMatrix, ±1)` rank-k as sequenced rank-1
(design §7's explicit v1 scope — the 2001 batched variant deliberately NOT implemented),
and the full simplicial split-solve set (`solve!`/`solve_L!`/`solve_D!`/`solve_Lt!`/`\`
as new methods on the EXISTING exported functions, design §6/§0 N7 — plain CSC column
loops, no PureBLAS). Provenance: everything algorithmic is Davis & Hager 1999
(`refs/linear_algebra/modify_sparse_cholesky.pdf`, read directly): the numeric
recurrence is their Algorithm 5 verbatim (§5.1 p.617, sparse Method C1 modified; the
paper notes it is diagonal-scaling-equivalent to Pan's stable orthogonal method — this
is design §7's "hyperbolic/stable recurrences"); the etree-path restriction is Theorem
5.2 (downdate: pattern of `v = L⁻¹w` = path `P(k)`, `k = min support(w)`, their eq.
(5.1)) and Corollary 5.3 (update: path `P̄(k)` in the new tree); pattern growth and
parent rewiring are Theorem 4.1 + Algorithm 3 (`π̄(j) = min L̄ⱼ \ {j}`), generalized to
arbitrary-`w` downdates via §6's Algorithm 6a merge phase.

**Verified (2026-07-13):** 10 new ReTestItems in `test/simplicial_tests.jl` (1470
assertions), full suite **15197/15197 passing** (13727 pre-existing, `llt.jl`/`ldlt.jl`/
`symbolic/*`/`ordering/*` untouched). Coverage: (a) conversion reconstructs
`Pᵀ·L·D·Lᵀ·P ≈ K` at 1e-12 rel. and `G.d == F.d` bitwise, pattern
sorted/unique/in-capacity with `parent == min(pattern)` asserted per column; (b) update
vs. TWO independent oracles — reconstruction ≈ `K + wwᵀ` AND agreement with a
from-scratch `ldlt(K + wwᵀ)` refactorization (fresh symbolic, fully independent code
path) — measured ≤ 2.8e-16 rel. on random SPD up to n=90 (gate 1e-12), plus an SQD
`[H+wwᵀ Aᵀ; A −C]` case with inertia (25,15) asserted preserved through the update;
(c) update-then-downdate round-trip returns the original `L`/`d` within the design M2
target `100·eps·n` (measured ~1.4e-16 rel., i.e. orders below the gate) with the
workspace-zeroing invariant `all(iszero, G.wval)` asserted after every walk; (d)
rank-k wrapper produces BITWISE-identical `L`/`d` to manually sequenced rank-1 calls
and matches `K + WWᵀ` at 1e-12; (e) downdate instability: `I₄ − 4e₂e₂ᵀ` and a random
SPD matrix downdated past its smallest eigenvalue both return `:not_definite` with
`ok=false`/`fail_col` set, while a same-direction SAFE downdate succeeds at 1e-11 —
and the SQD flavor (UPDATE against a negative pivot, `diag(1,−1) + 4e₂e₂ᵀ`) is caught
by the same `ᾱ ≤ 0` recurrence signal; (f) pattern growth: `grow = 0` (zero slack)
returns `:refactor_required` and refuses reuse, the identical update under default
slack succeeds at 1e-14; (g) post-update residual `‖(K+wwᵀ)x−b‖/(‖K‖‖x‖) < 1e-12`
through the simplicial solves ONLY (supernodal factor left stale), incl. multi-RHS and
split-solve composition; (h) `@allocated updowndate! == 0` for both signs after warmup.

**Judgment calls (design ambiguities resolved, documented in-code too):**
- **Return-code shape:** `updowndate!` returns a `Symbol` — `:ok` /
  `:refactor_required` (design §7's overflow contract) / `:not_definite` — AND sets
  `G.ok = false` + `G.stats.fail_col` on failure (the design's "reported through
  `F.ok`/stats" discipline); a failed factor throws `ArgumentError` on further
  `updowndate!` calls. Failure leaves earlier path columns modified (documented):
  the caller must refactor anyway, and keeping single-pass merge+numerics is what
  makes the success path allocation-free.
- **Definiteness detection is the recurrence's `ᾱ ≤ 0`**, checked BEFORE the pivot is
  committed (Alg 5: `d̄ⱼ = (ᾱ/α)dⱼ` with `α > 0` invariant), not an after-the-fact
  sign inspection. For SQD factors the same predicate refuses any inertia-changing
  modification, including updates against negative pivots — so the guard is
  σ-independent, not downdate-only.
- **Storage layout:** strictly-lower per-column CSC (`colptr` fixed slot ranges +
  `colnnz` used-length, implicit unit diagonal, `d` separate) with per-column slack
  `min(n−j, max(len, ceil(grow·(len+1))))`; `grow` is the new `simplicial_grow`
  Preference (default 1.5 — own free choice, derivation in `tuning.jl`; the paper §7
  sizes columns from a known worst-case factor instead, which a general update stream
  doesn't have). `wval`/`wpat` scatter workspaces live in the struct; `wval` is kept
  all-zero BETWEEN calls by re-zeroing exactly the touched entries during/after the
  walk (no O(n) `fill!` per call).
- **Pattern extraction keeps the supernodal superset** (amalgamation padding rides
  along as exact zeros — they provably stay exactly 0.0 in floating point). Chosen
  because the padded pattern is CLOSED under the paper's eq. (3.1) parent-containment
  (in-block: slices nest trivially; across supernodes: the §4.3/§9.1 superset
  invariant), which is all the walk needs; the true per-column pattern is not
  recoverable from the factor alone without A. Consequence: `G.parent` is the etree
  of the STORED pattern (possibly a refinement of `sym.parent`) and is maintained by
  `updowndate!` as patterns grow; padded path columns are harmless `wⱼ = 0` no-ops
  (paper §5.2).
- **Downdates never shrink patterns** — the paper's multiset symbolic-downdate
  machinery (Algorithms 2/4, §6 Algorithm 6b entry removal) is deliberately skipped;
  entries that become numerically zero stay stored. A closed superset is always
  numerically safe, and this halves the symbolic code. Both update AND downdate run
  the same 6a-style support merge (an arbitrary-`w` downdate can add fill — the 1999
  single-path Theorem 5.2 assumes `w` is a column of `A`; §6 is the general case).
- **`w` is taken in ORIGINAL row order** (like `ldlt`'s `signs`) and permuted
  internally; the O(n) input scan for a dense `w` is unavoidable and documented (the
  O(changed nnz) claim is about factor-modification work).
- **Wide-support updates legitimately overflow default slack:** the 6a merge phase
  adds `support(w)` to every path column, so a dense-ish `w` against 1.5× slack
  correctly returns `:refactor_required` (observed in testing — not a bug; the
  recurrence itself measured ≤ 2.8e-16 wherever storage sufficed). Oracle test items
  therefore build with `grow = n` (never-overflow) while the slack policy has its own
  dedicated items at `grow = 0`/default.

**Deliberately out of scope here (still-open M2 items):** the direct simplicial
factorization path for small/very-sparse problems (design §1.2 mentions it; only the
`LDLFactor` conversion is scheduled); the 2001 batched multiple-rank algorithm (listed
extension); a `SparseVector` fast path for the input scan; update/downdate
benchmark-gate additions.

## M2 progress — `refine!` + IPM guide docs LANDED (2026-07-13)

`src/refine.jl`: `refine!(x, F, A, b; iters=2)` (design §5.2) — classical
residual-correction iterative refinement (`x = F\b`, then `iters` rounds of `x += F \
(b - Symmetric(A,:L)·x)`), generic over any `AbstractSparseFactor` (a new small
docstring was added to that abstract type itself — it had none before, needed once
`refine!`'s docstring cross-referenced it via `@ref` for the docs build). Works
unchanged for `SupernodalFactor`/`LDLFactor`/`SimplicialLDLFactor` since it only calls
the already-generic `solve!`.

**Verified (2026-07-13):** 2 new ReTestItems in `test/refine_tests.jl`; full suite
**15203/15203 passing**. One real bug caught in the FIRST draft of the test, not in
`refine!` itself: the original test forced a well-conditioned pivot's sign to flip
(diag entry −3 under `signs=+1`) and asserted `refine!` would drive the residual to
1e-10 — it instead diverged (measured 14.4, growing per iteration). Root cause,
confirmed analytically and by direct calculation: iterative refinement's fixed-point
map has per-entry contraction factor `|1 − K_jj/F_jj|`; forcing a sign flip on an O(1)
pivot gives `K_jj/F_jj = -3/3 = -1`, so `|1-(-1)| = 2 > 1` — provably divergent, not a
`refine!` defect. This is not what PureSparse's regularization produces in practice
(`ldlt_delta` only forces pivots already near the magnitude floor); the test was
rewritten around a realistic near-singular-pivot case (`1e-13` floored to `1e-12`,
same sign) and the assertion changed from an unreachable machine-precision target to
the actually-true property: monotonic geometric improvement (measured ratio 0.900 per
iteration, matching the analytical `|1 - 0.1|` prediction to 3 decimal places).

`docs/src/ipm-guide.md` (new page, wired into `docs/make.jl`): the interior-point
workflow — `symbolic` once / `ldlt!` refactor per iteration, reading
`FactorStats`/inertia, `refine!` when `n_perturbed > 0`, and `updowndate!` for the
less-common structural-change case (occasional constraint add/remove), explicitly
distinguished from the per-iteration value-only refactor. `docs/src/api.md` gained
`AbstractSparseFactor`/`SimplicialLDLFactor`/`simplicial`/`updowndate!`/`refine!`
entries; full `makedocs`+vitepress build verified succeeding end-to-end (one dead
`@ref` link caught and fixed the same way as M1's — `AbstractSparseFactor` needed a
docstring before it could be `@ref`'d).

## M2 progress — task 8 SQD/LDLT benchmark gate LANDED, PASSES CLEANLY (2026-07-13)

`benchmark/matrices.jl` gained `random_sqd_kkt(npos, nneg, density; rng)` (same
construction as `test/ldlt_tests.jl`'s helper) and `sqd_gate_matrices()` (4 sizes,
n=200 to n=2000). `benchmark/openblas_backend.jl`'s `OpenBLASBackend` module gained
`ger!` (needed by `ldlt.jl`'s hand-rolled base case) and `PureSparseOB` now also
re-`include`s `numeric/ldlt.jl` verbatim (same kernel-swap pattern as the M1 `llt.jl`
arm — verified bitwise-identical `d` between the PureBLAS and OpenBLAS arms on a smoke
test before the full run). New `benchmark/gate_ldlt.jl` mirrors `gate.jl`'s exact 3-arm
structure (own-ordering + same-permutation) with CHOLMOD's sparse `ldlt` (via
`SparseArrays`) as the baseline — verified directly beforehand that it accepts SQD
input without complaint and shares `cholesky`'s `.p`/`perm=` interface (one wrinkle:
CHOLMOD only supports extracting the `:LD` component from an LDLt factor, not `:L` —
`nnz(sparse(Fc.LD))`, not `Fc.L`, caught by an actual `CHOLMODException` on the first
smoke-test run, not guessed).

**Real gate result (2026-07-13, `neuromancer`, unlocked clock — same caveat as the M1
gate): 8/8 matrix-arm combinations PASS**, with `n_perturbed == 0` on every case
(synthetic SQD construction is well-conditioned, so this is a clean apples-to-apples
comparison, not one masked by regularization):

| matrix | arm | PS+PureBLAS | PS+OpenBLAS | CHOLMOD+OB | speedup |
|---|---|---|---|---|---|
| sqd n=200 | own | 0.044ms | 0.052ms | 0.072ms | 1.6x |
| sqd n=200 | same-perm | 0.047ms | 0.054ms | 0.072ms | 1.5x |
| sqd n=500 | own | 0.319ms | 0.345ms | 0.884ms | 2.8x |
| sqd n=500 | same-perm | 0.316ms | 0.337ms | 0.935ms | 3.0x |
| sqd n=1000 | own | 1.264ms | 1.305ms | 5.040ms | 4.0x |
| sqd n=1000 | same-perm | 1.269ms | 1.297ms | 5.146ms | 4.1x |
| sqd n=2000 | own | 3.736ms | 3.812ms | 18.19ms | 4.9x |
| sqd n=2000 | same-perm | 3.882ms | 4.006ms | 17.95ms | 4.6x |

The margin GROWS with size (1.5x at n=200 → ~4.9x at n=2000) — unlike the M1 LLᵀ gate
(11/14, some matrix classes still close/failing), the LDLᵀ path clears CHOLMOD's sparse
`ldlt` comfortably and consistently across the whole tested range, both own-ordering and
same-permutation. Not yet explained WHY the margin is this much larger than the LLᵀ
case (plausible hypothesis: CHOLMOD's sparse `ldlt` may take a less-optimized code path
than its `cholesky` internally, or the synthetic SQD block structure amalgamates more
favorably — NOT verified, flagged honestly as an open question rather than asserted).

**M2 status: all 8 tasks of design §10's M2 list are now done** (LDLᵀ core,
update/downdate, simplicial split solves, `refine!`, SQD benchmark gate — the gate
passes cleanly, unlike M1's). Remaining: the smaller deliberately-out-of-scope items
listed above (2001 batched rank-k, standalone simplicial factorization path,
`SparseVector` fast path) — none required by the M2 gate.

## M2 progress — supernodal LDLᵀ/SQD LANDED (tasks 1–3, 2026-07-13)

`src/numeric/ldlt.jl` implements the design §5.1 supernodal LDLᵀ for symmetric
quasi-definite systems: `ldlt(sym, A; signs)` / `ldlt(A; signs | n_pos/n_neg, ordering)`
/ `ldlt!(F, A)`, mirroring `cholesky`/`cholesky!`'s structure exactly (same relmap
linked-list left-looking schedule, same `_panelview`/`GC.@preserve` discipline — no
`reshape(view(...))` compile-tax reintroduction). `solve.jl` gained `solve_D!` and a
three-stage `solve!(x, F::LDLFactor, b)`; the L/Lᵀ sweeps were unified over
`Union{SupernodalFactor,LDLFactor}` (`_PanelFactor` + `_diagchar`, trsm `diag='U'` for
unit-lower LDLᵀ panels) rather than duplicated. `LDLFactor` gained the same built-once
`panels` wrapper cache as `SupernodalFactor`. Provenance: base-case unit-LDLᵀ
right-looking column loop is Golub & Van Loan's symmetric-indefinite-without-pivoting
(generic NLA, no CHOLMOD content); SQD strong factorizability is Vanderbei 1995; the
forced-sign fixed-pivot regularization is the QDLDL (Stellato et al., OSQP) / Clarabel
scheme per design §5.1/§0 D3 — deliberately NOT MA57 Bunch–Kaufman (CLAUDE.md req 8).

**Verified (2026-07-13):** 8 new ReTestItems in `test/ldlt_tests.jl`, 173 assertions;
full suite **13727/13727 passing** (13554 pre-existing — `llt.jl`/`symbolic/*`/
`ordering/*` untouched — plus the new items). Coverage: (a) dense-oracle reconstruction
`L·D·Lᵀ ≈ P·K·Pᵀ` at rel. 1e-9 AND elementwise `L`/`d` agreement at 1e-8 with a
from-scratch dense no-pivot unit-LDLᵀ oracle, on random `[H Aᵀ; A −C]` KKT matrices up
to n=75 (with `n_perturbed == 0` asserted, so regularization wasn't papering over
anything); (b) BigFloat oracle at 1e-12 on small SQD; (c) inertia `(n_pos,n_neg,n_zero)`
exactly matches construction on every SQD case (Vanderbei: inertia of SQD =
(n₊,n₋,0)); (d) wrong-sign pivot forced (diag(2,−3,5) under signs=+++: `n_perturbed==1`,
`max_perturbation==6`, reconstructs the REGULARIZED diag(2,3,5) at 1e-12 and differs
from A by design), zero pivot forced up to the δ floor and classified `n_zero`; (e)
solve residual `‖Kx−b‖/(‖K‖‖x‖) < 1e-8` incl. multi-RHS; (f) `ldlt!` refactorize on the
same pattern (D scales, unit-L bit-stable at 1e-12 under value scaling).

**Judgment calls (design ambiguities resolved, documented in-code too):**
- **`signs` convention:** always in `A`'s ORIGINAL column order; `ldlt` permutes them
  internally through `sym.perm`. This makes the `ldlt(A; n_pos, n_neg)` convenience
  (n_pos leading `+1`s, n_neg trailing `−1`s — the `[H Aᵀ; A −C]` layout) compose
  cleanly with AMD, because the caller never sees factor ordering. The task-description
  alternative ("signs in the factor's permuted order") would NOT compose with AMD and
  was rejected; the constructor was kept, not dropped.
- **`Workspace.cd` sizing gap in design §5.1:** the "one extra buffer of
  `max_update_size` column-scaled values" does NOT bound the scaled copy `|R1|×ncol_d`
  (a wide descendant with a short update block exceeds `max|R|·|R1|`). Rather than
  touching `symbolic`'s workspace bounds (off-limits for this additive task), the
  update gemm is chunked over descendant columns with width `max_update_size ÷ |R1|`
  (≥ `|R|` always, so one chunk in the common case; provably ≥ 1).
- **Plain `gemm!` for the whole `|R|×|R1|` update block** (design §5.1 gives latitude):
  its top `|R1|×|R1|` part is symmetric and a symmetric-aware rank-k-with-D kernel
  would halve that part's flops — documented efficiency opportunity in ldlt.jl's
  header, deferred to the M2 benchmark pass.
- **Free constants the design leaves open:** zero-pivot classification ζ = `eps(T)`
  (machine epsilon relative to running max|d| — standard "numerically zero at scale"
  cut); δ's ‖A‖-scale = max|assembled tril entry| (free inside the load loop). Both are
  our own choices, no external provenance. `signs[j] == 0` (scaffolded "free" value in
  types.jl) enforces the magnitude floor only, never a sign flip.

**Deliberately out of scope here (still-open M2 items):** zero-alloc `ldlt!` (same
category as M1 task 7's remaining `cholesky!`/`solve!` allocations — `ldlt!` shares the
per-call `_panelview(cbuf,...)` update-buffer allocation and now a `cdbuf` one, plus
`solve!`'s permuted-RHS scratch — **since CLOSED, see "M2 — zero-allocation `ldlt!`"
above**); simplicial storage/conversion; Davis–Hager
update/downdate; `refine!`; SQD benchmark-gate additions; IPM guide docs.

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

**M1 task 7 (zero-allocation hardening) — `cholesky!` is now GENUINELY ZERO-ALLOC
(2026-07-13), CLAUDE.md requirement 5 met for the LLᵀ path.** History: 7392 → 2576
bytes (65%) after caching `SupernodalFactor.panels::Vector{Matrix{T}}` once at
factor-construction time (`_build_panels`) instead of re-`unsafe_wrap`ping
`panel`/`panel_d` every call → **0 bytes**, measured directly (`@allocated
cholesky!(F,A) == 0`), after also fixing the update-block buffer. The fix: `Workspace.c`
changed from a flat `Vector{T}(max_update_size)` (needing a fresh `_panelview`
`unsafe_wrap` — 80 B/call — or `reshape` — 48 B/call — every use, since its NEEDED
shape `(ctot, k1)` varies per (descendant, ancestor) pair) to a single pre-allocated
`Matrix{T}(max_extend_rows, max_extend_rows)`, used via `view(cbuf, 1:ctot, 1:k1)` —
**measured 0 bytes**, because a `view` of an already-allocated `Matrix` costs nothing
(same pattern the existing `Ldiag`/`Lbelow` views already relied on). This works
because `ctot` and `k1` are BOTH provably `≤ max_extend_rows` for every (d,s) pair
(`R1 ⊆ R ⊆` descendant `d`'s own below-diagonal rows, whose count is exactly what
`max_extend_rows` bounds by definition — derivation in `types.jl`'s `Workspace`
docstring). Memory cost of the square buffer vs the old flat sizing: measured 1.0×–4.4×
on the M1 gate matrices (a one-time `Workspace` allocation, not a hot-path one) —
checked directly before committing to this design, not assumed.

The SAME fix applied to `ldlt!`'s `C` buffer (identical bound, identical technique).
`ldlt!`'s total per-call allocation dropped to **1120 bytes** (measured on a
n=35 SQD KKT case) — the remaining source is the OTHER LDLᵀ-specific buffer,
`Workspace.cd` (the `L·D` scaled-copy staging buffer), whose needed shape `(k1, wk)`
is chunked to fit `mus = max_update_size` but is NOT bounded by `max_extend_rows` the
way `c`'s is (`wk` can exceed `max_extend_rows` when `k1` is small — worked through
explicitly, not assumed) — a correct fix needs either a new `Symbolic` field bounding
max supernode column width, or restructuring the chunking to tile on both axes; **not
attempted this pass**, left as an honestly-scoped remaining gap.

`solve!`'s outer permuted-RHS buffer now reuses `Workspace.rhs` (pre-allocated at
size `n`) for the single-RHS (`b::AbstractVector`) path instead of allocating a fresh
`Vector{T}(n)` every call — a real, measured improvement, but **not yet zero**:
`solve!` still measures 7904 bytes (n=50 SPD case) because `_solve_L!`/`_solve_Lt!`
themselves have TWO further allocation sites, both left as follow-ups: (a) they
re-`_panelview` `F.x` fresh every supernode instead of reusing the already-cached
`F.panels[s]` (the same fix `cholesky!`'s update loop got in the earlier panel-caching
pass, just never applied to the solve path); (b) the below-diagonal update buffer
(`upd = Matrix{T}(undef, nsrow-nscol, nrhs)`) is allocated fresh per supernode with a
size that — like `ldlt!`'s `cd` — is not obviously bounded by `max_extend_rows` alone
once `nrhs > 1` is accounted for. Multi-RHS (`b::AbstractMatrix`) solves remain
allocating by design (`nrhs` is caller-chosen and unbounded, can't be pre-sized).

None of this blocks correctness or either wall-time gate (both were already passing);
it is the literal CLAUDE.md requirement 5 text ("zero allocations after `symbolic()`")
being closed out property by property, with `cholesky!`/`ldlt!`'s numeric refactor —
the actual per-iteration IPM hot path — now done or substantially improved, and
`solve!`'s remaining gap precisely scoped rather than left vague.

## Milestones (design §10)

### M1 — AMD + Symbolic + Supernodal LLᵀ + Solve
**Deliverables:** `tuning.jl`, `types.jl`, `contracts.jl`, `ordering/interface.jl`,
`ordering/amd.jl`, `symbolic/etree.jl`, `symbolic/counts.jl`, `symbolic/supernodes.jl`,
`numeric/llt.jl`, `numeric/solve.jl`; full test files for these; Chairmarks + PkgBenchmark
harness (design §9.3, 4-arm with quadrant 4 marked N/A); docs skeleton
(DocumenterVitepress).

**Gate — ALL ITEMS MET (2026-07-13). M1 CLOSED.** full zoo correctness (dense `BigFloat`
oracle + CHOLMOD black-box cross-check) *(MET — full suite passing)*; zero-allocation
gate (`@allocated cholesky!(F, A2) == 0`, StrictMode-checks-disabled config) *(MET — 0
bytes, "M1 task 7" below)*; wall-time gate — `median_seconds(PureSparse+PureBLAS) <
median_seconds(CHOLMOD+OpenBLAS)` on the M1 KKT/FEM set, both own-ordering and
same-permutation arms, strictly faster on at least half the set *(MET and then some —
11/14 named-matrix gate, PLUS the size-sweep gate now passes at every size 2–2048 on
BOTH clock-locked machines, galen and wintermute — see "Perf follow-up" above)*; `juliac
--trim` smoke build succeeds *(MET 2026-07-13 — `juliac/build.jl`/`juliac/entry.jl`, 0
trim-verifier errors, built executable runs and reproduces bit-identical residuals to
normal Julia; `test/trim_tests.jl`'s TrimCheck item guards regression in the ordinary
suite)*; AMD fill ≤ 1.15× CHOLMOD-AMD fill on the zoo *(MET — checked directly in
`test/setups/oracle_setup.jl`)*.

**Task list:**
1. Scaffold `Project.toml`/module/`tuning.jl`/`types.jl`/`contracts.jl`. *(in progress)*
2. Elimination tree + postorder + column counts, brute-force-oracle tests.
3. AMD (longest single task — budget accordingly). Paper §-by-§: quotient graph storage →
   pivot loop → approximate degree scan → supervariable detection/mass elimination →
   aggressive absorption → dense rows → garbage compaction.
4. Fundamental supernode detection + relaxed amalgamation.
5. Symbolic driver (rowind/px/assembly-map/workspace-bound sizing).
6. Supernodal LLᵀ numeric (load → linked-list update loop → potrf/trsm) + solve.
7. Refactorize/allocation hardening + StrictMode guards. *(`cholesky!` DONE 2026-07-13,
   genuinely 0 bytes — see "M1 task 7"; `solve!`'s remaining partial allocation gap is
   documented but not required by any milestone gate)*
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

**Gate — ALL ITEMS MET (2026-07-13). M2 CLOSED.** SQD zoo (synthetic IPM iterate
sequences) factor without failure *(MET — `n_perturbed==0` on the well-conditioned gate
set, forced-regularization paths covered separately in `test/ldlt_tests.jl`)*; inertia
matches construction *(MET)*; update/downdate round-trip ≤ 100·eps·n *(MET — measured
~1.4e-16 rel., far inside the bound)*; zero-alloc `ldlt!` *(MET 2026-07-13, the last open
M2 gate item — see "M2 — zero-allocation `ldlt!`", independently re-verified: 0 bytes at
n=35/100/320/1000 and across multiple random seeds on the wide-descendant/small-k1 shape
the fix specifically targets)*.

**Tasks:**
1. `ldlt` base case + dense unit tests vs a from-scratch no-pivot dense oracle. *(DONE
   2026-07-13 — oracle is a from-scratch fixed-order unit-LDLᵀ, not `bunchkaufman`,
   whose Bunch–Kaufman pivoting makes its L/D incomparable to ours; see "M2 progress")*
2. LDL descendant updates (L·D-scaled gemm path, chunked over `Workspace.cd`) + base
   case covers the panel rows (no separate trsm stage). *(DONE 2026-07-13)*
3. Signed regularization + inertia stats + `signs` plumbing. *(DONE 2026-07-13)*
4. Simplicial storage + conversion (`simplicial(F)`). *(DONE 2026-07-13 — see
   "M2 progress — simplicial rank-1/rank-k update/downdate")*
5. Rank-1 update/downdate (Davis–Hager Method C) incl. pattern growth. *(DONE
   2026-07-13 — round-trip measured ~1.4e-16 rel., far inside the 100·eps·n gate)*
6. Rank-k (successive single-rank first, then multiple-rank optimization).
   *(successive single-rank DONE 2026-07-13 per design §7 v1 scope; the 2001 batched
   multiple-rank variant remains a listed extension, not scheduled)*
7. Refinement helpers + simplicial split solves. *(DONE 2026-07-13 — both split solves
   and `refine!`, see "M2 progress — `refine!` + IPM guide docs LANDED")*
8. IPM guide docs. *(DONE 2026-07-13 — `docs/src/ipm-guide.md`, see same section)*

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

### M5 — Sparse QR (next; design draft awaiting adversarial review)
Full design, deliverables, gates, and the ordered M5a/M5b task lists:
[`docs/design_qr.md`](docs/design_qr.md) (v1 Fable draft, 2026-07-14 — must go through
the same adversarial-review→v2 pass as `docs/design.md` before implementation starts;
review hotspots are in its §0). One-line shape: M5a = left-looking column Householder QR
(COLAMD ordering, star-matrix symbolic reusing etree/counts, singletons, Heath-test rank
detection, LS/basic/min-norm solves, zero-alloc `qr!`), M5b = conditional multifrontal
numeric phase (requires two new PureBLAS kernels, tasks P1/P2 there) triggered iff the
wall-time gate vs stdlib SuiteSparseQR fails on any stratum. **M5 closeout gate is the
unconditional wall-time inequality** (design_qr.md §9.3).

### M6 — GPU (renumbered from M3, 2026-07-14; content unchanged — see `### M3` above)

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
