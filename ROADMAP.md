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

**2026-07-16: M5 = sparse QR — MILESTONE CLOSED.** The non-negotiable wall-time
closeout gate (design_qr.md §9.3, CLAUDE.md req 2) is MET at 16/16 on both clock-locked
hosts (neuromancer + galen), warm PS `qr!` vs SPQR cold under D13, own-ordering AND
same-permutation arms. Both milestone-vs-gate remainders named in the gate entry below
are now resolved: the 7000×4000 flagship was re-measured on the corrected factorization
(parity with faer — the honest ceiling, not the withdrawn "2–6× win" bug artifact), and
this final design §10 checklist pass verified every deliverable against literal wording:
all six `src/qr` + `src/ordering` files, §9.1 layers (BigFloat oracle, SPQR black-box
agreement, H1/H2 executable invariants, zero-alloc, trim smoke), the §2.2 ordering-quality
bound (real test `colamd_tests.jl:144`, measured nnz(R) 1.002× ≤ 1.15× stdlib), docs
(§1.2 method-selection guidance + §5 `dropped_norm` honesty), drop-in forwarding, and all
13 M5a tasks + M5b P1/P2 + tasks 14–17. `solve_minnorm!` is rank-guarded (throws on
`n_dead>0`) AND zero-alloc (workspace `n1a`/`n1b`/`rblk` scratch, `@allocated==0` test)
— closed at task 10, re-verified this pass (413/413). Out of scope, filed as post-M5
follow-ups: #54 (ComplexF64 frontal, conjugate Householder) and #55 (BigFloat/non-isbits
frontal) — design_qr_m5b.md §A7.3 deliberately scoped generic-`T` to real isbits, so
neither blocks M5. **Next milestone: M6 (GPU).**

**2026-07-14: M5 = sparse QR (design CLOSED, implementation
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

**2026-07-15 later update: task 16e first lever CLOSED (allocation elimination) +
task 16d's `:auto` is now genuinely calibrated, not a placeholder.** Read faer
0.24.1's actual sparse QR source (`sparse/linalg/qr.rs`, MIT — freely readable per
CLAUDE.md req 1's SuiteSparse-only prohibition) via a research agent; found three
concrete techniques: (1) a `flops/nnz`-ratio simplicial-vs-supernodal dispatch
(threshold 40.0, `QR_SUPERNODAL_RATIO_FACTOR`), (2) a graded 4-tier amalgamation
schedule `(4,1.0),(16,0.8),(48,0.1),(∞,0.05)` shared across faer's QR/Cholesky/LU,
(3) per-PANEL (not per-front) block-size re-derivation as the staircase narrows.
Acted on (1) and a related allocation cleanup:

- **Zero-alloc pass** (`src/qr/frontal{,_assemble,_numeric,_solve}.jl`): moved every
  per-front scratch array (assembly's `push!`-grown gather buffers, factorization's
  `elim_col`/`piv_k`/`piv_rlo`, solve's `cc` and `apply_Q!`'s panel-boundary arrays)
  into `QRFrontWorkspace`, preallocated once from the symbolic's capacity bounds.
  `qr!`/`solve!`/`apply_Q!`/`apply_Qt!` are now genuinely 0-byte warm (new gate test
  in `qr_frontal_numeric_tests.jl`, both full-rank and rank-deficient instances).
  **Gate effect: 1/16 → 3/16 passing** (`grid_ls_40x30 own` and `staircase_n2000`
  both arms newly pass). **7000×4000 effect: none measurable** (PS frontal times
  within ~1-2% of the pre-optimization run at that scale — expected, since per-front
  allocation is a fixed cost per front, amortized away once fronts are large; it
  mattered at the smaller gate-set scale where per-call overhead is proportionally
  bigger, not at this scale where actual flops dominate).
- **`qr(A; method=:auto)` calibrated for real** (`src/tuning.jl`'s new
  `QR_AUTO_METHOD_RATIO=40.0`, `src/qr/numeric.jl`): dispatches on
  `sym.flops/sym.nnzR` (already computed by `symbolic_qr`, zero extra numeric work).
  Not a blind copy of faer's constant — independently verified against our own gate
  set: every `:column`-winning matrix sat at ratio ≤ 7, every `:frontal`-winning
  matrix at ratio ≥ 863, a wide enough margin that faer's own 40.0 is kept rather
  than picking an arbitrary number in the gap. Note this does NOT move the gate's
  own pass/fail count (the gate already takes best-of `:column`/`:frontal` per row),
  but it is the real task-16d deliverable and the mechanism a real caller gets.

Re-ran the 7000×4000 comparison after the zero-alloc pass (density-swept, galen):
faer 4.02-4.20s, SPQR 5.35-5.89s, PS frontal 5.44-6.97s, PS column 62.0-64.7s —
essentially unchanged from the pre-optimization run, confirming the allocation fix
doesn't matter at this scale (see above). PS frontal still edges out SPQR at the
highest density (5.44s vs 5.89s) and remains ~25-30% behind faer across the board.

**Remaining task 16e levers, NOT yet pulled** (paused here pending user direction):
(2) the QR-specific graded amalgamation retune — our current `AMALG_COLS`/
`AMALG_ZMAX` (3-tier, `(16,64,128)`/`(0.97,0.35,0.08)`) were swept for Cholesky, not
QR; faer's own 4-tier schedule is a concrete, well-grounded starting hypothesis to
re-verify on our own gate set, same discipline as the `:auto` ratio above. (3)
per-panel NB re-derivation, lower priority (§A5.3 already reuses faer's own
panel-split heuristic; this is a refinement on top, not a missing mechanism). Also
still open: strata-(i) tiny-matrix losses are noise-level (~0.05-0.15ms vs SPQR's
~0.04-0.15ms) and not explained by anything faer's source suggested — likely
PureSparse's own fixed per-call overhead (COLAMD/symbolic setup cost), a genuinely
separate investigation from the frontal-vs-column architecture question.

**2026-07-15: mechanical port of faer's supernodal QR numeric core landed
(`src/qr/frontal{,_numeric,_solve}.jl`)** — per user authorization (explicit, after
the SuiteSparse-AMD/COLAMD risk was flagged and the user chose the safe scope: symbolic
layer and ordering untouched, only the dense per-front factorization orchestration and
solve replay ported). Faithful translation of
`factorize_supernodal_numeric_qr_impl`'s dense phase (faer 0.24.1 `qr.rs:1246-1344`)
onto PureSparse's own symbolic layer and PureBLAS's `wy_t!`/`wy_apply!` — faer's
dense-KERNEL internals (`qr_in_place`'s own recursive blocking) are explicitly NOT
translated (§A7.4's boundary; PureBLAS's single-level compact-WY kernels substitute).
What IS translated: the staircase panel-split rule (row-scan, min-col-jump trigger,
`max(1, blocksize÷2)`), the block loop, the two trailing applies, post-factorization R
harvest, and pass-up bookkeeping. Every translated block cites its faer `qr.rs`
line range in a header comment in `frontal_numeric.jl`.

One real, load-bearing bug found and fixed mid-port (caught by Chairmarks, not
guessed): the first-draft translation misread faer's per-group `tau_block_size`
(qr.rs:1278-1279, the tier `recommended_block_size` value) as something that should
sub-split each staircase GROUP into multiple small (4-8-wide) WY blocks. It does not —
faer's own `block_count` increments exactly ONCE per staircase group (qr.rs:1283); the
tier value is fed only into `qr_in_place`'s own out-of-scope internal recursive
blocking, never used to multiply the number of `wy_t!`/`wy_apply!` calls at the
orchestration layer this port lives at. The buggy draft regressed `qr!` and `solve!`
by roughly 2x across the whole gate matrix set (measured, not assumed) before this was
found; fixed by dropping the inner sub-loop entirely and doing ONE `wy_t!`+`wy_apply!`
per staircase group, capped at the symbolic/workspace's NB storage capacity (the same
role the pre-port code's width cap played — own necessity, since PureSparse's
single-level WY block has no internal recursive absorption for an over-wide group the
way faer's `qr_in_place` does). A second, smaller regression from the same draft (an
attempted translation of faer's one-shot `householder_val.fill(zero())`,
qr.rs:1064-1066) was also reverted: faer's per-supernode capacity is exact (no rank
detection), so a one-shot zero costs exactly what's touched; PureSparse's `fmmax_f` is
a rank-deficiency-aware UPPER BOUND (§A3.2) that can exceed actual usage — the
pre-port per-front used-extent-only zero stays.

Verified: full suite (`Pkg.test`-equivalent `runtests(PureSparse)`) passes at
221,656/221,656 assertions, both with and without `--check-bounds=yes`; the
zero-alloc gate (`qr!`/`solve!`/`apply_Q!`/`apply_Qt!` warm) still holds. Directional
benchmark (`benchmark/qr_matrices.jl` gate set, this dev container — NOT clock-locked,
numbers are directional only) shows the flop-rich stratum genuinely improving over the
pre-port frontal code (`random_tall_n1200x300_d05`: qr! -39%, solve! -48%;
`dense_arrow_n800x200_d8dense`: solve! -28%) — the stratum where faer's real ~20-30%
architectural edge was measured (task 17's gate run above) — with strata (i)/(ii)
roughly flat (within this machine's noise floor). Not yet re-run on galen/wintermute
(clock-locked); task 17's full gate re-run is still the next real verdict.

Left out (deliberately, not chased): faer's own `colamd` module (SuiteSparse-AMD/
COLAMD BSD port) was never read for this task, per the explicit scope boundary —
ordering stays PureSparse's own independently-derived COLAMD/AMD, completely
untouched. faer's dense-kernel internals (`qr_in_place`'s recursive blocking,
`apply_block_householder_*`'s internals) were skimmed only enough to identify where
PureBLAS's `wy_t!`/`wy_apply!` substitute — not mechanically translated (§A7.4 is
explicit that this boundary is PureBLAS's own proven kernels, not a re-derivation
target). The per-panel block-size TIER value itself (`recommended_block_size`) has no
counterpart at all in the ported orchestration, for the reason above — this is an
intentional non-translation, not an oversight.

**2026-07-15 follow-up: mechanical port merged to master, bug fix found post-merge and
also merged, plus StrictMode/trim/@simd hygiene — re-measured on galen (clock-locked).**
Sequence: merged the mechanical port above (`cfdce7e`) after independent verification;
the agent then found (in its own follow-up worktree session) and fixed the
`tau_block_size` mis-translation described above — that fix (`f97e0b0`) was cherry-
picked onto master separately (`d66fc9f`) since it landed after the first merge. Also
closed two real gaps this session surfaced: (1) the frontal `qr!` had ZERO StrictMode
wiring (`check_refactor_shape`/`check_finite`, CLAUDE.md req 6) despite M5a's own
`qr!` having it since task 10 — fixed, `check_finite` covers `F.rval` only (not
`F.fval`/`F.tauv`, which are rank-deficiency-upper-bound-sized and can be partially
unwritten by design; checking them wholesale would false-positive on a correct
factorization, documented in place); (2) the frontal path had NO `TrimCheck`
`@validate` roots and no `juliac/entry.jl` smoke coverage at all (design_qr_m5b.md §A9
point 7, never done) — added both. Also added `@simd` to the three scalar hot loops
that stay in PureSparse rather than delegate to PureBLAS (`_front_form_reflector!`'s
scale loop, `_front_apply1!`'s dot-product+SAXPY, `_gather_panel_V!`'s copy) and
replaced a manual 2D zero-fill with `fill!`. Full suite: 221,656 assertions, 0 new
failures, throughout (`a2faf9b`).

Galen (clock-locked) re-measurements, in order:
- **Gate (`qr_gate_galen.json`)**: bounced 3/16 → 4/16 → 3/16 → 2/16 across the four
  re-runs in this sequence (zero-alloc pass, mechanical port, bug fix, `@simd`) — at
  this problem scale (matrices from a few hundred to a few thousand rows, sub-
  millisecond to tens-of-ms per call) `@be`'s per-config sample count is small enough
  that this bounce is at least partly measurement noise, not all real signal; no
  single run in this range should be read as the definitive verdict.
- **7000×4000 (the user's own benchmark target, `faer_vs_puresparse_7000x4000_galen.json`,
  simplified to 1%/10% density per request), REAL and REPRODUCIBLE (two back-to-back
  runs agreed to within 1%)**: a striking density-dependent swing appeared once the
  `tau_block_size` bug was fixed —
  - **10% density: PureSparse frontal DECISIVELY BEATS faer** — 2.26-2.32s vs faer's
    4.16-4.25s (~1.8x faster), vs SPQR's 5.71-5.76s (~2.5x faster). This is the best
    result of the whole M5b effort so far.
  - **1% density: regressed relative to the pre-bugfix measurement** — 8.7-9.4s,
    worse than the buggy draft's own 4.60s and worse than pre-mechanical-port's
    6.97s. `@simd` made no measurable difference either way (scalar loops are a small
    fraction of total work at this scale, as expected). **Not yet root-caused** — a
    reasonable hypothesis (not yet verified, do not treat as fact) is that faer's
    exact column-index-jump split trigger produces many more, smaller staircase
    groups on this specific low-density near-uniform-random pattern than the old
    row-count-based heuristic did, since low density means less inter-row column-
    pattern overlap and therefore larger jumps between consecutive rows' min-cols —
    each such jump would end a group early. Needs actual profiling (front/panel-count
    histograms at 1% vs 10% density) before acting on this, not a code change yet.
- **PureBLAS lever identified, relayed to the user (not acted on here — PureBLAS.jl's
  repo is off-limits this session, another agent has live work there)**:
  `PureBLAS.qr_block_size(m, n)` (`wy.jl:113`, what `_qr_faer_block_size`'s removal
  left as the only block-size input this port uses) ignores its own `m`/`n` args and
  returns the flat, self-flagged-as-a-shortcut `_QR_NB = 32` (`qr.jl:164`,
  `ponytail: hand-set for Zen4, tune if needed`) — even though PureBLAS already has a
  proven, cache-residency-derived formula for the SAME problem on its complex path
  (`_zqr_nb`, `qr.jl:181-182`, keyed on `_L3_BYTES`), just never extended to the real
  path `qr_block_size` calls into. Per PureBLAS's own CLAUDE.md req 8 ("every
  machine-dependent tuning parameter... MUST have a default that is a FORMULA...
  existing literals... are tech debt to migrate"), and given faer's own analogous
  `recommended_block_size` (read directly for this port) IS size-tiered, this is a
  concrete, scoped, evidence-backed candidate for whoever picks up PureBLAS's QR
  tuning next — not guessed, grounded in this session's own faer-source reading.

**2026-07-15: 1%-density regression ROOT-CAUSED AND FIXED — PureSparse now beats faer
at BOTH densities, decisively.** Diagnosed by directly comparing `F.pbs` panel-width
histograms across three git revisions (pre-port, buggy mechanical-port draft, the
first bug-fix) on the identical 7000×4000 @1% matrix — not guessed. Two real bugs:

1. The first bug-fix commit (which correctly removed an erroneous inner sub-blocking
   loop) over-corrected by ALSO deleting `_qr_faer_block_size`/`NBf` entirely. Re-
   reading faer's actual source (qr.rs:609-613, the symbolic-time per-front
   `max_block_size` computation, together with :1260-1265, its use in the split
   trigger) showed `max_block_size` legitimately feeds the split-trigger threshold —
   only a SEPARATE, group-local `bs` re-derivation (which feeds `qr_in_place`'s own
   internal recursive blocking, genuinely out of scope) has no counterpart here.
   Restored `_qr_faer_block_size`/`NBf`, correctly scoped to the split-trigger only
   this time.
2. The real culprit: `cap_trigger` included a row-count term (`(idx-k) >= NB`) that
   faer's own condition **does not have at all**, and that has no storage
   justification — `ws.wy.V`'s row capacity is `fsym.max_front_rows`, not `NB`; only
   the column/T-matrix dimension is `NB`-bounded. At 1% density, measured: ~13 rows
   share each distinct min-col on average (COLAMD-ordered 7000×4000 @1%) — row count
   accumulates far faster than column span, so the erroneous row-count trigger fired
   almost immediately regardless of `split_jump`, capping nearly every group to a
   handful of columns (measured: 491 panels, median width 1, 83% width-1). Removing
   it: 37 panels, median width 32 (every panel hits the full NB width, zero width-1),
   `qr!` 12.58s→2.11s locally (~6x) on the regressed case.

Galen (clock-locked) confirmation, 7000×4000, before → after this fix:
| | 1% density | 10% density |
|---|---|---|
| **PureSparse frontal** | 8.7-9.4s → **1.78s** | 2.3s → **0.99s** |
| faer | 4.02s | 4.34s |
| SPQR | 5.25s | 5.79s |
| **PureSparse vs faer** | was ~2.2x slower → now **~2.3x faster** | was ~1.8x faster → now **~4.4x faster** |

The gate set also jumped: `iii_flop_rich` (the stratum where faer's architectural
edge lives) went from 0-2/4 across every prior run this session to a clean **4/4**
(`dense_arrow_n800x200_d8dense`: 5.7ms→2.4ms; `random_tall_n1200x300_d05`:
10.2ms→4.2ms) — overall gate 3-4/16 → **6/16**, the best of the session.

Next: `ii_sparse_R` (0/6) and stratum-(i) tiny-matrix noise are still open; task 16e's
amalgamation retune (§A8) is now lower priority given amalgamation was directly
tested and ruled out as the 1%-density cause (collapsing to 1 front measured SLOWER,
not faster, in the course of this investigation) — the panel-split fix above was the
real lever. The gate's own noise floor at small/medium scale still needs a longer/
more-sampled run before any single verdict there is trusted.

**2026-07-15, session close-out: re-ran the gate solo on galen (confirmed no
contention from a concurrent PureBLAS-tuning session also using the fleet this
session) — result STABLE at 6/16, `iii_flop_rich` still a clean 4/4, matching the
prior run exactly (not a fluke).** Two more findings from this pass, both by
diagnosis rather than guessing, both concluding "not the lever" (recorded so nobody
re-chases them without new evidence):
- `ii_sparse_R`'s amalgamation was swept AGAIN, specifically for `grid_ls_70x50`
  (383 fronts currently): a more aggressive schedule collapsed it to 85 fronts but
  ran SLOWER locally (6.7ms→8.2ms) — nnzVF grew ~1.8x (420725→751242), padding waste
  outweighing the reduced per-front overhead. Confirms task 16e's amalgamation retune
  is not a live lever for either matrix regime tested so far (the earlier
  near-fully-dense 7000×4000 case, and now this genuinely-sparse grid-Laplacian
  case) — deliberately did NOT add QR-specific `qr_amalg_cols`/`qr_amalg_zmax`
  Preferences infrastructure (§A8 suggests this IF QR wants different values;
  it currently doesn't, by direct test, so the indirection isn't earned yet).
  `grid_ls_70x50`'s own margin against SPQR is now much tighter than it looked in
  earlier runs (own-arm ~1.09x, same-perm ~1.4x — the panel-split fix already helped
  here too, just not enough to flip the verdict).
- Stratum-(i) (`lp_slack`/tiny matrices) losses are confirmed OUT of M5b's scope: these
  matrices are singleton-heavy and structurally decompose the frontal path into ~1
  front per column (e.g. `lp_slack_n300x60`: 274 fronts) — but `qr(A; method=:auto)`
  already correctly routes them to `:column` (ratio 1.14-1.59, far under the 40.0
  threshold), so the gate's `best-of` already picks the right method; the residual
  loss is `:column`'s (M5a, already gated/shipped) own fixed per-call overhead at
  noise-level margins (~0.04-0.15ms either way) — not a frontal-path defect to fix.

**2026-07-15, further `banded_ls_n1500x500_bw15` digging (still 0/6 `ii_sparse_R`,
the worst-remaining offender at ~5-8x behind SPQR — unlike `grid_ls`, this one did
NOT meaningfully improve from the panel-split fix): two more hypotheses tested and
ruled out, both by direct local measurement, no single bug found this pass.**
- **NBf floor hypothesis (wrong, reverted):** this matrix's fronts are tiny (max
  25 cols × 65 rows — SMALLER than the workspace's own NB=32!), so
  `_qr_faer_block_size` naturally returns a tiny tier (4), giving `split_jump=2` —
  aggressive splitting. Reasoned that since PureSparse (unlike faer) has no scalar
  fallback for small tiers — it always pays `wy_t!`/`wy_apply!`'s BLAS-3 dispatch
  cost regardless of `NBf` — flooring `NBf` at `NB` (forcing wider groups, fewer
  calls) should help. Tested directly: REGRESSED (0.4695ms → 0.5683ms locally).
  Root cause of the regression: banded structure means true row support is tightly
  localized per column; forcing wider groups pulls in rows with no real overlap,
  paying extra padding-zero flops that outweigh the saved call count. Reverted
  immediately; `git diff` confirmed clean before moving on.
- **Sampling profile** (`Profile.@profile` over 20000 warm `qr!` calls): no single
  dominant hotspot — samples split roughly evenly across reflector formation
  (`nrm2`+`_front_form_reflector!`), the in-group scalar apply (`_front_apply1!`),
  `wy_t!`, and `wy_apply!`. Consistent with a genuine constant-factor cost from many
  small BLAS-3 call pairs (34 fronts × ~4 panels ≈ 136 `wy_t!`/`wy_apply!` pairs on
  operands this small) rather than one fixable algorithmic bug — the same territory
  as the `qr_block_size` gap already relayed to (and now being worked on by) the
  PureBLAS-side agent; may improve here too once that lands, not verified yet.

**2026-07-15: small-front scalar fallback landed — faer's `qr_in_place_unblocked`
precedent, not previously read.** Re-reading faer's actual dense QR kernel source
(`linalg/qr/no_pivoting/factor.rs`, which the M5b port had only skimmed enough to
identify the PureBLAS substitution point, never actually read) showed
`qr_in_place`/`qr_in_place_blocked` recursively drop to a PURE SCALAR Householder
loop — zero BLAS-3 calls — whenever a sub-problem falls under
`QrParams::auto().blocking_threshold = 48×48 = 2304` elements. `banded_ls`'s fronts
(max 25×65=1625) are well under that — faer itself never blocks them either.
Implemented the equivalent: fronts under `QR_FRONTAL_UNBLOCKED_THRESHOLD` (2304,
`tuning.jl`) now skip `wy_t!`/`wy_apply!` entirely, using a pure column-by-column
scalar pass instead (new `QRFrontFactor.fscalar` flag; solve-phase `apply_Qt!`/
`apply_Q!` replay via a new `_scalar_apply_to_vec!`, exploiting that a Householder
reflector is self-adjoint so the same formula covers both directions). R harvest/
pass-up needed no changes (already a shared post-loop pass keyed on `F.elimcol`).
Scope note: the actual scalar factorization logic stays in PureSparse (coupled to
dead-pivot/Heath rank-detection policy PureBLAS deliberately doesn't own, same
boundary as the pre-existing `_front_form_reflector!`/`_front_apply1!`); the
THRESHOLD decision is architecturally closer to `qr_block_size`'s role and could
migrate to PureBLAS later, but lives in PureSparse's own `tuning.jl` for now
(PureBLAS's repo is off-limits this session).

Verified: full test suite (221656 assertions, 0 new failures), 61-case random sweep
under `--check-bounds=yes` (every matrix in that sweep is small enough to exercise
the new path). Measured, galen:
- `banded_ls_n1500x500_bw15`: locally 0.47ms→0.36ms (~23% faster, all 34 fronts
  went scalar).
- `grid_ls_70x50` (342/383 fronts scalar): the GATE's own single-sample measurement
  initially looked like a regression (8.27ms→10.29ms) — resolved by a targeted
  2000-rep timing check: median `qr!` = **2.95ms** (min 2.94ms, but a long tail to
  6.29ms — this matrix has real, high sample-to-sample variance at this scale,
  something to keep in mind for future gate runs here specifically). The true
  median is well under SPQR's 5-8ms, meaning this matrix most likely actually
  passes now — the single-sample gate run caught a rare slow outlier, not a real
  regression. Full gate re-run recommended before trusting any single verdict on
  this specific matrix.
- `grid_ls_40x30`/`dense_arrow`/`random_tall`: no regressions; `random_tall`
  correctly stayed fully on the blocked path (0 fronts below threshold).

**2026-07-15: task 17 gate re-run on galen, now including PureBLAS's landed geqrf
`_qr_nb` fix (`el-oso/PureBLAS.jl@357db97`, "derive real `_qr_nb` from register count +
L2, req#8" — the flat-NB gap this session had relayed to the concurrent PureBLAS-tuning
agent).** Result: still **6/16** (`i_singleton` 2/6, `ii_sparse_R` 0/6, `iii_flop_rich`
4/4) — same shape as the pre-fix run. `grid_ls_40x30` is now within noise of passing
(frontal 2.582ms/1.583ms vs SPQR 2.476ms/1.475ms); `grid_ls_70x50`'s single-sample gate
reading (10.458ms/5.682ms) is again far above the 2000-rep true median (2.95ms) found
previously — same noise caveat, not re-investigated further here. Conclusion: the
`_qr_nb` fix doesn't move this gate's own matrix set, because none of the 8 gate
matrices are large enough to spend meaningful time in the blocked dense-panel GEMM path
`_qr_nb` tunes — they're either overhead-bound (stratum i) or sparsity-structure-bound
(stratum ii). The fix's actual target is the flagship 7000×4000 dense-panel case
(`benchmark/faer_vs_puresparse_7000x4000.jl`), re-measured separately (see below).

Operationally: galen's `~/Documents/claude/PureBLAS.jl` — the copy PureSparse's
`Manifest.toml` path-deps on — had gone stale (still flat `_QR_NB=32`, predating the
merge) and was a raw rsync copy with no `.git`, distinct from the other agent's actual
working checkout at `/home/el_oso/PureBLAS.jl`. Per user direction, turned it into a
real tracking checkout (`git init` + `git remote add origin ... + fetch + reset --hard
origin/master`) rather than rsyncing files — verified clean/no-uncommitted-work first,
consistent with the standing PureBLAS-repo-contention caution. Also root-caused two
unrelated infra footguns that silently killed background gate runs on galen for over an
hour before this: (1) `~/.julia/config/startup.jl` auto-loads OhMyREPL even under
non-interactive `-e`, which appears to crash/exit silently with no controlling TTY
(stdin `/dev/null`) — fix is `--startup-file=no`; (2) `benchmark/qr_gate.jl` (like most
of this repo's benchmark scripts) gates its `run_gate()` call behind `if
abspath(PROGRAM_FILE) == @__FILE__`, which is never true when the file is loaded via
`include(...)` from a `julia -e` string — must invoke it as `julia script.jl`, not
`julia -e 'include("script.jl")'`. Neither is a PureSparse code defect; both are
recorded here as galen-specific operational gotchas for future sessions.

Re-ran the flagship 7000×4000 comparison (`benchmark/faer_vs_puresparse_7000x4000.jl`,
galen, now with the geqrf fix live) to check where it *would* show up — the gate's own
matrices are all too small. Result, PS frontal cold median vs faer/SPQR:

| density | PS frontal | faer | SPQR | vs faer | vs SPQR |
|---|---|---|---|---|---|
| 1% | 1713.6ms | 4039.4ms | 5237.9ms | 2.36× | 3.06× |
| 10% | 893.7ms | 4214.3ms | 5657.5ms | **4.72×** | 6.33× |

vs the pre-fix numbers from the session that landed the panel-split root-cause fix
(1.78s / 0.99s), this is ~4% faster at 1% and ~10% faster at 10% — small but real, and
the 10%-density margin over faer improved past the previously reported ceiling
(4.4×→4.72×). Confirms the geqrf `_qr_nb` fix's benefit is real but scale-gated: it
shows up on the large dense panels this flagship case exercises, not on the gate's
small/medium matrix set. `PS column` remains unusably slow here as expected (61-65s —
this is exactly the flop-rich regime `:auto`/`:frontal` exists to avoid).

**2026-07-15: M5a task 13 (user-facing docs) closed — QR guide, API reference, and a
benchmarking page landed, with real comparison plots and a bibliography.** Delegated
doc/plot generation to a Fable-model agent (`docs/src/qr-guide.md`, the `## Sparse QR`
section of `api.md`, the `## Sparse QR (M5)` section of `benchmarking.md`,
`benchmark/plot_qr_comparison.jl`), then verified: fixed one Markdown-escaping bug in
`qr`'s docstring (a bare `design_qr.md` on the same line as a `[...](@ref)` link broke
both), corrected an imprecise `:column`-vs-`:frontal` speedup claim (36-74×, not the
agent's "35-65×"), and confirmed the docs build clean end to end
(`julia --project=docs docs/make.jl` — zero new warnings; the pre-existing
`TypeContractsDocumenterExt` load error is unrelated, present before this work). Added
`docs/src/refs.bib` (`DocumenterCitations`, 7 entries: AMD, COLAMD paper, Larimore
thesis, Gilbert-Ng-Peyton, SPQR paper, the Davis/Rajamanickam/Sid-Lakhdar survey, faer)
and `.github/workflows/{CI,docs}.yml` (this repo had none before — mirrored
PureFFT.jl's exact templates) so the new README badges (CI/Coverage/Docs/License,
matching the sibling projects' pattern) aren't pointing at nonexistent infrastructure.

Per explicit user direction, the comparison plots are **violin+boxplot overlays**
(project convention, see BlazingPorts.jl's `bench/harness.jl`), not bars of the median
alone — `qr_gate.jl` and `faer_vs_puresparse_7000x4000.jl` now persist each config's
raw Chairmarks sample vector (`*_samples` keys) alongside the median specifically so
this is possible from saved JSON without re-running. This immediately paid off: the
gate-strata violins visually confirm real, high sample-to-sample variance on several
`ii_sparse_R` matrices (not just the already-documented `grid_ls_70x50` — `banded_ls`
own-arm spans ~1ms to ~100ms across 20 samples), which a bar-of-median plot hides
entirely. The flagship 7000×4000 figure stays a **bar chart** (explicit user call):
faer/SPQR cost 4-6s/call there, so even a deliberately-bounded re-run (10 samples for
the plotted series, `:column` capped separately at 3 samples since it isn't plotted and
costs ~65s/call) is too thin a sample for a meaningful density estimate, and a full
violin-worthy count would cost ~30 min. That bounded re-run's first attempt actually
surfaced a real bug: at the original `SECONDS=2.0` budget, faer/SPQR (4-6s/call) each
got exactly **one** completed sample (Chairmarks finishes an in-progress rep before
checking budget), rendering as a degenerate flat-line "violin" — fixed by giving the
flagship script its own larger budget. Three back-to-back re-runs of the flagship case
now agree within ~1% (1.72-1.73s / 4.03-4.04s / 5.2-5.4s at 1% density; 0.88-0.89s /
4.21-4.23s / 5.66-5.69s at 10%) — the 6/16 gate figure itself also re-confirmed
unchanged on a fresh run (stratum pass/fail details shuffled slightly within the
already-documented noise, total unchanged).

Side finding while verifying the agent's report (measured directly, not just trusted):
warm `qr!` on the `:column` (M5a) path allocates **1056 bytes** at 200×50 — a real
CLAUDE.md req-5 zero-alloc gap the existing gate test doesn't catch (it only exercises
a 20×12 matrix, where this path happens to be 0 bytes). `:frontal`'s `qr!`/`solve!` are
0 bytes at both sizes — this is `:column`-specific. Filed as task #48, not fixed here
(out of scope for a docs task).

**2026-07-16: task #48 closed.** Root cause found via `Profile.Allocs` (not guessed —
the initial size-based hypothesis was wrong; allocation turned out non-monotonic in
`(m,n)` and seed-dependent at a FIXED size, ruling out a simple size threshold):
`numeric.jl`'s row-subtree gather (`qr!` step 2) calls `sort!(view(tsub, 1:len))` with
Base's default algorithm, which auto-selects `RadixSort` for `Int` arrays above a size
heuristic — `RadixSort` allocates scratch buffers (`Base.Sort.make_scratch`). `len` is
a per-column row-subtree size, not the array's static length, so whether any column of
any given matrix crosses that heuristic is genuinely data-dependent — explains why the
existing 20×12 gate test never caught this in M5a's whole lifetime (that size/seed
combination happens to never cross it) while ~14/15 random seeds at 100×50 did,
allocating 500–4000+ bytes depending on the exact values realized, not the shape.
Fixed with `alg=InsertionSort` (Base's guaranteed-zero-alloc algorithm at any size,
verified directly) — appropriate since `len` is small/tree-depth-bounded by
construction, not a full-array sort. Added a swept regression test (5
size/density/seed combinations, `test/qr_numeric_tests.jl`) specifically because a
single fixed case is exactly the failure mode that let this hide. Full suite:
221663/221663 assertions pass.

**2026-07-16: task #47 (`ii_sparse_R`) — real progress, gate now 8/16 (was 6/16),
`ii_sparse_R` now 2/6 (was 0-1/6), `grid_ls_40x30` PASSES both arms.** All prior
`ii_sparse_R` work (this file's many dated entries above) targeted the numeric
factorization loop in the multifrontal path. Direct measurement (galen, `@be`
medians) reframed the problem: for small/sparse gate matrices, the NUMERIC loop
isn't where the time goes — `symbolic_qr` (called fresh every cold `qr(A)` call,
correctly, since SPQR has no analyze-once/refactor split to compare against) was
53.9% of total time for `banded_ls_n1500x500_bw15`, more than ordering, front-tree
construction, and the numeric loop combined:

```
banded_ls_n1500x500_bw15 (before)      cold=1.6367ms  order=0.2616ms(16%)
  symbolic_qr-rest=0.8819ms(53.9%)  fsym=0.0568ms(3.5%)  numeric=0.4364ms(26.7%)
```

Handed this precise, quantified brief to a Fable-model agent (isolated worktree) to
find and fix the specific bottleneck within `symbolic_qr`'s non-ordering steps
(`star_pattern`→`symmetrized_upper`→`etree`→`postorder`→`relabel_pattern`→`etree`
again→`column_counts`→`row_leftcol`→`qr_row_structure`). The agent's session was
cut short by an API/network disruption before it finished its own verification, but
left a complete, well-reasoned fix in its worktree: `qr_row_structure` (§3.4's
`S_k`/`pivotslot` construction) previously built each column's row set via a
per-column `Vector` + `push!`/`append!` + `sort!` — measured (`Profile.Allocs`) to
be the dominant cost (~2350 allocations / ~8 MB churned per `banded_ls` call).
Rewrote it to write each `S_k` DIRECTLY into its final `vrowind` segment with no
sort at all, via a proof (own derivation, in the docstring) that the merge is
*already* sorted by construction: physical row numbers are grouped ascending by
`leftcol` block, `parent` is postordered so child subtrees are disjoint ascending
column intervals, and by induction each child's survivor sublist is both final and
sorted by the time its parent processes it — so "child survivors in ascending
child order, then column k's own consecutive block" is the answer with no sort
needed. Child traversal uses the same head/next linked-list idiom `postorder`
already uses elsewhere in this codebase (own precedent, not new).

I did NOT trust the agent's unfinished work — verified independently before
committing: full local suite (221663/221663 assertions, unchanged) confirms
correctness (including `test/qr_symbolic_tests.jl`'s 178550 H1/H2-style invariant
assertions, the most direct check on this exact function); re-ran the isolated
`symbolic_qr`-only breakdown on galen (clock-locked) and confirmed the targeted
component actually shrank (`banded_ls`: `symbolic_qr rest` 0.8819ms→0.337ms, -61.8%;
`grid_ls_70x50`: 1.9472ms→1.059ms, -45.6%); then ran the REAL gate script (not the
noisier subtraction-derived micro-breakdown) for the actual verdict: **6/16→8/16**,
`ii_sparse_R` 0-1/6→2/6, `grid_ls_40x30` now a clean PASS on both arms. `banded_ls`
itself still fails both arms (1.778ms/1.452ms vs SPQR 1.025ms/0.604ms, ~1.7-2.4×
behind, down from ~5-8× before ANY of this session's numeric-loop or symbolic-loop
work) — closer, not closed. `grid_ls_70x50` also still fails, within this matrix's
already-documented high sample-to-sample variance (own-arm improved 9.841ms→
7.158ms; same-perm read worse, 9.273ms→8.268ms — a single-sample comparison on a
matrix already known to need a longer run to trust any one verdict).

Next: `banded_ls` is now the clear worst offender and worth its own fresh
breakdown pass (its numeric share, ~0.44-1.0ms depending on run, has never been
attributed past "no single hotspot" from an earlier sampling profile — that
profile predates BOTH the scalar-fallback fix and this symbolic fix, worth
re-running fresh) — not done this session, flagged for the next one.

**2026-07-16 (later same day): followed up on the `banded_ls` breakdown flagged
above — found the gate had already effectively closed and the single-sample
verdicts were noise, not a real deficit; gate now 11/16, `ii_sparse_R` 5/6.**

A component-level attribution attempt (symbolic/frontal-symbolic/factor-construct/
numeric measured separately, summed, compared to the real total) fell apart with a
NEGATIVE unattributed residual for `banded_ls` (parts summed to MORE than the
whole) — direct evidence that at this timescale (~1ms total), separately-measured
sub-components carry more noise than the effect being chased, and further
decomposition here is not productive. Switched to high-sample-count (600-1000
samples, `seconds=8-10`) direct comparisons instead of decomposing further:

- `banded_ls_n1500x500_bw15`: PS frontal median **0.9961ms** vs SPQR median
  **1.0056ms** — PureSparse is marginally FASTER on the honest statistic (62.8% of
  individual PS samples beat SPQR's median). Down from ~5-8× behind at session
  start. The gate's own single-sample-per-config runs were catching noise on a
  photo-finish, not a real deficit — same failure mode already documented for
  `grid_ls_70x50` earlier this file, now also true here.
- `grid_ls_70x50`: genuinely different, NOT resolved to simple noise — the PS
  distribution is bimodal (min 5.52ms, actually below SPQR's own min of 5.80ms,
  but roughly half the samples land near 9ms instead of ~5.8ms, pulling the median
  to 9.07ms vs SPQR's 6.04ms). Plausible mechanism, not yet confirmed: GC-pause
  bimodality from the cold path's fresh per-call allocation (symbolic + front
  factor + workspace, all rebuilt every call) — a real fast mode exists and beats
  SPQR outright, something intermittently pushes ~2/3 of calls into a slow mode.
  Flagged as a genuine, distinct lead for a future session (GC-pause diagnosis,
  e.g. `GC.gc_num()` deltas per sample or `--heap-size-hint`), not resolved here.

Re-ran the actual gate script fresh (not the noisier decomposed breakdown) to get
an honest current snapshot: **11/16** (was 8/16 this morning), `ii_sparse_R`
**5/6** (was 2/6) — `banded_ls` own-arm now PASSES (1.005ms vs SPQR 1.033ms),
`grid_ls_70x50` PASSES both arms (6.062ms/4.747ms vs SPQR 8.396ms/5.025ms, a
favorable draw from the bimodal distribution characterized above — expect this
specific matrix's gate verdict to keep bouncing run-to-run until the bimodality
itself is diagnosed). Only `banded_ls` same-perm still fails, and only barely
(0.684ms vs SPQR 0.606ms, ~13% gap — well inside the noise band the 1000-sample
check above demonstrates for this matrix). `i_singleton` unchanged at 2/6,
already-documented out-of-scope `:column` per-call overhead (not re-investigated
this pass). `iii_flop_rich` still a clean 4/4.

**M5 gate status: 11/16, not yet passing — but `ii_sparse_R` has gone from the
worst stratum (0/6 for most of this session) to nearly resolved (5/6, remaining
gap noise-level) purely from this session's fixes (panel-split trigger, scalar
fallback, symbolic `sort!` elimination) without any matrix-specific tuning.**

**2026-07-16 (later still): `grid_ls_70x50`'s GC-pause bimodality diagnosed —
CONFIRMED (per-call `Base.gc_num()`/`Base.GC_Diff` deltas, not inferred), traced
to two allocation sites, root cause is genuine COLAMD-driven elimination fill for
a 2D-grid matrix under a greedy (non-nested-dissection) ordering, not implementation
waste — not fixed this session, real levers identified for next time.**

Direct per-call GC instrumentation (500 cold `qr(A; method=:frontal)` calls,
`Base.gc_num()` before/after each, `Base.GC_Diff` for the delta) confirms the
bimodality IS GC-pause-driven, unambiguously:

```
fast calls (293/500): mean time=5.824ms  gc_time=0.0015ms  pause_count=0.00
slow calls (178/500): mean time=30.661ms gc_time=22.6261ms pause_count=0.57
                                                            full_sweep=0.24
allocd per call: IDENTICAL either way, mean=15676.8 KiB (min=max=~15.68 MiB)
```

The critical fact: allocation VOLUME is constant regardless of fast/slow — this
isn't "slow calls allocate more," it's purely whether Julia's generational GC
happens to trigger (and whether it's a cheap minor collection or an expensive
full sweep, which ~24% of slow calls hit) during that particular call. A GC
pause alone averages 22.6ms — nearly 4× the ~5.8ms the factorization itself
takes.

`Profile.Allocs` on a single cold call traced the ~15.3 MiB total to two
dominant sites (69% of the total between them):
- `QRFrontFactor{T,Ti}(fsym)` construction (`frontal.jl:130`, the `fval` dense
  front-value array specifically) — 5.75 MiB, driven by `fsym.nnzVF` (dense
  front storage, 420725 entries for this matrix).
- `qr_row_structure`'s `vrowind` array (`symbolic.jl:243`, one single 4.87 MiB
  allocation) — driven by `sym.nnzV` (623736 entries).

Checked whether this is fixable waste or genuine fill: `nnzV/nnz(A) = 45.3×` —
a large fill ratio, but this is the well-known signature of a 2D-grid-shaped
matrix under a greedy minimum-degree-family ordering (COLAMD) rather than
nested dissection, not an implementation defect — SPQR defaults to the same
ordering family, so it likely carries comparable internal fill/allocation
volume for this exact matrix. The actual difference isn't "PureSparse allocates
more than it should," it's architectural: SPQR is a C library (malloc/free, no
garbage collector, so it never pays an unpredictable stop-the-world tax no
matter how much scratch memory it touches), while PureSparse runs on Julia's
generational GC, which intermittently pays a large, unpredictable pause for
processing that same allocation volume. This is a genuine Julia-vs-C runtime
tradeoff for this specific workload shape (many mid-size allocations on a cold,
one-shot path), not a code bug to patch away.

**Not attempted this session** (real, testable, NOT YET TRIED levers for next
time, roughly in order of how contained/low-risk they are):
1. GC tuning (no code change): does `--heap-size-hint` or a periodic explicit
   `GC.gc(false)` (cheap minor collection, preempting the rarer expensive full
   sweeps) change the pause-frequency/severity distribution? Purely a runtime
   flag/measurement question, testable without touching source.
2. Whether the rewritten `qr_row_structure` (this session's `sort!`-elimination
   fix) still has any avoidable slack in `vrowind`'s sizing/construction beyond
   the mathematically-necessary V-pattern — not re-audited since that rewrite
   landed.
3. Whether a different ordering (`AMDOrdering` on `AᵀA` instead of the default
   `COLAMDOrdering`) meaningfully reduces fill for 2D-grid-shaped matrices
   specifically — would need a careful, isolated evaluation (this project's own
   ordering-quality-vs-CHOLMOD gate exists for Cholesky, no QR-specific
   equivalent has been run for THIS comparison) since changing defaults broadly
   risks regressing other matrix shapes.

Next session: try lever 1 above (GC tuning, no code change, lowest risk) on
`grid_ls_70x50` and re-measure; if that doesn't move it, levers 2-3. Then a
clean multi-run gate confirmation before considering `ii_sparse_R` genuinely
closed.

**2026-07-16 (later still): correction to the above — GC tuning was the wrong
lever, tried and explicitly rejected; the `vrowind`/`pivotslot` allocation was
NOT genuine required fill, it was waste for this caller. Fixed by skipping its
construction; verified as a real (not GC-masking) improvement.**

Lever 1 (GC tuning) was tried: `--heap-size-hint=2G` moved GC time from 51.2%
to only 49.4% of wall time; `--gcthreads=4` changed the pause *shape* but not
the underlying allocation volume. Neither meaningfully helped, confirming GC
tuning treats the symptom, not the cause — standing rule now: never reach for
GC flags to fix an allocation-driven pause, find and remove the allocation
(saved to memory as `fix-allocation-not-gc`).

Re-investigating with "is this allocation actually used" rather than "is this
allocation's SIZE justified" (the question the earlier entry above asked)
surfaced a different, more actionable finding: exhaustive grep across
`src/qr/frontal*.jl` confirms `sym.vrowind`/`sym.pivotslot`/`sym.vptr` (built
by `qr_row_structure` inside `symbolic_qr`) are **never read anywhere** on the
`:frontal` path — the frontal numeric loop builds its own front-local V
storage via `symbolic_qr_frontal`'s `fsym.nnzVF`, entirely independently.
`sym.nnzV`/`max_vcol`/`flops` (which `:frontal` DOES use, for its own stats)
are computed from `vptr`/`vcount` only, never from `vrowind`/`pivotslot`
directly — so those stay exact even without materializing the arrays. The
fill itself is genuine (COLAMD's real elimination structure for this matrix
shape), but computing and storing the V-PATTERN CONTENTS for a caller that
never consumes them is pure waste, independent of whether the fill amount is
itself reducible via a better ordering (a separate, still-open question).

Fix: threaded a `build_v::Bool = true` keyword through `qr_row_structure` →
`symbolic_qr`, defaulting to `true` everywhere (the `:column` path's two call
sites in `numeric.jl` need the real `vrowind`/`pivotslot` contents and are
untouched); `qr_frontal`'s own `symbolic_qr` call passes `build_v = false`.
When `false`, `qr_row_structure` returns empty `Ti[]` placeholders for
`vrowind`/`pivotslot` instead of running the child-merge loop that fills them.

Verified, not assumed:
- `sym.nnzV`/`max_vcol`/`flops`/`rperm`/`riperm`/`mb` bit-identical between
  `build_v=true` and `build_v=false` on `grid_ls_70x50` (direct comparison).
- Solve correctness: `‖Aᵀr‖ / (‖A‖₁‖r‖) ≈ 8.7e-20` after the fix — unaffected.
- Full local test suite: 221663/221663 pass (this touches shared symbolic
  code reused for both QR paths).
- Cold `qr_frontal(A)` allocation on `grid_ls_70x50` (`@allocated`, 30 calls,
  local): 14.77 MiB → 9.93 MiB (removes exactly the ~4.84 MiB `vrowind`
  allocation, matching the earlier `Profile.Allocs` estimate).
- galen, same `Base.gc_num()`/`GC_Diff` instrumentation as the diagnosis above
  (500 cold `qr(A; method=:frontal)` calls, back-to-back baseline-then-fix on
  the SAME synced checkout, no GC flags):

```
                  baseline (pre-fix)     post-fix
median time       6.029ms                5.474ms
mean time         14.494ms               9.16ms
slow calls (>1.5x median)  33.8%         8.2%
total gc_time     3796.4ms (52.4%)       1724.1ms (37.6%)
full sweeps       39                     18
p90 / p99 / max   20.8 / 92.8 / 101.0ms  7.4 / 87.4 / 99.3ms
```

Mean time down ~37%, slow (GC-bimodal) call fraction down 4×, GC time share
down ~15 points, full-sweep count roughly halved — a real reduction from
removing an actual allocation, not from masking pauses with GC flags. The
bimodality is NOT fully eliminated (still 8.2% slow calls, still 37.6% GC
share) — the remaining ~9.93 MiB per cold call (`fval` and other dense front
storage, driven by real `fsym.nnzVF` fill) is genuinely consumed by the
frontal numeric loop, so removing it further is a fill-reduction problem
(ordering quality), not a waste-removal one — the still-open lever 3 from the
list above.

Full 16-matrix gate re-run (galen, post-fix, `benchmark/qr_gate.jl`, cold
median seconds, `benchmark/results/qr_gate_galen.json`) confirms the fix at
the actual gate level, not just the diagnostic:

```
grid_ls_70x50  ii_sparse_R  own        5.495ms (PS frontal) vs 8.398ms SPQR  PASS
grid_ls_70x50  ii_sparse_R  same-perm  4.396ms (PS frontal) vs 7.259ms SPQR  PASS
```

Both arms now PASS with real margin (PS frontal ~35-40% faster than SPQR),
not a marginal/noise-level pass. `ii_sparse_R` stands at 5/6 — the remaining
gap moved to `grid_ls_40x30` same-perm (PS frontal 1.674ms vs SPQR 1.509ms,
~10% behind), a tight single-sample margin consistent with this stratum's
established noise floor (`banded_ls`'s own earlier back-and-forth), not
re-diagnosed this session. Overall gate: still 11/16 (`i_singleton` remains
2/6, gated behind task #50's same-perm/singleton-peeling issue; `iii_flop_rich`
remains 4/4, better than H4's "may lose" expectation). `grid_ls_70x50` is now
solidly closed as an open item.

**2026-07-16 (later still): task #50 fixed — same-perm arm's 3.4x fill inflation
on `lp_slack` matrices was a real bug (`GivenOrdering` forced singletons off), not
inherent to the matrix shape. `ii_sparse_R` now 6/6.**

Root cause, confirmed (not guessed): `GivenOrdering` carries a permutation sized
for the FULL, pre-peel matrix (SPQR's own `pcol`), but singleton pre-elimination
(§2.3) hands `order_columns` only the `n - n1` surviving columns of `A22` —
`order_columns(::GivenOrdering, ...)`'s length check then throws `DimensionMismatch`
unless the caller disables peeling entirely (`singletons=false`), which is exactly
what the gate's same-perm arm did, at the cost of `nnzR` inflating 3.4x
(`lp_slack_n800x150`: 4491 with singletons vs 15255 without) since the whole point
of peeling is to strip away the trivially-solvable diagonal-shaped block before the
main factorization ever sees it.

Fix (`src/qr/singletons.jl`, `_restrict_ordering`): before `_qr_compose_singletons`
hands `A22` to `_qr_block`, restrict the given permutation to just the entries
naming a surviving column (drop the peeled ones), preserving relative order, then
relabel each from its original column index to its local index within `A22`.
`AMDOrdering`/`COLAMDOrdering`/`NaturalOrdering` need no such adaptation — their
`order_columns` recomputes a fresh permutation from whatever pattern it's handed,
so a reduced `A22` was never a problem for them; `_restrict_ordering` is a no-op
for anything but `GivenOrdering`. Since `peel_column_singletons` is a pure function
of `A`'s own pattern/values (never `ordering`), `n1`/`nnzR` now come out
bit-identical between the "own" and "same-perm" arms — confirmed directly
(`lp_slack_n800x150`: both arms `n1=800`, `nnzR=4491`), and the composed factor's
solve residual matches the existing own-arm path bit-for-bit too (same code, same
numbers, just relabeled). Full local test suite: 221663/221663 unchanged.

`benchmark/qr_gate.jl` updated to stop forcing `singletons=false` on the same-perm
arm (`ps_singletons = true` unconditionally now) — it was measuring an artificially
crippled configuration that doesn't reflect real product behavior (the default is
always `singletons=true`).

galen, back-to-back gate runs, same matrices:

```
                            same-perm arm, cold median
                    pre-fix (forced no-singleton)   post-fix
lp_slack_n300x60    0.132ms                          0.045ms   (SPQR 0.038-0.039ms)
lp_slack_n800x150   0.879ms                          0.155ms   (SPQR 0.103-0.105ms)
```

`lp_slack_n800x150`'s same-perm arm: 0.879ms → 0.155ms, ~5.7x faster, now within
~50% of SPQR instead of 8.5x behind. `ii_sparse_R` reached 6/6 this run (was 5/6;
`grid_ls_40x30` same-perm flipped to PASS, within this stratum's known noise
floor). `i_singleton` did NOT close — still 1-2/6 depending on draw — but the
REMAINING gap is now a genuinely different, much smaller problem: these are
sub-millisecond matrices where SPQR's constant per-call overhead (C, malloc/free,
no GC, no dispatch) wins outright regardless of algorithmic fill (`lp_slack_n300x60`
own-arm: PS 0.048ms vs SPQR 0.044-0.046ms — already near-parity, not a fill
problem). Filed as task #51 — separate from #50, which is now closed: the fill bug
is fixed, what's left is call-overhead on tiny problems, not an ordering/singleton
defect. Overall gate: still 11/16 (numbers moved between strata, total unchanged) —
`ii_sparse_R` and `iii_flop_rich` both now solid; `i_singleton`/task #51 is the
sole remaining open stratum.

**2026-07-16 (later still): task #51 closed — the "call overhead" WAS still an
allocation problem, just a different one. `i_singleton` jumps to 6/6, overall gate
11/16 → 14/16.**

Profiled `qr(lp_slack_n800x150; singletons=true)`'s cold-call breakdown
(BenchmarkTools, function-barrier-isolated so each phase compiles/measures
independently): full call ~196μs median, 2580 allocations, 668.80 KiB. Split into
`peel_column_singletons` (~44μs, 49 allocs), `_qr_block(A22)` on the (degenerate,
0-row for this matrix — every row consumed by a slack singleton) surviving block
(~20μs, 161 allocs), leaving `_qr_compose_singletons`'s OWN bookkeeping at ~59% of
total wall time and the bulk of the allocation count.

Root cause: the R11/R12 harvest loop (gathering each peeled row's original entries,
mapped to final column position, sorted ascending) allocated a fresh `Tuple{Ti,T}[]`
(grown via `push!`) PER PEELED ROW, then `sort!`ed it — for `lp_slack_n800x150`'s 800
peeled rows, ~2400 of the function's 2531 allocations traced directly to this one
loop. Fixed: write each row's (column, value) pairs directly into their already-
correctly-sized final `r1colind`/`r1val` slice, then insertion-sort that slice in
place (a `_insort_row!` helper) — zero allocation, and for LP-slack's typically-small
row degrees O(deg²) insertion sort costs far less than one heap allocation would
have. Correctness: unaffected either way, since both produce the same ascending
column order the array's own contract requires — confirmed via the full local test
suite (`qr_singleton_tests.jl`/`qr_singleton_compose_tests.jl` and everything else,
221663/221663 pass) rather than an ad hoc script (see below).

`_insort_row!` had to be its OWN standalone function, not inlined at the call site:
fusing its `while`-inside-`for` into `_qr_compose_singletons`'s own triple-nested
loop caused a genuine LLVM compile-time explosion (LoopStrengthReduce/SCEV, observed
hung for minutes via a live stack trace) on first compilation — a known LLVM
pathology for deeply-nested generic loops in one large function, not a runtime bug;
isolating it as a small function gave LLVM a much smaller unit to analyze and
resolved it immediately. (A separate red herring while chasing this: an ad hoc
verification script's own `for` loop at top-level/global scope, referencing several
global variables across many iterations, hit the *same* LLVM pathology independent
of any PureSparse code — confirmed by reproducing it against the unmodified,
already-committed `sort!`-based version too. Standard Julia practice — wrap script
loops in a function — didn't fully avoid it either in that throwaway script; the
existing project test suite was the correct verification tool all along, not a new
script, and gave a clean, fast, unambiguous answer.)

Measured (BenchmarkTools, `lp_slack_n800x150`, 300 samples, local): full cold
`qr(...)` call 195.9μs → 123.8μs median (~37% faster), 668.80 KiB → 448.85 KiB
(~33% less), 2580 → 266 allocations (~90% fewer). galen, full 16-matrix gate
(same matrices, back-to-back with the #50 fix already in place):

```
i_singleton     1/6 → 6/6   (staircase_n2000 and both lp_slack matrices now PASS
                              or WIN outright both arms, e.g. lp_slack_n300x60
                              own-arm: PS 0.034ms vs SPQR 0.044ms)
overall gate    11/16 → 14/16
```

`ii_sparse_R` dipped 6/6 → 4/6 in this same draw (`grid_ls_40x30` failed both arms)
— consistent with this specific matrix's already-documented tight-margin/noise-floor
behavior earlier in this file (it has flipped PASS/fail across single-sample runs
several times this session with no code change in between), not a regression from
this fix. `iii_flop_rich` unaffected, still 4/4. M5's overall gate is now 14/16 —
the closest it has been all session; `grid_ls_40x30`'s single-sample noise and a
fresh multi-sample confirmation are the natural next steps before considering
`ii_sparse_R` fully closed alongside `i_singleton`.

**2026-07-16 (M5b P2 LANDED — the frontal path is now generic over real isbits `T`,
AD-traceable; no new kernel was needed).** Scoped by the user to the real-generic/AD
target (CLAUDE.md req 3), not complex/BigFloat. Key finding: PureBLAS's `gemm!` had
*already* become generic (AD-traceable triple-loop for non-BLAS `T`) and `wy_t!`/
`wy_apply!` are `AbstractMatrix{T}`-generic and call it — so the frontal numeric loop +
kernels were already generic; the only barrier was the `T === Float64` routing gate.
Empirically scoped first (all measured, not assumed): Float64/Float32/Float16 factor
correctly through `:frontal`; BigFloat SEGFAULTS (non-isbits — pointer-based front
storage); ComplexF64 BoundsErrors (real-reflector storage assumption + needs conjugate
Householder). So: relaxed the gate to `_frontal_capable(T) = isbitstype(T) && T <: Real`
(`Float32`/`Float16`/`ForwardDiff.Dual` route to `:frontal`; complex/BigFloat fall to the
generic `:column`). Found + fixed one real Float64 hardcoding: `QRStats.dropped_norm =
Float64(sqrt(dropped_sq))` (a rank diagnostic, off the differentiable path) broke Duals
because `Float64(::ForwardDiff.Dual)` is deliberately undefined — replaced with
`_stat_f64` (base `Float64(::Real)`) plus a weak-dep `ext/PureSparseForwardDiffExt.jl`
supplying the Dual primal, so `src` keeps NO ForwardDiff dependency (trim-safe: the
extension is never in the `--trim` path; base `_stat_f64` is a plain `Float64` cast).
Verified: the CLAUDE.md req-3 headline — a least-squares solve differentiated THROUGH the
frontal QR (`ForwardDiff.derivative`) matches a central finite difference (rtol 1e-4);
plus Float32/Float16 `RᵀR = AᵀA` at precision-appropriate tolerance; new test item
"P2: generic over real isbits T", 12/12. ForwardDiff added as a TEST dep + a weakdep
(UUID from the resolver, not guessed). ComplexF64 (task #54) and BigFloat (task #55) are
filed follow-ups, both out of the AD scope. **This closes the last substantive M5
engineering gap** — only the §10 closeout-checklist verification remains before M5 is a
fully-stamped milestone.

**2026-07-16 (7000×4000 perf autopsy — the tie with faer is the honest ceiling, not a
gap we can close from the PureSparse side).** Chased where the ~6.5s goes after the
flagship re-measure showed parity with faer (not the withdrawn 2-6×). Findings, all
measured:

- **Ordering is NOT the gap.** PS/SPQR nnz(R) ratio = 1.001 (1%) / 1.000 (10%) — our
  COLAMD produces bit-competitive fill vs the SuiteSparse C reference (and faer, which
  also uses COLAMD, confirmed by reading its qr.rs). No fill/ordering deficit exists;
  the premise "faer's COLAMD is better" is false.
- **Profile (Julia sampler, both densities identical):** ~97% of time is the numeric
  factorization; ~93% is inside PureBLAS's `gemm` microkernel, reached entirely through
  `wy_apply!` (the compact-WY trailing block update, `frontal_numeric.jl:281`). 0%
  OpenBLAS leakage — PureBLAS genuinely is the kernel. Leaf self-time is the SIMD FMA
  (`muladd`) in `_microkernel_db!`, with load/store ≈ 45% of FMA cost — the fingerprint
  of skinny-K gemm (K = panel width), low arithmetic intensity.
- **NB (panel width) sweep at 7000×4000** (added a `front_block_size` override to
  `qr_frontal`/`symbolic_qr_frontal`, default `nothing` = qr_block_size, bit-identical):
  GFlop/s is FLAT — 1%: {16:12.87, 24:12.80, **32:13.01**, 48:12.96, 64:12.96, 96:12.77};
  10%: flat-to-declining, best at 16-32. `qr_block_size`'s default (32 for these fronts)
  is already the optimum. Widening does nothing — ~13 GFlop/s is genuinely shape-limited
  (skinny-K QR trailing updates can't reach square-gemm peak), which is exactly why faer
  sits at the same rate.

Conclusion: PureSparse matches faer's ordering AND per-flop throughput, and beats SPQR
~30% per-flop, at this scale. The only remaining lever is the PureBLAS dense-gemm
microkernel itself (owner pursuing separately as a PureBLAS task) — but even that is
constrained by the skinny-K shape here, so upside is uncertain. The `front_block_size`
knob is kept as the re-calibration hook for after any microkernel change (tested:
`test/qr_frontal_numeric_tests.jl` block-size item, correctness at a forced non-default NB).

**2026-07-16 (flagship 7000×4000 re-measured on the corrected factorization — the old
"decisive win" was itself a bug artifact).** The prior flagship numbers (PS frontal
2.3–6.5× faster than faer/SPQR) were withdrawn because they timed the broken blocked
path — which dropped ~2/3 of columns, so it clocked a fraction of the real work.
Re-measured on neuromancer (clock-locked) after verifying correctness at scale first
(frontal LSQ residual 1.1e-19 @1% / 3.7e-20 @10%, full rank 4000/4000 — the discipline
that was missing the first time). Honest numbers (cold factorize-only, faer via the
factorize-only `faer_sparse_qr_factor` shim, 10/8/10 samples, median):

```
density   PS frontal   faer          SPQR
1%        6466ms       6618 (1.02x)  8253 (1.28x)
10%       6692ms       6929 (1.04x)  8985 (1.34x)
```

So at 7000×4000 PureSparse's multifrontal QR is essentially TIED with faer (a mature
Rust lib) and ~30% faster than SuiteSparseQR — a genuine, respectable pure-Julia
result, but NOT the inflated 2–6× the broken path fabricated. A clean bookend to the
correctness thesis: the bug made even the flagship look better than reality. Results:
`benchmark/results/faer_vs_puresparse_7000x4000_neuromancer.json`. Public artifact's
flagship section restored with these numbers.

**2026-07-16 (M5 GATE MET — 16/16, both clock-locked hosts): warm singleton refactor
closes `i_singleton`; the M5 closeout wall-time gate PASSES.** Extended `qr!` to
warm-refactor a singleton-composed factor (`sym.n1>0`) — the last thing blocking the
gate. D13 gates the warm path, but `qr!` had rejected singleton factors, forcing the
singleton-dominated stratum onto a `singletons=false` warm path (full work on
trivially-peelable columns) → `i_singleton` stuck at 3/6. Key realization: the peel
set's STRUCTURAL half ("exactly one live nonzero") is pattern-only, hence
refactor-invariant (a refactor shares the pattern by contract); only the magnitude
test is value-dependent. So warm singleton refactor is safe: reuse the fixed peel set,
refresh A22's values zero-alloc, re-harvest R11/R12 with a per-pivot magnitude guard
(a numerically-dead pivot folds into the existing `n_dead`/`dropped_norm` accounting).
design_qr.md §2.3's one-shot-only restriction lifted (owner-authorized). Implemented by
a Fable agent (4 pre-alloc `QRFactor` fields at compose time, unified
`_qr_block_numeric!`), INDEPENDENTLY verified here: warm LSQ residual ~1e-20 matching
the `:column` oracle (lp_slack_n300x60/n800x150, own + same-perm), rank exact,
`@allocated qr!`==0, `--check-bounds=yes` clean; the sweep test's oracle change
(cold-composed vs cold-nosingletons) confirmed legitimate — the two cold paths already
disagree on 5/294 `tol=0` degenerate matrices ON MASTER, a pre-existing accounting
difference unrelated to warm reuse. Full suite 222217/222217.

**Gate result (D13, warm PS vs SPQR cold), confirmed on BOTH clock-locked hosts:**

```
                neuromancer   galen
i_singleton     6/6           6/6      (was 3/6 — warm singletons closed it)
ii_sparse_R     6/6           6/6
iii_flop_rich   4/4           4/4
OVERALL         16/16 PASS    16/16 PASS
```

`lp_slack_n800x150` warm `:column` 0.662ms → 0.018ms (10× once warm singletons are on);
vs SPQR cold 0.18ms. Strata ii/iii are n1==0, structurally unchanged from the 13/16 run.
Two-host clock-locked confirmation = the project's own bar for a gate verdict
([[reference_benchmark_machines]]). **M5's wall-time closeout gate (design_qr.md §9.3,
CLAUDE.md req 2) is MET.** The public artifact (claude.ai/code/artifact/1e1c9658-...)
updated to 16/16. Remaining before calling the whole M5 MILESTONE closed (vs the gate):
the still-withdrawn 7000×4000 flagship needs re-measurement on the corrected path, and a
final design §10 checklist pass — but the non-negotiable perf gate itself is now green.

**2026-07-16 (resolution): both frontal bugs FIXED + verified; first trustworthy M5
gate under D13 = 13/16.** Bug 1 (ftau OOB) fixed via single-source `QRFrontSymbolic.nb`
(commit 0364f4b). Bug 2 (blocked path dropped columns past NB per split trigger — one
panel emitted per trigger instead of ⌈width/NB⌉) root-caused + drafted by a Fable agent,
INDEPENDENTLY verified here (my own diagnostic, `--check-bounds=yes`: blocked LSQ residual
60×60 2.8e-3→5.9e-18, 100×80→2.0e-18, 400×384→1.1e-18, matching the `:column` oracle;
full ranks restored; suite 221669/221669), committed da4a2c3. Regression test's
correctness checks flipped `@test_broken`→`@test`. Then re-ran the ACTUAL gate on the
corrected factorization under D13 (neuromancer, clock-locked,
`benchmark/results/qr_gate_neuromancer.json`):

```
                          PS front warm    SPQR cold    gate
banded_ls own             0.345ms          1.602ms      PASS (4.6x)
banded_ls same-perm       0.344ms          0.919ms      PASS
grid_ls_40x30 own/sp      1.31ms           2.87/2.32ms  PASS
grid_ls_70x50 own/sp      4.98/4.97ms      10.72/7.61ms PASS (2.2x)
dense_arrow own/sp        2.26/1.99ms      4.49/3.72ms  PASS
random_tall own/sp        8.70/8.78ms      16.94/14.84  PASS
i_singleton               3/6 (holdout)
```

`ii_sparse_R` 6/6, `iii_flop_rich` 4/4 — both SOLID and DETERMINISTIC (warm path is
zero-alloc, so the grid_ls_70x50 GC-bimodality that plagued the cold-vs-cold gate is
gone by construction — D13's second payoff). Overall 13/16; the 3 failures are all
`i_singleton` (lp_slack sub-0.25ms matrices where SPQR's C per-call overhead wins — a
separate, known issue, not a frontal defect). M5 still open on i_singleton, but for the
first time held open by a REAL, correctly-measured number. Public M5 gate artifact
(claude.ai/code/artifact/1e1c9658-...) rewritten around this correctness story + the
first trustworthy numbers; prior 7000×4000 flagship withdrawn (it too timed the broken
path — re-measurement pending on a 2nd clock-locked host).

**2026-07-16 (CRITICAL, changes the whole M5 picture): the M5b BLOCKED frontal path
produces NUMERICALLY WRONG results, and every M5 gate number this session was timing
a broken factorization. Two distinct bugs found, one fixed, one open + delegated.**

Chain of discovery. While closing the gate: (a) the user correctly rejected gating on
a cold path that triggers GC pauses ("if there is a GC-call on any calculation, that
gate CANNOT be closed"); (b) reading `design.md`/`design_qr.md` §9.3 showed M1/M2/M4
already gate on the WARM (zero-alloc, StrictMode-verified) refactor path, and
`design_qr.md`'s cold-vs-cold choice for M5 rested on a flawed inference — "SPQR has no
refactor mode, so we must compare cold" doesn't follow; SPQR's cold IS its best case,
so the correct gate is our-warm-vs-SPQR-cold (recorded as design_qr.md **D13**,
§9.3 corrected, `qr_gate.jl` reworked to gate on `min(:column, :frontal warm) < SPQR
cold`); (c) re-running that corrected gate SEGFAULTED on neuromancer (Zen5) at
`dense_arrow_n800x200`, while galen (Zen3) completed — a march-portability violation,
exactly the class the user flagged as a contract breach.

**Bug 1 (FIXED): ftau slab out-of-bounds write.** `--check-bounds=yes` turned the
segfault into a clean `BoundsError` at `frontal_numeric.jl:284`. Root cause: two
inconsistent block sizes for the same `ftau` T-slab — the symbolic phase budgeted it
with `qr_block_size(0, 0)` (=8) under a "one query suffices" assumption, but
`qr_block_size` is dimension-dependent and the numeric phase capped panels at
`qr_block_size(max_front_rows, max_front_cols)` (=16 for an 800×169 front). The numeric
loop packed `pcount×pcount` T's (pcount ≤ 16) into a slab budgeted for pcount ≤ 8 →
overflow. Undefined behavior: benign adjacent-heap write on Zen3 (silent), unmapped-page
SIGSEGV on Zen5. Fixed by making `NB` a single source of truth: a new `QRFrontSymbolic.nb`
field computed once (after the max front dims are known) as `qr_block_size(max_front_rows,
max_front_cols)`, used by BOTH the `ftauptr` slab sizing AND `QRFrontWorkspace.Tm`
(`frontal.jl` now reads `fsym.nb` instead of recomputing). Verified: repro clean under
`--check-bounds=yes` (cold + 2 warm refactors), full suite 221663 pass. Regression test
added (`qr_frontal_numeric_tests.jl`) asserting the single-source-of-truth invariant
(`size(ws.Tm,1) == fsym.nb`) on a matrix deliberately large enough to hit nb==16 — no
prior frontal test did.

**Bug 2 (OPEN, delegated to a Fable agent): the blocked multifrontal numeric loop is
numerically wrong on ANY large front.** Fixing Bug 1 un-crashed the blocked path but
exposed this: `qr_frontal` on a front over `QR_FRONTAL_UNBLOCKED_THRESHOLD`
(=2304 elements → the `wy_t!`/`wy_apply!` BLOCKED path, not the scalar fallback) returns
a wrong R / least-squares solution. Measured least-squares residual `‖Aᵀ(b-Ax)‖/(‖A‖₁‖b‖)`:

```
front 40x25   (scalar,  nb=8)   frontal 3.3e-18   column 5.3e-18   OK
front 60x60   (blocked, nb=8)   frontal 2.8e-3    column 8.7e-18   WRONG
front 100x80  (blocked, nb=8)   frontal 1.5e-3    column 2.3e-18   WRONG
front 400x384 (blocked, nb=16)  frontal 1.5e-4    column 1.3e-18   WRONG
```

Wrong at nb=8 AND nb=16, so it is NOT the OOB and NOT nb-specific — a genuine defect in
the blocked kernel orchestration (trailing-apply / R-harvest / pass-up), pre-existing,
independent of everything else this session. **Why it was invisible until now:** every
existing frontal correctness test uses matrices under ~1000 elements → all take the
SCALAR fallback; the blocked path — the entire point of multifrontal, the BLAS-3 core of
M5b — was NEVER correctness-tested end-to-end. And the gate measures only TIMING, so this
session's "16/16 / 15/16 / 14/16 / 12/16" verdicts were **timing a factorization that
produces wrong answers — those numbers are meaningless**. Regression test marks the
blocked-path correctness checks `@test_broken` (suite stays green, auto-alerts when fixed);
delegated to a Fable-model agent per CLAUDE.md's hard-algorithmic-piece rule.

**What is NOT affected:** M1/M2/M4 (Cholesky/LDLᵀ — entirely separate code). The M5a
`:column` path (correct at every size, independently gated, ~1e-18 residual throughout —
it is the trustworthy oracle that exposed Bug 2). No CLOSED gate is invalidated (M5 was
never closed); but the whole M5 gate is now blocked on Bug 2 and cannot be meaningfully
re-run until the blocked path is correct.

**2026-07-16 (M1/M2, CLAUDE.md req 5): a real, previously-unverified zero-alloc gap
found and fixed in `solve!` for `cholesky!`/`ldlt!` — not M5b, but landed via M5b's
own StrictMode infrastructure being extended to the Cholesky/LDLT paths.** Added
`benchmark/audit/` (StrictMode `@assert_noalloc`, `checks_enabled=true`, isolated
from `test/` since `cholesky!`/`ldlt!`/`qr!` call StrictMode's own runtime checks
internally — mirrors PureFFT.jl's `bench/audit/` precedent). This guarantee is
strictly stronger than the existing `@allocated == 0` tests (those only prove the
one input exercised is alloc-free; `@assert_noalloc`'s static AllocCheck mode proves
every path the compiler can enumerate — though for functions with a lazily-grown-
then-cached scratch buffer, static mode false-positives on the growth branch
regardless of warm-up state, the same class PureBLAS's own test suite already
documented; used `static = false`, the empirical mode, for exactly that reason,
matching PureBLAS's own precedent).

First run found `solve!` on `SupernodalFactor`/`LDLFactor` allocated **5920 bytes**
— genuinely never gated before (only `cholesky!`/`ldlt!` themselves had a zero-alloc
test; nobody had written one for their `solve!`). Root cause: `_solve_L!`/
`_solve_Lt!` were re-`unsafe_wrap`ping `F.x` fresh every call (instead of the
already-cached `F.panels`, built once for exactly this — `cholesky!`'s own hot path
already used it) and allocating a fresh scratch `Matrix` per supernode for the
off-diagonal update. Fixed both: `F.panels[s]` directly, and a new `Workspace.
rhs_blocks` field (same caching technique as `F.panels`, wrapping the persistent
`F.ws.rhs` buffer) plus reuse of the existing `ws.c` scratch (already sized
`max_extend_rows × max_extend_rows`, more than enough, same bound already proven for
its factorize-time use) via `view`. Split into `_solve_L_cached!`/`_solve_L_generic!`
(and the `Lt` equivalents) rather than an inline per-call branch: `view(ws.c,...)`
(`SubArray`) and `Matrix{T}(undef,...)` are different concrete types, so a ternary
between them would make the hot loop's `yblk`/`upd` Union-typed — a real type
instability, not just style. `solve_L!`/`solve_Lt!` are also exported directly
(split-solve consumers, e.g. iterative refinement) and CAN be called with an
arbitrary caller-owned vector, not just `F.ws.rhs` — an object-identity check
(`y === F.ws.rhs`) dispatches to the unchanged generic path there, so that path's
correctness is untouched.

Verified: `solve!` now 0 bytes for both `cholesky!`/`ldlt!` (was 5920), machine-
precision residuals unchanged, full `@assert_noalloc` audit passes for all three
factor types (`cholesky!`, `ldlt!`, `qr!(::QRFrontFactor)`) and their `solve!`s,
full local suite 221663/221663 assertions unchanged. `qr!`/`solve!` on
`QRFrontFactor` (M5b) needed no fix — already 0 bytes going in.

**2026-07-15: user flagged a suspected apples-to-oranges bug in the faer comparator —
real, but not the mechanism suspected, and the measured margin holds.** User's
hypothesis was that faer materializes explicit Q/R while PureSparse only stores R
explicitly with Q implicit (Householder reflectors), needing an extra step. Checked:
false — both store Q implicitly (`QRFrontFactor.fval`/`ftau` vs faer's `Qr.indices`/
`numeric`, same convention as SuiteSparseQR); PureSparse never materializes Q unless
`apply_Q!` is called. The REAL asymmetry: `BlazingPorts.jl`'s `faer_sparse_qr` shim
timed factorize **and** `solve_lstsq_in_place` together, because faer's sparse `Qr`
exposes no direct `.R()` accessor and a solve was the only way the original author
found to stop the Rust compiler eliding the factorization as dead code — while
PureSparse's/SPQR's own wrappers here (`_ps_frontal_cold`/`_spqr_cold`) are
factorize-only. Confirmed by reading faer 0.24.1's own source
(`sparse/solvers.rs::Qr::try_new_with_symbolic`, what `sp_qr()` calls): it does ONLY
`factorize_numeric_qr`, no solve of any kind — the solve genuinely was pure
benchmark-shim overhead, not part of faer's own factorization cost.

Fixed with `std::hint::black_box` (Rust's actual idiom for "keep this alive without
doing extra work," stable since 1.66) instead of a real solve — added
`faer_sparse_qr_factor` to `BlazingPorts.jl` (`f370a3a`), rebuilt the cdylib on galen,
switched the comparator wrapper to it (`260f7c1`), re-measured. Result: the corrected
numbers are within ~0.7% of the pre-fix ones (1% density: faer 4002.5ms vs prior
~4029.7ms avg; 10%: 4231.1ms vs prior ~4216.9ms avg) — well inside the run-to-run
noise already established for this benchmark. **The solve was cheap relative to the
factorization itself**, so the bug was real (worth fixing — the comparator now
measures the same scope on both sides) but empirically didn't inflate PureSparse's
reported margin over faer. `docs/src/benchmarking.md`'s flagship table updated to the
re-measured numbers (2.3×/4.8× vs faer, 3.0×/6.5× vs SPQR — materially unchanged from
before).

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

### M3 — GPU  ⚠️ SUPERSEDED by `### M6` below (2026-07-16)
**This section is stale pre-M1 content and contradicts design.md §8** (Fable M6 review,
F1: KA-kernels+level-set+reported-not-gated here vs vendor-BLAS+per-supernode-staging in
§8). Kept for history only. The live M6 plan is in the `### M6` section further down and
`docs/design_gpu.md` (in progress). Original stale text preserved below:

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

### M6 — GPU (the last milestone; supersedes the stale `### M3` above)
**Kicked off 2026-07-16 after a Fable design review (all 3 factual findings verified).**
Full design: `docs/design_gpu.md` (in progress; goes through the same v1 → two independent
adversarial reviews → v2 process as design.md/design_qr.md — user-approved).

**Scope (user decision):** Cholesky + LDLᵀ together (M6a Cholesky, M6b LDLᵀ — shared
scheduler). Sparse QR is OUT of M6 (different multifrontal-WY arch, its gate already
closed). Update/downdate stays CPU (latency-bound).

**Architecture (Fable review):** device-resident factor (12 GB fits any winnable
problem); symbolic-time **upward-closed etree frontier** splitting CPU subtrees from GPU
supernodes (one-way, once-only panel uploads — left-looking supports it without
restructuring), replacing the underived per-supernode `gpu_flop_threshold=2e9` (F2: it
never fires on any existing gate matrix); exact symbolic-time capacity check with loud CPU
fallback. The offload must be derived from `llt.jl` AS IT IS (F3: it has a contiguity β=1
fast path that §8/§4.3 predate), not from the stale doc.

**Kernel strategy (user requirement, 2026-07-16 — DUAL track):**
1. **Wire cuBLAS/cuSOLVER first** (gemm/syrk/trsm/potrf) — a working, gate-passing GPU
   path and the proven-fast baseline to beat. Clean-room-fine (closed binary, used
   black-box; we never read its source). NOT cuDSS/cusolverSp (NVIDIA's *sparse* solvers
   stay black-box baselines like CHOLMOD — the sparse orchestration stays pure Julia).
2. **Tune pure-Julia kernels to BEAT cuBLAS**, and make them **vendor-portable**
   (KernelAbstractions → AMD ROCm / Intel oneAPI). Rationale: PureBLAS already beats
   OpenBLAS AND MKL on CPU, so beating cuBLAS FP64 is a legitimate target, not a fantasy;
   and portable pure kernels are a strategic win cuBLAS can't give. Phase-0 measured (galen
   RTX 4070): cuBLAS FP64 ~305 GF (67% of ~455 GF peak); naive AND 4×4-register-blocked
   pure kernels both ~148 GF (0.48×), NO register spills (diagnostic) → not a pure-Julia
   ceiling, an un-profiled bottleneck (occupancy/FP64-ILP). Pure-kernel optimization is a
   live background R&D track (Fable agent on galen, NCU-profiled).
   **Prior wrong turn (corrected):** an earlier read concluded "pure can't beat cuBLAS" from
   the 0.48× naive number — the exact "benchmarked a slow path → wrong 'it's dead' verdict"
   trap; register blocking giving 0 improvement was the tell the KERNEL was wrong, not pure
   Julia. See [[feedback_anchor_proven_fastest_path]]-style discipline.

**Gate (Fable, needs the two contract amendments below signed off):** ≥2× median warm
refactor vs our OWN single-thread CPU PureSparse on a NEW large-matrix stratum (fits ~9 GB)
+ ≤ noise regression on the existing gate set (auto threshold) + still beats CHOLMOD+
OpenBLAS on the stratum. cuDSS = reported context arm (like faer was for QR).

**Two contract amendments requiring explicit user sign-off (design.md hard reqs):**
(i) req-5 zero-alloc GPU wording (kernel launches/CUBLAS allocate host bytes — need
"0 device bytes after setup + measured/bounded host bytes"); (ii) req-2 GPU gate baseline
(the ≥2×-vs-own-CPU definition above). Not yet approved.

**Hardware:** galen (RTX 4070, 12 GB, sm_89), CUDA.jl 6.2.1 in `~/Documents/claude/
gpu_probe/`. Probes: `benchmark/gpu/phase0_probe.jl`, `kernel_diag.jl`.

**Progress (2026-07-17):** Multifrontal GPU engine BUILT and verified machine-precision on
galen — Cholesky (all-GPU + hybrid + device solve) and LDLᵀ (blocked device-LDL, signed
regularization, order-free inertia, hybrid factor + device solve end-to-end). Pure KA
kernels beat cuBLAS FP64 1.14× via `muladd` (Julia is IEEE-strict — won't fuse `a*b+acc`
without it; that was the 0.48× "un-profiled bottleneck"). Path B (multifrontal) fixes the
launch-bound ceiling Path A (left-looking) hit. **Bounded stack-with-compaction arena
(§M.3) DONE** (commit `4d55fd9`): work slot + bounded stack, 5.9× smaller than monotonic
at grid3d_44 (ratio grows with size) — closed the large-KKT OOM. LDLᵀ hybrid vs CPU
`ldlt!` on 3D-grid KKTs: 24³ 1.70×, 28³ 2.80×, 36³ 3.98×, 40³ (nnzL 27M) 4.45×, 44³ (nnzL
46M) 5.04× — 3× target comfortably exceeded; the two largest previously OOM'd. Oracles:
`benchmark/gpu/gpu_{multifrontal,solve,ldlt,ldlt_e2e}_test.jl`; CPU-testable @testitems in
`test/gpu_multifrontal_tests.jl`.

**Optimization push (2026-07-17, opts 1–3):** (1) **device buffers hoisted to persistent
kwargs** (`d_emap`/`d_W`/`d_dummy`/`d_Anz`/`d_info`) — amendment-A zero-alloc; only residual
device alloc is the LDLᵀ strided-diag D2H staging (commit `20644e7`). (2) **multifrontal CPU
fronts routed through PureBLAS** (design §M.4; commit `e35629b`). (3) **pure device
potrf+trsm** (Fable-authored, galen-validated ≤3.9e-16; commits `379fb12`/`9cb075d`) wired
into both Cholesky (potrf+trsm) and LDLᵀ (trsm) GPU paths — cuSOLVER/cuBLAS now used only by
the retained left-looking reference arms; failure via deferred `d_info` (= amendment D).
**Fable optimization round DONE** (fused `gpu_front!`: single-launch small-front path +
in-shared MAGMA diag-inverse + trapezoidal trailing update; `benchmark/gpu/pure_potrf_opt.jl`
→ `ext/gpu_dense.jl`) — weighted 1.13× vs vendor on the real crown mix. **Integrated** (commit
`159b866`): `gpu_front!` on both Cholesky paths, `gpu_trsm_rlt_opt!` on both LDLᵀ paths; all
oracles machine-precision. **Gate target raised 3× → 5×** (user, after vendor measured 5.04×).
**PERF SPLIT MEASURED:** the **fusion is load-bearing** (separate potrf+trsm lose 0.67×) — so
the **SPD Cholesky recovered to vendor parity** (grid3d 44³ 2.91×, ≥ old cuSOLVER) but the
**SQD LDLᵀ can't fuse** (its diagonal factors on CPU via signed `_ldl_block!`) → only the
standalone trsm applies → stuck at **3.41×** (vs 5.04× vendor), missing 5×. **Pure device signed-LDL FRONT BUILT + INTEGRATED** (commit `71bda3c`, `ext/gpu_ldlt_dense.jl`):
`gpu_ldlt_front!` fuses the signed-LDL diagonal (fixed-pivot signed reg + order-free inertia,
amendment E) with the panel solve in ONE kernel — the LDLᵀ analogue of `gpu_front!` — removing
the nscol³ CPU-diag round-trip that pinned the KKT path. Machine-precision (relL 1e-18, inertia
EXACT); flop-weighted 4.42× vs the vendor front. **PURE LDLᵀ NOW MEETS 5×:** KKT hybrid vs CPU
`ldlt!` — 36³ 4.29×, 40³ 4.44×, **44³ 5.08×** (vs 5.04× vendor — pure now beats it). **Both SPD
(fused Cholesky front, vendor parity) and SQD (fused signed-LDL front, ≥ vendor) are fully
pure/portable AND ≥ vendor speed → kernel-policy resolved as pure-primary, no backend dispatch.**
Removes the last residual device alloc (strided-diag D2H staging).

**AMD PORTABILITY PROVEN (2026-07-17, commit `c0fcf04`).** The entire pure-kernel path runs
on real AMD hardware (neuromancer, Radeon 840M / **gfx1151**, RDNA3.5, ROCm 7.14, via
AMDGPU.jl) at machine precision: gemm (1e-17), fused Cholesky front (relL/relP 1e-16, both
v1/v2), fused signed-LDL front (relL/relD 1e-16, **inertia EXACT**, both v1/v2). Genericization:
CUDA intrinsics in the shared kernel files (`gpu_dense.jl`, `gpu_ldlt_dense.jl`) replaced by
KernelAbstractions-portable equivalents. **Blocker found + fixed:** gfx1151's GPUCompiler
segfaults in `check_ir!` on any USE of an atomic-rmw's return value (bare Int32 atomic is fine;
Int64 crashes even bare) — so the "last-group-writes-diagonal" election was replaced by an
election-free **group-1-to-disjoint-scratch** write + driver copy (no atomic, deadlock-free,
perf-neutral on CUDA). Validation harness `benchmark/gpu/amd_kernel_test.jl` (env
`~/Documents/claude/amd_probe`). **CUDA (galen) re-verification of this change DONE + PASSED** — the group-1-to-scratch redesign
preserves the CUDA path: all fused oracles machine-precision (relL 1e-18, inertia exact,
solve-res 1e-16) and the 44³ LDLᵀ perf held at **5.10×** (was 5.08×, within noise). Validated on
BOTH hosts (gfx1151 machine-precision + galen 5.10×).
Note: gfx1151 is an FP64-weak iGPU, so absolute AMD perf is low — the deliverable is the
portability PROOF, not AMD speed.

**FORMAL §8 GATE RUN (2026-07-17, galen RTX 4070, `benchmark/gpu_gate.jl`, results
`benchmark/results/gpu_gate_galen.json`).** Three arms (CPU-PureSparse / pure-GPU /
CHOLMOD+OpenBLAS), SPD+SQD stratum, **factor+solve** timed region (amendment B). Result:
**clause 1 (≥5× vs our CPU): 0/8 FAIL; clause 3 (beats CHOLMOD): 4/8** — the GPU LDLᵀ
CRUSHES CHOLMOD on all SQD/KKT (up to 23×, the target IP workload) but SPD is ~parity.
**Root cause (breakdown `benchmark/gpu/gpu_gate_breakdown.jl`, SQD 40³): the DEVICE SOLVE
is the bottleneck** — factor 553ms, make-solve-ready 20ms, RHS 0.1ms, **solve 555ms** (as
slow as the factor). The solve walks all 10468 supernodes with ~63k tiny per-supernode
trsv/gemv/scatter launches → **launch-bound**, the SAME pathology multifrontal fixed for
the factor. The factor-only 5× is real; the solve was never optimized. **DEVICE SOLVE BATCHED (commit `864567a`, `ext/gpu_solve.jl`).** Level-scheduled pure-KA solve
(elimination-tree levels computed once at symbolic time; one batched kernel per level — fwd
trsv+gemv+bare-atomic scatter, D⁻¹, bwd gather+gemvᵀ+trsvᵀ). SQD 40³: **device solve 555ms →
26.2ms (21× faster)**, 33 levels → 67 launches (was ~63k); now 4.7% of the factor, no longer
the bottleneck. Machine-precision both hosts (galen res ≤ 8e-16 + inertia exact; gfx1151
compiles+matches res ≤ 6e-16 — the fast solve is portable too, bare-atomic scatter). **§8 GATE RE-RUN (batched solve, stratum to 44³, `gpu_gate_galen.json`).** Batched solve
flipped the result: **clause 3 (beats CHOLMOD+OpenBLAS) now 10/10** — GPU beats CHOLMOD on
every SPD (1.4–2.7×) AND every SQD (19–51× faster than CHOLMOD's slow sparse ldlt). **Clause 1
(≥5× vs our own CPU): 0/10, best 4.48× (SQD 40³), 4.06× (44³).** The 5× target was set from an
OPTIMISTIC number (factor-only, min-of-4, best-cutoff = 5.10×@44³); the rigorous **factor+solve
median** the gate measures tops out ~4.5×. Honest state: **beats CHOLMOD everywhere (up to 51×
on the target KKT workload) + ~4.5× vs our own single-thread CPU.** **DECISION (user): re-scope clause 1 from "5× vs our CPU" to "≥1.0× the CUDA vendor equivalent
(cuBLAS/cuSOLVER)".** Under that target M6 PASSES: pure ≥ vendor on every op — gemm 1.14×,
**Cholesky front 1.14–2.00× at EVERY size** (`chol_front_sweep.jl`, independently re-verified on
a quiet galen; worst row 1.14× at nscol=1536/below=186), LDLᵀ front 4.42×, batched solve 21×;
end-to-end factor+solve ~1.5× the vendor path. The Cholesky all-sizes win came from
`_front_fused64_v3!` (commit `f662b69`, fully register-resident fused factor+solve rank-1 sweep;
closed the potrf-dominated large-nscol gap where cuSOLVER used to win 0.86–0.97×). **§8 GATE re-run with a VENDOR GPU arm (commit `229c02e`, `gpu_gate_galen.json`) — ONE CLEAN TABLE,
PASSES 10/10.** Arm 4 = same multifrontal, vendor fronts (cuSOLVER potrf+cuBLAS trsm / CPU-diag+cuBLAS
trsm) + per-supernode cuBLAS solve; frontmode kwarg defaults to the shipped pure path (oracles
re-verified unaffected). **Pure-GPU factor+solve beats the vendor-GPU equivalent at EVERY stratum
size:** SPD 4.35×(28³)→2.25×(44³), SQD 3.32×(28³)→**1.92×(44³, worst margin)**; vendor-arm residual
≤8.62e-16. Clause 3 (beats CHOLMOD) also 10/10. **M6 GATE CLOSED on the re-scoped target (pure ≥
vendor everywhere + beats CHOLMOD everywhere).** **NUMBERS PUBLISHED** to `docs/src/benchmarking.md`
(commit `7685d03`, figures `gpu_gate_ratios`/`gpu_chol_allsizes`/`gpu_arena` from saved JSON).
**eGPU cross-host bar DEFERRED** — the only spare GPU is a weaker NVIDIA 3050 (redundant tick on an
already-solid galen gate, not the AMD-perf prize), and neuromancer's nvidia driver stack is broken
(DKMS sources purged from /usr/src, 5 conflicting versions, mainline kernel 7.0) — not worth a
kernel/driver fight for a redundant data point. AMD portability already proven on the gfx1151 iGPU
(correctness + machine precision; FP64-weak so no speed story). Optional: pin the frontier
auto-policy. **NEXT PHASE (user, 2026-07-18): optimize LU and QR** (scope TBC — GPU-accelerate the
existing sparse QR (M5, CPU) same as Cholesky/LDLᵀ; "LU" scope to confirm — PureSparse is currently
symmetric-only, no unsymmetric LU).
(pinned SPD+SQD stratum ≥6, both req-2 arms incl the `PureSparse+PureBLAS` vs `CHOLMOD+OpenBLAS`
CPU baseline, still-beats-CHOLMOD, two-host galen + neuromancer-eGPU bar).

**2026-07-18: M7 (GPU sparse QR) — EVALUATED AND SHELVED BY MEASUREMENT. A pure-Julia GPU-QR front
cannot beat cuSOLVER `geqrf`; the M6-style "pure beats the CUDA vendor" milestone is not achievable
for QR.** Full design process ran first (`docs/design_qr_gpu.md` v1 → two independent adversarial
reviews `design_qr_gpu_review{,_fable}.md` → v2; then a Fable TSQR redesign `design_qr_gpu_v3_fable.md`
→ Opus adversarial review `design_qr_gpu_v3_review_opus.md`). Opus's key move: verified all arithmetic
against the raw probe JSON and identified that the true crux is **γ, the batched WY-apply gemm ratio**,
not the panel. Three Phase-0 probes on galen (RTX 4070, Float64, warm `CUDA.@elapsed` medians;
`benchmark/gpu/qr_{panel_phase0,front_project,gamma_phase0}.jl`) settled it before any production code:
(1) pure single-workgroup Householder **panel** is 3–10× slower than `geqrf` standalone (1 SM of 46,
occupancy-bound); (2) **front-level projection** best-NB 1.49–2.70× slower; (3) **γ_WY ≈ 0.77** — pure
*loses* the trailing WY-apply 1.3×. The WY-apply is two equal-flop gemms: pure WINS the standard `nn`
shape (γ_nn=1.15, as M6) but LOSES the `tn` tall-K/tiny-output shape badly (γ_tn=0.575) — cuBLAS's
most-tuned batched-split-K regime. Even at hypothetical `tn` parity the trailing caps ~1.07× and the
panel is parity-at-best → both places pure could win are losses or ties. ROOT CAUSE: M6's Cholesky win
came from `potrf`/`syrk` fitting a fused register-resident pure kernel; QR's Householder tall-skinny
panel + tall-K WY-apply is exactly what cuSOLVER/cuBLAS are most tuned for — the win does not transfer.
The measurement-first process worked as intended: ~1 day of probes killed a milestone that would have
cost weeks of TSQR building. **CPU sparse QR (M5) remains CLOSED and gate-passing — only the GPU-QR
milestone is shelved.** A *non-pure* GPU-QR (vendor fronts) was considered and rejected: our pure
kernels already beat CPU SuiteSparseQR on GPU throughput (GPU-vs-CPU, like M6 vs CHOLMOD), so calling
the vendor buys nothing and abandons the pure thesis. NEXT (pending owner consult): LU via PureKLU.jl
(SciML MIT, evaluated 0-alloc/1e-15) — a collaboration, not this repo's to start unilaterally.
**Current focus: consolidate M1/M2/M4/M5/M6 and prepare a release** (tag v0.1.0 now; General
registration waits on the deps-first chain PureBLAS → StrictMode/TypeContracts → PureSparse, since the
Manifest currently dev-tracks PureBLAS master).

**2026-07-18: StrictMode 0.3.9 + guarantee gates.** Bumped StrictMode 0.3.8 → 0.3.9 (compat lower
bound raised to 0.3.9). Adopted two of 0.3.9's `@assert_*` guarantee macros as a new CI gate,
`test/strictmode_guarantees_tests.jl` (an isolated **checks-ON** subprocess — the main suite runs
checks-OFF for the zero-alloc gate, and `assert_enabled()` ⇒ `checks_enabled()`, so the guarantees
must run in their own temp env with AllocCheck+JET; ~168 s/run): (1) **`@assert_concurrency_safe
cholesky/ldlt(sym, A)`** — machine-proof that the allocating factor treats its `Symbolic` argument
read-only, i.e. one immutable `Symbolic` is safe to share by reference across concurrent
factorizations (the analyze-once thesis, req 7 — previously prose-only); (2) **`@assert_typestable
solve!`** for all three factor kinds (req 3). Targets are the **check-free** `solve!` + allocating
constructors, so the checks-ON body the macro analyzes equals the shipped checks-OFF path.
**Finding (tracked, not a release blocker):** `@assert_noalloc` PASSES for the QR `solve!` but
FAILS for Cholesky/LDLᵀ `solve!` — AllocCheck (all-paths, `ignore_throw`) finds a non-throw
allocation path the happy-path `@allocated == 0` gate does not exercise; the shipped hot path is
alloc-free (that gate passes), so tightening the Chol/LDLᵀ solve tree to AllocCheck-clean is a
follow-up. `@assert_trim_safe` also passes on `solve!` but only as a static heuristic (not juliac-
authoritative), and `TrimCheck.@validate` already covers `solve!` positionally, so it was not
adopted. Full suite green on 0.3.9 (225 950/225 950) incl. the new item; juliac --trim=safe green.

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
