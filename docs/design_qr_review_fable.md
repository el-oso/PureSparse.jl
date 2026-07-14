# Adversarial review of `design_qr.md` (v1) — Fable, independent pass

Reviewer: Fable (second, independent reviewer; no other review consulted).
Scope: full read of `design_qr.md` @ ff1150f, checked line-by-line against the five
archive PDFs (COLAMD 2004 read in full, all 24 pp.; SPQR 2011 read in full, all 25 pp.;
survey pp. 52–66 + 130–136; GNP92 §1; Larimore thesis TOC + printed pp. 19–22 and
27–33), against PureBLAS source (`src/qr.jl`, `src/cabi_lapack.jl`, `src/svd.jl`), and
against **computational brute-force checks run for this review** (Julia 1.12.6):

- **H1 check**: 3,000 random rectangular patterns (m ∈ 1..14, n ∈ 1..12, density
  0.05–0.65): boolean-elimination filled graph, etree, and column counts of
  pattern(AᵀA) vs. the star matrix — **identical in all 3,000 trials**.
- **H2 checks**: (a) 4,000 random-valued trials simulating the design's §4.1/§5.2
  numeric semantics (reflector over the symbolic S_k pattern, designated pivot,
  dead-pivot parking): the §9.1(3c) **superset invariant held in all 4,000 trials**
  (no numeric nonzero ever landed outside S_k); (b) 1,500 trials of the row-path
  property (§3.4 consistency property) — 0 failures; (c) vcount recurrence in its
  live-children form — 0 failures; (d) targeted counterexamples for the findings
  below (reproductions given inline).
- §9.2 environment facts re-verified live (`SparseArrays.SPQR.QRSparse{Float64,Int64}`,
  `.R/.Q/.prow/.pcol`, `ORDERING_FIXED/AMD/COLAMD/METIS/DEFAULT`, `rank(F)`, `F\b`).

Verdict up front: **no clean-room violations found** (every name/constant traced to a
permitted source — details in "Verified clean" below), H1 is correct (one small proof
gap, D3), and the H2 recurrence is *safe* (superset holds) — but three genuine
correctness errors sit in the §3.4/§4.4 bookkeeping, exactly where §0 predicted.

---

## BLOCKER

### B1. §3.4 `vcount` recurrence produces negative counts when an etree child is structurally dead

The stated recurrence — `vcount[k] = a_k + Σ_{c child of k} (vcount[c] − 1)` — subtracts
1 for **every** etree child, including dead children (`vcount[c] == 0`), which
contribute no pivot row to retire. Dead columns *can* be etree children.

Reproduction (run for this review): `A = [1 1 1]` (1×3). Column etree is the chain
1→2→3; `S = [{row 1}, ∅, ∅]`. The literal formula yields `vcount = [1, 0, −1]`
(actual `|S_k| = [1, 0, 0]`). A negative `vcount` corrupts `vptr`/`nnzV` construction
(§1.4) and every workspace bound derived from it (§3.5).

Fix: `vcount[k] = a_k + Σ_c max(vcount[c] − 1, 0)` (equivalently: sum over live
children only). This is precisely the "off-by-one that's *sometimes* right" failure
mode §0 H2 warns about — the safety-net discussion in §0 H2 should also note that the
§9.1 superset invariant does **not** catch this bug (a negative count fails earlier
and differently, at allocation time, not as a pattern violation).

Related paper note: COLAMD eq. (2) (paper p. 359) has a single global "−1" while the
design's recurrence has a per-child "−1"; they are consistent only through the offset
`vcount[k] = l_k + 1`, and eq. (2)'s absorption machinery never sums a dead child the
way the design's etree-children form does. §2.2 pt 1's claim that the two are
"*identical algebra*" obscures exactly this (see N1).

### B2. §3.4/§1.4/§4.5 staircase row numbering is unrepresentable when dead pivots exist, and always when m < n

The scheme "pivot row of column k gets number k; non-pivot rows receive numbers > n"
cannot be realized in the declared data structures (`rperm::Vector{Ti}` of length m,
§1.4; `x::Vector{T}` of length m, §4.5):

- **m < n (a declared goal, §1.1 "any shape")**: a live pivot column with index k > m
  needs permuted number k > m. Reproduction: `A = [0 1]` (1×2) — column 2 is live,
  its pivot row must be numbered 2, but m = 1.
- **m ≥ n with dead columns (the §5 rank-deficient path)**: with d dead columns there
  are `(m − n) + d` non-pivot rows but only `m − n` numbers above n. Reproduction:
  `A = [1 1; 0 0]` (2×2): column 2 is dead; row 2 must take *some* number, and the
  only consistent choice is the dead column's number 2 — but then §1.4's invariant
  "`vrowind` column k starts with k, the pivot slot" and §4.1's harvest indexing
  (`R[i,k] = x[i]` at pivot slot i) collide with a live row parked at a dead pivot
  number in the general case.

The design says "when m < n dead pivots absorb the shortfall" — that acknowledges the
counting identity but not the representation problem: as specified, `rperm` is not a
permutation of 1..m, and `x` (length m) has no slot for pivot numbers k > m. The fix
is mechanical but must be *designed*, not improvised in task 5: either (i) a virtual
row space of size `max(m, n)` (x, riperm sized accordingly; virtual rows structurally
empty), or (ii) an explicit `pivotslot[k]` indirection decoupling "R row k" from
"permuted row k". Whichever is chosen, §1.4's `vrowind` comment, §4.1 steps 1/3/4,
and §4.5's sizes all need the same convention. (Both reproductions verified in code
for this review.)

### B3. §4.4 Householder kernel divides by zero on a numerically empty live-pattern column when `tol ≤ 0`

`beta = 2/(vᵀv)` with `v = x + sign(x[pivot])·‖x‖·e`: if `x` is exactly zero on the
whole pattern, `v = 0` and `beta = 2/0` → Inf/NaN poisons R, V, and every later
column that touches them. This input is **reachable, not hypothetical**: a
symbolically live S_k can be numerically all-zero without any value cancellation
(survivor rows whose remaining column pattern is empty — the same structural
early-death mechanism as D2). In the 4,000-trial simulation, **78 columns** across
the trials were live-pattern-but-numerically-zero.

With rank detection ON, the §5.1 τ-test intercepts these (`‖x‖ ≤ τ` → dead). But §5.3
explicitly supports `tol ≤ 0` ("disables rank detection entirely — exact structural
behavior; structurally-dead pivots still handled"): in that mode nothing guards the
division. "Structurally-dead pivots still handled" covers only `vcount == 0` columns,
not live-pattern/zero-value ones. Fix: the kernel itself must set `beta = 0` (H = I)
when `‖x‖ == 0`, independent of the rank-detection setting (the same convention the
design already relies on for dead columns). One sentence in §4.4; without it the
supported `tol ≤ 0` mode produces NaN factors on rank-deficient input.

---

## DEFECT

### D1. §2.2 pt 5 / §1.6: "the paper prescribes no threshold" misstates the COLAMD paper; "shape is ours" misstates the provenance chain

COLAMD paper p. 362 (checked): "Determining how dense a row or column should be for
it to be withheld is problem dependent. **We used the same default threshold used by
MATLAB's COLMMD, 50%**, which is probably too high for most matrices." The paper
*used and shipped* a 50% default; it did not decline to prescribe one. The thesis
(§4.2.3, printed p. 32, checked) likewise has default dense knobs. The design's "the
paper prescribes none and flags COLMMD's 50% as too high" (§1.6) is an inaccurate
paraphrase presented as the provenance basis for choosing our own default.

Separately, the `max(16, 10.0·√n)` **shape** is not "ours" (§1.6 "threshold shape is
ours"): it is the existing PureSparse AMD heuristic verbatim, whose declared
provenance (checked: `src/tuning.jl:54`, design.md §0 N5/§2.2 pt 6) is the **AMD
package User Guide default** — a permitted source, so no clean-room violation, but
the chain should be stated as "inherited from design.md §2.2 pt 6 (AMD User Guide)"
rather than claimed as an independent derivation. As written, the claim would not
survive the project's own "where did this come from?" test, even though the number
itself is legitimately sourced.

### D2. §3.4 exactness claim: `pattern(V_k) = S_k` is an upper bound, not the exact numeric support

"Column k's reflector acts on exactly the rows S_k" — the reflector is *applied* over
S_k by construction, but S_k strictly overpredicts the true nonzero support whenever a
survivor's remaining column pattern is empty (structural early death; no value
cancellation involved). Measured: **118 of 4,000** random trials contain at least one
strictly overpredicting column (138 columns total). Reproduction from the run: an
11×5 pattern where rows {4, 9} both have pattern {3}: after column 3's reflector both
rows are structurally exhausted, yet the recurrence carries row 9 into S_4.

George–Ng's published statement (survey p. 56: "the pattern V_k is the union of the
pattern V_c of each child c … and also the entries in the lower triangular part of
A") has the same upper-bound character; the exactness results quoted nearby (survey
p. 55, Ostrouchov/George–Ng "exactly represents the intermediate fill-in") carry a
zero-free-diagonal/square assumption (survey p. 61: "assuming A is square with a
zero-free diagonal") that general LS matrices do not satisfy.

Consequences the design must state: (a) V stores structurally-guaranteed zeros —
harmless for correctness (superset invariant verified, 0/4,000 violations) and the
§3.5 flop count remains exact *as performed work* (the loop really does apply over
S_k), but `nnzV` overcounts true nonzeros; (b) any test comparing V's pattern for
*equality* against a numeric oracle will fail — §9.1's superset formulation is the
right one and must not be "tightened" later; (c) B3 above is the sharp edge of the
same phenomenon.

### D3. §3.2 fill-path proof has an uncovered case (conclusion still true; verified by brute force)

The converse direction argues the detour v_j–v₁–v_k stays legal because "v_j is an
*interior* vertex of the path (so v_j < min(a,b))". If the replaced clique edge is the
first or last edge of the fill path, v_j (or v_k) **is an endpoint**, not interior,
and the stated inequality doesn't apply to it. The proof still goes through — the
needed fact is only that the *newly added interior vertex* v₁ satisfies
v₁ < min(v_j, v_k), and since each of v_j, v_k is either an endpoint of the path or
interior (< min(a,b)), v₁ < min(a,b) in every case — but the case analysis as written
is incomplete. One-line fix. (Empirically: filled graph, etree, and counts identical
on 3,000 random patterns, including empty rows/columns and m < n; H1 stands.)

### D4. §4.6/§7.2 (P1): the existing PureBLAS kernel implements only C := Q·C — the multifrontal front update needs QᵀC

Checked against `svd.jl:641–671`: `_apply_reflectors_left!` iterates blocks
right-to-left with the forward-columnwise (dlarft-style) T, i.e. it computes
**C := Q·C only** (the SVD back-transform direction). The M5b front factorization the
design describes (§4.6: "factor the pivotal column block, then apply its reflectors to
the non-pivotal columns") is the **transposed** application — dormqr's 'T' case
(Tᵀ, blocks left-to-right) — which does not exist in PureBLAS today. Same for §7.3's
"apply_Qt! becomes per-front larfb sweeps". The adaptation remains small (transpose
the T triangle, reverse the block order), so P1's "adapt, don't derive" conclusion
survives, but §4.6's "it is the real thing" and §11's "verified present and correct"
overstate: the *role* is present, the *direction M5b actually needs* is not. P1's
scope should say so explicitly. (The §4.6 `_QR_WS` grow-on-demand observation is
accurate — confirmed at `qr.jl:264–271`; both quoted source comments, `qr.jl:7` and
`cabi_lapack.jl:14`, are verbatim.)

### D5. §2.2 pts 1–2: the condensed C_j update drops Algorithm 2's `l_k = 0` branch

Paper p. 361, Algorithm 2 (checked): after the subtraction pass, `K = {k}`, **but if
`l_k = 0` then `R_k = ∅; K = ∅`** — the pivot row is discarded and `{k}` is *not*
added to any C_j. The design's condensation "`C_j = (C_j \ Cₖ) ∪ {k}` for every
j ∈ Rₖ" silently loses this branch. The paper notes `l_k = 0` "can occur for k < n if
the matrix is not strong Hall" — i.e. routinely, for exactly the rectangular/
rank-deficient inputs this milestone targets. An implementer following §2.2's formula
instead of the paper's Algorithm 2/3 verbatim would insert phantom `{k}` references.
Task 3 does mandate implementing from the paper's algorithms, which mitigates; the
design text should still carry the branch.

### D6. Solve-method numbering cited to the wrong SPQR section

§6.2 ("paper §3.3 method (3)"), §6.3 ("SPQR paper §3.3 method (2)"), §5.2 ("SPQR
paper §3.3 method (3) semantics"): the (1)/(2)/(3) enumeration is in **§5.1 "The
methods"** (paper p. 15), not §3.3. §3.3 contains the LS formula `x=P*(R\(Q'*b))`
(that citation is correct) but no numbered method list. Content of all three claims
is otherwise accurate against §5.1 (checked).

### D7. §1.2: "α heuristic documented there" — the survey documents no α heuristic

Survey §7.5 (p. 64, checked) says only "Replacing I with a scaled identity matrix αI
can improve the conditioning"; §11.5 (p. 131, checked) says α's "optimal value is only
approximated through heuristics" — and gives none. The guidance table promises users
a documented heuristic that the cited source does not contain. Either cite a source
that actually gives one (e.g. Björck's literature, if added to the archive) or weaken
the table entry.

### D8. §6.3 transpose-factorization identity has P and Pᵀ swapped

From `Aᵀ·P = QR` follows `A = P·Rᵀ·Qᵀ`, not "`A = Pᵀ·Rᵀ·Qᵀ`" as written. The
*operational* formulas that follow are correct (solve `Rᵀz = Pᵀb`, then `x = Q·[z;0]`
— matches SPQR §5.1 method (2), `x=Q*(R'\(P'*b))`, checked), so this is a display
error, but in a docstring-bound formula where a sign/side error would send an
implementer debugging the wrong thing.

### D9. §3.4: "row k of R is structurally empty" for a dead column is false

A structurally dead column (vcount = 0) can still have `rcount[k] > 1`: for
`A = [1 1 1]`, column 2 is dead yet row 2 of the Cholesky-of-AᵀA pattern is {2,3}
(rcount = 2). Row k of R is **numerically** zero but structurally sized by `rcount`
(which is what §1.4/§3.3 allocate, and what Heath-style handling *requires* per SPQR
§2.3 — the design's own §1.1 argument). A test written from §3.4's sentence
(asserting structural emptiness or `rcount[k] == 1` for dead k) would be wrong.
Reworded: "row k of R is numerically empty (its structural slots stay zero)".

---

## NIT

- **N1.** §2.2 pt 1 "identical algebra … no accident": eq. (2) and the vcount
  recurrence are offset-equivalent (`vcount[k] = l_k + 1`), not identical; the +1
  offset is exactly where B1's off-by-one lives. State the offset explicitly.
- **N2.** §2.2 pt 3 "rejected … by the paper's own experiments" is loose for two of
  the four: exact degree (§4.2) was rejected on cost, explicitly *not tested* ("We
  thus did not test this method"); approximate deficiency (§4.6) results were "mixed
  … about the same", not experimentally rejected.
- **N3.** §2.2.6 "AMD wins most of its large LS set" (SPQR Table VI): in the "Best"
  column AMD takes 5 of 11 (plurality; METIS 4, COLAMD 2) — "most" overstates.
- **N4.** §3.4 pivot rule, "else the inherited row of smallest current number":
  inherited rows have no permuted number at that point in the pass (non-pivot numbers
  are assigned later), so "current number" is undefined; presumably "smallest original
  index". Make the deterministic rule precise — it feeds `rperm` and hence bitwise
  reproducibility (§9.1 pt 5's `qr!`-equals-`qr` test).
- **N5.** §7.3 "upper bounds when τ > 0": SPQR (p. 9) says exact when τ < 0
  (disabled), upper bound when **τ ≥ 0** — the τ = 0 boundary belongs to the
  upper-bound side.
- **N6.** §9.2/§9.3: if SuiteSparseQR turns out to run TBB parallelism that cannot be
  disabled, the design records-and-reports but never says what then closes the gate.
  Decide the rule now (e.g. gate against the recorded parallel baseline anyway, or
  add a taskset/env control), or M5 closeout inherits an ambiguity.
- **N7.** §7.1 cites the staircase to "paper §3.1"; SPQR defines it in §2.3 (p. 7)
  and illustrates it in §3.1. Cosmetic.
- **N8.** §1.4: `parent` is block-local (length n−n1) while `rcount`/`rptr`/`vptr`
  are global (length n/n+1); the offset convention between the two index spaces is
  never stated — a known implementation trap when n1 > 0.

---

## Verified clean (checked, no finding)

Reported so the passing checks are on the record; each was checked against the
primary source, not the design's paraphrase.

1. **Clean-room provenance (H5): no violations found.**
   - `beta`: survey §7.3's own pseudocode variable (`Beta`, p. 54/61 listings) ✓.
   - COLAMD routine names in §2.2/task 3 (`init_rows_cols`, `init_scoring`,
     `find_ordering`, `order_children`, `garbage_collection`, `detect_super_cols`):
     all are **published thesis content** (TOC §§4.2.1–4.2.7; body pp. 27–33 checked) ✓.
   - Thesis-attributed details all real: index array of size `2·nnz + n_col` with
     merged-row construction + garbage collection (§4.1.3, verbatim); degree list and
     hash table sharing one head array with `headhash` collision handling (§4.1.4);
     shared-variable overlays (§4.1); newly-null column detection and natural-order
     tie-breaking via degree-list insertion order (§4.2.3); §3.2 carries the
     initial-metric COLMMD-vs-AMD finding at derivation depth (pp. 19–21, incl.
     Fig. 3.1 and the "quite by accident" account matching the journal's §4.8 story) ✓.
   - `S_k` notation: the survey itself uses S_k for Oliveira's active sets (p. 57) ✓.
   - `R11/R12/A22`, `P₂`, staircase, singleton terminology: SPQR paper ✓.
   - τ default (§5.3) and reflector convention (§4.4): declared own derivations,
     structurally distinct from anything in the read papers ✓ (no source available to
     me that they accidentally match; the design's own "must not drift" warning is
     the right posture).
   - Dense thresholds: chain is legitimate (AMD User Guide via design.md) — only the
     *description* is wrong (D1).
2. **§2.2 vs COLAMD paper**: eq. (1) (incl. "regular row absorption" term), eq. (2),
   eq. (3) + O(|A|) initialization, eq. (4) incl. the `Rs`/"most recent pivot row"
   condition, Algorithm 3's `w`/`v`/monotone-`t` semantics ("at the end of the first
   phase, wᵢ − t = ‖Rᵢ \ Rₖ‖, vᵢ − t = ‖Aᵢ \ Rₖ‖" — verbatim), §4.7 aggressive
   absorption (deletion when i ∉ Cₖ; "costs almost nothing" quote), §4.8 recommended
   variant (initial COLMMD + AMD update + no initial aggressive absorption +
   aggressive-during-elimination + super-rows/columns) and the kept-"bug"/8% story,
   super-column hash (Ashcraft) + mass elimination + immediate elimination when
   C_j = {k}, storage argument, complexity — **all accurate**.
3. **§3.2/H1**: construction quote is near-verbatim survey §7.1 (p. 54); GNP92 §1
   states the QR application verbatim as quoted; `counts.jl` reuse claim consistent;
   brute force passed 3,000/3,000.
4. **§3.4/H2 core**: superset invariant 4,000/4,000; row-path property 1,500/1,500;
   Oliveira's published convention (survey p. 57: "One row is selected as a pivot,
   and the remainder are sent to the parent") already covers the retire-one/pass-rest
   scheme — §3.4's provenance is *stronger* than the design claims for itself.
5. **§5 vs sources**: Heath description matches SPQR §3.2 (incl. the "2-norm of
   column 6" quote and Theorem 1 + its induction proof as summarized); survey §7.4
   "least accurate" quote exact; Foster–Davis phase-1 quote exact (survey p. 63);
   Pierce–Lewis can't-keep-Q rationale matches SPQR §3.2; Tikhonov fallback with
   γ = 10⁻¹²·max_j‖A_{*j}‖₂ is indeed benchmarked in SPQR §5.2. H3's honesty framing
   is fair.
6. **§2.3 singletons vs SPQR §2.1**: 215/353 LP statistic, threshold-τ definition,
   trapezoidal R11 case, reuse-path disable quote (verbatim), cost model
   (O(|R11|+|R12|) + O(n) scan + O(|A|) prune), breadth-first characterization,
   transpose-needed-anyway observation — all accurate.
7. **§1.3/§7 vs sources**: "very competitive when R remains very sparse" (SPQR §1) ✓;
   2.49 vs 2.67 GFlops single-core (§5.5) ✓; flop-accounting caveat (§5.5: reflections
   counted one-at-a-time, applied blocked) ✓; COLAMD-iff-m≤2n default (§5.4) ✓;
   supernode two-condition + relaxed amalgamation (§2.3) ✓; workspace-allocated-
   before-parallel-phase (§4, p. 13) ✓; MA49/Matstoms landscape (survey §11.5) ✓;
   eq. (11.3) ✓.
8. **§4.6 vs PureBLAS source**: `geqrf!` blocked compact-WY for Float64
   (`qr.jl:275`) and BlasComplex (`qr.jl:197`); both quoted comments verbatim
   (`qr.jl:7`, `cabi_lapack.jl:14`); `_apply_reflectors_left!` at `svd.jl:641` with
   `SVDWorkspace`-coupled `bt_T/bt_G/bt_W/bt_Yb` scratch exactly as described;
   `_QR_WS` grow-on-demand global (`qr.jl:264`); generic-`T` path confirmed absent.
   All §4.6 claims accurate **except** the direction gap in D4.
9. **§9.2 baseline facts**: re-verified live in this environment (all pass).
10. **§9.3 gate**: wall-time medians, locked clocks, single-thread, both permutation
    arms, per-stratum unconditional inequality, GFlops demoted to diagnostic —
    conforms to CLAUDE.md req 2 / design.md D2. Cold-vs-cold is justified (stdlib
    exposes no analyze-once path — confirmed: each `SparseArrays.qr` call reanalyzes)
    and warm `qr!` reported-but-not-gated is the honest treatment. Only open edge is
    N6 (TBB contingency undefined).
11. **Archive-index correction** (§11): `refs/linear_algebra/chapter-direct.pdf`
    first page is indeed "CHAPTER 3 — Computing the Cholesky Factorization of Sparse
    Matrices" in support-preconditioner context (checked via pdftotext); the design's
    correction of design.md §11's label is right.

## Summary counts

- BLOCKER: 3 (B1 vcount dead-child off-by-one; B2 row-numbering unrepresentable for
  m < n / dead pivots; B3 beta = 2/0 on zero live-pattern columns when tol ≤ 0)
- DEFECT: 9 (D1–D9)
- NIT: 8 (N1–N8)
- Clean-room: **no violations**; one provenance *description* error (D1).

The three blockers are all in the §3.4/§4.4 bookkeeping layer the design itself
flagged as H2 — the self-diagnosis was accurate, and all three have small, local
fixes. H1, the COLAMD specification, the SPQR/survey sourcing, and the gate design
survive adversarial checking essentially intact.
