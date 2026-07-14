# Adversarial review of `design_qr.md` (v1 draft) — findings

Reviewer: Opus, hostile pass, mirroring the `design.md` v1→v2 review process (that pass
found "2 BLOCKERs, 7 DEFECTs, algorithmic core verified sound"). Review-only: no code
written, `design_qr.md` not edited. Clean-room policy honored — only the five archived
papers, the existing PureSparse/PureBLAS source, and black-box `SparseArrays` output were
consulted; no SuiteSparseQR/CHOLMOD source in any form.

## Headline

**1 BLOCKER, 5 DEFECTs, 6 NITs. The two load-bearing derivations are otherwise sound:
H1 (star matrix) is fully confirmed by independent brute force, and H2's set-recurrence /
pivot bookkeeping produces correct V patterns — EXCEPT its `vcount` allocation formula,
which is wrong (goes negative) whenever a child column structurally evaporates. That is
the BLOCKER, and it lands squarely in H2, the design's own top-flagged hotspot.**

Method note: H1 and H2 were verified with a self-contained Julia brute force (no PureSparse
dependency): symbolic Cholesky of both `AᵀA` and the star matrix `S` for filled-graph /
etree / column-count equality, and dense staircase Householder QR reading off each
reflector's actual row support, over ~2400 random rectangular patterns plus hand cases.
Scripts in scratchpad (`verify_qr*.jl`).

---

## Self-flagged hotspot verdicts (H1–H6)

- **H1 (§3.2 star-matrix fill-equivalence): CONFIRMED SOUND.** Re-derived independently and
  agree; brute force found `etree(star)==etree(AᵀA)`, equal column counts, AND full
  filled-graph equality in 252/252 + 2424/2424 cases (0 failures). The survey quote
  ("the kth row and column of the star matrix is the union of rows in A whose leftmost
  nonzero appears in column k") is verbatim and correctly attributed to
  Gilbert–Li–Ng–Peyton 2001. The missing primary paper (BIT 41(4)) is adequately mitigated
  by the own proof + this executable check; not a blocker. One NIT on the proof wording
  (N1 below). **The Gilbert–Li–Ng–Peyton 2001 gap does NOT need to block implementation.**
- **H2 (§3.4 V-recurrence + pivot bookkeeping): REAL PROBLEM FOUND.** The *set* recurrence
  `S_k = assigned(k) ∪ ⋃_child survivors(c)` and the pivot/retire discipline are correct:
  with the design's own pivot-assignment staircase, `pattern(V_k) ⊇ actual reflector
  support` held in ALL 2302 non-evaporation cases (0 violations), exact in 93%, the rest
  legitimate upper-bound over-prediction. BUT the `vcount` allocation recurrence is wrong —
  see BLOCKER-1.
- **H3 (§5 rank policy): CONFIRMED SOUND (honest).** The `dropped_norm² = Σ discarded x[k]²`
  quantity IS the squared Frobenius residual `‖A−QR‖_F²` of the dropping — a correct
  a-posteriori certificate. The Foster–Davis phase-1 quote and Heath-"least-accurate" quote
  are verbatim in survey §7.4; the design correctly notes its own method is even less
  accurate than Heath (drops what Heath rotates) and the `\` basic-solution = SPQR §3.3
  method (3). Error bound and `\`-semantics are stated honestly. One clarity NIT (N2).
- **H4 (§1.3/§9.3 architecture & gate): MOSTLY SOUND, one real gap (DEFECT-3).** Left-looking
  v1 with multifrontal as gate-triggered M5b is well-justified and not a dodge — the
  escalation trigger ("M5a loses any stratum") is concrete and the milestone gate stays
  unconditional. The soft spot is the *baseline*, not the escalation: the gate is undefined
  against a possibly-TBB-parallel stdlib SPQR (DEFECT-3).
- **H5 (naming/constant provenance): ONE REAL DEFECT (DEFECT-2), rest OK.** `qr_tol_mult=8.0`
  is genuinely own (does not match SPQR's `20`); `beta`/`rowind`/`*_ptr` follow package
  convention. But `colamd_dense_row_mult=10.0` with `√n` scaling plausibly coincides with
  the COLAMD *package* default and is labeled "own, no external provenance" — the exact
  B2 trap from the original review (DEFECT-2).
- **H6 (§2.2 COLAMD vs sources): CONFIRMED SOUND on the journal paper; one citation DEFECT.**
  All 12 checked §2.2 claims (eq (1)/(2)/(4), Algorithm 3 `w`/`v`/`t` tag bookkeeping, §4.7
  aggressive absorption, §4.8 recommended variant incl. the deliberate COLMMD-initial /
  AMD-maintained mismatch and the "kept bug"/8% story, dense-threshold non-prescription,
  complexity, super-column hashing, and the "not-AMD-on-a-different-graph" meta-claim) are
  faithful to Davis–Gilbert–Larimore–Ng 2004. The Larimore thesis ch.3–4 (read in full by a
  sub-reviewer) confirms the implementation-depth claims and surfaces several precision
  points the condensed §2.2 omits (fine — §2.2 explicitly defers these to the thesis + task
  3), see NIT N6. The one defect is a survey mis-citation for Liu 1986c (DEFECT-1).

---

## BLOCKER

### BLOCKER-1 — §3.4 `vcount` formula under-allocates V (goes negative) on structural rank deficiency

**What.** §3.4 states:

> `vcount[k] = a_k + Σ_{c child of k} (vcount[c] − 1)` … `nnzV = Σ vcount`

and, separately, that a structurally dead pivot has `vcount[k] == 0` (Oliveira 2001
evaporation). These two statements are mutually inconsistent: a child `c` with
`vcount[c] == 0` passes **0** rows up (its `S_c` is empty, survivors empty), but the
formula subtracts `vcount[c] − 1 = −1` for it. Every evaporated child therefore
over-decrements its parent by 1.

**Evidence.** Brute force over 2424 random rectangular patterns: the formula as written
disagrees with the true `|S_k|` in 29 cases, and **every** disagreement is a case with at
least one evaporated column; the guarded form `Σ_{c: vcount[c]>0}(vcount[c]−1)` (equivalently
`max(vcount[c]−1, 0)`) matches `|S_k|` in **100%** of all 2424 cases with zero set-overlaps.
Minimal witness: column etree `1→2`, column 1 evaporates (`vcount[1]=0`), `a_2=0` ⇒ the
design formula gives `vcount[2] = 0 + (0−1) = −1`. A negative allocation.

**Why it matters.**
1. It is a correctness bug in the single most-flagged hotspot (H2). An implementer
   transcribing §3.4 verbatim ships it.
2. It triggers on structurally rank-deficient blocks — a **first-class** scenario for this
   design (§5 rank handling is a headline feature), not an exotic corner. ~5% of random
   rectangular patterns here contained an evaporation.
3. `nnzV = Σ vcount` under-counts (or is negative) ⇒ `vptr`/`vval` under-sized ⇒ the numeric
   loop writes `V_k` past its allocation into the next column's slots. That is a
   memory-safety bug and a direct violation of the zero-alloc / exact-sizing contract
   (CLAUDE.md req 5), precisely on the inputs §5 exists to serve.
4. **The stated safety net does not catch it.** §3.4/§9.1 name the "superset invariant" as
   H2's protection, but that invariant checks *pattern(V_k) ⊇ produced nonzeros* — an
   OVER-allocation guard. Under-allocation from this bug is the opposite failure and slips
   straight through it. (This is a direct answer to H2's own question "review whether the
   invariant as stated actually catches a wrong pivot convention": for this failure mode,
   no.)

**Fix.** Clamp the child contribution: `vcount[k] = a_k + Σ_{c child of k} max(vcount[c] − 1, 0)`
(equivalently, only sum over children with `vcount[c] > 0`). Add a first-class test that
constructs a matrix with a known evaporated column and asserts `nnzV == Σ|S_k|` and
`all(vcount .>= 0)`. Also add an *under*-allocation guard to §9.1 (e.g. assert the numeric
loop's per-column V write count equals `vcount[k]`), since the current superset invariant is
one-sided.

---

## DEFECTs

### DEFECT-1 — §3.4 / §11 provenance: "Liu 1986c's row-merge tree gives the counting view" is not supported by the cited source
The survey (the design's stated route to Liu 1986c) attributes Liu 1986c's row-merge tree to
block-Row-Givens merging (§7.2) and to deriving the etree of `AᵀA` without forming it
(§11.5) — **never** to a "counting view", and it does not appear in the §7.3 V-pattern
discussion at all. The "counting" row-merge tree the design actually leans on is the COLAMD
paper's (Liu 1991, via Davis–Gilbert–Larimore–Ng), a *different* citation. This is the exact
B1/B2 discipline the project enforces: a citation must survive "where did this come from?".
As worded it reads as survey-sourced but is the author's own synthetic bridge. **Fix:** drop
the Liu-1986c-for-counting clause from §3.4's provenance list and the §11 table, or relabel
it explicitly "own inference, by analogy to the COLAMD row-merge count recurrence (Liu 1991)".

### DEFECT-2 — §1.6/§2.2 pt 5: `colamd_dense_row_mult = 10.0` × √n is labeled "own" but plausibly matches the COLAMD package default
The paper legitimately prescribes no threshold ("problem dependent"; COLMMD's 50% "probably
too high") — confirmed verbatim. But the design's chosen default, `max(16, 10·√n)` rows /
`max(16, 10·√m)` cols, uses mult=10 with √-of-dimension scaling, which is (to the reviewer's
knowledge) the shape and constant of COLAMD's *shipped* default dense knob. Calling a value
that coincides with the real implementation's default "our own free tunable, no external
provenance" is exactly the failure mode the original review caught with the `0.8/0.1/0.05`
amalgamation thresholds. **Fix:** check `10·√n` against the COLAMD **user guide** (a permitted
source — not the C source). If it matches the package default, relabel honestly ("matches the
COLAMD package default, adopted deliberately") rather than "own"; if it doesn't, keep "own"
but say what it was derived from (the note "same as our AMD `AMD_DENSE=10`" is a start but
does not explain the √n or the `16` floor — see NIT N3). Treat as BLOCKER-tier if the match
is confirmed, per the original review's precedent.

### DEFECT-3 — §9.3 gate is undefined against a possibly-parallel SPQR baseline (H4)
§9.2 honestly flags that the harness must check whether stdlib SuiteSparseQR runs TBB tree
parallelism and, if so, "record and report" it. But §9.3 states the milestone gate as an
unconditional single-thread wall-time inequality with "no fudge-factor gates, no 'within 2×
is fine'". If SPQR does run multi-threaded and cannot be pinned to one thread from Julia, the
inequality "single-thread PureSparse cold < multi-thread SPQR cold" may be unwinnable for
reasons orthogonal to the algorithm, and the design gives no rule for that case — "report" is
not a gate decision. **Fix:** define the rule now: attempt to force SPQR to a single thread
(document the mechanism, e.g. env var, and whether it is even honored), and state explicitly
that the gate is *single-thread-SPQR vs single-thread-PureSparse*; if SPQR cannot be pinned,
state that the gate is evaluated against single-thread-SPQR wall time as separately measured,
not against the parallel run. (The Cholesky gate accepted an analogous CHOLMOD situation via
the 4-arm design; QR needs the equivalent explicit resolution.)

### DEFECT-4 — §9.1 superset invariant is mis-scoped as H2's safety net
Folded into BLOCKER-1 but worth its own line because it is a *documentation/testing* defect
independent of the fix: the design repeatedly presents the "superset invariant" as the thing
that protects H2's novel bookkeeping. It only guards over-allocation / stray fill; it is
structurally incapable of catching an under-count (BLOCKER-1) or an under-sized `vptr`. **Fix:**
add a complementary lower-bound / exact-count check (per-column produced-V-nnz `== vcount[k]`;
`nnzV == Σ|S_k|`) and stop describing the one-sided superset test as *the* H2 safety net.

### DEFECT-5 — §1.4 `rcount` length inconsistency across the singleton split
`QRSymbolic.rcount` is commented "length n" and equated with "colcount of L(AᵀA)", but
everything in §3 runs on the *non-singleton block* of size `n−n1`, and `parent` is
correctly declared `length n−n1`. `rcount` (and `rptr`, `vptr`, `vrowind` sizing) is over the
block, not full `n`, whenever `n1 > 0`. As written an implementer cannot tell whether
`rcount` is indexed by original column or block column. **Fix:** state the indexing basis
explicitly for every `QRSymbolic` array (original-`n` vs block-`(n−n1)`), and make `rcount`'s
declared length consistent with `parent`'s.

---

## NITs

- **N1 — §3.2 proof under-argues the path-endpoint edge.** The fill-path detour argument is
  stated for a clique edge `(v_j,v_k)` that is "interior" (`v_j < min(a,b)`). It does not
  explicitly handle the clique edge incident to a path *endpoint* (where `v_j = a` or
  `v_k = b`). The result still holds there — `v₁ ≤` the other, interior end of that edge
  `< min(a,b)` — and brute force confirms filled-graph equality universally, so this is
  wording only. Add the one-clause endpoint case for completeness.
- **N2 — §5.2 clarity.** "The per-column tail dropped at detection is itself ≤ τ" is true only
  for the drop *at the detecting column*; the later per-column discards (`x[k]` for `j > k`
  with dead `k ∈ T^j`) can exceed τ and are the bulk of `dropped_norm`. The text accumulates
  them correctly but the sentence could be read as bounding all drops by τ. Reword to make
  clear only the detection-time tail is τ-bounded; the certificate sums the (unbounded) rest.
- **N3 — §1.6/§2.2 the `16` floor and `√n` scaling in the dense threshold have no stated
  derivation.** Even setting aside DEFECT-2's provenance question, `max(16, …)` — why 16? —
  and the √-scaling are asserted, not derived. Give a one-line rationale or mark them
  calibration placeholders like the amalgamation tunables were in design.md B2.
- **N4 — §4.4 reflector `sign(0)` edge case unspecified.** With `v[pivot] = x[pivot] +
  sign(x[pivot])·‖x‖`, if `x[pivot] == 0` but `‖x‖ > 0`, `sign(0)=0` gives `v[pivot]=0` and
  `R[k,k]=0`, mis-forming the reflector. PureBLAS's own `qr.jl` handles this (`head ≥ 0 ?
  nrm : −nrm`, i.e. `sign(0):=+`). Specify the `sign(0):=+1` convention.
- **N5 — §2.2 pt 3 "16 variants … recommends this combination (… super-rows/super-columns
  ON)".** Per the paper, super-rows/super-columns are an always-on implementation feature,
  not one of the four binary dimensions of the 16-variant sweep. Minor: don't imply
  super-rows/columns were a swept dimension. (Also: the paper's Algorithm 2 uses `K = ∅`, not
  `{k}`, in the degenerate `l_k = 0` case — the design's `∪ {k}` silently assumes the generic
  case, same as the paper's own prose; harmless but worth a footnote.)
- **N6 — §2.2 is journal-condensed; several thesis-level precision points are absent.** The
  full read of Larimore ch.3–4 surfaced details a bare journal reading would get subtly
  wrong: the per-row `−1` in the *initial* degree `d_j = Σ_{i∈C_j}(|R_i|−1)`; that aggressive
  row absorption (row-level, inline during the set-difference scan) and further mass
  elimination (column-level, after degree summation) are two distinct mechanisms; that final
  scoring MUST follow super-column detection (`d_j = d_j + |R_r| − |j|`, `|j|` = grown
  supercolumn size); the exact hash `(Σ_{i∈C_j} i) mod n_col`; the lazy sentinel reset of
  `w`; and the ones'-complement garbage-collection row marker. §2.2 *explicitly* defers this
  to the thesis + task 3, so this is not a defect — but the task-3 checklist should name
  these specific items so they are not lost.

---

## Checked and found sound (so a re-reviewer need not redo)

- **PureBLAS §4.6/§7.2 (re-verified against source):** `svd.jl:_apply_reflectors_left!`
  (line 641) exists and is the real compact-WY apply `C −= V·(T·(Vᵀ·C))` via `gemm!`,
  Float64, coupled to `SVDWorkspace{Float64}` scratch (`bt_T`/`bt_G`/`bt_W`/`bt_Yb`) — the
  design's corrected description is accurate, including that it is `_`-private and
  SVD-workspace-bound. `geqrf!`/`qr_unblocked!` are Float64 + BlasComplex only; the generic
  `T<:Real` path is genuinely absent (`cabi_lapack.jl:14`, `qr.jl:7`). §4.6/§7.2 P1/P2
  scoping is correct. (One residual PureBLAS note the design already makes: `geqrf!` uses a
  module-global grow-on-demand `_QR_WS`; flagged for M5b — fine.)
- **M5b symbolic reuse (§7.1):** `fundamental_supernodes`, `relaxed_amalgamation`,
  `supernode_tree`, `supernode_rowind`, and the `AMALG_*` tunables all exist in
  `symbolic/supernodes.jl` and take `(n, parent, colcount)`; feeding `(n−n1, parent, rcount)`
  is valid because `rcount == colcount(L(AᵀA))` (H1) and the fundamental-supernode predicate
  `colcount[j]==colcount[j+1]+1` carries over. Reuse claim holds.
- **`column_counts` reuse (§3.1/§3.2):** `symbolic/counts.jl` is the Gilbert–Ng–Peyton
  skeleton-leaf algorithm and is pattern-generic; its output on the star pattern is `rcount`.
  Sound. (Minor provenance nuance: `counts.jl`'s own header attributes it to GNP *1994* as
  presented in Davis's book, while §3.2/§11 cite the GNP *1992* ORNL tech report as "the
  primary source … verified `counts.jl` … is that algorithm" — same authors/algorithm, the
  1992 report being the pre-print of the 1994 journal paper; not a real discrepancy.)
- **Types/contracts (§1.4/§6.5):** `QRFactor <: AbstractSparseFactor{T}` satisfies the
  existing `contracts.jl` surface (`solve!`, `issuccess`); a separate `QRSymbolic` rather
  than extending `Symbolic` is justified. The new `order_columns` second ordering entry point
  does not conflict with the existing `order` contract. No inconsistency beyond DEFECT-5.
- **Survey/SPQR paraphrases (H1/H2/H3/§6):** George–Ng 1986/87, George–Liu–Ng 1988 row-path,
  Oliveira evaporation, the `qr_left_householder` dense pseudocode + row-subtree sparse spec,
  George–Heath–Ng 1984 min-norm, §7.5 three-way alternatives guidance, the SPQR numbers
  (215/353 singletons; Theorem 1; §3.3 methods (1)/(2)/(3); 2.49 vs 2.67 GFlops) — all
  verified faithful (only the Liu-1986c clause, DEFECT-1, and the loosely-cited §7.5-for-QR
  row, folded into N-level, were off).
```
