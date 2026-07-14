# PureSparse.jl — Sparse QR Design Document (v1, DRAFT — awaiting adversarial review)

Adds sparse QR factorization (least-squares / minimum-norm / rank-revealing-lite) to
PureSparse.jl as **milestone M5**, scheduled before the GPU backend, which is renumbered
**M3 → M6** (see §1.0). Clean-room reimplementation of the SuiteSparseQR algorithm
family **from published papers only** — SuiteSparseQR and CHOLMOD source are GPL and are
never read, in any form (§11; same absolute policy as `design.md` §11).

This is a **v1 Fable draft**, produced by the same process that produced the original
`design.md`: a first comprehensive design, to be adversarially reviewed against the
actual papers and the actual PureBLAS/PureSparse source before implementation. §0 lists
the decisions most in need of hostile review. Nothing here is implemented yet.

Companion documents: [`design.md`](design.md) (Cholesky/LDLᵀ, the canonical reference
for conventions used here), [`../ROADMAP.md`](../ROADMAP.md) (milestone status).

---

## §0 Review status and hotspots (v1)

No changelog yet — this is the unreviewed draft. Decisions the reviewer should attack
first, in order of blast radius:

- **H1 — the star-matrix fill-equivalence derivation (§3.2).** The whole symbolic layer
  reuses the existing Cholesky pipeline on a stand-in pattern; the correctness of that
  reuse rests on the ~10-line fill-path argument written out in §3.2. It is our own
  derivation (the primary paper, Gilbert–Li–Ng–Peyton 2001, is not in the archive; only
  the survey's two-sentence description of the construction is). If the argument has a
  hole, everything downstream is wrong. Hand-check it against small matrices by brute
  force.
- **H2 — the V-pattern recurrence and pivot-row bookkeeping (§3.4).** Derived from the
  survey's statement of George–Ng 1987 plus the row-merge view (Liu 1986c via survey).
  The exact "which row retires at column k" convention is ours; an off-by-one here
  produces V patterns that are *sometimes* right — the same failure mode design.md §3.3
  N6 warns about for column counts. The §9.1 superset invariant is the safety net; review
  whether the invariant as stated actually catches a wrong pivot convention.
- **H3 — rank-deficiency handling is *not* Heath's exact method (§5).** v1 drops dead
  columns' sub-threshold tails (reporting the dropped mass) instead of Heath's exact
  Givens row-zeroing. §5 argues this is the Foster–Davis phase-1 strategy and bounds the
  error; review whether the bound and the `\`-semantics consequences are stated honestly.
- **H4 — the architecture decision (§1.3): left-looking v1, multifrontal only if the
  gate forces it.** The gate philosophy (CLAUDE.md req 2) is non-negotiable wall-time;
  §9.3 predicts where left-looking wins and loses and defines the escalation trigger.
  Review whether M5's gate as written is a real gate or a dodge.
- **H5 — every identifier and constant against the B1/B2 test** ("where did this come
  from?" — design.md §0). Particular attention: the τ default formula (§5.3, our own),
  the reflector sign convention (§4.4), field names in §1.4, the COLAMD dense-row/column
  thresholds (§2.2 pt 5, ours — the paper explicitly declines to prescribe one).
- **H6 — the COLAMD specification (§2.2) against the primary sources.** §2.2 was
  written from the actual Davis–Gilbert–Larimore–Ng 2004 paper (read in full), the same
  way design.md §2.2 was written from the AMD paper — and AMD's §2.2 needed an
  adversarial pass against the paper to get the degree bookkeeping right (design.md §0
  D4/N4/N5). Verify §2.2's set-difference/tag-array bookkeeping and the recommended-
  variant choices against the paper's Algorithm 3 and §4.8 line by line; COLAMD is
  *not* AMD on a different graph (no quotient-graph elements, different absorption
  rules) and any place where §2.2 silently reads like design.md §2.2 is suspect.
  Additionally: Larimore's 1998 thesis (the paper's own stated detail reference, in the
  archive) was only **spot-checked** for this draft (§2.2 status note) — the reviewer
  should read its Chapters 3–4 and flag anything the condensed journal prose made §2.2
  get subtly wrong.

---

## §1 Overview and architecture

### 1.0 Milestone placement and numbering

**M5 = sparse QR** (this document). **M6 = GPU** (identical content to the old M3, see
`design.md` §8 — deliberately deferred last because the dev box has no NVIDIA GPU;
ROADMAP 2026-07-13 note). Completed milestones keep their numbers (M1 LLᵀ, M2 LDLᵀ, M4
drop-in; M3 was never started under that number). All references to "M3 (GPU)" in older
ROADMAP text should be read as M6 from now on; `ROADMAP.md` carries the renumber note.

Why QR now, before GPU: (a) GPU work is unverifiable on this machine and is already
parked pending `galen` (ROADMAP); (b) QR completes the CPU factorization triad —
PureSparse already has both classical *alternatives* to QR (normal equations via
`cholesky`, augmented-system via `ldlt`; survey §7.5) but not QR itself, which is the
method of choice for ill-conditioned and rank-deficient least squares (survey §7 intro);
(c) the symbolic machinery reuse (§3) is maximal while the M1–M2 code is warm.

### 1.1 Goals and non-goals

**Goals (M5).** A pure-Julia sparse QR of an m×n real matrix A (any shape, m ≥ n or
m < n): `A·P_c = Q·R` with `Q` held implicitly as sparse Householder vectors `V` plus
coefficients `beta` (survey §7.3: keeping Householder form is *the* advantage of the
column method over Row-Givens), `R` sparse upper triangular (trapezoidal when
rank-deficient, "squeezed" semantics per §5). Least-squares solve (`min ‖Ax−b‖₂`,
m ≥ n), basic solution (m < n or rank-deficient), minimum-norm solve via the documented
factor-Aᵀ pattern (§6.3; George–Heath–Ng 1984 via survey §7.2). Rank detection with a
magnitude threshold τ (§5). Column-singleton pre-elimination (§2.3; SPQR paper §2.1).
Same hard requirements as the rest of the package: analyze-once/factorize-many
(`qr!(F, A2)` zero-alloc, CLAUDE.md reqs 5/7), generic over `T<:Real` with Float64 the
tuned path (req 3), trim-compatible (req 4), wall-time gate (req 2, §9.3).

**Non-goals (M5).**
- **No complex element types** — matches design.md §1.1's existing non-goal (code
  generic over `T<:Real` from day one; complex needs conjugation plumbing throughout
  and a complex-τ Householder convention — a later, mechanical extension). SPQR
  supports complex; we defer.
- **No column pivoting for rank detection** (destroys sparsity — survey §7.4; Heath's
  fixed-column-order approach instead, §5).
- **No exact Heath row-zeroing / no second-phase null-space machinery** (Pierce–Lewis
  1997, Foster–Davis 2013 phase 2, Ng 1991 — §5.2). v1 reports rank and dropped mass;
  it is not a full rank-revealing QR.
- **No BTF (block-triangular-form) pre-permutation.** The Cholesky-of-AᵀA pattern is
  then only an upper bound on R for non-strong-Hall A (Coleman–Edenbrandt–Gilbert 1986
  via survey §7.1) — accepted, because Heath-style rank handling *requires* R sized to
  the column-etree pattern anyway (SPQR paper §2.3: "Heath's method requires R to
  accommodate any nonzero entry in the Cholesky factorization of AᵀA"; SPQR itself
  skips full BTF for the same reason). Singletons (the 1×1 BTF blocks) are exploited
  (§2.3), matching SPQR's compromise.
- **No dense-row withholding** (Björck 1984 via survey §7.2/§7.4). A dense row of A
  makes AᵀA — and hence R — completely full; v1 documents this failure mode and points
  the caller at `ldlt` on the augmented system (which PureSparse already has, and which
  the survey §7.5 recommends precisely for the dense-row case, Arioli–Duff–de Rijk
  1989). Listed extension, not scheduled.
- **No row/column update-downdate of a QR factor** (Edlund 2002 territory). Out of
  scope entirely.
- **No parallelism** (SPQR's TBB tree parallelism is its headline feature; PureSparse
  is single-threaded by project convention and the gate is measured single-threaded).

### 1.2 When *not* to use QR (guidance the docs must carry)

Survey §7.5, condensed, because PureSparse uniquely already ships both alternatives:

| Situation | Recommended PureSparse tool |
|---|---|
| Well-conditioned LS, no rank worries | normal equations: `cholesky(AᵀA)` (Google/Ceres precedent, survey §7.5) — fastest, least memory |
| Moderately ill-conditioned, or dense rows in A | augmented system `[αI A; Aᵀ 0]` via `ldlt` + `refine!` (survey §7.5, eq. 7.1; α heuristic documented there) |
| Ill-conditioned, rank-deficient, or robustness required | `qr` (this document) |

### 1.3 Architecture decision: left-looking column Householder v1, multifrontal as the gated escalation

Two published architectures were considered seriously (Row-Givens — George–Heath 1980 —
was ruled out immediately: keeping Q in Givens form is impractical, survey §7.3, and it
underuses everything this package already has):

**(a) Left-looking column-oriented Householder** (Davis 2006, full dense-case pseudocode
`qr_left_householder` reproduced in survey §7.3; the sparse version replaces the
`for i=1:k-1` loop with a traversal of the k-th *row subtree* of the column etree —
survey: "the only prior Householder vectors that need to be applied correspond to the
nonzero pattern of the kth column of R"). Simplicial: one column of R and one Householder
vector per step, no fronts, no contribution blocks, no stack.

**(b) Multifrontal Householder QR** (Matstoms 1994/1995; Amestoy–Duff–Puglisi 1996
"MA49"; SPQR paper §3, all via published papers). Each supernode of the Cholesky factor
of AᵀA becomes a dense frontal matrix; assemble child contribution blocks + newly
arriving rows of A, dense partial Householder QR of the front's pivotal columns, push
the trailing contribution block to the parent; BLAS-3 throughout.

Honest trade-off table:

| Axis | (a) left-looking | (b) multifrontal |
|---|---|---|
| Reuse of existing code | **maximal**: entire symbolic pipeline (§3) reused on a stand-in pattern; numeric loop is structurally the sibling of `llt.jl`'s left-looking scatter loop (scattered work vector + pattern-driven updates + per-column harvest), and of `simplicial/updown.jl`'s column storage discipline | symbolic layer same as (a); numeric layer is new machinery: frontal assembly, extend-add, contribution-block stack, staircase exploitation, in-front rank handling |
| PureBLAS dependencies | **none new** (§4.6 — the per-column work is sparse-indexed level-1, which is PureSparse's own domain; PureBLAS `nrm2` used on packed segments) | **two new kernels required** (§7.2): apply-stored-block-reflectors-to-external-C (LAPACK dlarfb/dormqr role) and a generic-`T` `geqrf!` fallback — both verified missing from PureBLAS today (§7.2) |
| Zero-alloc-after-symbolic | natural (exact V/R sizing from symbolic, §3.4; no dynamic structures) | needs a preallocated contribution-block arena sized by a symbolic stack simulation (SPQR paper §2.3/§3.1 describes exactly this simulation; doable, more machinery) |
| Flop rate | BLAS-1/2-grade; wins when fronts are small / R very sparse (SPQR paper §1: row/column methods "are very competitive when R remains very sparse") | BLAS-3; SPQR reaches a substantial fraction of dense-DGEQRF speed (paper §5.5: 2.49 vs 2.67 GFlops single-core) |
| Rank handling | drop-with-reported-error (§5) — simple, zero-alloc | Heath-per-front, exact, contribution block can grow (SPQR §3.2 + Theorem 1) |
| Implementation size / risk | small (one new numeric file + symbolic extension) | large (the biggest single numeric component in the package if built) |

**Decision.** M5 lands in two stages:

- **M5a (committed): left-looking column Householder.** It is the right v1 for exactly
  the reasons the trade-off table shows: it converts ~all of its budget into *shared*
  infrastructure (symbolic analysis, singleton handling, ordering, types, solves, rank
  policy, tests, benchmark harness — every one of which multifrontal needs unchanged),
  and it needs nothing from PureBLAS that doesn't exist. It is **not throwaway** under
  any outcome.
- **M5b (conditional, gate-triggered): multifrontal numeric phase** replacing only the
  numeric loop, keeping M5a's symbolic layer, API, and tests. Trigger: the §9.3
  benchmark shows M5a losing the wall-time gate on any stratum of the gate set. The
  milestone-level gate (§9.3) is unconditional — M5 does not close while any stratum
  loses — so this is a sequencing decision, not a gate waiver. §7 sketches M5b far
  enough to prove the M5a symbolic layer feeds it without rework, and lists its
  PureBLAS prerequisite tasks explicitly so they are scheduled work, not silent
  assumptions.

What we explicitly do **not** do: build multifrontal first because it is the impressive
option (it would stall the milestone on two new PureBLAS kernels and a frontal-assembly
layer before a single least-squares problem gets solved), or build a hybrid
"supernodal-left-looking-QR" of our own invention (no published basis — a multifrontal
front factorization scheduled left-looking is just multifrontal with worse storage
discipline; if BLAS-3 is needed, do the published thing).

### 1.4 Core types

Naming note (B1 discipline): field names below follow this package's own established
conventions (`rowind`/`*_ptr`/`px` from `types.jl`; `beta` is the survey §7.3
pseudocode's own name for the Householder coefficients). None are copied from any
SuiteSparse internal (which we have never seen).

```julia
struct QRSymbolic{Ti<:Integer}
    m::Int
    n::Int
    # --- singleton block (§2.3); n1 == 0 when disabled or none found ---
    n1::Int                        # number of pre-eliminated column singletons
    # --- permutations ---
    cperm::Vector{Ti}              # column permutation (singletons first, then
    ciperm::Vector{Ti}             #   fill-reducing ∘ postorder on the rest), length n
    rperm::Vector{Ti}              # row permutation: staircase sort + pivot-row
    riperm::Vector{Ti}             #   assignment (§3.4), length m
    # --- column elimination tree of the non-singleton block (postordered) ---
    parent::Vector{Ti}             # length n-n1; 0 = root
    # --- factor structure ---
    rcount::Vector{Ti}             # nnz of row k of R  (= colcount of L(AᵀA)), length n
    rptr::Vector{Ti}               # row-of-R pointers (CSC of Rᵀ), length n+1
    vptr::Vector{Ti}               # V column pointers, length n+1
    vrowind::Vector{Ti}            # V row patterns (permuted rows; first entry of
                                   #   column k is k, the pivot slot — §3.4)
    # --- workspace sizing ---
    max_rrow::Int                  # max rcount — sizes the row-subtree gather buffer
    max_vcol::Int                  # max V column length — sizes the packed reflector buffer
    nnzR::Int
    nnzV::Int
    flops::Float64                 # §3.5 — exact when rank detection is off
end

mutable struct QRStats
    nnzR::Int
    nnzV::Int
    flops::Float64
    rank::Int                      # live pivots after §5 dead-column handling
    n_dead::Int                    # dropped columns
    dropped_norm::Float64          # ‖dropped tails‖_F (§5.2 — the Foster–Davis phase-1
                                   #   error report); 0.0 when full rank
end

mutable struct QRFactor{T<:Real,Ti<:Integer} <: AbstractSparseFactor{T}
    sym::QRSymbolic{Ti}
    # R stored ROW-wise (CSC of Rᵀ): row k of R owns slots rptr[k]:rptr[k+1]-1.
    rcolind::Vector{Ti}
    rval::Vector{T}
    # Q implicit: V column-wise on sym.vptr/vrowind; beta[k] == 0 ⇒ dead/trivial
    # reflector (H_k = I), which makes §5's dead-column skip a plain no-op.
    vval::Vector{T}
    beta::Vector{T}
    ws::QRWorkspace{T,Ti}          # §4.5
    stats::QRStats
    ok::Bool
end
```

`QRFactor <: AbstractSparseFactor{T}` satisfies the existing `contracts.jl` contract
surface (`solve!(::Self, x, b)`, `issuccess(::Self)::Bool`) — `solve!` with
least-squares semantics (§6). A separate `QRSymbolic` (rather than extending `Symbolic`)
is deliberate: the two share no fields' meaning (`Symbolic` is square/symmetric,
supernode-partitioned; `QRSymbolic` is rectangular, row-permuted, V-patterned), and M5b
adds front structure to `QRSymbolic` without disturbing the Cholesky type (§7.1).

### 1.5 Module layout (additions)

```
src/
  qr/singletons.jl    # §2.3
  qr/symbolic.jl      # §3 (star pattern, V/R structure, staircase; drives the
                      #     EXISTING etree.jl/counts.jl functions — no reimplementation)
  qr/numeric.jl       # §4 (M5a left-looking loop)
  qr/solve.jl         # §6 (apply_Qt!/apply_Q!, solve_R!/solve_Rt!, solve!, \)
  qr/frontal.jl       # §7 (M5b only; absent in M5a)
```

`ordering/colamd.jl` (§2.2) and `ordering/ata.jl` (§2.2.6) join the ordering
directory. Tunables → `tuning.jl` (§1.6).
Contracts → `contracts.jl`; StrictMode runtime checks → the same layer as the rest of
the package (design.md §9.1 D6 separation applies unchanged).

### 1.6 Tunables (all Preferences.jl, same mechanism as design.md §1.4)

| Preference | Default | Meaning |
|---|---|---|
| `qr_tol_mult` | `8.0` | c_τ in the rank threshold τ = c_τ·max(m,n)·eps(T)·max_j‖A[:,j]‖₂ (§5.3 — **own derivation**, free tunable, no external provenance; B2 discipline) |
| `qr_singleton_mult` | `1.0` | singleton magnitude threshold = this × τ (§2.3) |
| `colamd_dense_row_mult` | `10.0` | COLAMD withholds rows with nnz > max(16, mult·√n) (§2.2 pt 5 — threshold shape is ours; the paper prescribes none and flags COLMMD's 50% as too high) |
| `colamd_dense_col_mult` | `10.0` | COLAMD withholds (and orders last) columns with nnz > max(16, mult·√m) (§2.2 pt 5, same provenance status) |

---

## §2 Ordering

### 2.1 Interface

Adds `COLAMDOrdering <: AbstractOrdering` (the QR default, §2.2) alongside the existing
`AMDOrdering`/`GivenOrdering`/`NaturalOrdering` (design.md §2.1).
`symbolic_qr(A; ordering=COLAMDOrdering())`. Because QR orders *columns of a
rectangular A* rather than a symmetric graph, the ordering interface gains a second
entry point, `order_columns(o, m, n, colptr, rowval) -> cperm` (contract added to
`contracts.jl`): `COLAMDOrdering` implements it natively on A's pattern;
`AMDOrdering` implements it by forming pattern(AᵀA) and delegating to the existing
symmetric `order` (§2.2.6); `GivenOrdering` passes its permutation through — again the
escape hatch (METIS-on-AᵀA if the user wants it) and the mechanism for the
same-permutation gate arm (§9.3).

### 2.2 v1 default: COLAMD, from the primary paper

Sources, both in the archive:

- **Primary (this section is written from it, read in full):** Davis, Gilbert,
  Larimore, Ng, *A Column Approximate Minimum Degree Ordering Algorithm*, ACM TOMS
  30(3):353–376, 2004
  (`refs/linear_algebra/QR/davis_gilbert_larimore_ng_2004_colamd.pdf` — the companion
  "Algorithm 836" software paper is not needed; the algorithm content is here).
- **Implementation-depth companion (for task 3):** Larimore, *An Approximate Minimum
  Degree Column Ordering Algorithm*, MS thesis, University of Florida, 1998
  (`refs/linear_algebra/QR/larimore_1998_colamd_thesis.pdf`, 171 pp.) — the full
  derivation the journal paper was condensed from, and the reference the paper itself
  defers to twice ("Details … are given in Larimore's [1998] thesis"). Its Chapter 4
  specifies the working data structures and routine decomposition at implementation
  precision (row/column structs and their shared-variable overlays; the single index
  array of size 2·nnz+n_cols with in-place merged-row construction and garbage
  collection; the degree list and supercolumn hash table sharing one head array with
  collision handling; routines init_rows_cols → init_scoring → find_ordering →
  order_children, plus garbage_collection and detect_super_cols; dense/null row and
  column pre-elimination with newly-null column detection; natural-order tie-breaking
  via degree-list insertion order), and its §3.2 carries the derivation behind the
  journal §4.8 initial-metric finding. Status honesty: for this draft the thesis was
  **spot-checked** (front matter/TOC + Chapter 4's opening, pp. 29–32 — verified to be
  the same algorithm at greater depth and consistent with the journal paper), not read
  cover-to-cover; the v1→v2 review pass and the task-3 implementer must read Chapters
  3–4 in full and treat the thesis as the tiebreaker wherever the journal prose is
  compressed. Practical note: the PDF's embedded text layer is garbled (custom font
  encoding) — read it via rendered pages, not pdftotext.

§2.2 is written from these the way design.md §2.2 was written from the AMD paper —
**published documents only, never the COLAMD C source** (§11; the thesis describes the
same design the C library implements, which is exactly what makes a faithful clean-room
implementation possible without ever opening that source — same relationship as the
SPQR paper to SPQR). COLAMD computes the column ordering directly from the pattern of A
without ever forming AᵀA — the same property the star matrix gives the rest of the
symbolic phase (§3.2) — and is what makes the whole pipeline AᵀA-free end to end.

COLAMD is **not** "AMD run on a different graph": it is a *symbolic LU factorization
with column selection* on row/column set structures, with no quotient-graph
variables/elements and different absorption rules. The paper's development, condensed
to what we implement:

1. **Row-merge symbolic LU (paper §3).** Maintain, for each original row i, its
   pattern set `Aᵢ`, and for each pivot step k a pivot-row bound `Rₖ`; the row-merge
   tree (Liu 1991, via paper §3) organizes them. At step k,
   `Rₖ = (∪_{k = min Rᵢ} Rᵢ ∪ ∪_{k = min Aᵢ} Aᵢ) \ {k}` — every candidate pivot row's
   upper-bound pattern collapses onto `Rₖ`, so the used sets are discarded (*regular
   row absorption*, paper eq. (1)). The pivot-column count bound is
   `lₖ = Σ_{k = min Rᵢ} lᵢ + |{i : k = min Aᵢ}| − 1` (paper eq. (2)) — note this is
   *identical algebra* to §3.4's `vcount` recurrence, which is no accident: both are
   the row-merge tree. Storage never grows above O(|A|) (paper's §3 argument: each
   `C_j` update replaces `C_j` by `(C_j \ Cₖ) ∪ {k}`, never larger).
2. **Column sets for column selection (paper Algorithm 2).** To permute columns during
   the elimination, maintain for each column j the set `C_j` referencing exactly those
   row sets (`Rᵢ` bold / `Aᵢ` plain, in the paper's notation) containing j; initially
   `C_j = Struct(A[:,j])`. The symbolic update rewrites `C_j = (C_j \ Cₖ) ∪ {k}` for
   every j ∈ Rₖ.
3. **Pivot metric (paper §4.3/§4.8, Algorithm 3).** At step k, select the candidate
   column c minimizing the maintained metric `d_c`, swap into position k. The
   **recommended COLAMD variant** (paper §4.8, adopted verbatim as our v1): initial
   metric = the COLMMD-style loose bound (paper eq. (3), O(|A|) to initialize); metric
   maintained during elimination = the AMD-style approximate external row degree bound
   (paper eq. (4): `‖Rₖ‖ ≤ ‖Rs \ {k}‖ + Σ_{i∈Cₖ\{s}}‖Rᵢ \ Rs‖ + Σ_{i∈Cₖ}‖Aᵢ \ Rs‖`,
   with `Rs` the most recent pivot row that modified the column), computed with the
   paper's Algorithm-3 tag-array bookkeeping (`w`/`v` arrays + monotone tag `t`; after
   the first pass `wᵢ − t = |Rᵢ \ Rₖ|`, `vᵢ − t = |Aᵢ \ Rₖ|`); **no initial aggressive
   absorption; aggressive row absorption during elimination ON** (paper §4.7: when the
   AMD-metric pass finds `|Rᵢ \ Rₖ| = 0`, delete `Rᵢ`/`Aᵢ` even when i ∉ Cₖ — "costs
   almost nothing to detect" there); super-rows and super-columns ON. The paper tested
   16 variants and explicitly recommends this combination (§4.8, including the
   deliberately kept initial-metric "bug" story — initial COLMMD beat initial AMD by
   ~8% flops); we do not re-litigate that experiment. Rejected metrics (exact degree,
   Householder-update size, approximate Markowitz, approximate deficiency — paper
   §4.2/§4.4–4.6) are documented as rejected *by the paper's own experiments*, not
   re-tested by us.
4. **Super-columns / mass elimination (paper §4).** Columns in `Rₖ` with identical
   pattern (hash-bucketed, Ashcraft-style hash per the paper) merge into
   super-columns; selecting a super-column mass-eliminates all its members; a column
   whose post-update pattern equals `{k}` is eliminated immediately. Same role — and
   same test discipline — as supervariable detection in our AMD (design.md §2.2 pt 3),
   but the detection site (within `Rₖ` after the symbolic update) is COLAMD's own.
5. **Dense rows/columns (paper §4).** Dense rows destroy the bound (one dense row ⇒
   the A⁽¹⁾ bound is fully dense) and are withheld from the ordering; dense columns
   only cost time and are withheld and placed **last** in Q. The paper prescribes no
   threshold (explicitly: "problem dependent"; MATLAB COLMMD's 50% is called "probably
   too high"). Our defaults are therefore our own free tunables (§1.6, B2 discipline):
   withhold rows with > max(16, `colamd_dense_row_mult`·√n) entries and columns with >
   max(16, `colamd_dense_col_mult`·√m) entries — the same shape as our AMD dense-row
   heuristic, chosen for the same reason, calibrated in the M5 benchmark pass. A
   withheld dense row still densifies R itself (§1.1 non-goals — Björck 1984
   withholding is the real fix and is out of scope); the ordering just stops being
   poisoned by it.
6. **Complexity** (paper §3): time O(Σ_j |A[:,j]|·υ_j) (υ_j = bound-of-U column
   counts), storage O(|A|), both typically far below numeric factorization.

Output: `cperm` for the non-singleton block; the column etree + postorder are then
computed by §3 (COLAMD does not need to produce a tree; the paper's l_k/R_k machinery
is *internal* to the ordering and is discarded — §3 recomputes structure on the star
pattern, keeping the two components independently testable).

**Ordering-quality guardrail** (mirrors M1's "AMD fill ≤ 1.15× CHOLMOD-AMD" gate
item): nnz(R) under our COLAMD ≤ 1.15× nnz(R) under stdlib SPQR's default ordering
(black-box, §9.2), on the §9.4 zoo. Not equality — tie-breaking and variant details
legitimately differ; the same-permutation gate arm (§9.3) keeps ordering quality
orthogonal to factorization throughput, exactly as in M1.

### 2.2.6 `AMDOrdering` on the explicit pattern of AᵀA (supported alternative)

Kept as a first-class option, not a placeholder: MA49 orders exactly this way (SPQR
paper §2.2), and SPQR's own measured default *prefers* AMD-on-AᵀA for m > 2n (paper
§5.4, Table VI — AMD wins most of its large LS set). `ata_pattern(A)` builds
pattern(AᵀA) column-by-column with a marker array from the row-form copy of A (already
needed by §2.3/§3.4; the SPQR paper §2.1 makes the same "transpose needed anyway"
observation), then delegates to the untouched `ordering/amd.jl`. Cost: worst-case
O(Σ_i nnz(row i)²) time and O(|AᵀA|) memory, paid once per symbolic — the price COLAMD
avoids; the M5 benchmark task measures both orderings across the gate set and records
whether an SPQR-style shape-based default (COLAMD iff m ≤ 2n) earns its keep, rather
than assuming it.

### 2.3 Column-singleton pre-elimination

SPQR paper §2.1, reimplemented from the paper's description: a column singleton is a
column with exactly one nonzero whose magnitude exceeds a threshold; permute it (and its
row) to the front, delete both, repeat. Result: `A·P = [R11 R12; 0 A22]` with R11 upper
triangular (upper trapezoidal when a singleton column has no surviving row —
structurally rank-deficient case, paper's example), and the QR of the singleton block
requires **no numerical work and no fill**. The paper documents the payoff (215/353 of
the collection's LP problems become *entirely* singletons) and the algorithm shape
(breadth-first peeling on the row-form copy, O(|R11|+|R12|) plus O(n) scan, prune in
O(|A|)); the queue-based peeling implementation is ours from that description.

Two policy points, both taken from the paper's own reasoning:

- **Values, not just pattern:** the magnitude test makes singleton detection a
  *numeric*-phase-coupled decision. Therefore — exactly as SPQR does — **singletons are
  exploited only in the one-shot `qr(A)` path and disabled when the symbolic is built
  for reuse** (`symbolic_qr` + repeated `qr!`): a singleton set chosen for A's values is
  invalid for A2's (paper §2.1: "If the symbolic analysis is to be reused ... singletons
  are not exploited because they conflict with how rank-deficient matrices are
  handled"). `QRSymbolic.n1 == 0` in the reuse path.
- Threshold: `qr_singleton_mult × τ` (§1.6), so the singleton and rank thresholds move
  together (a "singleton" below the rank tolerance would be a rank-deficiency dodge).

---

## §3 Symbolic analysis

Everything in this section runs on the non-singleton block A22 (or all of A when
singletons are off); `m, n` below refer to that block. Pipeline: star pattern → existing
etree → existing postorder → existing column counts → V/R structure + staircase row
permutation. Total O(|A|·α + n) beyond the one-time §2.2 ordering cost.

### 3.1 What is being computed, and why the Cholesky machinery applies

George–Heath 1980 (via survey §7/§11.5, eq. 11.3): `AᵀA = Rᵀ(QᵀQ)R = RᵀR`, so **the
pattern of R equals the pattern of Lᵀ for the Cholesky factorization of AᵀA** (exact
when A is strong Hall; an upper bound otherwise — Coleman–Edenbrandt–Gilbert 1986 via
survey §7.1 — and we *want* the upper bound for rank handling, §1.1 non-goals). The
column elimination tree of A = the etree of AᵀA. Row k of R ↔ column k of L, so the
existing `column_counts` output *is* `rcount`. The row subtree T^k (nodes i < k with
R[i,k] ≠ 0) drives the numeric loop (§4.2).

### 3.2 The star pattern: running the existing pipeline without forming AᵀA

With COLAMD as the default ordering (§2.2), nothing in the pipeline needs AᵀA at all
(the optional `AMDOrdering` path, §2.2.6, is the one exception); naively feeding AᵀA's
pattern (potentially ≫ |A|) through etree/counts would make the whole symbolic phase
O(|AᵀA|). Gilbert–Li–Ng–Peyton 2001 (primary paper unavailable; construction as
described in survey §7.1) avoid this with a **star matrix** S with O(|A|) entries whose
Cholesky factorization has the same pattern as that of AᵀA: *"the kth row and column of
the star matrix is the union of rows in A whose leftmost nonzero entry appears in column
k."*

**Independent correctness derivation (H1 — review this).** Fix the column order. The
graph of AᵀA is the union over rows i of A of a clique on C_i = pattern(row i). The
graph of S replaces each clique C_i by a star centered at its minimum vertex
v₁ = min C_i (edges v₁–v_j for all v_j ∈ C_i). Claim: G(AᵀA) and G(S) have the same
filled graph, hence the same etree, counts, and factor pattern. By the fill-path theorem
(Rose–Tarjan; as used throughout design.md §3), edge (a,b) is in the filled graph iff
there is a path a→b whose interior vertices are all < min(a,b).
- Every S-edge is a clique edge, so any S-fill-path is an AᵀA-fill-path: filled(S) ⊆
  filled(AᵀA).
- Conversely, replace any clique edge (v_j, v_k), j,k ≥ 2, appearing in an
  AᵀA-fill-path by the detour v_j–v₁–v_k. Since v₁ = min C_i < v_j and v_j is an
  *interior* vertex of the path (so v_j < min(a,b)), the detour's interior vertices
  remain < min(a,b): the path is still a fill path in G(S). Hence filled(AᵀA) ⊆
  filled(S). ∎

Consequences, all for free:
- `etree(S)` via the existing `etree.jl` = the column elimination tree of A.
- `column_counts(S)` via the existing `counts.jl` = `rcount` (row sizes of R). That
  implementation was built from the Gilbert–Ng–Peyton algorithm
  (`refs/linear_algebra/QR/gilbert_ng_peyton_1992_ornl_tm12195.pdf` is the primary
  source with full pseudocode and Lemmas 1–4; verified: `counts.jl`'s
  first-descendant/maxfirst/prevleaf/path-halving structure is that algorithm) — and
  GNP92 itself states the QR application in its introduction ("Our algorithms can be
  used also to predict the row counts and column counts of the upper triangular factor
  R, since the structure of R is always contained in the structure of the Cholesky
  factor of AᵀA").
- Building S: one pass over the row-form copy of A (already built, §2.2): for each row,
  find its leftmost (permuted) column k and add its entries to column k's list; dedupe
  with a marker array; |S| ≤ |A|. Feed the strict-upper part through the existing
  `symmetrized_upper`-shaped entry points unchanged.
- Postorder: existing `postorder` + `relabel_pattern`, composed into `cperm` exactly as
  design.md §3.2 does. (No amalgamation priority needed in M5a — no supernodes; M5b
  reuses the priority mechanism when fronts arrive, §7.1.)

### 3.3 R structure

R is stored by rows (CSC of Rᵀ, §1.4), sized exactly by `rcount`. The column indices of
row k of R are *not* precomputed — the numeric loop appends them left-to-right as
columns arrive (§4.3): row k of R receives entry (k, j) exactly when k ∈ T^j, and j is
processed in ascending order, so each row's entries arrive already sorted with a per-row
write cursor. `nnzR = Σ rcount`. This mirrors the Row-Givens observation (survey §7.1)
that R's *final* pattern is known but fills in over time — except the left-looking
column order makes the arrival order per-row monotone, so no intermediate-fill concern
exists (contrast survey §7.2's row-ordering discussion, which is about row methods).

### 3.4 V structure: the row-merge recurrence and the staircase row permutation (H2)

Published basis, all via survey §7.1/§7.3: George–Ng 1986/1987 define the column
patterns of V and show V fits in the space of L(AᵀA) for square zero-free-diagonal A;
George–Liu–Ng 1988 show each *row* of V is a path in the column etree starting at the
column of that row's leftmost nonzero; Liu 1986c's row-merge tree gives the counting
view; the survey's §7.3 closing line fixes the contract: "the pattern Vk of the kth
column of V is computed in the symbolic factorization phase."

Our formulation (own derivation on top of those statements — this is hotspot H2):

- **Row assignment.** Each row r of A is assigned to column `leftcol(r)` = its leftmost
  nonzero (permuted) column. `a_k` = number of rows assigned to k.
- **Active sets.** Process columns in ascending order, maintaining disjoint row sets.
  `S_k` = (rows assigned to k) ∪ (non-pivot rows inherited from each child of k in the
  column etree). Column k's reflector acts on exactly the rows S_k, so
  **pattern(V_k) = S_k**; one row of S_k retires as the *pivot row* of column k (it is
  where row k of R physically lives after the reflector), and the remaining |S_k| − 1
  rows pass to parent(k).
- **Counts** (for exact allocation): `vcount[k] = a_k + Σ_{c child of k} (vcount[c] − 1)`
  — one bottom-up O(n) pass; `nnzV = Σ vcount`. A column with `vcount[k] == 0` is a
  *structurally dead* pivot (Oliveira 2001's "row k evaporates when S_k is empty",
  survey §7.1): `beta[k] = 0` permanently, row k of R is structurally empty, and if
  structural rank matters the caller learns it from `stats.rank` (§5).
- **Row permutation `rperm`.** Rows are numbered so that the pivot row of column k gets
  number k (a staircase sort by `leftcol`, then pivot selection; SPQR's analysis phase
  performs the same leftmost-sort, paper §2.3, "P₂"). Pivot selection rule (ours,
  deterministic): the assigned row with the smallest original index if a_k > 0, else
  the inherited row of smallest current number. Non-pivot rows receive numbers > n in
  arrival order (for m > n; when m < n dead pivots absorb the shortfall). In permuted
  numbering, `vrowind` for column k starts with k itself (the pivot slot) followed by
  the inherited/assigned rows in ascending permuted order.
- **Patterns** (`vrowind`): a second bottom-up pass materializes each S_k. Each row
  lives in exactly one active set at a time, so threading rows through per-column
  linked lists (head/next arrays, the same idiom as `llt.jl`'s descendant lists) builds
  all patterns in O(nnzV) total, then one sort pass per column (counting-free: rows
  can be emitted in ascending order by merging children's already-sorted survivor lists
  with the assigned-rows list — children's lists are sorted by induction).
- **Consistency property** (tested, §9.1): for every k, applying the recurrence's set
  algebra must reproduce George–Liu–Ng 1988's row-path characterization — for every
  row r, {k : r ∈ S_k} is a contiguous ascending path in the column etree starting at
  leftcol(r) and ending where r retires as a pivot (or at a root). A cheap exact
  cross-check on every zoo matrix, and the property the numeric loop's correctness
  leans on.

### 3.5 Flops and workspace bounds

Applying reflector i to a column costs 4·vcount[i] flops (one dot + one axpy over
pattern(V_i)); constructing reflector k costs ~3·vcount[k]. Reflector i is applied once
for every j with i ∈ T^j — that multiplicity is `rcount[i] − 1`. So
`flops = Σ_i (4·vcount[i]·(rcount[i]−1) + 3·vcount[i])`, computed in the counts pass —
exact when rank detection is off (dead columns only *remove* applications; same
upper-bound stance as SPQR paper §2.3). `max_rrow = max rcount` sizes the row-subtree
gather buffer; `max_vcol = max vcount` sizes the packed reflector staging buffer (§4.5).

---

## §4 Numeric factorization (M5a): left-looking column Householder

### 4.1 Statement

Direct sparse transcription of survey §7.3's `qr_left_householder`, with the
`for i = 1:k-1` loop replaced by the ascending row-subtree traversal, exactly as the
survey specifies. For k = 1..n:

1. **Scatter** column k of A (rows permuted by `rperm`) into the dense work vector
   `x` (length m, kept all-zero between columns — the `SimplicialLDLFactor.wval`
   discipline, re-zero only what was touched).
2. **Row subtree.** Collect T^k = {i < k : R[i,k] ≠ 0}: for each j in
   pattern(S[:,k]) (star matrix column, available from §3.2's structures), climb
   `parent[]` marking with stamp k until an already-stamped node or k; then produce
   T^k in **ascending order** (reflectors do not commute; the dense reference applies
   i = 1,…,k−1 ascending). In-place, allocation-free ordering of the gathered set into
   the `max_rrow` buffer (insertion into runs or in-place quicksort — implementation
   detail, but the no-allocation requirement is contractual).
3. **Apply prior reflectors.** For i in T^k ascending, skip if `beta[i] == 0` (dead or
   trivial), else: `w = beta[i] · Σ_{r ∈ V_i} vval[r]·x[r]` (sparse dot), then
   `x[r] -= w·vval[r]` for r ∈ V_i (sparse axpy). Harvest `R[i,k] = x[i]` (the pivot
   slot of i) into row i's cursor position and zero `x[i]`.
4. **Form reflector k** (§4.4) from x on pattern(V_k) = `vrowind` column k; write
   packed values into `vval`, coefficient into `beta[k]`, diagonal into `R[k,k]`;
   zero x on the pattern. Rank test happens here (§5).

Structural sibling of `llt.jl`'s loop (pattern-driven pending work + scatter/harvest on
preallocated storage + per-step dense-ish kernel), with Householder apply where LLᵀ has
`syrk!/gemm!` — which is what makes M5b a kernel swap rather than a rewrite.

### 4.2 Correctness anchor

The applied set (T^k) and the harvest positions are exactly the survey's specification;
the invariant that x's nonzeros after step 3 are confined to T^k ∪ pattern(V_k) is
George–Ng's theorem (§3.4), enforced as a StrictMode postcondition (checks-enabled
configuration only, design.md §9.1 layer-2 discipline) and as the §9.1 superset test.

### 4.3 `qr(A)`, `qr!(F, A2)`, and zero allocations

`qr(A; ordering, tol)` = singletons (§2.3) + `symbolic_qr` + numeric. `qr!(F, A2)` for
pattern-identical A2: reset cursors/stats, replay §4.1 — **zero allocations**
(CLAUDE.md req 5; gated in the StrictMode-checks-disabled configuration, same as
`cholesky!`). No assembly map is needed (unlike design.md §4.2): the scatter is a direct
CSC walk through `riperm`, already O(nnz) with no searches. Note the reuse-path caveat
from §2.3: `n1 = 0` under reuse.

### 4.4 Householder convention (documented, independently derived)

Textbook reflector (Golub–Van Loan-style; also the survey's `gallery('house')`):
`H = I − beta·v·vᵀ`, `v[pivot] = 1` implicit? — **No: v is stored in full with its pivot
entry**, `beta = 2/(vᵀv)`, and the sign choice `v[pivot] = x[pivot] + sign(x[pivot])·‖x‖`
avoids cancellation. `R[k,k] = −sign(x[pivot])·‖x‖`. Rationale for storing v unnormalized
with explicit pivot entry rather than LAPACK's implicit-1 convention: the sparse apply
(§4.1 step 3) then never special-cases the pivot slot, and `beta` absorbs the
normalization — one fewer branch in the innermost loop. ‖x‖ is computed by packing the
pattern values into the `max_vcol` staging buffer first and calling PureBLAS `nrm2` on
the packed view (overflow/underflow-safe lassq accumulation — PureBLAS req 6 — for free;
the packed copy is then reused as the source for `vval`). This convention is
self-contained here and tested against `H·x = (R_kk, 0…)ᵀ` directly; it deliberately
does not need to match LAPACK/faer/anything else since V never crosses an ABI.

### 4.5 `QRWorkspace{T,Ti}`

Preallocated once per factor from `QRSymbolic` sizes: `x::Vector{T}` (m, zero-kept),
`stamp::Vector{Ti}` + `tsub::Vector{Ti}` (row-subtree stamps and gathered/sorted T^k,
`max_rrow`), `pack::Vector{T}` (`max_vcol`, §4.4), `rcursor::Vector{Ti}` (n, per-row
append cursors into `rcolind`/`rval`), `rhs::Vector{T}` (m, solve scratch — §6).

### 4.6 PureBLAS dependency check — result (checked against PureBLAS source, 2026-07-14)

Verified by reading `/home/el_oso/Documents/claude/PureBLAS.jl/src/qr.jl` and
`cabi_lapack.jl` directly:

- PureBLAS **has** dense QR: `geqrf!(A, tau)` — blocked compact-WY (dlarft/dlarfb-style
  T-matrix construction + `gemm!`/`trmm!` trailing update) over a tuned unblocked panel
  (`qr_unblocked!`, faer-port, fused rank-2 apply), for **Float64** and **BlasComplex**.
  This is the proven-fast path (BlazingPorts-derived, gated on galen).
- PureBLAS **lacks**, as of today: **(a)** any apply-stored-reflectors-to-an-external-
  matrix kernel (the LAPACK dlarf/dlarfb/dormqr role — its larft/larfb logic exists only
  inlined inside `geqrf!`'s own trailing update, not callable on a separate C);
  **(b)** a generic `T<:Real` fallback for `geqrf!`/`qr_unblocked!`
  (`cabi_lapack.jl:14`: "getrf!/geqrf!/gesvd! are Float64-only kernels"; `qr.jl:7`:
  "Float64-only … ponytail: generic/AD QR deferred").
- **M5a needs neither.** The left-looking method has no dense panels: its per-column
  work is sparse-indexed level-1 on scattered/packed vectors, which is sparse-domain
  code and belongs in `src/` by the same boundary that puts `updown.jl`'s column loops
  there (the CLAUDE.md "dense kernels via PureBLAS" rule governs dense block work; the
  only dense-contiguous operation here, the packed-segment norm, goes through PureBLAS
  `nrm2`, which is generic). M5a is therefore **not blocked on PureBLAS at all**.
- **M5b needs both (a) and (b).** They are scheduled as explicit PureBLAS prerequisite
  tasks in §10 (M5b tasks P1/P2), not assumed. (a) is the front factorization's partial
  QR: factor the pivotal column block, then apply its reflectors to the non-pivotal
  columns — precisely the DLARFG/DLARF/DLARFT/DLARFB decomposition the SPQR paper
  §3.2 names as its LAPACK usage. (b) is CLAUDE.md req 3 (generic hot paths) applied to
  the front kernel, mirroring how `potrf!` already has a generic path.

One further PureBLAS observation for M5b planning: `geqrf!` uses module-global grow-on-
demand workspace (`_QR_WS`) — fine after warmup, but the M5b zero-alloc gate must warm
it up at the maximal front size first, or the apply kernel (a) should take
caller-provided workspace. Flagged for the M5b design pass.

---

## §5 Rank-deficiency policy

### 5.1 Detection: Heath's threshold test

Heath 1982 (via survey §7.2/§7.4 and SPQR paper §3.2): fixed column order (column
pivoting would invalidate the symbolic analysis and destroy sparsity), and at step k the
pivot magnitude is tested against a threshold τ. In our column formulation the test is
`‖x[pattern(V_k)]‖₂ ≤ τ` at §4.1 step 4 — the same quantity SPQR tests per-front ("the
2-norm of column 6 drops below the threshold τ", paper §3.2).

### 5.2 Handling (v1): dead-column drop with reported error — *not* Heath's exact row-zeroing (H3)

What the published methods do with a dead pivot:

- **Heath 1982 (Row-Givens):** the dead row of R is zeroed *exactly* via Givens
  rotations up the etree path and deleted ("squeezed" R). SPQR Theorem 1 (paper §3.2 —
  read, short induction on the etree path; each rotation partner's pattern contains the
  dying row's, so no fill beyond the Cholesky-of-AᵀA pattern) guarantees this stays
  inside the symbolic pattern.
- **SPQR (multifrontal):** skip the reflector inside the dense front; the un-eliminated
  row stays in the contribution block and rises to the parent — exact, no dropped mass,
  contribution blocks can grow a row (paper §3.2). Natural for fronts; not available
  without fronts.
- **Foster–Davis 2013 (phase 1, via survey §7.4):** dead columns are *dropped*, "but
  the method computes the Frobenius norm of the small errors that occur from this
  dropping", and dead columns are permuted last.

Heath's exact rotation transplants poorly to the left-looking column method: row k of R
does not exist yet when column k dies (its entries arrive with future columns), so the
row-vs-row rotation has no second operand. The multifrontal variant needs fronts. **v1
therefore adopts the Foster–Davis phase-1 strategy:** on a dead pivot, set
`beta[k] = 0` (all later applications of H_k become no-ops — no pattern growth, no
allocation), leave row k of R structurally present but empty-below-diagonal
(R[k,k] = 0; entries R[k,j], j > k, that later columns would have written against pivot
k are *the dropped mass*: each later column j with k ∈ T^j discards `x[k]` at harvest
time, accumulating `dropped_norm² += x[k]²`), and count k in `n_dead`. The per-column
tail dropped at detection is itself ≤ τ by the test. `stats.rank`, `stats.n_dead`, and
`stats.dropped_norm` report the outcome; `\` computes the **basic solution** (dead
columns' unknowns set to zero, back-substitution over live rows only — SPQR paper §3.3
method (3) semantics).

Honest consequences, documented for the user: this is the least accurate of the
published rank strategies (the survey says exactly this of Heath's method, §7.4, and
ours drops what Heath would rotate); `dropped_norm` is the a-posteriori certificate.
When it is not small relative to ‖A‖, the docs point to (i) Tikhonov regularization
(append γI and refactor — the SPQR paper itself benchmarks this fallback, §5.2) or
(ii) the augmented-system `ldlt` path (§1.2). M5b upgrades to the exact per-front SPQR
behavior for free when fronts exist (§7.3); the exact Heath/Givens variant and the
second-phase methods (Pierce–Lewis 1997; Foster–Davis 2013 phase 2; Ng 1991;
Bischof–Hansen 1991) remain non-goals (§1.1) — all are either dynamic-restructuring
(breaking the static-pattern/zero-alloc contract; SPQR paper §3.2 explains Pierce–Lewis
requires update/downdate of R and can't keep Q) or second-factorization machinery out
of v1 scope.

### 5.3 Threshold default (own derivation — B2 discipline)

`τ = qr_tol_mult · max(m,n) · eps(T) · max_j ‖A[:,j]‖₂`, `qr_tol_mult = 8.0` free
tunable (§1.6). Shape rationale (ours): a backward-stable orthogonal reduction perturbs
each column by O(#ops · eps · ‖column‖); max(m,n) is the generic ops-per-column scale
and the max column norm makes the test scale-invariant per matrix, not per column
(a per-column τ would misclassify well-scaled small columns in badly scaled problems).
The constant 8 is a starting point to be calibrated in M5's test pass against the
BigFloat oracle's exact ranks — it has **no external provenance** and must not drift
toward any implementation's default it was never derived from. `tol ≤ 0` disables rank
detection entirely (exact structural behavior; structurally-dead pivots still handled).

---

## §6 Solve phase and API

### 6.1 Building blocks (all exported, mirroring the split-solve convention of design.md §6)

- `apply_Qt!(y, F)` / `apply_Q!(y, F)`: y ← Qᵀy / Qy by applying reflectors k = 1..n
  ascending / n..1 descending over pattern(V_k) (dense y, length m; multi-RHS variants
  loop columns). Sparse-indexed level-1, same kernels as §4.1 step 3.
- `solve_R!(x, F, c)`: back-substitution over rows of R descending, live rows only
  (dead ⇒ x[k] = 0); `solve_Rt!` the forward mirror (needed by minimum-norm and by
  CSNE-style consumers).

### 6.2 Least squares (m ≥ n) and basic solutions

`solve!(x, F, b)` / `F \ b` / `ldiv!`: y ← rperm-permute(b); `apply_Qt!`;
`solve_R!` on y[1:n]; x ← cperm-unpermute. Exactly SPQR paper §3.3's
`x = P·(R \ (Qᵀb))`, with the singleton block's `R11/R12` triangular solve prepended
when `n1 > 0`. For m < n or rank-deficient F the same path yields the basic solution
(dead/absent columns zero — paper §3.3 method (3)). Residual-norm helper
`lsq_residual(F, b)` = ‖tail of Qᵀb‖ comes free from the same apply.

### 6.3 Minimum-norm solve (m < n)

Published pattern (George–Heath–Ng 1984 via survey §7.2; SPQR paper §3.3 method (2)):
factor **Aᵀ** (tall), then from Aᵀ·P = QR follows A = Pᵀ·Rᵀ·Qᵀ… i.e. solve
`Rᵀ·z = (Pᵀb)` forward (`solve_Rt!`), then `x = apply_Q!([z; 0])`. Provided as
`solve_minnorm!(x, F_of_At, b)` with the factor-the-transpose requirement in its
docstring and checked by StrictMode (dimension test distinguishes misuse). Q must be
kept for this — and V *is* always kept in v1 (Q-less/discard-Q mode is a listed
extension: SPQR paper §3.3 and MA49's seminormal-equations mode show what it buys;
YAGNI until someone needs the memory).

### 6.4 Public surface (M5)

```julia
S  = symbolic_qr(A; ordering=COLAMDOrdering())     # analysis, allocates; NO singletons (§2.3)
F  = qr(A; ordering=COLAMDOrdering(), tol=nothing) # singletons + symbolic + numeric
F  = qr(S, A; tol=nothing)                      # numeric into fresh factor sharing S
qr!(F, A2)                                       # zero-alloc refactor, same pattern
x  = F \ b ; solve!(x, F, b) ; ldiv!(x, F, b)   # LS (m≥n) / basic solution
solve_minnorm!(x, F, b)                          # §6.3 (F from qr of Aᵀ)
apply_Q!(y, F); apply_Qt!(y, F); solve_R!(x, F, c); solve_Rt!(x, F, c)
rank(F); issuccess(F)                            # rank from stats; ok flag
SparseArrays.sparse(F.R) / F.V extraction        # M4-parity-style extraction (§6.5)
```

`qr` follows the same stdlib-name discipline as `cholesky` did (`PureSparse.jl`'s
`import LinearAlgebra` note): our `qr` is PureSparse's own function; the deliberate
drop-in forwarding of `LinearAlgebra.qr(::SparseMatrixCSC)` is a **separate,
Preferences-gated** step exactly like M4's `dropin.jl`, listed as an M5 task with the
M4 checklist as its template (property surface observed black-box from the stdlib
factor object: `.R`, `.Q`, `.prow`, `.pcol`, `rank`, `\` — verified available in this
environment, §9.3).

### 6.5 Contracts and runtime checks

`contracts.jl`: `QRFactor` inherits the `AbstractSparseFactor{T}` contract
(`solve!`/`issuccess`) — LS semantics satisfy the existing signature; add
`qr!(::QRFactor{T,Ti}, ::SparseMatrixCSC{T,Ti}) -> QRFactor{T,Ti}` and the §6.4 surface
with concrete inferred return types (precompile-time only, trimmed away — design.md
§9.1 D6 separation unchanged). StrictMode layer: dimension/pattern-match preconditions,
the §4.2 scatter-pattern postcondition, `issorted` on every `vrowind`/per-row R column
run, x-is-all-zero-between-columns.

---

## §7 M5b sketch: multifrontal numeric phase (built only if §9.3 triggers it)

Just enough here to prove M5a's symbolic layer feeds it unchanged and to scope the
prerequisites; a short dedicated design addendum precedes implementation if triggered.

### 7.1 Fronts from the existing supernode machinery

Run the **existing** `fundamental_supernodes` + `relaxed_amalgamation` +
`supernode_tree` (unchanged code) on (parent, rcount) from §3.2 — SPQR paper §2.3: each
supernode of L(AᵀA) = a set of rows of R with (near-)identical pattern = one frontal
matrix; the paper notes SPQR uses the two-condition (non-fundamental) variant and
relaxed amalgamation, i.e. exactly the knobs `supernodes.jl` already has (including the
`AMALG_*` tunables, recalibrated for QR fronts in an M5b task). Front f's pivotal
columns = the supernode's columns; its rows = rows of A assigned to those columns
(§3.4's `a_k` lists) + children's contribution-block rows; the *staircase* (first
structural zero per column — paper §3.1) falls out of the same assembly simulation the
SPQR paper §2.3 describes, which also yields exact front sizes and the
contribution-block **stack** high-water mark for a postorder schedule → preallocated
arena, zero-alloc numeric (paper §3.2: fronts factorized in postorder, contribution
blocks on a stack; §4: "all workspace … allocated before" the parallel phase — same
discipline, minus the parallelism).

### 7.2 PureBLAS prerequisites (from §4.6's verified gaps)

- **P1 `larfb`-role kernel:** apply a stored block of reflectors (V panel + tau/T) to
  an external matrix C, compact-WY (`C -= V·(Tᵀ·(Vᵀ·C))` — the identical triple
  PureBLAS's `geqrf!` already performs internally on its own trailing block; the task
  is exposing it for caller-provided V/T/C with caller-provided workspace). Float64
  fast path + generic fallback.
- **P2 generic `geqrf!`:** a `T<:Real` generic unblocked path (potrf! precedent in the
  same file family), so PureSparse's front loop stays one generic implementation
  (CLAUDE.md req 3).

Both land in PureBLAS with its own gates (OpenBLAS-parity per its CLAUDE.md), before
M5b's numeric work starts.

### 7.3 What M5b changes and what it keeps

Keeps: §2 ordering+singletons, §3 symbolic (plus front structure), §5 τ policy —
upgraded to exact SPQR-style per-front dead-pivot handling (skip reflector inside the
dense front; the row stays in the contribution block; C may grow a row — paper §3.2 +
Theorem 1's no-fill guarantee; the symbolic stack/size bounds become upper bounds when
τ > 0, exactly the paper's stated trade), §6 API and solves (V storage gains a
per-front panel form; `apply_Qt!` becomes per-front `larfb` sweeps). Replaces: §4's
per-column loop with assemble→partial-QR→push-C. The M5a scalar path remains as the
generic-`T` fallback and small-problem path (mirroring the width-1/2 fast-path
philosophy in `llt.jl`).

---

## §8 Trim compatibility

Nothing new in kind: no runtime eval, no `Vector{Any}`, tunables are Preferences-backed
consts (`tuning.jl` pattern), all recursion in symbolic passes already iterative
(etree/postorder/counts reused; the §4.1 subtree climb is a bounded while-loop). The M1
`juliac/entry.jl` smoke gains a least-squares block (`symbolic_qr → qr → solve! → qr!`)
and `test/trim_tests.jl` gains the corresponding TrimCheck `@validate` roots (Float64/
Int64, kwarg-default paths) — same pattern as the existing gate.

---

## §9 Verification and benchmarking

### 9.1 Test strategy (same 7-layer structure as design.md §9.1; QR-specific items)

1. **TypeContracts**: §6.5 surface precompiles; negative shadow-module test as in M1.
2. **StrictMode runtime checks**: §6.5 list, checks-enabled configuration.
3. **Invariants (first-class):** (a) star-pattern equivalence — on every zoo matrix,
   `etree(star(A)) == etree(pattern(AᵀA))` and `column_counts` agree (brute-force AᵀA
   formed *in the test only*; this is H1's executable check); (b) V row-path property
   (§3.4); (c) R/V superset property — every numeric nonzero produced lands inside the
   symbolic pattern, both full-rank and rank-detecting modes.
4. **Oracles:** (a) dense **BigFloat Householder QR** of the permuted matrix,
   elementwise |R| comparison (sign-freedom per row: compare R up to row signs) on
   small/medium matrices, and Q via applying stored V to I; (b) residual gates:
   `‖Aᵀ(Ax−b)‖/(‖A‖²‖x‖)` for LS, `‖Ax−b‖` + minimality (‖x‖ vs oracle) for min-norm;
   `‖QᵀQ−I‖` on modest sizes; (c) **SuiteSparseQR black-box** via `SparseArrays.qr`
   (§11 policy: outputs only): LS/basic/min-norm solutions agree to residual-level
   tolerance, `rank(F)` agrees on the zoo (where τ conventions allow — compare at
   matching explicit `tol`), nnz(R) within the §2.2 ordering-quality bound.
5. **Property/fuzz:** random sparse rectangular (both shapes), constructed
   rank-deficient (duplicate/scaled columns, known rank — assert `stats.rank`,
   `dropped_norm` small), singleton-rich LP-like generators, permutation invariance via
   `GivenOrdering`, `qr!` refactor equals fresh `qr` bitwise on same values.
6. **Matrix zoo:** extend the existing downloader (same lockfile + atomic-rename
   requirements) with a rectangular set: LS problems and underdetermined/LP problems
   from the SuiteSparse Collection (candidate names from SPQR paper Tables II/V —
   published names; implementer verifies availability/size before pinning), plus
   synthetic generators needing no download.
7. **Zero-alloc gate:** `@allocated qr!(F, A2) == 0` and `@allocated solve!(x, F, b)
   == 0` after warmup, StrictMode-checks-disabled configuration (design.md §9.1 D7
   split unchanged) — including a rank-deficient instance (the §5 path allocates
   nothing by construction; prove it).

### 9.2 Baseline facts verified in this environment (2026-07-14, Julia 1.12.6)

Verified by running, not assumed: `using SparseArrays; qr(A::SparseMatrixCSC)`
dispatches to `SparseArrays.SPQR.qr` and returns `SPQR.QRSparse{Float64,Int64}`;
keyword surface is `qr(A; tol, ordering)`; `SparseArrays.SPQR` exposes ordering
constants including `ORDERING_FIXED`, `ORDERING_AMD`, `ORDERING_COLAMD`,
`ORDERING_METIS`, `ORDERING_DEFAULT`; the factor object exposes `.R`, `.Q`, `.prow`,
`.pcol` properties; `F \ b` and `rank(F)` work. So the stdlib **does** ship a full
SuiteSparseQR baseline by default, and **the same-permutation gate arm is feasible in
both directions**: (→) run stdlib `qr` with `ordering=SPQR.ORDERING_FIXED` on
column-pre-permuted A to impose our permutation; (←) feed stdlib's chosen `F.pcol` into
PureSparse via `GivenOrdering`. One open empirical item for the harness (checked at
bench time, not assumed): whether Julia's SuiteSparseQR build runs TBB tree-parallelism
internally — the harness must confirm single-threaded execution (BLAS threads pinned to
1 as usual; if SPQR spawns TBB threads regardless, that is *recorded and reported* with
the results, and the gate comparison notes it — we do not silently gate against a
parallel baseline, nor silently ignore that it is one).

### 9.3 Benchmark matrix and performance gate

Same methodology as design.md §9.3 (Chairmarks medians, locked clocks, single thread,
results→JSON, plots from JSON; PkgBenchmark self-regression). Configurations:

| # | Factorization | Notes |
|---|---|---|
| 1 | PureSparse QR (M5a; M5b when built) | primary |
| 2 | SuiteSparseQR via `SparseArrays.qr` (stock) | baseline |
| 3 | 1 vs 2 under identical column permutation (both directions, §9.2) | part of the gate, not supplementary (design.md D2 discipline) |
| 4 | PureSparse `cholesky(AᵀA)` normal equations | context arm (not a gate): quantifies the §1.2 guidance |

**Gate (M5 closeout, non-negotiable, wall-time):** on each gate matrix,
`median_seconds(PureSparse qr(A)+solve, cold) < median_seconds(SparseArrays.qr(A)+solve,
cold)`, own-ordering **and** same-permutation arms, on a gate set stratified into
(i) singleton-dominated (LP-like), (ii) sparse-R/small-front LS, (iii) flop-rich/large-
front LS. Cold-vs-cold is the honest comparison unit because stdlib exposes no
analyze-once/refactorize path at all — our warm `qr!` numbers are **reported** (they are
the IPM/NLLS-relevant numbers and a genuine product advantage) but not gated against a
counterpart that doesn't exist.

**Stated expectation and the escalation trigger (H4):** by the published record, M5a
should win stratum (i) (no numerical work at all after singleton peeling) and is
competitive on (ii) (SPQR paper §1's own concession for very sparse R); stratum (iii)
is where multifrontal BLAS-3 earns its keep (paper §5.5: SPQR ≈ dense-DGEQRF rates) and
where M5a *may* lose. If, on locked-clock measurement, any stratum loses the gate, M5b
(§7) is triggered and M5 stays open until the full gate passes. No fudge-factor gates,
no "within 2× is fine" — the milestone closes on the same inequality every other
PureSparse milestone closed on. GFlops remains a secondary diagnostic only (design.md
D2; doubly gameable here because flop counts differ by Householder-vs-blocked
application accounting — the SPQR paper §5.5 makes exactly this point about its own
flop counts).

### 9.4 Gate set

Stratum (i): LP-constraint matrices (SPQR paper §2.1's singleton statistics identify
the class); (ii)/(iii): the paper's Table II least-squares problems
(psse0/psse2/graphics/Kemelmacher/deltaX/ESOC/Rucci1-class, availability verified at
implementation time) split by measured front-size distribution, plus synthetic:
2-D-grid surveying-type LS, random tall sparse, and rank-deficient constructions for
the §5 path. Sizes capped to CI-tolerable downloads per the existing zoo rules; the
large stratum-(iii) instances live in the performance set, not CI.

---

## §10 Milestones and task list

### M5 — Sparse QR (this document)

**M5a deliverables:** `qr/singletons.jl`, `qr/symbolic.jl`, `qr/numeric.jl`,
`qr/solve.jl`, `ordering/colamd.jl`, `ordering/ata.jl`, types/contracts/tuning
additions, tests per §9.1,
rectangular zoo extension, benchmark harness arms (§9.3), trim smoke extension (§8),
docs page (least-squares guide incl. §1.2 guidance + §5 honesty), drop-in forwarding
(M4-pattern, Preferences-gated).

**M5a gate:** §9.1 layers all green (BigFloat oracle, SPQR black-box agreement,
invariants H1/H2 executable checks, zero-alloc, trim); §9.3 measured on the full gate
set with the stratified verdict recorded in ROADMAP.

**M5 closeout gate (unconditional):** §9.3 wall-time inequality on every stratum, both
permutation arms + §2.2 ordering-quality bound. If M5a's measurement already satisfies
it, M5b is not built (recorded as such); otherwise M5b is mandatory scope.

**Task list (ordered):**
1. Types + tunables + contracts (`QRSymbolic`/`QRFactor`/`QRStats`/`QRWorkspace`,
   §1.4/§1.6/§6.5), incl. the `order_columns` contract (§2.1).
2. `ata_pattern` + AMD-on-AᵀA ordering path (§2.2.6) — small, lands first so every
   downstream task has a working ordering while task 3 proceeds; ordering-quality
   check vs stdlib baseline wired into tests.
3. **COLAMD** (§2.2) — the longest single M5a task, exactly as AMD was M1's (M1 task-3
   precedent: budget accordingly). **Prerequisite reading: Larimore thesis ch. 3–4 in
   full** (§2.2 sources — the thesis is the implementation-precision reference; the
   journal paper is the condensed spec this section was drafted from). Then §-by-§:
   row/column set storage + row-merge update (paper §3, Algorithms 1–2; thesis §4.1
   data-structure layout incl. the 2·nnz+n_cols index array + garbage collection) →
   metric bookkeeping (Algorithm 3 tag arrays, initial COLMMD metric, AMD metric in
   the update; thesis §4.2 init_scoring/find_ordering decomposition) → super-columns/
   mass elimination (thesis: hash table sharing the degree-list head array) →
   aggressive row absorption → dense/null row and column withholding (thesis §4.2.3).
   Tests: brute-force exact-minimum on tiny matrices, quality-vs-stdlib-COLAMD bound
   (§2.2), H6 review pass against both sources before merge.
4. Star pattern builder + reuse of etree/postorder/counts (§3.2); H1 brute-force
   equivalence tests **first** (they are cheap and everything depends on them).
5. Staircase row assignment + V counts/patterns (§3.4) + row-path property test (H2).
6. Numeric left-looking loop (§4) + Householder kernel (§4.4) + BigFloat oracle tests.
7. Solve phase (§6: apply_Q!/apply_Qt!/solve_R!/solve_Rt!/solve!/`\`/min-norm) +
   residual gates + SPQR black-box solution agreement.
8. Rank detection + dead-column path (§5) + constructed-rank tests + `dropped_norm`
   certificate tests.
9. Singleton pre-elimination (§2.3) + LP-class tests + the reuse-path (`n1=0`)
   interaction test.
10. `qr!` refactor hardening: zero-alloc gate, StrictMode layer, trim smoke + TrimCheck
    roots (§8).
11. Zoo extension + benchmark arms + **gate measurement and stratified verdict**,
    incl. the COLAMD-vs-AMD-on-AᵀA default decision (§2.2.6) and dense-threshold
    calibration (ROADMAP entry; this task decides M5b).
12. Drop-in forwarding + stdlib-parity property checks (observed surface, §6.4/§9.2).
13. Docs (least-squares guide, API reference, benchmark page from saved JSON).

**M5b (conditional) task list:**
- P1. PureBLAS: block-reflector apply kernel (larfb-role, §7.2) — in PureBLAS, with its
  own OpenBLAS-parity gate.
- P2. PureBLAS: generic-`T` `geqrf!` fallback (§7.2).
- 14. M5b design addendum (front assembly/stack simulation details; §7.1 scope) —
  reviewed before code, like this document.
- 15. Front-structure symbolic extension (existing supernode code on (parent, rcount);
  assembly simulation; stack arena sizing; staircase).
- 16. Frontal numeric loop (assemble → partial QR via geqrf!+P1 → push C), per-front
  Heath handling (§7.3), amalgamation recalibration for QR fronts.
- 17. Re-run §9.3; M5 closes on the unconditional gate.

### M6 — GPU (renumbered from M3, content unchanged — design.md §8, ROADMAP "M3" section)

---

## §11 Clean-room provenance policy (QR-specific restatement)

Identical policy to design.md §11, with SuiteSparseQR added explicitly to the prohibited
set: **never read CHOLMOD or SuiteSparseQR source code, headers, comments, or commit
history — directly or indirectly** (search snippets, LLM recall of source text,
third-party ports). Never reuse a SuiteSparse identifier, struct field name, or numeric
constant unless independently derivable — every name and constant in this document must
survive "where did this come from?" with a paper citation, a user-guide/interface
citation, or an in-document derivation (the τ formula §5.3, the star-matrix proof §3.2,
the pivot-row convention §3.4, and the reflector convention §4.4 are the in-document
derivations; H5 asks the reviewer to hunt for accidental matches). **Permitted and
used:** published papers/books; official interface documentation; black-box observation
of `SparseArrays.qr`'s API surface, outputs, and performance (§9.2's probes were
reflection on a running session — kwargs, property names, constants' *names* — never
wrapper or library source).

**Provenance table** (every component, its allowed source):

| Component | Source |
|---|---|
| R-pattern = Cholesky(AᵀA) pattern; column etree | George–Heath 1980; Coleman–Edenbrandt–Gilbert 1986 (upper-bound caveat) — both via survey §7.1/§11.5 |
| Star-matrix AᵀA-free symbolic | construction: Gilbert–Li–Ng–Peyton 2001 as described in survey §7.1 (primary paper unavailable — declared gap); correctness: **own fill-path derivation, §3.2** |
| Row/column counts | Gilbert–Ng–Peyton, ORNL/TM-12195 1992 (`refs/.../QR/gilbert_ng_peyton_1992_ornl_tm12195.pdf`, pseudocode + Lemmas 1–4; QR applicability stated in its §1) — already implemented in `symbolic/counts.jl`, reused |
| Left-looking column Householder | Davis 2006 as presented in survey §7.3 (full dense pseudocode + sparse row-subtree specification quoted there) |
| V patterns / row paths / row-merge counting | George–Ng 1986, 1987; George–Liu–Ng 1988; Liu 1986c; Oliveira 2001 (evaporation) — all via survey §7.1/§7.3; set-algebra formulation + pivot convention: **own, §3.4 (H2)** |
| Householder reflector convention | textbook (Golub–Van Loan-style; survey's `gallery('house')` reference); packing + explicit-pivot storage: own, §4.4 |
| Singleton pre-elimination | SPQR paper (Davis, TOMS 2011) §2.1 — description-level; queue implementation ours |
| COLAMD (v1 default ordering) | Davis–Gilbert–Larimore–Ng, ACM TOMS 30(3), 2004 (`refs/.../QR/davis_gilbert_larimore_ng_2004_colamd.pdf`, read in full — §3 symbolic LU/row-merge, §4 Algorithms 2–3 + metrics, §4.8 recommended variant); implementation depth: Larimore MS thesis, UF 1998 (`refs/.../QR/larimore_1998_colamd_thesis.pdf`, spot-checked this draft, full ch. 3–4 read scheduled for task 3/review — §2.2); row-merge tree: Liu 1991 via the paper; dense thresholds: **own, §2.2 pt 5 (H5)** |
| Ordering alternative (AMD on AᵀA) | existing `ordering/amd.jl` (design.md §2.2 provenance); precedent MA49 + SPQR options/default, SPQR paper §2.2/§5.4 |
| Rank detection threshold test | Heath 1982 via survey §7.2/§7.4 + SPQR paper §3.2; τ default formula: **own derivation, free tunable, §5.3** |
| Dead-column drop + error report | Foster–Davis 2013 phase-1 strategy as described in survey §7.4; left-looking adaptation ours (§5.2, H3) |
| No-fill guarantee for Heath-style handling | SPQR paper §3.2 Theorem 1 (proof read and summarized §5.2) |
| Multifrontal QR (M5b) | Matstoms 1994/1995; Amestoy–Duff–Puglisi 1996 (MA49); SPQR paper §2.3/§3 — survey §11.5 for the landscape |
| Solve formulas (LS/basic/min-norm) | SPQR paper §3.3; George–Heath–Ng 1984 via survey §7.2 |
| Alternatives guidance (§1.2) | survey §7.5 (normal equations / augmented system / Peters–Wilkinson) |
| Fronts-from-supernodes, staircase, stack | SPQR paper §2.3/§3.1–3.2 |
| Dense QR kernels (M5b) | PureBLAS `geqrf!` (verified present, §4.6); P1/P2 additions specified §7.2 |

Local reference archive (gitignored): `refs/linear_algebra/QR/` holds the five primary
PDFs cited throughout: `davis2011_spqr_toms.pdf`,
`davis_rajamanickam_sidlakhdar_survey_2016.pdf` (§7 + §11.5),
`gilbert_ng_peyton_1992_ornl_tm12195.pdf`, and
`davis_gilbert_larimore_ng_2004_colamd.pdf` — all read in full for the sections cited —
plus `larimore_1998_colamd_thesis.pdf` (spot-checked; full ch. 3–4 read is scheduled
work, §2.2 status note).
The one remaining source gap is Gilbert–Li–Ng–Peyton 2001 (BIT 41(4), the QR/LU
row-count extension) — mitigated by the §3.2 own derivation plus its brute-force
equivalence test, and by GNP92 (its stated basis) being in the archive.
One correction to the existing archive index while auditing sources: design.md §11
describes `refs/linear_algebra/chapter-direct.pdf` as "Davis's book" — its actual
content is a support-preconditioner Cholesky chapter (checked: first page reads
"CHAPTER 3, Computing the Cholesky Factorization of Sparse Matrices", support-
preconditioner context; it contains none of the counts/QR material). Nothing in *this*
document cites it; the design.md §11 label should be corrected in a follow-up, and any
implementer told to "see Davis's book" must obtain the actual book (Davis, *Direct
Methods for Sparse Linear Systems*, SIAM 2006) rather than that file.
