# PureSparse.jl — M5b Design Addendum: Multifrontal Sparse QR Numeric Phase

Fleshes out `design_qr.md` §7's M5b sketch into an implementable design, per the M5b
task list's own task 14 ("M5b design addendum … reviewed before code, like this
document"). Trigger context: the M5a wall-time gate (design_qr.md §9.3) was measured on
two clock-locked machines and **M5a loses 3–10× on strata (ii)/(iii)**; profiling
confirmed the hot cost is the indirect gather/scatter in `_apply_reflector!`
(`src/qr/numeric.jl`: `y[sym.vrowind[pp]]` — non-contiguous, non-vectorizable), i.e.
the architectural limit design_qr.md §1.3's trade-off table predicted for the
left-looking method, not a codegen problem. Per §9.3/H4, M5b is now **mandatory
scope**; M5 stays open until the unconditional gate passes on every stratum.

This document extends `design_qr.md` (which remains the M5a record, unmodified). Its
§-numbers are prefixed **A** to avoid collision. Everything in design_qr.md §§1–6
stands except where §A1.2 explicitly says "replaced."

**Sources and provenance (same absolute policy as design_qr.md §11 — CHOLMOD /
SuiteSparseQR source remains prohibited in every form; only published papers, permitted
implementations, and in-document derivations appear below):**

- **SPQR paper** (Davis, *Multifrontal multithreaded rank-revealing sparse QR
  factorization*, ACM TOMS 2011; `refs/linear_algebra/QR/davis2011_spqr_toms.pdf`) —
  §2.3 (supernodes→fronts, the two-condition supernode test, relaxed amalgamation from
  counts alone, the leftmost-nonzero row sort, the assembly simulation and staircase,
  exact-if-τ<0 / upper-bound-otherwise), §3.1 (front assembly, Figures 2–3, pivotal /
  non-pivotal column terminology, upper-trapezoidal contribution blocks), §3.2 (the
  DLARFG/DLARF/DLARFT/DLARFB kernel decomposition, T with leading dimension b,
  staircase-limited panel application, Heath per-front dead-pivot handling and C
  growth, Theorem 1, the postorder stack discipline), §3.3 (solve formulas, the
  apply-b-during-factorization option). Re-read for this addendum; pages cited
  per claim below.
- **`faer` 0.24.1 sparse QR source** (MIT — the permitted-source category design_qr.md
  §0/§11 already established; read directly for this addendum at
  `~/.cargo/registry/src/index.crates.io-*/faer-0.24.1/src/sparse/linalg/qr.rs`,
  `supernodal` module: `SymbolicSupernodalHouseholder`,
  `ghost_factorize_supernodal_householder_symbolic`,
  `factorize_supernodal_numeric_qr`'s worker). Used for implementation-level mechanics
  the SPQR paper describes only at survey depth: permanent-front storage (no
  contribution-block stack), the per-front row-count recurrence with the trapezoid
  clamp, numeric-time per-front min-col sorting, the staircase-group panel split rule,
  post-factorization min-col rewrite, survivor pass-up via per-front cursors, and
  stored per-panel T blocks. Every reuse is cited inline as "faer `qr.rs`."
- **BlazingPorts.jl** (`/home/el_oso/Documents/claude/BlazingPorts.jl/src/Factorizations.jl`,
  sibling repo, own code) — prior art for the P1 kernel *implementation approach* only
  (§A7.4): its dense blocked QR beat faer's dense QR at all benchmarked sizes
  512–2048 after finding that the decisive lever was gemm orchestration (reading the
  trailing operand **in place, unpacked**, matching faer's own `pack_rhs=false`
  choice), not algorithm structure. Different problem (dense n×n vs per-front blocks);
  cited as feasibility proof and contract guidance, never reused directly.
- **Explicitly rejected source:** PureKLU.jl was suggested as possible prior art and
  was **not read** — its own README declares it "a direct, line-by-line port of …
  SuiteSparse C sources" (LGPL-derived), which is precisely the "third-party port
  derived from that source" CLAUDE.md requirement 1 prohibits. (It is also
  architecturally irrelevant: simplicial left-looking LU, no fronts, no stack.)

Everything below marked **own derivation** has no external source and must be judged
on the argument given; everything marked **empirical** is a free tunable to be
measured, not assumed.

---

## §A1 Architecture

### A1.1 What M5b is

Replace M5a's per-column numeric loop (design_qr.md §4) with the multifrontal scheme
(SPQR paper §3; Matstoms 1994/1995 and MA49 via the survey §11.5 for the landscape):
each supernode of the Cholesky factor of AᵀA — computed by the **existing** supernode
machinery on the **existing** star-pattern outputs — becomes one dense **frontal
matrix**; fronts are assembled and factorized in postorder; each front's dense partial
Householder QR runs through PureBLAS BLAS-3 kernels; the trailing **contribution
block** C is consumed by the parent front's assembly. All indirect-indexed work is
confined to assembly (O(front entries) data movement); all flops are dense.

### A1.2 What stays / what is replaced

**Kept unchanged (design_qr.md §7.3, confirmed here field-by-field in §A4):** §2
ordering + singleton pre-elimination and their one-shot/reuse policy; §3 symbolic
pipeline through `qr_row_structure` (star pattern, etree, postorder, `rcount`,
`rperm`/`riperm`/`mb`, `leftcol` row assignment); §5's τ policy and default; §6's API
surface (`qr`/`qr!`/`solve!`/`\`/`apply_Q!`/`apply_Qt!`/`solve_R!`/`solve_Rt!`/
min-norm); §9's test layers and the §9.3 gate definition. The M5a numeric path
(`qr/numeric.jl`) **remains in the tree** as the generic-`T` fallback and
small-problem path (§7.3; method selection in §A5.6).

**Replaced (for the Float64 tuned path):** the per-column left-looking loop and its
V-pattern-indexed storage. M5b introduces per-front dense storage (§A4.2); the M5a
fields `vptr`/`vrowind`/`vval`/`beta`/`pivotslot`-as-symbolic are not used by the
frontal path (the frontal analog of `pivotslot` is numeric-time, §A5.4). `sptr`/`sind`
(the T^k seed structure) is not needed by the frontal path either — the front tree
replaces the row-subtree traversal.

### A1.3 Storage decision: permanent fronts, no contribution-block stack

The SPQR paper (§3.2, pp. 11–12) describes a stack discipline: fronts factorized in
postorder, C blocks pushed to a stack, R and H then *compressed in place* to reclaim
the C space and the staircase zeros. `faer`'s sparse QR (`qr.rs`) makes a different,
simpler choice: **every factorized front is stored permanently, uncompressed**, in one
contiguous per-front column-major rectangle; the parent reads its children's C blocks
directly out of the children's stored fronts, and no separate stack exists at all
(`SymbolicSupernodalHouseholder.col_ptr_for_val` sizes `Σ_f rows_f × cols_f` up
front).

**M5b adopts faer's layout.** Rationale:

- Zero-alloc-after-symbolic (CLAUDE.md req 5) falls out immediately: every offset is a
  symbolic prefix sum; there is no in-place compression step and no stack whose
  high-water mark must be simulated. (The SPQR-style simulation is still performed —
  §A3.4 — but only for row-count/size bounds, not for a stack arena.)
- V must be kept anyway (Q is required by §6's solves; Q-less mode is a non-goal,
  design_qr.md §6.3), and the solve phase (§A6) replays fronts against their stored
  reflectors — permanent fronts are the natural representation for that.
- The memory cost vs SPQR's compressed layout is the retained C copies and staircase
  zeros — bounded by the same `Σ_f m_f·n_f` the numeric phase must touch anyway. The
  trade is memory for simplicity; recorded as revisitable if a gate matrix shows
  memory pressure (the compress-and-stack variant is a storage-layer change, not an
  algorithm change).

This is an implementation-architecture choice informed by reading a permitted source
(faer, MIT), not by SPQR's implementation; the SPQR *paper*'s stack description is the
published alternative we deliberately did not take.

---

## §A2 Fronts from the existing supernode machinery

### A2.1 Inputs — everything already exists

The frontal partition is computed on `(parent, rcount)` from `symbolic_qr`
(design_qr.md §3.2: `parent` = postordered column etree of the block, `rcount` =
column counts of L(AᵀA) = row sizes of R), by the **existing** functions in
`src/symbolic/supernodes.jl`:

- `fundamental_supernodes(n′, parent, rcount)` — with one **new variant flag**, §A2.2;
- `relaxed_amalgamation(n′, nsuper, super, parent, rcount)` — reused verbatim
  (tunable recalibration in §A8);
- `supernode_tree` / `supernode_rowind(n′, sptr, sind, parent, nsuper, super)` — the
  latter fed with the star matrix's postordered strict-upper pattern (`sym.sptr`/
  `sym.sind`, exactly the shape it consumes). Its `rowind` output, read in QR terms,
  is **front f's column list**: the union of the R-row patterns of f's pivotal
  columns. (In Cholesky terms it is the supernode's row pattern of L; row k of R ↔
  column k of L, design_qr.md §3.1, so the same array serves both readings.)

SPQR paper §2.3 (p. 5) is the published basis for the whole reuse: "Each supernode in
the Cholesky factor L represents a group of adjacent columns with identical or nearly
identical nonzero pattern, which is the same as a set of rows of the R factor for a QR
factorization. Each supernode from [the Cholesky analysis] becomes a single frontal
matrix" — and, on amalgamation, "[the] relaxed amalgamation [is] based solely on the
nonzero counts of L," which is exactly what `relaxed_amalgamation` consumes.

### A2.2 The two-condition supernode variant (new flag, small)

SPQR paper §2.3 (p. 5): "Two columns j and j+1 are in the same supernode if
parent(j) = j+1 and |L∗j| = |L∗,j+1| + 1 … For these two columns to reside in the same
*fundamental* supernode, j must also be the only child of j+1; SuiteSparseQR does not
use this restriction." The existing `fundamental_supernodes` enforces the third
(only-child) condition. M5b adds a keyword:

```julia
fundamental_supernodes(n, parent, colcount; fundamental::Bool = true)
```

`fundamental = false` skips the `childcount[j+1] == 1` test (the QR default;
Cholesky callers unchanged, default `true`). Two consequences, both checked here:

- **Pattern identity still holds.** The two remaining conditions already imply
  identical patterns (paper, same sentence: "which implies that the two columns have
  the same nonzero pattern (except for j itself…)"), so front columns/staircase logic
  below is unaffected.
- **`relaxed_amalgamation`'s exact-height derivation survives.** Its docstring's
  single-range-root argument needs only that a supernode is a `parent[j] == j+1`
  chain — condition 1, retained. Condition 3 never enters that argument
  (re-checked against the docstring's own induction while writing this addendum;
  own re-derivation, no code change needed).

The §A3.3 worked example shows the variant mattering: with the only-child condition
the example splits into 3 fronts, without it into 2.

### A2.3 Front tree and children lists

`supernode_tree` gives `fsnode` (column → front) and `fparent`. M5b additionally
stores the inverted children lists in CSC form (`fchildptr`/`fchildren`, children in
ascending front order — which is postorder, since fronts inherit the postordered
column order). Same head/next construction `supernode_rowind` already uses
internally; faer stores the identical structure (`child_head`/`child_next`,
`qr.rs`) — here materialized flat because it is symbolic (built once) and flat CSC is
trim-friendlier to iterate.

---

## §A3 Front assembly and the assembly simulation

### A3.1 Definitions

Everything is in **block space** (columns `1..n′ = n−n1`, physical rows `1..mb`) —
design_qr.md §1.4's D5 convention, unchanged. For front `f`:

- **Pivotal columns**: the supernode's own columns `fsuper[f] : fsuper[f+1]−1`
  (SPQR §3.1's term, p. 8; MA49 calls them fully-summed, Pierce–Lewis internal —
  paper's own attribution). Width `p_f`.
- **Front columns**: `supernode_rowind`'s sorted list for `f` — pivotal columns first
  (they are the smallest, being the supernode's own contiguous range), then the
  non-pivotal columns. Width `n_f`; non-pivotal count `c_f = n_f − p_f`.
- **A-rows of f**: physical rows whose `leftcol` is one of f's pivotal columns. SPQR
  §3.1 (p. 8): "A row i of P₂AP is assembled in the frontal matrix whose pivotal
  columns contain the leftmost column index j of row i"; M5a's `rperm` is exactly
  P₂'s leftmost sort (design_qr.md §3.4/B2), and `qr_row_structure`'s bucket
  boundaries `aptr` (now saved as `arowptr`, §A4.1) make these rows the contiguous
  physical range `arowptr[fsuper[f]] : arowptr[fsuper[f+1]]−1`. Count `a_f`.
- **Contribution block C of f**: the trailing, fully-triangularized rows of the
  factorized front over the non-pivotal columns — upper trapezoidal (SPQR §3.1 p. 8:
  "In general, the contribution block C can be upper trapezoidal"), passed to
  `fparent[f]`. Row count `cr_f` (numeric-time; bounds below).
- **min-col of a front row**: the front column of that row's first (leftmost)
  structural nonzero. A-rows of pivotal column k have min-col k. Row t (1-based) of a
  child's triangular C has min-col = the child's (p_c + t)-th front column.
- **Staircase**: per front column j, `stair[j]` = number of front rows whose min-col
  is ≤ column j, once rows are sorted ascending by min-col — "the row index of where
  the zero entries start in each column" (SPQR §2.3, p. 7, which defines it; §3.1
  Fig. 3 illustrates it — the N7-corrected citation from design_qr.md §0).

### A3.2 The row-count recurrence

Full-rank / rank-detection-disabled sizes, per front in one ascending (postorder)
pass:

```
m_f  = a_f + Σ_{c child of f} cr_c          # assembled rows
e_f  = min(m_f, n_f)                        # eliminations (full triangularization)
r_f  = min(p_f, m_f)                        # retired rows of R
cr_f = min(m_f − r_f, c_f)                  # contribution rows (trapezoid clamp)
```

The trapezoid clamp is the load-bearing part: rows beyond the `n_f`-th elimination are
structurally zero after full triangularization and are dropped, so a child never
passes more than `c_f` rows upward. Published/verified basis: SPQR §3.1's "C can be
upper trapezoidal" plus Fig. 2 (6×5 front → 3×3 triangular C) give the rule at
description level; **faer `qr.rs` implements the identical recurrence** —
`non_zero_count[parent] += min(max(s_count, panel_width) − panel_width, s_col_count)`
where `s_count = m_f`, `panel_width = p_f`, `s_col_count = c_f` (its
`ghost_factorize_supernodal_householder_symbolic`), and its numeric phase clamps the
pass-up cursor at `n_f` (`col_end_for_row_idx_in_panel[s] = min(…, s_ncols +
s_pattern.len())`). Note this is the supernodal generalization of the same formula
design_qr.md §0/B1 already cross-checked for the width-1 case.

**Rank-detection-aware capacity bounds (own derivation — the SPQR paper states only
*that* its analysis "provides an upper bound in case A is rank-deficient, and is exact
if rank-detection is disabled" (§2.3 p. 7, §3.1 p. 9), not the formula; faer has no
rank detection at all).** Under τ ≥ 0 a dead pivotal column retires no row, so C can
grow (SPQR §3.2 p. 11: "in general its size can increase … the loss of one column
would cause C to grow in size by one row"), still clamped by `c_f` (the column count
of C never changes — same page). Worst case all `p_f` pivots die:

```
crmax_f = min(mmax_f, c_f)                          # C capacity
mmax_f  = a_f + Σ_{c child of f} crmax_c            # row capacity
```

All storage (§A4.2) is sized by `mmax_f`/`crmax_f`; the exact recurrence above is kept
as the τ<0 diagnostic and drives the §A9 exact-count test. One honest caveat: M5b (like
M5a, `numeric.jl`'s B3 branch) treats an *exactly zero* pivot column as dead even when
τ<0, so "exact when τ<0" additionally assumes no exactly-zero transformed pivot
columns — a deliberate divergence from a strict reading of SPQR's exactness claim,
consistent with M5a's semantics (design_qr.md §4.4 B3). Capacities never rely on
exactness.

### A3.3 Worked example

(Verified numerically against the existing `star_pattern`/`etree`/`column_counts`/
`fundamental_supernodes`/`supernode_rowind` code before inclusion — not hand-derived
only.) A is 7×5, already in final permuted column order; row patterns:

```
r1: {1,2,4}   r2: {1,2,5}   r3: {2,4}   r4: {3,4}
r5: {3,5}     r6: {4,5}     r7: {4}
```

Symbolic pipeline (M5a, unchanged): star columns `1:{2,4,5}, 2:{4}, 3:{4,5}, 4:{5}`;
etree `parent = [2,4,4,5,0]`; `rcount = [4,3,3,2,1]`; `leftcol = [1,1,2,3,3,4,4]`,
so `a = [2,1,2,2,0]`, physical rows 1–7 = r1,r2 | r3 | r4,r5 | r6,r7, `mb = 7`.

Supernodes: the two-condition test merges `{1,2}` (parent(1)=2, 4=3+1) and `{3,4,5}`
(parent(3)=4, 3=2+1; parent(4)=5, 2=1+1). The *fundamental* (3-condition) partition
would instead give `{1,2},{3},{4,5}` because column 4 has two etree children (2 and
3) — the §A2.2 flag in action. Two fronts; front tree 1 → 2.

**Front 1**: pivotal {1,2}, front columns {1,2,4,5} (`p=2, n_f=4, c_f=2`); A-rows =
physical 1,2,3 (`a=3`), no children → `m=3`. Assembled (rows sorted by min-col;
`x` = structural entry, `.` = staircase-interior zero, blank = outside staircase):

```
            cols:  1  2  4  5          min-col   staircase stair = (2,3,3,3)
  phys 1 (r1):     x  x  x  .             1
  phys 2 (r2):     x  x  .  x             1
  phys 3 (r3):        x  x  .             2
```

`e = min(3,4) = 3` eliminations (front columns 1,2,4), `r = min(2,3) = 2` rows of R
(pivots 1,2), `cr = min(3−2, 2) = 1`: C is the 1×2 block over columns {4,5} — row 3
after triangularization, min-col 4.

**Front 2** (root): pivotal {3,4,5} = front columns (`p = n_f = 3, c_f = 0`); A-rows =
physical 4,5,6,7 (`a=4`) + front 1's C row → `m = 5`. Sorted assembly (A-rows before
child rows at equal min-col — §A5.2's stable bucket rule):

```
            cols:  3  4  5           min-col    stair = (2,5,5)
  phys 4 (r4):     x  x  .              3
  phys 5 (r5):     x  .  x              3
  phys 6 (r6):        x  x              4
  phys 7 (r7):        x  .              4
  c-row (f1):         c  c              4
```

`e = 3`, `r = 3`, `cr = min(2,0) = 0` (root; nothing passed). Padded-R sizes
(§A5.5): front 1 rows own 4 and 3 slots, front 2 rows 3+2+1 — total 13 = `Σ rcount`
here (patterns nest exactly; no padding on this example). Front value storage
(capacities, §A3.2 with `crmax_{f1} = min(3,2) = 2` → `mmax_{f2} = 6`):
`3·4 + 6·3 = 30` entries.

**Dead-pivot variant** (mechanics of §A5.4): suppose in front 1 the transformed
column 2 dies at elimination cursor k=2 (‖rows 2..3 of column 2‖ ≤ τ). Then the
reflector for column 2 is skipped, no row is consumed, and eliminations continue:
column 4 at cursor 2 (rows 2..3), column 5 at cursor 3 (row 3). Result: `r = 1` (pivot
1 only; R row 2 stays a zero-filled slot, the D9 convention), and C is now **2×2**
upper triangular over {4,5} — rows 2,3, min-cols 4 and 5 — i.e. C grew by exactly one
row, hitting the `crmax = 2` capacity, and front 2 assembles `m = 6 ≤ mmax = 6`. The
dropped mass for column 2 is its residual norm at detection, ≤ τ (§A5.4's frozen-
residual argument).

### A3.4 The assembly simulation (symbolic pass)

One ascending pass over fronts (postorder), O(total front-column count) = O(|R|'s
supernodal representation) — the SPQR paper's own complexity for this step (§2.3
p. 7: "proportional to the number of integers required to represent the supernodal
pattern of L"):

```
for f in 1:nfront                    # ascending = postorder
    a_f    = arowptr[fsuper[f+1]] − arowptr[fsuper[f]]
    mmax_f = a_f + Σ_{c ∈ children(f)} crmax_c
    r_f^ub = min(p_f, mmax_f)
    crmax_f = min(mmax_f, c_f)
    fvalptr[f+1]  = fvalptr[f]  + mmax_f * n_f          # front rectangle
    frowptr2[f+1] = frowptr2[f] + mmax_f                # row-id / min-col lists
    ftauptr[f+1]  = ftauptr[f]  + NB * min(mmax_f, n_f) # stored-T slab (§A5.3)
    # exact-mode (τ<0) sizes m_f/e_f/r_f/cr_f computed alongside for the
    # §A9 exact-count invariant and the fflops estimate (§A3.5)
end
```

No stack simulation is needed for storage (§A1.3); the *exact* per-front staircase is
not stored either — it is recomputed numerically per front during assembly in
O(m_f + n_f) (§A5.2), and used symbolically only inside the `fflops` estimate. This
deliberately deviates from design_qr.md §7.1's sketch wording ("stack high-water
mark … → preallocated arena"), superseded by the §A1.3 storage decision; the
*simulation* survives as the row-capacity recurrence above.

### A3.5 Flops estimate

Per front in exact mode, with the symbolic staircase (rows sorted by min-col, `s_j` =
stair at column j): elimination `e` at front column `j = j(e)` has support length
`ℓ = s_j − e + 1`; cost `3ℓ` (reflector) + `4ℓ·(n_f − j)` (apply to trailing columns).
`fflops = Σ_f Σ_e [3ℓ + 4ℓ(n_f − j)]`, same accounting family as design_qr.md §3.5
(and the same caveat: a diagnostic and method-selection input, never a gate — GFlops
gaming, design.md §9.3 D2).

---

## §A4 Symbolic extension: exact shapes

### A4.1 What is reused verbatim vs new

Reused with **zero changes**: `QRSymbolic` itself and everything that builds it
(`symbolic_qr`'s whole pipeline); `relaxed_amalgamation`; `supernode_tree`;
`supernode_rowind`; `csc_transpose`/`row_leftcol`. Changed with a **flag only**:
`fundamental_supernodes` (§A2.2). Genuinely new: one struct + one builder
(`qr/frontal_symbolic.jl` or folded into `qr/frontal.jl`; layout per design_qr.md
§1.5, which already reserves `qr/frontal.jl` for M5b).

M5a's `qr_row_structure` computes `aptr` internally and drops it; the frontal builder
needs it (§A3.1's A-row ranges). Rather than widening `QRSymbolic`, the builder
recomputes it from `leftcol` in O(m) (one counting pass — it is two lines) and stores
it in the frontal struct as `arowptr`. `QRSymbolic` stays byte-identical for M5a
users.

### A4.2 `QRFrontSymbolic{Ti}`

Composition, not mutation: the frontal symbolic *contains* the M5a symbolic (which the
fallback path and all shared fields — permutations, `rcount`, `rptr` for the
*unpadded* pattern, `mb` — keep serving). Every array below is block space.

```julia
struct QRFrontSymbolic{Ti<:Integer}
    base::QRSymbolic{Ti}           # M5a symbolic, shared by reference (§A4.1)
    # --- front partition & tree (post-amalgamation; §A2) ---
    nfront::Int
    fsuper::Vector{Ti}             # length nfront+1: pivotal column ranges
    fsnode::Vector{Ti}             # length n′: column -> front
    fparent::Vector{Ti}            # length nfront (0 = root)
    fchildptr::Vector{Ti}          # length nfront+1 ─┐ children lists (CSC over the
    fchildren::Vector{Ti}          #                  ┘ front tree, ascending order)
    # --- front column structure (supernode_rowind output, §A2.1) ---
    fcolptr::Vector{Ti}            # length nfront+1
    fcolind::Vector{Ti}            # per-front sorted global column lists; the first
                                   #   p_f entries are the pivotal columns
    # --- A-row assignment & row-form access (§A3.1, §A5.2) ---
    arowptr::Vector{Ti}            # length n′+1: physical rows arowptr[k]:arowptr[k+1]-1
                                   #   have leftcol == k (M5a's aptr, now retained)
    rowptr::Vector{Ti}             # length mb+1  ─┐ row-form PATTERN of the block A:
    rowcol::Vector{Ti}             #               ┘ permuted column ids per physical row,
                                   #   ascending (csc_transpose + relabel, built once)
    atrans::Vector{Ti}             # length nnz(A_block): CSC nonzero position (walked in
                                   #   cperm column order, as qr! step 1 already does) ->
                                   #   row-form slot; 0 for entries in null rows. Fills the
                                   #   numeric row-form value buffer in one O(nnz) pass per
                                   #   qr! — the amap idiom (design.md §4.2), QR-shaped.
    # --- capacities from the §A3.4 simulation (τ-robust upper bounds) ---
    fmmax::Vector{Ti}              # per-front assembled-row capacity mmax_f
    fcrmax::Vector{Ti}             # per-front contribution-row capacity crmax_f
    fvalptr::Vector{Ti}            # length nfront+1: front rectangles in fval (mmax_f×n_f,
                                   #   column-major, ld = mmax_f)
    frowptr2::Vector{Ti}           # length nfront+1: per-front row-id/min-col segments
    ftauptr::Vector{Ti}            # length nfront+1: stored-T slabs (§A5.3)
    fpanelptr::Vector{Ti}          # length nfront+1: per-front panel-descriptor capacity
                                   #   (≤ n_f panels; §A5.3)
    frptr::Vector{Ti}              # length n′+1: PADDED R row pointers (§A5.5) — row k
                                   #   owns n_f − (position of k in its front) + 1 slots
    # --- scalars ---
    nnzVF::Int                     # fvalptr[end]-1: total front storage
    nnzRF::Int                     # frptr[end]-1: padded R storage
    max_front_rows::Int            # max mmax_f  ─┐ workspace / kernel scratch
    max_front_cols::Int            # max n_f     ─┘ sizing (§A5.1, §A7)
    fflops::Float64                # §A3.5, exact-mode estimate
end
```

Naming note (design.md §0 B1 discipline): `fsuper`/`fsnode`/`fparent`/`fcolptr`/
`fcolind` extend this package's own `super`/`snode_of`/`sparent`/`rowind_ptr`/`rowind`
conventions (`src/types.jl`) with an `f` prefix; `arowptr`/`rowptr`/`atrans` extend
`aptr`/`csc_transpose`/`amap` precedents. None are taken from any SuiteSparse
internal, which we have never seen. faer's names for the analogous quantities
(`col_ptr_for_val`, `min_col_in_panel`, …) were read (MIT, permitted) but not adopted.

### A4.3 Numeric factor type

```julia
mutable struct QRFrontFactor{T<:Real,Ti<:Integer} <: AbstractSparseFactor{T}
    fsym::QRFrontSymbolic{Ti}
    # per-front storage (capacities from fsym; "used" extents below)
    fval::Vector{T}                # factorized fronts: V below the elimination profile
                                   #   (implicit unit diagonal), R rows / C block above.
                                   #   Zeroed per qr! on the used extents only.
    ftau::Vector{T}                # per-panel compact-WY T slabs (§A5.3)
    tauv::Vector{T}                # per-elimination LAPACK-convention tau (capacity
                                   #   Σ min(mmax_f, n_f); rebuildable T + generic path)
    frowind::Vector{Ti}            # per-front physical-row-id lists (frowptr2 layout),
                                   #   FILLED AT NUMERIC TIME in assembled (sorted) order
    fmincol::Vector{Ti}            # per-row min-col, same layout (assembly + pass-up)
    fm::Vector{Ti}                 # ACTUAL m_f of the last qr! (≤ fmmax)
    fr::Vector{Ti}                 # actual retired-row count r_f
    fnpanel::Vector{Ti}            # actual panel count per front
    pnrows::Vector{Ti}; pncols::Vector{Ti}; pbs::Vector{Ti}
                                   # per-panel descriptors (fpanelptr layout): rows/cols/
                                   #   T-block size of each staircase panel (§A5.3) —
                                   #   the solve replay reads these (faer stores the
                                   #   identical triple: tau_block_size/householder_
                                   #   nrows/ncols, qr.rs)
    fpivotrow::Vector{Ti}          # length n′: physical row retired as column k's pivot
                                   #   row, 0 = dead — the NUMERIC-TIME analog of M5a's
                                   #   symbolic pivotslot (§A5.4/§A6)
    rval::Vector{T}                # padded R values (frptr layout; §A5.5)
    rowval::Vector{T}              # row-form value buffer (length nnz), refilled per qr!
    ws::QRFrontWorkspace{T,Ti}     # §A5.1
    stats::QRStats                 # unchanged type
    ok::Bool
end
```

### A4.4 Workspace

```julia
struct QRFrontWorkspace{T,Ti<:Integer}
    g2l::Vector{Ti}       # length n′: global column -> local front column (0 = absent);
                          #   stamped/reset per front like faer's col_global_to_local
    cg2l::Vector{Ti}      # length n′: child-column -> child-local, during extend-add
    bucket::Vector{Ti}    # length max_front_cols+1: counting-sort cursors for the
                          #   min-col staircase sort (§A5.2 — allocation-free, stable)
    stair::Vector{Ti}     # length max_front_cols: numeric staircase of current front
    # P1 kernel scratch (§A7.2): sized once from max_front_rows/cols and NB
    wyV::Matrix{T}        # max_front_rows × NB   (explicit-unit V panel copy)
    wyVt::Matrix{T}       # NB × max_front_rows   (transposed copy, unpacked-gemm path)
    wyG::Matrix{T}        # NB × NB
    wyW::Matrix{T}        # NB × max_front_cols
    yqt::Vector{T}        # length max_front_rows: solve-phase gather buffer (§A6)
    rhs::Vector{T}        # length m: solve scratch (mirrors QRWorkspace.rhs)
end
```

`NB` is PureBLAS's derived block size via the P1 block-size query (§A7.2) — **not** a
PureSparse literal, and specifically not SPQR's published default b = 32 (§3.2 p. 10),
which we deliberately do not adopt (PureBLAS req 8: derive, don't hardcode).

---

## §A5 Frontal numeric loop

### A5.1 Top level

`qr!(F::QRFrontFactor, A; tol)` — same contract surface as M5a's `qr!`
(pattern-identical A, `n1 == 0` enforced, zero allocations after construction):

```
τ = _qr_threshold(A, tol)                       # reused verbatim (numeric.jl)
fill!(dropped/rank counters); fill used extents of fval with zero
rowval[atrans[p]] = A.nzval[p] for every nonzero p     # O(nnz) row-form values
for f in 1:nfront                               # ascending = postorder
    assemble!(f)                                # §A5.2
    factorize_front!(f, τ)                      # §A5.3–§A5.4
    harvest_R!(f); pass_up!(f)                  # §A5.5
end
finalize stats (rank, n_dead, dropped_norm, nnzR/nnzV/flops)
```

### A5.2 Assembly (per front f)

O(m_f + n_f + entries moved); all maps are preallocated, stamped or explicitly
un-set after use (faer `qr.rs` resets its global-to-local maps the same way).

```
1  build g2l: for (i, k) in enumerate(front cols of f): g2l[k] = i
2  gather min-cols of the m_f incoming rows:
     A-rows:  physical p ∈ arowptr[k]:arowptr[k+1]-1 for each pivotal k → min-col k
     child c: its stored survivor rows p_c+1 .. p_c+cr_c with their fmincol values
              (already ascending within each child, §A5.5)
3  staircase sort (stable counting sort, no allocation): count rows per local
   min-col into ws.bucket, prefix-sum, then place row ids + min-cols into
   frowind/fmincol at their bucket cursors — A-rows first, then children in
   ascending child order, preserving each source's internal order.  Numeric
   staircase: ws.stair[j] = bucket prefix sums (cumulative rows with min-col ≤ j).
   [faer sorts the same key with a comparison sort per front (qr.rs,
   sort_unstable_by_key on min_col_in_panel); the counting sort is our
   allocation-free equivalent — same result, stability makes it deterministic.]
4  scatter values into the front rectangle Ff = view(fval @ fvalptr[f],
   1:m_f, 1:n_f) (ld = fmmax[f]):
     A-row at local row i: for q in rowptr[p]:rowptr[p+1]-1:
         Ff[i, g2l[rowcol[q]]] = rowval[q]
     child C row at local row i: build cg2l for the child's columns once per child;
         copy the stored child front's row (columns ≥ that row's min-col) through
         g2l — the extend-add. Child columns map into parent front columns by the
         etree column-inclusion property (the same containment supernode_rowind's
         child-merge already relies on; design.md §3.6).
5  un-set g2l/cg2l entries touched (O(n_f) / O(n_c)).
```

The row-form input requirement is structural, not incidental: faer takes Aᵀ as its
numeric input outright (`qr.rs`, `AT` parameter); SPQR's paper notes the transpose "is
needed later … so this work is not wasted" (§2.1 p. 3). Here the pattern half is
symbolic (`rowptr`/`rowcol`) and only values are refilled per call (`atrans`), keeping
`qr!` zero-alloc.

### A5.3 Dense front factorization: staircase-blocked panels

Full triangularization of the assembled `m_f × n_f` front — pivotal *and* non-pivotal
columns (SPQR §3.1 Figs. 2–3: the C block emerges upper-triangularized; this is the
paper's "modified Strategy 3 from MA49", §3.1 p. 8 — the MA49 paper itself is not in
the archive, so Strategy 3 is used here exactly as far as SPQR's own text and figures
describe it, no further). The kernel decomposition is the one SPQR names (§3.2 p. 10):
per-column reflector generation + in-panel application, then blocked T + block-apply
to the trailing columns — DLARFG/DLARF/DLARFT/DLARFB roles, all via PureBLAS (§A7).

Panel loop (per front; `k` = elimination row cursor, `j` = column cursor):

```
k = 1; j = 1
while j ≤ n_f and k ≤ m_f
    # panel column range [j, j2): extend while the staircase permits — grow j2 until
    # width hits NB, or stair[j2] jumps past stair-of-panel-start by ≥ max(1, NB÷2)
    # (the split rule faer uses, qr.rs: current_min_col + max(1, max_block_size/2);
    # adopted as-is, MIT-cited; a free heuristic, revisit only on measurement)
    rows = k : stair[j2−1]                      # staircase-limited row extent
    (a) IN-PANEL, column by column jj = j..j2−1 (rank-policy-coupled → lives in
        PureSparse, calling PureBLAS level-1/2 primitives):
          ℓ = stair[jj] − k + 1
          xnorm = nrm2(Ff[k:stair[jj], jj])
          if jj is PIVOTAL and (xnorm == 0 or (τ > 0 and xnorm ≤ τ)):
              DEAD PIVOT (§A5.4): dropped_sq += xnorm²; n_dead += 1; record dead;
              continue          # k does NOT advance; no reflector; no tau slot
          if xnorm == 0:        # non-pivotal or τ-off exact-zero: identity reflector
              tauv[slot] = 0; advance k; continue      # B3 convention, §A3.2 caveat
          form reflector (LAPACK convention: v[k]=1 implicit, tau; sign(0):=+1 —
              matches PureBLAS qr_unblocked!'s own convention, design_qr.md §4.4);
              R diagonal −sign·xnorm at Ff[k, jj]
          apply H to the remaining panel columns (gemv! + ger!, rows k:stair[j2−1])
          record fpivotrow: if jj pivotal, fpivotrow[global(jj)] = frowind[k-th row]
          k += 1
    (b) PANEL→TRAILING: build T for the panel's live reflectors (P1a wy_t!,
        stored into ftau at this panel's slab; descriptors pnrows/pncols/pbs
        recorded), then C_trail := Qᵀ·C_trail via P1b wy_apply!('T') over
        Ff[panel k0 : stair[j2−1], j2:n_f]
    j = j2
end
e_f = k − 1   # eliminations performed
```

τ<0-and-full-rank fast path: step (a) degenerates to exactly `qr_unblocked!`'s job on
the panel view; the implementation may call it directly there (proven-fast path,
CLAUDE.md "anchor on the proven-fastest path") and keep the column-by-column loop for
the τ≥0 / generic-`T` cases. The in-panel dead-pivot test is on the **transformed**
column — the same quantity M5a tests (design_qr.md §5.1) and the same one SPQR tests
per front (§3.2 p. 11: "suppose the 2-norm of column 6 drops below the threshold τ"
*after* the previous reflection was applied).

Compact-WY T's are **stored** (`ftau` slabs), not rebuilt per solve: they were computed
during factorization anyway, and the §A6 replay then needs no per-apply `syrk!`+
recurrence. faer stores the identical objects (`tau_val` slabs sized
`max_block_size × min(rows, cols)` per supernode, `qr.rs`). The per-elimination scalar
`tauv` is also kept (T rebuild for the generic path, and StrictMode cross-checks).

### A5.4 Dead-pivot mechanics and the dropped-mass bound

Mechanics (from §A5.3's loop): a dead pivotal column is skipped — no reflector, no row
consumed, no R row (its padded R slots stay zero, the D9 convention; `fpivotrow` stays
0) — and elimination continues with the next column at the same cursor `k`. This is
SPQR's per-front Heath handling verbatim at description level (§3.2 p. 11: "The
Householder reflection for column 6 is skipped, and this front holds one less row of
R"), with Theorem 1 (same page; proof read and summarized in design_qr.md §5.2)
guaranteeing the squeezed R stays inside the Cholesky-of-AᵀA pattern — which is what
makes the padded, symbolically-sized storage safe under any death pattern.

Consequences, each already sized for in §A3.2: `r_f` shrinks by one per dead pivot;
`cr_f = min(m_f − r_f_live, c_f)` can grow by one per dead pivot up to the `c_f`
clamp (the §A3.3 variant shows it hitting the clamp); the parent's actual `m` grows
correspondingly, within `mmax`.

**Dropped-mass accounting (own derivation — neither the SPQR paper nor faer covers
the error-norm bookkeeping):** at detection, the dead column's residual (rows
`k:stair[j]`) has norm ≤ τ. Every *later* reflector is applied only to columns
strictly to the **right** of its own elimination column; a dead column, once skipped,
sits to the left of every subsequent elimination and is therefore **never touched
again** — its residual values are frozen. Everything subsequently dropped on that
column's account (the un-harvested sub-column entries of later-retired R rows, the
column's absence from C, the discarded all-zero trailing rows) is exactly a partition
of that frozen residual vector. Hence the per-dead-column contribution to
`stats.dropped_norm` is the *detection-time* `xnorm ≤ τ`, added once — a strictly
stronger guarantee than M5a's, where the post-detection discards accumulate
un-τ-bounded mass across every later column (design_qr.md §5.2's N2 honesty note).
This upgrade is a documented behavior change: M5b's `dropped_norm ≤ √n_dead · τ`
always. The §A9 tests pin it.

Bookkeeping consequence for `QRStats`: fields unchanged; `rank = Σ_f r_f_live`,
`n_dead`, `dropped_norm` as above.

### A5.5 Harvest and pass-up

**R harvest.** The retired row for live pivot k (global column) is the front row
consumed at its elimination: copy `Ff[row, pos(k):n_f]` into `rval` at `frptr[k]` —
a **padded** row: its column indices are implicit (the front's `fcolind[f]` tail from
k's position), so no per-row `rcolind` copy exists in the frontal factor; `solve_R!`
walks `(frptr, fcolind)` jointly. Padding = amalgamation's explicit zeros, the same
convention the Cholesky panels already carry (design.md §3.5); the padding ratio
`nnzRF / nnzR` is recorded per matrix as an §A8 calibration diagnostic. Dead rows: the
slots exist and stay zero (D9).

**Pass-up.** Rows `r_f_live+1 : e_f` of the factorized front are the C rows. Their
min-cols are rewritten to their post-triangularization values — row consumed by
elimination `e` at column `j(e)` has first entry at front column `j(e)`; survivor row
`t` of the C block gets the global column of front column `j(r_f_live + t)`'s
elimination (faer performs the same rewrite: the `current_min_col` loop after
factorization, `qr.rs`). Their physical row ids (`frowind`) and rewritten min-cols are
what the parent's step-2 gather reads; the values are read from the stored front
directly (§A5.2 step 4), so "pass-up" writes no values anywhere — it is bookkeeping
only. Rows beyond `e_f` (structurally zero after full triangularization, or frozen
dead-column residue) are dropped: not passed, not read again; their mass is already
counted (§A5.4).

### A5.6 Method selection and the generic fallback

`qr(A; method = :auto | :frontal | :column)` (and `symbolic_qr` mirroring it).
`:column` = M5a unchanged; `:frontal` = this design (Float64 first — the PureBLAS
tuned path; other `T<:Real` route through P2's generic kernels when they land, else
fall back to `:column`, which is fully generic today). `:auto`'s split is **empirical**
(an M5b benchmark-task deliverable, not designed here): the candidate predictor is a
front-quality statistic from the symbolic (e.g. `fflops / nnzRF` or mean
`mmax_f · n_f`), with the honest expectation from §9.3's measured strata that
`:frontal` wins (iii), likely (ii), and `:column` may keep stratum-(i)-like tiny
problems below kernel-call overhead. Do not guess the threshold — measure it on the
gate set.

---

## §A6 Solve phase

Q is the postordered product of per-front dense Q's, acting on each front's assembled
row set. Applications replay fronts against the **stored** panels + T's; because rows
are tracked by physical id end-to-end and each physical row belongs to exactly one
front's retired set (or terminates untouched), the replay can operate **in place on a
full-length work vector**, no solve-phase stack (own correctness argument below;
storage layout enables it, §A1.3).

```
apply_Qt!(y)   # y ← Qᵀy, length mb
for f in 1:nfront                        # postorder, same as factorization
    gather  ws.yqt[1:m_f] = y[frowind[rows of f]]     # assembled (sorted) order
    for panel b = 1 .. fnpanel[f]:                    # forward, trans = 'T'
        wy_apply!('T', panel view of stored front, stored T_b, ws)   # §A7.2
    scatter ws.yqt back to y[frowind[...]]
end
```

`apply_Q!` is the mirror: fronts in **reverse** postorder, panels within each front in
reverse order with `trans = 'N'`. Correctness of the in-place scheme (own argument,
recorded for the reviewer): front f's row-id list contains its A-rows and its
children's C-row ids — all still "live" when f runs, because a row id is retired at
most once (it becomes some front's pivot row, after which no ancestor lists it — the
pass-up passes only rows `> r_f_live`), and postorder guarantees every child completed
before f gathers. After the full sweep, `y[fpivotrow[k]]` holds `(Qᵀb)` entry for R
row k, and every never-retired physical slot holds a tail component —
`lsq_residual` = norm over the non-retired slots (design_qr.md §6.2's helper,
frontal edition).

`solve_R!` / `solve_Rt!`: back/forward substitution over the padded rows —
row k's entries are `rval[frptr[k] : frptr[k+1]−1]` against implicit columns
`fcolind[f][pos(k):n_f]`; dead rows (`fpivotrow[k] == 0`) ⇒ `x[k] = 0`, exactly M5a's
basic-solution semantics (design_qr.md §5.2/§6.2, SPQR paper §5.1 method (3)).
`solve!`/`\`/min-norm compose these identically to M5a (§6.2–§6.4) — the singleton
block's `r1*` fields and composition logic are untouched (frontal path slots in as the
block factorizer inside `qr`'s existing singleton flow).

Multi-RHS: loop columns (allocating multi-RHS staging is permitted exactly as
design.md's `Workspace.rhs` note permits it; single-vector solves are zero-alloc).

Deliberately **not** adopted: SPQR's apply-b-during-factorization option (§3.3 p. 12,
factor `[A b]` and discard Q) — it conflicts with analyze-once/solve-many (CLAUDE.md
req 7) and V is kept regardless; listed as a possible later memory-mode extension,
same status as Q-less mode (design_qr.md §6.3).

---

## §A7 PureBLAS kernel contracts (P1a / P1b / P2)

Specified here so PureBLAS work proceeds independently; each lands in PureBLAS with
its own OpenBLAS-parity gate (PureBLAS CLAUDE.md req 1) before M5b numeric work
begins. Verified current state (PureBLAS source read 2026-07-14, this addendum's own
pass — confirming design_qr.md §4.6's D8-corrected findings):

- `qr.jl`: `geqrf!(A, tau; nb)` Float64 blocked compact-WY over `qr_unblocked!`
  (faer-port panel); its **inline** trailing update already computes the
  `C −= V·(Tᵀ·(Vᵀ·C))` **'T'-direction** product (lines ~284–332), including the
  measured µarch-split unpacked-Vt path and a SIMD `trmm!` for the T-apply — but
  workspace comes from the module-global grow-on-demand `_QR_WS` (line ~264).
- `svd.jl`: `_apply_reflectors_left!` (line ~645) computes the **'N' direction**
  (`Y = T·W`, blocks right-to-left ⇒ `C := Q·C`) with scratch hardwired to
  `SVDWorkspace` fields.

So **both directions already exist as proven inline code**; P1 is extraction and
unification, not derivation — an even smaller task than design_qr.md §7.2's already-
rescoped description, and the D8 conclusion ("adapt, don't derive") is doubly
confirmed.

### A7.1 Conventions to pin (cross-package ABI decisions)

- **Reflector/tau convention:** LAPACK-style — `V` unit-lower-trapezoid with implicit
  unit diagonal, `H = I − tau·v·vᵀ`, `sign(0) := +1` (PureBLAS `qr_unblocked!`'s
  existing convention). Note the two *internal* T-recurrence variants in PureBLAS
  today differ (`geqrf!` builds T with `λ = 1/tau` for the faer `H = I − vvᵀ/tau`
  form; `svd.jl` uses tau directly): P1 must expose **one** documented convention and
  reconcile both call sites. M5a's own `beta = 2/vᵀv, explicit-pivot` convention
  (design_qr.md §4.4) stays confined to the `:column` path — the two never mix (V
  never crosses between methods).
- **Block size:** export `qr_block_size(m, n)::Int` (the derived-const `_QR_NB`
  logic, made queryable) so PureSparse's `NB` (§A4.4) is PureBLAS-derived, never a
  PureSparse literal. faer exposes the same query (`recommended_block_size`,
  `qr.rs`) — precedent, not spec.

### A7.2 P1a — T construction + block apply, caller-owned workspace

```julia
# dlarft role: T (bs×bs upper) from panel V (rows m, cols bs; implicit unit,
# read from the packed panel in place) and tau. G is bs×bs scratch (VᵀV via syrk!).
wy_t!(Tm::AbstractMatrix{T}, Apanel::AbstractMatrix{T}, tau::AbstractVector{T},
      G::AbstractMatrix{T}) where {T}

# dlarfb/dormqr role, ONE block:  C := Q·C ('N') or Qᵀ·C ('T'),
# Q = I − V·Tm·Vᵀ.  ws carries V/Vt/W views (per-shape minimums below).
wy_apply!(trans::Char, C::AbstractMatrix{T}, Apanel::AbstractMatrix{T},
          Tm::AbstractMatrix{T}, ws::WYApplyWorkspace{T}) where {T}
```

Workspace shape (exactly the five `_QR_WS` slots, caller-owned):
`V (m×bs)`, `Vt (bs×m)`, `G (bs×bs)`, `Tm (bs×bs)`, `W (bs×ncols(C))`. Multi-block
order is the **caller's** responsibility (PureSparse loops panels forward for 'T',
reverse for 'N') — the kernel is single-block, which is what dissolves D8's "reversed
block order" concern into caller-side looping.

**Orchestration contract (this is a requirement, not advice):** the trailing operand
`C` must be **read in place — never packed** — for the `W = Vᵀ·C` product on the fast
path, preserving `geqrf!`'s existing measured behavior (the explicit `Vt` copy + the
unpacked no-trans gemm / µarch split at `qr.jl` ~293–299). Two independent
confirmations that this is the decisive lever: faer's own gemm sets `pack_rhs=false`
for this shape (read in BlazingPorts' campaign from faer's MIT source), and
BlazingPorts' dense QR (`Factorizations.jl` `_qr_VtC_unpacked!`, comment: "do NOT
pack the huge trailing C — that ~mp·nt pack was the whole gap vs faer") went from
losing to **beating faer at every benchmarked size 512–2048** on exactly this change,
with pure SIMD.jl and no inline asm — i.e. the kernel-formulation gap, not the
language, was the whole story (the same conclusion the M5a gate post-mortem reached
for the sparse side). BlazingPorts additionally found the T-apply (`Y = Tᵀ·W`) worth
vectorizing — PureBLAS's `geqrf!` already does (`trmm!`, comment "was a scalar
latency-bound triple loop"); `wy_apply!` must go through `trmm!`, not a scalar loop.
BlazingPorts' two-level fat-dlarfb driver (inner ib-panels + one wide k=48 outer
apply) is noted as a possible later upgrade for very large fronts — **not** in P1
scope. (Prior art / feasibility only: BlazingPorts' code is a different package and a
different problem shape — dense n×n vs staircase fronts — nothing is imported.)

Acceptance (in PureBLAS): equality-to-`dormqr`-semantics oracle vs OpenBLAS on random
(m, n, bs) sweeps, both `trans`; zero allocations with caller workspace; parity gate
per PureBLAS methodology. Migration follow-up (PureBLAS-side, non-blocking for M5b):
re-point `geqrf!`'s inline update and `svd.jl`'s `_apply_reflectors_left!` at the
extracted kernels to avoid triplicated T-recurrences.

### A7.3 P2 — generic-`T` unblocked QR

`qr_unblocked!(A::AbstractMatrix{T}, tau::AbstractVector{T}) where {T<:Real}` —
generic scalar path (potrf!-precedent in the same file family; `cabi_lapack.jl:14`
documents today's Float64-only status). Blocked generic falls out if `gemm!`/`trmm!`/
`syrk!` generic paths hold up — their signatures are `AbstractMatrix`-generic
(verified: `gemm.jl:2063`, `level3.jl:1004/2741`) but **generic-`T` correctness/perf
through these exact call shapes must be verified at P2 implementation time, not
assumed here.** Until P2 lands, non-Float64 `qr` routes to `:column` (§A5.6), which
is generic today — so P2 gates only req-3 uniformity of the *frontal* path, not M5b's
Float64 gate.

### A7.4 Explicitly out of P-scope

The staircase panel loop, Heath testing, and dead-pivot skipping stay in PureSparse
(`qr/frontal.jl`): rank policy is sparse-QR-coupled, and PureBLAS kernels remain
rank-agnostic (boundary per CLAUDE.md's "dense per-supernode work through PureBLAS"
rule — orchestration is the sparse library's own domain, as in `llt.jl`).

---

## §A8 Amalgamation recalibration for QR fronts

**What is paper-grounded:** reusing the count-based relaxed amalgamation *unchanged in
mechanism* is SPQR's own documented choice (§2.3 p. 5: relaxed amalgamation "based
solely on the nonzero counts of L"), so `relaxed_amalgamation` runs verbatim on
`(parent, rcount)`. The two-condition supernode variant is likewise the paper's
(§A2.2).

**What is genuinely different for QR (claims, each with its basis):**

- The z-fraction the existing gate tests is the density of the *R-pattern rectangle*
  (L-pattern in Cholesky terms). For Cholesky that is exactly the padded panel the
  kernels touch; for QR the front's workload also scales with its **row count**
  (`m_f`: A-rows + child C rows), which the z-test never sees. The test is therefore a
  coarser proxy in QR than in Cholesky — *mechanism reused, meaning shifted* (own
  observation; SPQR reuses the same proxy, so this is not a defect, just a reason the
  numeric thresholds cannot be assumed transferable).
- Merging changes different costs: in Cholesky, padding wastes `potrf!`/`syrk!` flops;
  in QR a merge also **deletes an assembly round-trip** (the absorbed child's C is
  never materialized/re-gathered — pure memory-traffic savings, the very cost M5a
  died of) while adding staircase-padded flops in a taller front. Directionally this
  argues QR tolerates *more* padding than Cholesky at equal width (own reasoning —
  **not** a measured claim).
- The M5a→M5b padding also shows up as `nnzRF / nnzR` (padded R storage, §A5.5) — a
  new memory-side cost Cholesky's calibration never priced.

**What must be measured, not assumed (M5b task 16e):** the tier values. Protocol =
the ROADMAP task-7b' sweep, re-run for QR: Chairmarks medians on the §9.4 gate set
over a grid around the current Cholesky-calibrated `AMALG_COLS = (16,64,128)` /
`AMALG_ZMAX = (0.97,0.35,0.08)`, plus the 2-vs-3-condition supernode flag as a swept
axis, scored on the §9.3 wall-time arms with `nnzRF/nnzR` recorded. If QR wants
different values than Cholesky, add QR-specific Preferences (`qr_amalg_cols`/
`qr_amalg_zmax`, defaulting to the shared ones) rather than perturbing the Cholesky
calibration. Provenance guard (design.md §0 B1/B2 discipline): the existing names and
numbers are already our own swept values with in-file derivation comments
(`tuning.jl`); whatever the QR sweep selects must keep that property — a swept
number with the sweep recorded, never a remembered default from any implementation.

---

## §A9 Verification additions (extends design_qr.md §9.1; layers unchanged)

1. **Exact-count / capacity invariants** (the B1-class guard, frontal edition): on
   τ<0, no-zero-column inputs, the numeric `fm/fr/cr` per front equal the §A3.2 exact
   recurrence, and every `frowind` segment fills to exactly `m_f`; on rank-deficient
   inputs (constructed, §A3.3's variant included verbatim as a test), `m_f ≤ mmax_f`,
   `cr_f ≤ crmax_f`, with at least one case *hitting* the clamp.
2. **Cross-method agreement:** M5a vs M5b on the same matrix and τ — R rows equal up
   to row signs on live rows, identical `rank`/`n_dead` away from the τ boundary,
   solutions to residual tolerance; `qr!` refactor bitwise-equals fresh `qr` per
   method (M5a's §9.1 pt 5 property, now per method).
3. **Dropped-mass upgrade pinned:** constructed dead-column cases assert M5b's
   `dropped_norm ≤ √n_dead · τ` (§A5.4) and that M5a's ≥ M5b's on the same input.
4. **Staircase/assembly invariants (StrictMode layer):** `fmincol` non-decreasing
   after step 3; every scattered entry lands inside its front's column set; g2l/cg2l
   fully un-set after each front; `fpivotrow` values distinct-or-zero and within
   `1..mb`.
5. **Orthogonality/oracle:** the existing BigFloat-QR and SuiteSparseQR black-box
   oracles (§9.1 pts 4a–4c) run unmodified against the frontal factor through the
   public API — no frontal-specific oracle needed (the API surface is unchanged).
6. **Zero-alloc gate:** `@allocated qr!(F, A2) == 0` and single-vector
   `solve!/apply_Q!/apply_Qt! == 0` for `QRFrontFactor`, including a rank-deficient
   instance (checks the counting sort, pass-up bookkeeping, and stored-T replay paths
   allocate nothing) — StrictMode-checks-disabled configuration, as ever.
7. **Trim smoke:** the §8 juliac entry gains the frontal path (method kwarg
   exercised both ways); TrimCheck `@validate` roots for the new types.

---

## §A10 Task list (supersedes design_qr.md §10's M5b items; numbering continued)

- **P1** (PureBLAS): `wy_t!` + `wy_apply!` (both `trans`), caller workspace,
  `qr_block_size` query, tau-convention reconciliation, unpacked-C contract, parity
  gate (§A7.1–§A7.2). *Blocks 16b.*
- **P2** (PureBLAS): generic-`T` `qr_unblocked!` + generic-path verification of
  `gemm!`/`trmm!`/`syrk!` for the wy shapes (§A7.3). *Blocks nothing Float64; gates
  req-3 uniformity of the frontal path.*
- **14.** This addendum, reviewed before code (adversarial pass, same process as
  design_qr.md §0 — reviewer instructions: hunt accidental SuiteSparse rhymes in
  names/constants, and check §A3.2's capacity bounds and §A5.4's frozen-residual
  argument, the two load-bearing own-derivations).
- **15a.** `fundamental_supernodes` two-condition flag + tests (Cholesky callers
  regression-checked); front tree/columns wiring (`supernode_rowind` on
  `sptr`/`sind`); §A3.3's example as the first fixture.
- **15b.** `QRFrontSymbolic` builder: assembly simulation (§A3.4), row-form +
  `atrans`, `arowptr`, capacities/prefix sums; §A9.1 exact-count tests (they are the
  B1-class guard — land with the builder, not after).
- **16a.** Assembly + staircase counting sort + extend-add (§A5.2); leaf-front
  assembly tests vs dense gather oracle.
- **16b.** Front factorization (§A5.3, needs P1): staircase panels, dead-pivot
  handling, stored T; per-front BigFloat oracle + §A9.2/§A9.3 agreement tests.
- **16c.** Harvest/pass-up (§A5.5), padded-R `solve_R!`/`solve_Rt!`, per-front
  `apply_Q!`/`apply_Qt!` replay, `fpivotrow` (§A6); residual gates + zero-alloc gate
  (§A9.6).
- **16d.** Method selection (`:auto` heuristic measured on the gate set, §A5.6) +
  generic-`T` routing; drop-in/dropin parity re-checks.
- **16e.** Amalgamation + NB calibration sweep (§A8); `nnzRF/nnzR` recorded; ROADMAP
  entry with the swept grid and verdict.
- **17.** Re-run §9.3 in full (all strata, both permutation arms, faer context arm)
  on clock-locked machines (rsync-verify the remote checkout first — standing rule);
  M5 closes only on the unconditional gate.

---

## §A11 Provenance additions (extends design_qr.md §11's table)

| Component | Source |
|---|---|
| Supernodes→fronts; two-condition test; count-based relaxed amalgamation reuse; leftmost row sort; assembly simulation + staircase; exact-iff-τ<0 | SPQR paper §2.3 (pp. 5–7) |
| Front assembly, pivotal/non-pivotal terms, trapezoidal C, full triangularization ("modified MA49 Strategy 3" — used only as far as SPQR's own text/figures describe it; MA49 paper not in archive, declared gap) | SPQR paper §3.1 (pp. 8–9, Figs. 1–3) |
| Kernel decomposition (larfg/larf/larft/larfb roles), T with ld = b, staircase-limited panel apply, per-front Heath skip + C growth, Theorem 1, postorder/stack discipline | SPQR paper §3.2 (pp. 10–12) |
| Permanent-front storage (no C stack); row-count recurrence w/ trapezoid clamp; numeric per-front min-col sort + post-factorization rewrite; cursor pass-up; stored per-panel T; staircase-group split rule `max(1, NB÷2)`; Aᵀ numeric input | `faer` 0.24.1 `src/sparse/linalg/qr.rs` (MIT — permitted category per design_qr.md §0/§11), read directly for this addendum |
| Rank-aware capacity bounds (`mmax`/`crmax`, §A3.2); frozen-residual dropped-mass ≤ τ bound (§A5.4); in-place solve-replay correctness (§A6); padded-R storage + implicit colind (§A5.5); counting-sort staircase assembly (§A5.2); 2-condition amalgamation-height re-check (§A2.2) | **own derivations**, arguments in-document — review targets |
| P1 unpacked-C orchestration requirement; pure-SIMD ≥ faer feasibility | BlazingPorts.jl `src/Factorizations.jl` (own sibling code; faer `pack_rhs=false` read from faer's MIT source during that campaign) — prior art/feasibility, nothing imported |
| Deliberately not read | CHOLMOD/SuiteSparseQR source (prohibited, absolute); PureKLU.jl (offered as prior art, rejected: self-declared line-by-line SuiteSparse port ⇒ CLAUDE.md req 1's "third-party port" clause) |

Tunables introduced: none with literal values. `NB` is PureBLAS-derived (§A7.1);
the staircase split rule is faer's cited heuristic; QR-specific amalgamation
Preferences appear only if the §A8 sweep demands them, carrying the sweep as their
derivation. The SPQR paper's published b = 32 default is explicitly **not** adopted.
