# PureSparse.jl — Design Document (v2, final)

Pure-Julia supernodal sparse Cholesky/LDLᵀ solver. Clean-room reimplementation of the
CHOLMOD algorithm family (from published papers only) on top of PureBLAS.jl. This is the
canonical implementation document.

This is the second revision (v2) of the design. The first draft (v1) was adversarially
reviewed against the AMD paper, against PureBLAS's real source (signatures and semantics
verified line by line), and by hand-tracing the trickiest part of the algorithm (the
left-looking update schedule). The review found 2 BLOCKERs (clean-room provenance issues)
and 7 DEFECTs and verified the algorithmic core sound; v2 fixes every BLOCKER and DEFECT.
See `§0` below for the changelog.

For milestone status, see [`../ROADMAP.md`](../ROADMAP.md).

---

## §0 Changelog from the review of v1

- **B1** Renamed `Symbolic.maxcsize`/`maxesize` → `max_update_size`/`max_extend_rows`
  (§1.2, §3.6); quantities now derived by independent reasoning from the §4.3 schedule,
  with derivation comments, no external-implementation naming.
- **B2** Amalgamation thresholds are no longer presented as paper-grounded; §1.4/§3.5 now
  derive a starting point from first-principles flop-inflation/L2-residency reasoning, use
  our own numbers, and mark them as free Preferences-overridable tunables to be calibrated
  in M1's benchmark pass.
- **D1** §9.3: CHOLMOD+PureBLAS benchmark quadrant marked N/A (blocked on PureBLAS
  host-runtime double-init of libjulia under `lbt_forward` in any live Julia process);
  gate restated to not depend on it.
- **D2** §9.3: primary gate is now median wall-time on identical problems (not GFlops),
  and the same-permutation (`GivenOrdering`) run is part of the gate, not supplementary;
  GFlops demoted to secondary diagnostic.
- **D3** §5.1 relabeled "QDLDL/Clarabel-style signed diagonal regularization" (was
  inaccurately "matches MA57-in-IPOPT"); inertia counts `(n_pos, n_neg, n_zero)` added to
  `FactorStats` (§1.2); §5.2 notes downstream IPOPT-style consumers read reported inertia
  for their own regularization loop.
- **D4** §2.2: mass elimination is no longer a standalone numbered test; folded into
  supervariable/indistinguishability detection per the AMD paper's actual Algorithm 1.
- **D5** §3.4: fundamental-supernode predicate gains the required single-child condition;
  additionally the superset invariant `rowind[s] ⊇ true L-pattern of every column in s`
  is now a first-class tested property (§9.1).
- **D6** §9.1: TypeContracts (compile/precompile-time, trimmed away) cleanly separated
  from StrictMode runtime pre/postcondition checks (gated on
  `StrictMode.checks_enabled()`); test items rewritten to match the correct mechanism.
- **D7** §9.1: matrix-zoo cache download now specified as atomic temp-file-then-rename
  under a lock file; zero-allocation gate specified (§9.1 point 7).
- **N4** §2.2(2): degree-bound notation corrected to subtract the whole supervariable
  `𝐢`, not the singleton `{i}`.
- **N5** §2.2(6): dense-row threshold attributed to the AMD package user guide default
  (`AMD_DENSE = 10`), not the 1996 paper.
- **N6** §3.3: column-count sketch explicitly deferred to the full `cs_counts`-style
  presentation in Davis's book; sketch marked not-for-direct-implementation.
- **N7** §6: split solves (`solve_L!`/`solve_D!`/`solve_Lt!`) explicitly required for
  `SimplicialLDLFactor` as well (M2 deliverable), needed for post-update/downdate
  iterative refinement.
- **N9** §10 (M4): `SparseMatrixCSC` extraction of `F.L`/`F.U`/`F.p` added to the
  stdlib-parity checklist.

Analysis-time pass (symbolic analysis measured against faer's `SymbolicLlt` on the 2D
100×100 and 3D 30×30×30 Laplacians):

- **S1** §2.2(3): the supervariable hash table is capped at 4096 buckets, with the
  paper's hash value kept per node as the comparison key, and only buckets holding two
  or more variables are scanned.
- **S2** §2.2(7): AMD's node arrays use the narrowest integer type that holds the
  workspace addresses (`Int32` below ~7·10⁸ pattern entries); the same algorithm and the
  same permutation in both arms.
- **S3** §3.1: the permuted pattern is formed once, in its final labeling, as the
  below-diagonal pattern that column counts and row structure read; the etree is read
  off the full pattern through the ordering. Pattern columns are no longer sorted —
  no consumer depends on their order.
- **S4** §3.2: the postorder visits children in ascending column order; the
  preliminary column-count pass that ordered siblings by count is gone (nnz(L)
  unchanged by construction; nsuper 1016 → 1013 on the 2D and 4201 → 4148 on the 3D
  Laplacian).
- **S5** §3.4/§3.5: supernode partition arrays are allocated at their exact size.

Large-PDE pass (symbolic analysis measured against `CHOLMOD.symbolic` on the 2D 300×300
and 3D 30³/45³ Laplacians, galen core 8, Chairmarks medians; before → after):
lap2d 300² 38.4 → 32.4 ms (CHOLMOD 27.2), lap3d 30³ 19.9 → 18.1 ms (CHOLMOD 18.7),
lap3d 45³ 81.6 → 72.7 ms (CHOLMOD 65.5). AMD alone is at parity with `cholmod_l_amd`
measured directly (15.9 vs 16.2, 12.6 vs 13.2, 46.4 vs 46.5 ms) and is 50–65% of the
total; the rest of the gap is the per-call allocation (37–46 MB of transient arrays,
which CHOLMOD keeps in its persistent workspace) and, on the 3D cases, the etree and
row-structure stages.

- **S6** §3.1: the pattern is built from the strictly-lower entries of `A` only (the
  numeric phase reads nothing else), every stage allocates exact sizes, and the
  permuted below-diagonal pattern carries each entry's slot in `A`, so the assembly
  map (§4.2) is written straight from it instead of bucketing `A`'s entries again
  (lap2d 300²: 3.4 → 1.2 ms, 10.1 → 4.3 MB).
- **S7** §3.5: the amalgamation fixpoint evaluates only the supernodes whose outcome
  can have changed, with the same result as sweeping all of them (10 and 14 sweeps on
  the 2D and 3D Laplacians: 1.4 → 1.6 and 2.4 → 1.5 ms, no sorting, no pass count).
- **S8** §3.6: supernodal row structure is climbed from row subtrees in ascending row
  order, sized exactly from the column counts, so no row list is sorted and no marker
  array over the columns is needed (lap3d 45³: 6.5 → 3.8 ms).
- **S9** §2.2: AMD's node status is encoded in the arrays its scans read anyway, the
  list rewrite needs no copy of `A_i`, the supervariable key is masked rather than
  reduced modulo `n − 1`, and the hash-bucket list is bounded by the bucket count.
  Output permutations are unchanged; the gain is 3–5%. Workspace compaction walks the
  live slices in address order (variables in node order, then elements in elimination
  order) instead of sorting them: 3% on the 3D Laplacians, which compact 58–152 times.
- **S10** §3.1: when `A` carries 64-bit indices and `n`, `nnz(A)` and `nnz(L)` fit in
  `Int32`, the analysis runs on `Int32` arrays and widens only the stored `Symbolic`
  fields: 37 → 31 MB per call on lap2d 300², 46 → 40 MB on lap3d 45³, results
  identical (the same code runs on either index type).

After this pass (galen core 8, medians): lap2d 300² 32.0 ms (CHOLMOD 27.1), lap3d 30³
18.0 ms (CHOLMOD 18.6), lap3d 45³ 71.0 ms (CHOLMOD 65.2). With the collector disabled
or a 16 GB heap hint the medians do not improve, and the non-collection time of a
20-call loop is 29.5 / 18.6 / 70.1 ms, so the remaining gap is compute, not
allocation: AMD at parity with `cholmod_l_amd`, and on the 3D case the given-ordering
pipeline (21.5 ms against CHOLMOD's 17.9 with a user permutation).

Simplicial path (§4.5):

- **P1** `symbolic` chooses between the supernodal path and a column-by-column path
  over the exact pattern of `L` from the work-per-nonzero predictor `flops / nnzL`,
  with a measured threshold (`simplicial_ratio`, §1.4) and a `method` override. The
  choice is a `Bool` in `Symbolic`; the factor types are unchanged, so `cholesky(A)`
  and `ldlt(A)` stay type-stable. The simplicial numeric phase is up-looking, the
  solves are two scatter sweeps over `L` held by columns and by rows, and the
  rank-1 update/downdate conversion accepts either mode.

Unchanged (verified sound by review): AMD core mechanics (§2.2 points 1–2, 4–6 as
corrected), etree/symbolic pipeline, the §4.3 left-looking `relmap` schedule
(hand-traced), every PureBLAS kernel mapping in §4.3/§5.1 (signatures and β=0 overwrite
semantics verified against PureBLAS source), update/downdate approach, GPU design,
milestones, tooling.

---

## §1 Overview and architecture

### 1.1 Goals and non-goals

**Goals.** A pure-Julia, statically-compilable (juliac/trimming-compatible),
allocation-free-after-setup sparse symmetric solver: supernodal LLᵀ for SPD, supernodal
LDLᵀ (1×1 pivots, signed regularization) for symmetric quasi-definite (SQD) systems,
simplicial LDLᵀ with rank-1 update/downdate, fill-reducing AMD ordering. Dense kernels
exclusively from PureBLAS.jl. Competitive with CHOLMOD+OpenBLAS in wall-time on the
target matrix classes (interior-point KKT systems, FEM stiffness matrices, graph
Laplacians).

**Non-goals (v1.0).** No 2×2 Bunch–Kaufman pivoting (hence no general indefinite systems
— SQD only, see §5). No nested dissection ordering (accept `GivenOrdering` from METIS.jl
if users want it). No complex element types initially (`Float64`/`Float32`; the code is
generic over `T<:Real` from day one). No distributed memory.

**Design stance.** Left-looking supernodal factorization (papers: Ng–Peyton 1993;
Rothberg–Gupta 1993). Left-looking beats right-looking/multifrontal here because it needs
one bounded update workspace instead of a frontal-matrix stack, which suits the
allocation-free-after-symbolic requirement.

### 1.2 Core types

```julia
struct Symbolic{Ti<:Integer}
    n::Int
    perm::Vector{Ti}              # fill-reducing permutation
    iperm::Vector{Ti}
    parent::Vector{Ti}            # elimination tree (postordered)
    colcount::Vector{Ti}          # nnz per column of L
    # --- supernode partition ---
    nsuper::Int
    super::Vector{Ti}             # super[s] = first column of supernode s (length nsuper+1)
    rowind_ptr::Vector{Ti}        # CSC-style pointers into rowind
    rowind::Vector{Ti}            # concatenated row patterns, one block per supernode
    snode_of::Vector{Ti}          # column -> supernode
    sparent::Vector{Ti}           # supernode elimination tree
    # --- workspace sizing, derived from the §4.3 schedule ---
    # max_update_size: the left-looking loop materializes, for each
    # (descendant d, ancestor s) pair, an update block C of dimension
    # |R| x |R1| (R = rows of d at/below first(s); R1 = those inside s's
    # column range). This field is max over all such pairs of |R|*|R1|,
    # computed by scanning each supernode's rowind against the supernode
    # partition during symbolic analysis. It sizes the single preallocated
    # update buffer; no numeric-phase allocation is ever needed.
    max_update_size::Int
    # max_extend_rows: max over supernodes of the row count strictly below
    # the diagonal block (|R2|). Sizes off-diagonal panel staging (GPU/batched
    # paths, §8) and the trsm panel views.
    max_extend_rows::Int
    nnzL::Int
    flops::Float64
end

mutable struct SupernodalFactor{T,Ti} <: AbstractSparseFactor{T}
    sym::Symbolic{Ti}
    px::Vector{Ti}                # per-supernode offsets into x
    x::Vector{T}                  # dense column-major storage, one block per supernode
    ws::Workspace{T,Ti}           # update buffer (max_update_size), relmap, head/next lists
    stats::FactorStats
    ok::Bool
end

mutable struct LDLFactor{T,Ti} <: AbstractSparseFactor{T}
    # same layout as SupernodalFactor; unit-lower L blocks + separate D
    sym::Symbolic{Ti}
    px::Vector{Ti}
    x::Vector{T}
    d::Vector{T}                  # diagonal of D (1x1 pivots only)
    signs::Vector{Int8}           # expected pivot signs (+1/-1), permuted
    ws::Workspace{T,Ti}
    stats::FactorStats
    ok::Bool
end

mutable struct FactorStats
    nnzL::Int
    flops::Float64
    # inertia: pivot-sign counts observed BEFORE any regularization
    # perturbation, accumulated for free in the base-case column loop (§5.1).
    n_pos::Int
    n_neg::Int
    n_zero::Int
    n_perturbed::Int              # pivots forced by signed regularization
    max_perturbation::Float64
    rcond_est::Float64            # cheap min|d|/max|d| estimate
end
```

`SimplicialLDLFactor{T,Ti}` (M2): plain CSC-pattern L (columns individually addressable),
`d::Vector{T}`, plus the etree — the representation Davis–Hager update/downdate operates
on. Produced by `simplicial(F::LDLFactor)` (one-time conversion, allocates) from a factor
in either mode.

`Symbolic` further carries `simplicial::Bool` and, in simplicial mode, the exact pattern
of `L` by columns (`lcolptr`/`lrowval`) and by rows (`lrowptr`/`lcolval`) plus the
permuted upper triangle of `A` (`ucolptr`/`urowval`); the two factor types carry the
matching values `lx`/`ltx`. See §4.5.

### 1.3 Module layout

```
src/
  PureSparse.jl        # module, exports, Preferences load
  ordering/amd.jl      # §2.2
  ordering/interface.jl# AbstractOrdering, GivenOrdering, NaturalOrdering
  symbolic/etree.jl    # §3.2
  symbolic/counts.jl   # §3.3
  symbolic/supernodes.jl # §3.4–3.6
  symbolic/pattern.jl  # §4.5, exact pattern of L by columns and rows
  numeric/llt.jl       # §4
  numeric/ldlt.jl      # §5
  numeric/solve.jl     # §4.4, split solves
  simplicial/numeric.jl # §4.5, up-looking LLᵀ/LDLᵀ and scatter solves
  simplicial/updown.jl # §7
  contracts.jl         # TypeContracts declarations (compile-time only)
  strict.jl            # StrictMode-gated runtime checks
ext/
  PureSparseCUDAExt.jl # §8 (M3)
```

### 1.4 Tunables

All via Preferences.jl (compile-time constants under juliac, no runtime dispatch cost),
each with a keyword-argument override on `symbolic`/`cholesky`:

| Preference | Default | Meaning |
|---|---|---|
| `amalg_cols` | `(8, 32, 128)` | merged-width tiers for relaxed amalgamation (§3.5) |
| `amalg_zmax` | `(0.9, 0.15, 0.03)` | max zero-fraction allowed per tier (§3.5) |
| `amd_dense_mult` | `10.0` | dense-row multiplier (§2.2 pt 6) |
| `ldlt_delta` | `1e-12` (rel.) | signed-regularization floor (§5.1) |
| `simplicial_ratio` | measured, see `tuning.jl` | `flops / nnzL` at or below which `symbolic` takes the simplicial path (§4.5) |
| `gpu_flop_threshold` | `2e9` | per-supernode flop count above which GPU path engages (§8) |

**Provenance note (B2):** `amalg_cols`/`amalg_zmax` are free heuristics with *no* claimed
paper provenance. The tier structure follows from the tradeoff stated in §3.5 (relaxed
amalgamation as a concept is from Ashcraft–Grimes 1989 / Ng–Peyton 1993; those papers do
not prescribe numeric thresholds). The specific numbers are our own first-principles
starting point (derivation in §3.5) and are expected to move during M1's calibration
pass. They were chosen from the cache/flop math below, not from any existing
implementation's tuned values.

---

## §2 Fill-reducing ordering

### 2.1 Interface

```julia
abstract type AbstractOrdering end
struct AMDOrdering <: AbstractOrdering; dense_mult::Float64; aggressive::Bool; end
struct GivenOrdering{Ti} <: AbstractOrdering; perm::Vector{Ti}; end
struct NaturalOrdering <: AbstractOrdering end
```

`symbolic(A; ordering=AMDOrdering())`. `GivenOrdering` is the escape hatch for
METIS/ND permutations and the mechanism for the same-permutation benchmark gate (§9.3).

### 2.2 AMD — approximate minimum degree

Source: Amestoy, Davis, Duff, *An Approximate Minimum Degree Ordering Algorithm*, SIAM J.
Matrix Anal. Appl. 17(4), 1996 (paper only; §11 policy). Implemented on the quotient-graph
representation (variables + elements sharing one workspace, garbage-compacted in place).

1. **Quotient graph.** After eliminating pivot p, p becomes an *element* whose adjacency
   `L_p` is the union of its variable neighbors and its neighboring elements' patterns;
   neighboring elements absorbed into p (element absorption). Storage never exceeds the
   original graph plus O(n); periodic in-place compaction when the free tail is exhausted.

2. **Approximate degree.** For variable supervariable `𝐢` adjacent to freshly formed
   element p, the exact external degree is replaced by the upper bound

   `d̄_𝐢 = min( n − k,  d_𝐢^prev + |L_p \ 𝐢|,  |A_𝐢 \ 𝐢| + |L_p \ 𝐢| + Σ_{e ∈ E_𝐢, e≠p} |L_e \ L_p| )`

   Note the set differences subtract the **whole current supervariable `𝐢`**, not just
   the singleton `{i}` — subtracting only `{i}` inflates degrees when supervariables have
   grown. The `|L_e \ L_p|` terms are computed in one scan per pivot using the
   external-degree workspace `w[]` trick from the paper (set `w[e]` while scanning `L_p`,
   then per-variable pass reads it off).

3. **Supervariable detection (and mass elimination, which falls out of it).** After
   forming `L_p`, key each variable in it by the sum of its adjacency-list entries (the
   paper §5 reduces that sum modulo n−1; see below), bucket by key, and pairwise-compare within
   buckets; indistinguishable variables (identical adjacency once both lie in `L_p`)
   are merged into one supervariable. The key stored per node is the list sum masked
   to 30 bits rather than reduced modulo `n − 1` as in the paper: it only has to agree
   for identical lists, and the remainder cost an integer division per variable of
   `L_p`. The key is compared before the lists are; the bucket table has
   `min(nextpow(2, n), 4096)` entries (16 KB of `Int32`, L1-resident) and the key is
   folded into it by masking, since only the `|L_p|` variables hashed in one pivot
   share the table. Only buckets that receive a second variable are scanned, each
   listed once (on the 2D Laplacian 44k of 46k hashed variables sit alone in their
   bucket); each survivor in `L_p` then clears its own bucket. Mass elimination is **not a separate test**: variables that
   become indistinguishable from the pivot's supervariable are merged into it and hence
   eliminated together with the pivot, as a consequence of this detection. Do not
   implement a standalone "mass elimination" pass.

4. **Aggressive absorption.** While scanning to compute `|L_e \ L_p|`: if it comes out 0
   (i.e. `L_e ⊆ L_p`), absorb element e into p even though e is not adjacent to the pivot
   in the usual sense. Toggleable via `AMDOrdering(aggressive=false)` for A/B testing;
   default on.

5. **Tie-breaking and postorder.** Minimum approximate degree with the paper's
   degree-list bucket structure (O(1) extract-min over buckets). After elimination order
   is fixed, build the etree and postorder it (§3.2); AMD's output permutation is
   composed with the postorder so that supernode detection sees contiguous children.

6. **Dense rows.** Rows/columns with degree exceeding `max(16, amd_dense_mult · √n)` are
   stripped before the main loop and appended (ordered last). Attribution: this is the
   AMD *package's* documented user-guide default (`AMD_DENSE = 10` as the multiplier),
   not part of the 1996 paper's algorithm text; the paper's algorithm has no dense-row
   special case. `16` floor is ours (avoids stripping anything in tiny problems).

7. **Node storage.** Node ids, list entries, degrees and the shared list workspace are
   stored in the narrowest integer type that can address the workspace: `Int32` when
   `3·nnz + 3n + 2 ≤ typemax(Int32)` (the workspace never grows past `2·nnz + 3n + 1`),
   `Int64` otherwise, chosen once by a two-arm static dispatch so each arm is a concrete
   specialization of the same code. Hash sums are accumulated in `Int` before the
   modulus. The `w[]` generation offset advances by the largest `|L_e|` seen plus one,
   not by `n + 1`, so it does not overflow `Int32` on large problems; the mark and `w[]`
   arrays are cleared only when the next step would overflow their element type.
   Measured on the 2D Laplacian (n = 10⁴), the narrow arm saves 1 MB of the 2.2 MB the
   ordering allocates and about 3% of its time: the algorithm is bound by the number of
   list entries it touches (≈4.5·10⁵ for 7515 pivots), not by cache footprint.
   A node's status is read off the arrays the list scans touch anyway: a variable's
   `nv` is positive while live, negated while it is a member of the current `L_p`, and
   zero once merged; an absorbed element has `ptr == 0` and `w == −1`. This encoding,
   and the in-place rewrite of a variable's lists with the pivot's slot reserved and
   the kept entries written one step behind the read, follow faer's AMD (a port of
   SuiteSparse AMD, §11). Measured on galen it is worth 3–5%; the scans are bound by
   short-loop branch behavior rather than by the number of arrays they read.

Complexity: O(nnz · α) expected in practice; worst cases bounded by the compaction
discipline. Output validated in tests against AMD.jl's permutation *quality* (nnz(L)
within a small factor — not equality; tie-breaking differs) and against exact-minimum-
degree on tiny graphs by brute force.

---

## §3 Symbolic analysis

Pipeline: full pattern of `A+Aᵀ` → ordering → etree (through the ordering, without
forming the permuted pattern) → postorder → permuted below-diagonal pattern → column
counts → fundamental supernodes → relaxed amalgamation → supernodal row patterns and
workspace bounds. All O(nnz + n·α) except pattern union O(nnz).

### 3.1 Pattern setup

Mirror the strictly-lower entries of `A` (`i > j`; the numeric phase reads nothing
else, so any stored upper-triangle entry is ignored throughout) into full CSC form, no
diagonal — the ordering's input. Duplicates are removed with a per-column stamp, not a
sort, and only when some column of `A` is not strictly increasing; no pattern column is
sorted anywhere in the pipeline: the ordering, the etree and the column-count passes
read columns as sets, and `rowind` (§3.6) is produced in order. The permuted pattern
is materialized once, after the postorder, as the below-diagonal pattern in final
labels (column `c` = rows `> c`) built directly from `A`'s lower triangle, each entry
carrying its slot in `A`: column counts (§3.3) read it, its transpose feeds the row
structure (§3.6), and the slots make the assembly map (§4.2) a single write per entry.
Every array is allocated at its final size; buffers a stage no longer needs (the full
pattern after the etree, the postorder arrays after the relabeling) are handed to the
later stages as scratch. When `A`'s index type is wider than `Int32` and `n`,
`nnz(A)` and `nnz(L)` fit, all of this runs on `Int32` arrays (the same generic
code, chosen by a static two-arm dispatch like AMD's); only the arrays stored in the
`Symbolic`, plus the panel offsets and assembly map that address the numeric buffer,
are built on `A`'s index type. A factor whose `nnz(L)` does not fit is analyzed again
on `A`'s index type. On lap2d 300² the pattern setup went from 4.3 ms / 7.2 MB to
1.3 ms / 3.6 MB and the whole analysis from 52 MB to 31 MB allocated per call.

### 3.2 Elimination tree

Liu's algorithm (Liu 1986, *A compact row storage scheme for Cholesky factors using
elimination trees*; survey in Liu 1990, *The role of elimination trees in sparse
factorization*):
single pass over the strict-upper entries of the permuted matrix, read off the full
pattern through `perm`/`iperm` (column `k` of `PAPᵀ` is column `perm[k]` of A; entries
with `iperm[i] < k` are its strict-upper part), with a path-compressed `ancestor[]`
array. Then compute a postorder (children-first DFS via first-child/next-sibling arrays
built from `parent`, children visited in ascending column order) and relabel `perm` and
`parent` by it. Postordering is load-bearing: fundamental supernodes require
consecutive columns. The sibling order is not derived from column counts: doing so
needs a preliminary column-count pass (0.3 ms and 1.5 MB on the 2D Laplacian, 1.7 ms on
the 3D) and, measured on both, yields slightly *more* supernodes after amalgamation
(1016 vs 1013; 4201 vs 4148) at identical nnz(L).

### 3.3 Column counts

`colcount[j] = |{i ≥ j : L[i,j] ≠ 0}|` computed in O(nnz·α(nnz,n)) without forming L, via
the skeleton-graph leaf-counting algorithm (Gilbert–Ng–Peyton 1994).

**Implementation directive (N6):** the sketch here is intentionally *not* self-contained
and must not be coded from directly. Implement from the full `cs_counts`-style
presentation in Davis, *Direct Methods for Sparse Linear Systems* (SIAM, 2006), ch. 4 —
it specifies the auxiliary arrays the sketch omits: the first-descendant array
(`first[]`), `maxfirst[]` initialization, previous-leaf (`prevleaf[]`) and
previous-neighbor tracking, and the path-halving disjoint-set `ancestor[]` structure used
to find least common ancestors. Getting any of these subtly wrong yields counts that are
*sometimes* right — the property tests in §9.1 compare against counts extracted from an
explicitly-computed L on the full zoo, not just toy cases.

The same pass accumulates `nnzL` and the exact factorization flop count (`Σ colcount[j]²`
for LLᵀ), stored in `Symbolic.flops` and used by §8's GPU threshold and §9.3's GFlops
diagnostic.

### 3.4 Fundamental supernodes

Columns j and j+1 belong to the same fundamental supernode iff **all three** hold:

1. `parent[j] == j+1` (j's etree parent is the next column),
2. `colcount[j] == colcount[j+1] + 1` (patterns nest exactly), and
3. **j+1 has exactly one etree child** (`childcount[j+1] == 1`).

Condition 3 is required by the Liu–Ng–Peyton definition of *fundamental* supernodes (Liu,
Ng, Peyton 1993); conditions 1–2 alone can merge a column whose parent has other
children, which still yields a valid superset partition but is not fundamental and
changes update-list granularity. `childcount[]` is one O(n) pass over `parent`.

Independent of the predicate, the following is a first-class tested invariant (§9.1),
asserted on every zoo matrix: **for every supernode s and every column j ∈ s, `rowind[s]`
restricted to rows ≥ j is a superset of column j's true L-pattern.** The numeric loop
(§3.6/§4.3) is correct for any partition satisfying this; the invariant is what actually
protects correctness under both fundamental detection and the amalgamation below.

### 3.5 Relaxed amalgamation

Concept: Ashcraft–Grimes 1989; Ng–Peyton 1993 (merge small supernodes with their parent
even when patterns don't nest, padding with explicit zeros, to fatten dense blocks for
BLAS-3). **Those papers motivate the transformation but prescribe no thresholds; the
thresholds below are our own free tunables (see §1.4 provenance note).**

Bottom-up FIXPOINT over the supernodal etree (see the `relaxed_amalgamation`
docstring in `src/symbolic/supernodes.jl` for the full algorithm and the exact
union-height row estimate): the result is that of re-sweeping all supernodes in
ascending order until no merge happens, but only the supernodes whose outcome can
have changed are re-evaluated — after a merge of s into t, those are t itself and the
block ending just before s's — tracked in a bit set, so the cost is one sweep plus the
merges (the sweep count on the 2D and 3D Laplacians is 10 and 14). Merge child c into
parent s if the zero-fraction of the merged block,

`z = 1 − (true nnz of merged columns) / (merged block cells)`,

is at or below the tier limit for the merged width `nc`:

| merged width `nc` | max z | rationale |
|---|---|---|
| ≤ `amalg_cols[1] = 16` | `amalg_zmax[1] = 0.97` | Blocks in this range are at most 2× PureBLAS's Float64 microkernel register tile (~8 columns); per-supernode fixed costs (kernel dispatch, relmap scatter setup, update-list traversal) dominate regardless of density, so merge nearly always. |
| ≤ `amalg_cols[2] = 64` | `amalg_zmax[2] = 0.35` | Update flops scale ~quadratically in width; a zero-fraction z inflates flops by roughly 1/(1−z)² on the padded block. |
| ≤ `amalg_cols[3] = 128` | `amalg_zmax[3] = 0.08` | Wide panels already run near peak; padding is increasingly pure loss, and panel growth starts pressuring the update-buffer's cache residency, so only near-nesting qualifies. |
| > 128 | no merge | — |

**Empirically recalibrated** from the original
starting-point numbers (`(8,32,128)`/`(0.9,0.15,0.03)`, picked from flop-inflation/L2
arithmetic alone before any real measurement was possible) against the M1 wall-time gate
(§9.3) once the row-count estimate feeding the z-test became exact instead of a proxy —
see `src/tuning.jl`'s `AMALG_COLS`/`AMALG_ZMAX` derivation comment for the sweep that
produced these values and the measured gate-pass-count before/after. They still carry no
external provenance and no correctness weight — any thresholds satisfy the §3.4 superset
invariant — and remain free tunables, not hardware- or paper-derived constants.

### 3.6 Supernodal structure and workspace bounds

For each (post-amalgamation) supernode s with columns `super[s]:super[s+1]-1`,
`rowind[s]` = the sorted union of the member columns' L-patterns = the diagonal-block
rows followed by the below-diagonal rows, which equal `struct(L[:, last column of s])`
(every member's etree path leaves the block through its last column, §3.5). Its size is
therefore `ncols(s) + colcount[last column] − 1`, known from §3.3, and the lists are
filled by Liu's row subtrees at supernode granularity: row `i` is climbed from the
supernodes of its upper-triangle entries along `sparent` until a supernode already
stamped with `i` or `i`'s own supernode. Rows are visited in ascending order, so every
list comes out sorted, and the stamp array is indexed by supernode. `sparent[s]` =
supernode containing `parent[last column of s]`. On lap3d 45³ the row structure takes
3.8 ms.

**Workspace bounds (independent derivation — B1).** These fall directly out of the §4.3
schedule; they are computed here so the numeric phase never allocates:

- For each supernode d and each ancestor supernode s that d updates, let `R = {r ∈
  rowind(d) : r ≥ first(s)}` and `R₁ = R ∩ columns(s)`. The update block C materialized
  in §4.3 is `|R| × |R₁|`. `max_update_size = max |R|·|R₁|` over all (d, s) pairs.
  Computed by walking each supernode's rowind once against the supernode partition (each
  rowind entry crosses each ancestor boundary once; total cost O(nnz of rowind arrays)).
- `max_extend_rows = max over s of (length(rowind(s)) − ncols(s))` — the tallest
  below-diagonal panel; sizes trsm panel views and GPU staging (§8).

---

## §4 Numeric supernodal LLᵀ

### 4.1 Data layout

Each supernode is one dense column-major block of dimension `length(rowind(s)) ×
ncols(s)`, stored contiguously in `F.x` at offset `px[s]`. Rows ordered as in
`rowind(s)`: diagonal block first (lower triangle meaningful), below-diagonal panel
after. Explicit zeros from amalgamation live in the block and are factored as ordinary
values.

### 4.2 Assembly

`cholesky!(F, A)` scatters the permuted lower triangle of A into the blocks (zeroing
first), using a per-column binary search over `rowind(s)` — or, faster, a precomputed
assembly map (`Vector{Ti}` of destination offsets, one per nnz of tril(A), built once in
`symbolic` since the pattern is fixed). We use the precomputed map: O(nnz) assembly, no
searches, and it is what makes repeated `cholesky!` on new values allocation-free and
cheap.

### 4.3 Left-looking update loop (relmap scheduling)

Verified by review via hand-trace; carried forward unchanged.

State: `head[nsuper]` / `next[nsuper]` intrusive linked lists (descendants pending for
each supernode), `dptr[nsuper]` (per-descendant progress pointer into its rowind),
`relmap::Vector{Ti}` of length n, update buffer `C` of `max_update_size` elements.

For each supernode s in ascending order:

1. **relmap fill:** for local index k in `rowind(s)`, set `relmap[rowind(s)[k]] = k`. *No
   clearing between supernodes:* every descendant row r that gets looked up satisfies `r
   ∈ rowind(d), r ≥ first(s)`, and by the etree/pattern containment property such rows
   are a subset of `rowind(s)` — exactly the invariant asserted in §9.1. Stale entries
   for rows outside `rowind(s)` are never read.
2. **Apply pending updates:** for each descendant d popped from `head[s]`: let `q =
   dptr[d]` point at the first row of `rowind(d)` that is ≥ `first(s)`; split the tail
   rows into R₁ (within s's columns) and R₂ (below). With `L₁ = L_d[R₁, :]`, `L₂ =
   L_d[R₂, :]` (views into d's block):
   - `syrk!('L', 'N', -one(T), L₁, zero(T), C₁)` — diagonal part of C,
   - `gemm!('N', 'T', -one(T), L₂, L₁, zero(T), C₂)` — off-diagonal part,
   both into disjoint views of the workspace C. β=0 semantics: PureBLAS `syrk!`/`gemm!`
   *overwrite* C when β==0 (verified against PureBLAS source) — no NaN hazard from
   uninitialized workspace, and no workspace clearing needed.
   - **Scatter-add** C into s's block: destination row/col local indices via
     `relmap[rowind(d)[·]]`. Plain `+=`; the additions land inside s's diagonal block
     and panel only.
   - Advance `dptr[d]` past R₁; if rows remain, compute d's next ancestor supernode `s' =
     snode_of[rowind(d)[dptr[d]]]` and push d onto `head[s']`.
3. **Factor diagonal block:** dense in-place Cholesky of the `nc × nc` diagonal block via
   PureBLAS `potrf!('L', D)`. A nonpositive pivot sets `F.ok = false`, records the
   failing column in stats, and returns (caller decides; `\` throws
   `PosDefException`-equivalent).
4. **Panel solve:** `trsm!('R', 'L', 'T', 'N', one(T), D, P)` on the below-diagonal panel
   P.
5. If s has below-diagonal rows, set `dptr[s]` past the diagonal block, compute s's first
   ancestor, push s onto its head list.

All kernel calls are on `view`s of preallocated storage; zero allocations after
`symbolic` (gated in §9.1).

### 4.4 Triangular solves

`solve!(x, F, b)`: permute, `solve_L!`, (`solve_D!` for LDLᵀ), `solve_Lt!`, unpermute.
Forward solve walks supernodes ascending: dense `trsv!`/`trsm!` on the diagonal block
(PureBLAS), then `gemv!`/`gemm!` of the panel into a gather/scatter workspace indexed by
`rowind(s)`. Backward solve is the mirror, descending. Multi-RHS (`B::Matrix`) uses the
gemm forms. Split solves are exported (§6) — needed by iterative refinement, by SQD
consumers wanting `L`-only preconditioning, and by §7.

### 4.5 Simplicial path and the automatic choice

A factor whose `L` carries little dense work per stored entry gains nothing from
supernodes: amalgamation pads it with explicit zeros and every solve pays a
dense-kernel call per supernode, each on a block a few columns wide. Such factors
(narrow banded matrices, small grids) are factored and solved column by column over
the exact pattern of `L`.
The CHOLMOD *User Guide* documents that CHOLMOD likewise switches between its
simplicial and supernodal factorizations on a work-per-nonzero ratio; that fact is the
only thing taken from it (§11) — the predictor's form here follows from the column
counts PureSparse already has, and the threshold is PureSparse's own measurement.

**Switch criterion.** `symbolic` computes `nnzL = Σⱼ colcount[j]` and `flops = Σⱼ
colcount[j]²` from the column counts (§3.3) before any supernode work. Their quotient
`flops / nnzL` is the mean number of multiply-adds spent per stored entry of `L`, i.e.
how much dense work each nonzero is amortized over; it is independent of `n` and of
the ordering's quality. The simplicial path is taken when

`flops ≤ simplicial_ratio · nnzL`

(written without a division so an empty matrix is well defined), with
`simplicial_ratio` a Preferences-backed tunable (§1.4). `method = :simplicial` or
`:supernodal` on `symbolic`/`cholesky`/`ldlt` overrides the choice; `:auto` is the
default. The GPU extension always asks for `:supernodal` — its engine is built on the
supernode partition.

**Threshold.** The value of `simplicial_ratio` is the crossover measured with
`benchmark/simplicial_threshold.jl`: both paths are forced on a set of SPD matrices
whose predictor spans roughly 3 to 500 (banded matrices of growing bandwidth, 2D and
3D grid Laplacians of growing size, random sparse SPD matrices), and the warm
refactorization, the warm single-vector solve and the analysis are timed for each. The
switch is set on refactor+solve, the quantity the target users pay per iteration
(§1.2: hundreds of refactorizations per analysis). On the Zen 5c core the runs were
pinned to (four runs, shared machine), the simplicial path wins refactor+solve for
every matrix with a predictor of at most 9, 2D grids keep winning to about 18, the
first losses are banded matrices at 13 and 17 (their long chain supernodes suit the
supernodal path), and from 25 on every matrix loses. The refactorization is what
crosses over early: the up-looking factorization reads every column of `L` once per
nonzero of its row structure, so its memory traffic is `Σ colcount²`, which the
supernodal blocking reuses from cache instead; at the 2D 100×100 Laplacian (predictor
55) it is 1.85–2.03× slower, which no solve gain repays. The default is the integer
above the last predictor that wins in every run and below the first observed loss:
**10**. The 2D 100×100 Laplacian therefore stays supernodal; the simplicial solve
alone would beat the supernodal one there, but the supernodal solve's small-block
`trsv` kernel (PureBLAS) is the right place to close that gap. A solve-dominated
workload can raise the preference (the simplicial solve alone stays ahead on 2D grids
to a predictor of about 70) or pass `method = :simplicial`. The measured table is in
`src/tuning.jl`.

**Symbolic analysis in simplicial mode.** After the ordering, the etree, the postorder
and the column counts (§3.1–§3.3) — which both modes share — the simplicial mode
computes, instead of §3.4–§3.6:

- `ucolptr`/`urowval`: the permuted upper triangle of `A` with its diagonal, as a CSC
  pattern in factor order (column `k` holds the rows `i ≤ k` of `P·A·Pᵀ`), together
  with the assembly map `amap` from every stored entry of `A` to its slot — the §4.2
  replay aimed at a value buffer `ux` instead of the panels.
- `lcolptr`/`lrowval`: the exact pattern of `L` by columns, each column's diagonal
  first and its rows ascending; and `lrowptr`/`lcolval`: the same pattern by rows,
  each row's diagonal last, the other columns in an order in which every column
  precedes its etree ancestors. Both come from one pass over the rows: the structure of
  row `k` of `L` is the row subtree of the etree (Liu 1990, §3–§4; Davis 2006, ch. 4),
  climbed from every row index of column `k` of the upper triangle until a node already
  reached for row `k`. Appending `k` to each reached column in row order sorts the
  columns; the climbed paths, each a descendant-to-ancestor chain, are stacked with the
  latest path first, which is the stored row order and a topological order of the
  etree. The supernode fields are left at `nsuper = 0` with length-1 pointer arrays.

The analysis therefore never forms supernodes, and its cost is the shared pipeline plus
`O(nnz(A) + nnz(L))` for the two patterns.

**Factor storage.** `SupernodalFactor` and `LDLFactor` hold, in simplicial mode, the
values of `L` twice: `lx` in the by-column pattern and `ltx` in the by-row pattern,
both of length `nnz(L)`, with the diagonal stored (its value for LLᵀ, `1` for LDLᵀ) and
`d` separate for LDLᵀ as in §5. `Workspace` adds the value buffer `ux`, a per-column
write cursor, and `invdiag`, the reciprocal of each pivot (`1/L[j,j]`, or `1/d[j]`),
so that the per-entry divisions of the factorization and the solves are
multiplications; the dense scatter vector is the workspace's `rhs`. The panel fields
are empty. Nothing is allocated after `symbolic` (requirement 5).

**Numeric factorization: up-looking** (Davis 2006, ch. 4; the LDLᵀ form as in QDLDL,
Stellato et al.). Row `k` of `L` is one sparse triangular solve, `L[k, 1:k-1]ᵀ =
L[1:k-1, 1:k-1]⁻¹ a_k` with `a_k` the upper triangle of column `k` of `P·A·Pᵀ`,
followed by `L[k,k]² = a_kk − ‖L[k, 1:k-1]‖²` (LDLᵀ: `d_k = a_kk − Σᵢ L[k,i]² dᵢ`). The
solve runs over the row structure in the stored topological order, scattering column
`i`'s already-computed entries (rows `< k`, a prefix of the column tracked by the
cursor) into the dense vector, which is all-zero between rows. Each `L[k,i]` is written
to its by-column slot (appended, so columns fill in row order) and its by-row slot. The
LDLᵀ pivot goes through the same inertia accounting and signed regularization as the
supernodal base case (§5.1, shared code), and a non-positive LLᵀ pivot sets `F.ok =
false` with the exact failing column in `stats.fail_col`. Total work is `Σⱼ
colcount[j]²/2` multiply-adds, as for the supernodal path. This is sparse per-entry
arithmetic, not dense block work, so it does not go through PureBLAS (CONTRIBUTING.md
requirement 9 concerns dense kernels).

**Solves.** Both triangular sweeps are scatter loops: the forward sweep down the
columns of `L` (`lx`), the backward sweep along its rows (`ltx`, i.e. the columns of
`Lᵀ`). Holding `L` by rows is what makes the backward sweep a scatter; as a per-column
dot product over the by-column storage its accumulator serializes the floating-point
adds, measured at 1.5–2× the scatter form. `solve_D!` is unchanged. The split solves,
`\`, `refine!`, `sparse_L`, `det`/`logdet` and `simplicial(F::LDLFactor)` all work on
either mode (`simplicial` copies the exact pattern instead of the padded supernodal
one).

**Type stability.** `cholesky(A)` returns `SupernodalFactor{T,Ti}` and `ldlt(A)`
returns `LDLFactor{T,Ti}` whatever the choice: the mode is a `Bool` field of
`Symbolic`, read at run time by `cholesky!`/`ldlt!`/`solve!`, which branch once per
call into the supernodal or the simplicial kernels. The alternatives were rejected
because they cannot make an automatic choice type-stable: a type parameter fixed by
the `Symbolic` would make `symbolic(A)`, and hence `cholesky(A)`, return one of two
types depending on the data; separate entry points would push the choice onto the
caller. One branch per call is free, the unused path's fields are empty vectors, and
every downstream consumer (drop-in, GPU, update/downdate) keeps dispatching on the two
factor types it already knows.

---

## §5 LDLᵀ and quasi-definite systems

### 5.1 Supernodal LDLᵀ with signed diagonal regularization

Target: symmetric quasi-definite (SQD) KKT systems `[H Aᵀ; A −C]` (Vanderbei 1995: SQD
matrices are strongly factorizable — LDLᵀ with 1×1 pivots exists under any symmetric
permutation). Same supernodal structure and update loop as §4.3 with three changes:

- Updates carry D: the syrk/gemm calls use `L·D` scaled copies (one extra buffer of
  `max_update_size` column-scaled values; `dimm`-style column scaling is a trivial loop,
  no new PureBLAS kernel needed).
- Diagonal-block factorization is a base-case unit-LDLᵀ column loop (right-looking within
  the block, dense, PureBLAS `ger!`/`gemv!` for the rank-1 trailing updates — this is the
  one place we hand-roll a small dense factorization rather than call `potrf!`).
- **Signed regularization, QDLDL/Clarabel-style** (this is *not* the MA57 model — MA57
  does threshold Bunch–Kaufman 1×1/2×2 pivoting and reports exact inertia; we do fixed
  pivot order with forced signs, which is the QDLDL (Stellato et al., OSQP) / Clarabel
  approach): the caller supplies expected pivot signs (`signs::Vector{Int8}`, or the
  convenience constructor `ldlt(A; n_pos, n_neg)` for block-structured KKT). In the
  base-case column loop, before using pivot `d_j`:
  1. **Inertia accounting (free):** classify the *pre-perturbation* `d_j` as
     positive/negative/zero (`|d_j| ≤ ζ·maxabs(D so far)`) and bump `stats.n_pos/n_neg/n_zero`.
     This is two comparisons per column inside a loop we already run.
  2. If `sign(d_j) ≠ signs[j]` or `|d_j| < δ_j` (with `δ_j = ldlt_delta · ‖A‖-scale`), set
     `d_j ← signs[j] · max(δ_j, |d_j|)`, bump `n_perturbed`, track `max_perturbation`.

PureSparse never throws on SQD input: regularization guarantees the factorization
completes with the requested inertia. `F.ok` stays true; the perturbation record in
`FactorStats` tells the caller how much forcing occurred.

### 5.2 API and downstream consumers

`ldlt(A; signs)`, `ldlt!(F, A)`, `solve!`, split solves. Iterative refinement helper
`refine!(x, F, A, b; iters=2)` recommended whenever `n_perturbed > 0`.

**Downstream regularization loops:** an IPOPT-style consumer that wants MA57-like
behavior — factor, inspect inertia, add `δI`/`−δI` and refactor until inertia is correct
— can implement that loop *on top of* PureSparse by reading `(n_pos, n_neg, n_zero)` from
`F.stats` after each `ldlt!` (with `ldlt_delta = 0` to disable internal forcing, or with
it enabled as a safety net). PureSparse itself does not run such a loop; it reports and
continues.

---

## §6 User-facing API

```julia
S = symbolic(A; ordering=AMDOrdering(), method=:auto)   # analysis, allocates; method ∈ :auto/:simplicial/:supernodal (§4.5)
F = cholesky(A; ordering, method)            # symbolic + numeric
cholesky!(F, A2)                             # refactor, same pattern, zero alloc
F = ldlt(A; signs, method) ; ldlt!(F, A2)
x = F \ b ; solve!(x, F, b) ; ldiv!(x, F, b)
solve_L!(x, F, b); solve_D!(x, F, b); solve_Lt!(x, F, b)
G = simplicial(F::LDLFactor)                 # -> SimplicialLDLFactor (M2)
updowndate!(G, w, ±1)                        # rank-1 (M2, §7)
```

Split solves (`solve_L!`/`solve_D!`/`solve_Lt!`) are exported for **all three** factor
types: `SupernodalFactor`, `LDLFactor` (§4.4/§5.2), **and `SimplicialLDLFactor`** — the
simplicial versions are required for iterative refinement after update/downdate and are
an explicit M2 deliverable, not an afterthought (they are simple CSC column loops, no
PureBLAS needed).

`FactorStats` is public API (inertia, perturbation record, nnzL, flops, rcond estimate).

---

## §7 Rank-1 update/downdate (simplicial)

Davis–Hager, *Modifying a Sparse Cholesky Factorization* (SIAM J. Matrix Anal. Appl.
1999) and the multiple-rank follow-up (2001) — papers only. Operates on
`SimplicialLDLFactor`: for `A ± wwᵀ`, the modified columns are exactly those on the etree
path from `min(support(w))` to the root; the algorithm updates L's columns and d along
that path in O(changed nnz) with the hyperbolic/stable recurrences from the paper.
Pattern growth on update is handled by symbolic elbow room: `simplicial()` allocates each
column with slack (grow-factor Preference), and a pattern overflow triggers a documented
refactor-required return code rather than reallocation (allocation-free contract
preserved; caller refactors). Multiple-rank = sequenced single-rank in v1 (the 2001
batched variant is a listed extension, not scheduled). Downdate can destroy
positive-definiteness — detected via the recurrence and reported through `F.ok`/stats,
same discipline as §4.3 step 3.

---

## §8 GPU offload (design only until M3)

Extension package (`PureSparseCUDAExt`) — core stays dependency-free and trimmable.
Left-looking loop unchanged; for supernodes whose update flop count (from
`Symbolic.flops` per-node breakdown) exceeds `gpu_flop_threshold`, steps 2–4 of §4.3 run
on device: descendant panels staged through pinned buffers sized by `max_extend_rows ×
max panel width`, syrk/gemm/trsm/potrf via the device BLAS, scatter-add as a trivial
kernel using an on-device copy of relmap. Small supernodes stay on CPU (PureBLAS); the
split point is the standard hybrid design (motivated by the published GPU-supernodal
literature, e.g. Rennich et al. 2016 — concept-level only). Nothing in M1/M2 depends on
this section.

---

## §9 Verification and benchmarking

### 9.1 Test strategy

ReTestItems.jl, parallel workers. Layers:

1. **Compile-time interface contracts — TypeContracts.jl.** `contracts.jl` declares the
   public method surface and inferred return types (e.g. `cholesky!(::SupernodalFactor{T,Ti},
   ::SparseMatrixCSC{T,Ti}) -> SupernodalFactor{T,Ti}`). These are *precompile-time*
   assertions, eliminated by the trimmer, with **no runtime cost and no runtime failure
   mode** — a violation is a precompile error. The corresponding test item asserts that
   the package *precompiles* with contracts on, and (negative test) that a deliberately
   broken contract in a test-only shadow module fails precompilation. TypeContracts is
   **not** a runtime precondition mechanism and the tests do not pretend it is.
2. **Runtime pre/postconditions — StrictMode.jl.** `strict.jl` holds runtime checks
   (argument dimensions, pattern-match between `F.sym` and `A`, aliasing of `x`/`b`,
   `issorted(rowind)` post-symbolic, the §3.4 superset invariant, finite-values
   post-factor), every one gated behind `if StrictMode.checks_enabled()` so they cost
   nothing — no branches taken, **no allocations** — in production/trimmed builds. A test
   item runs the suite with checks enabled and asserts each violated precondition throws
   loudly.
3. **Invariant tests (first-class, per D5):** on every zoo matrix, compute the exact
   per-column L-pattern via a reference simplicial symbolic pass and assert `rowind[s] ⊇`
   every member column's pattern, for both fundamental-only and amalgamated partitions.
4. **Oracles:** (a) dense `BigFloat` Cholesky/LDLᵀ of the permuted matrix, elementwise
   comparison of L and d on small/medium matrices; (b) residual gates
   `‖Ax−b‖/(‖A‖‖x‖)` on the full zoo; (c) CHOLMOD *results* via SparseArrays as a numeric
   cross-check (observing another implementation's output is black-box and
   clean-room-safe, §11 — its *source* is never read).
5. **Property/fuzz:** random SPD (Laplacian + shift), random SQD KKT with known inertia
   (assert `FactorStats` inertia matches construction), random permutations through
   `GivenOrdering`, update/downdate round-trips (`update then downdate ≈ original`, and
   vs. refactorization).
6. **Matrix zoo:** curated SuiteSparse Collection subset, downloaded once into a
   Scratch.jl scratchspace. **Concurrency requirement (ReTestItems runs parallel test
   processes):** the downloader must take a lock file (`mkpidlock`/`FileWatching.Pidfile`)
   around first-run population and write via temp-file-then-atomic-`rename` —
   otherwise concurrent first runs race and can corrupt the cache. This is specified
   behavior, not an implementation nicety.
7. **Zero-allocation gate.** The in-place entry points (`cholesky!`, `ldlt!`, `qr!`,
   `solve!`) must not allocate. Two checks, both in the single checks-on test
   configuration (StrictMode's default, which StrictModeTest requires):
   - **Authoritative:** StrictModeTest's `@test_noalloc`, which proves the absence of
     allocation on every path with AllocCheck. StrictMode's own `noalloc` verdicts are
     static heuristics that only warn and do not count as this gate.
   - **Measured:** `@allocated f(...) == 0` after warmup, on the path the test executes.
   Disabling checks only removes code, so a pass here also holds for a checks-off
   deployment. The allocating constructors (`cholesky`, `ldlt`, `qr`) are not held to
   this gate.

### 9.2 Reference oracles

As in 9.1(4). BigFloat oracle tolerances: `‖L−L_ref‖ ≤ c·n·eps(T)·‖A‖^{1/2}`-style
bounds, loose enough for reordering-of-sums differences, tight enough to catch scatter
bugs (calibrated on known-good dense potrf first).

### 9.3 Benchmark matrix and performance gate

BenchmarkTools/Chairmarks medians, fixed thread count, zoo subset stratified by class
(KKT/FEM/Laplacian) and size. Configurations:

| # | Factorization | Dense kernels | Status |
|---|---|---|---|
| 1 | PureSparse | PureBLAS | primary |
| 2 | PureSparse | OpenBLAS (via LBT) | kernel-attribution arm |
| 3 | CHOLMOD (SparseArrays) | OpenBLAS (stock LBT default) | baseline |
| 4 | CHOLMOD | PureBLAS | **N/A — blocked.** PureBLAS's documented limitation: `BLAS.lbt_forward`-ing its juliac-built `.so` from inside a live Julia process aborts (signal 6) due to double initialization of libjulia; running CHOLMOD in a subprocess is still a live Julia process, so the same abort applies. This is a host-runtime-init limitation, not missing symbols. Revisit if/when PureBLAS ships a host-runtime-init story; nothing gates on it. |

Attribution logic without quadrant 4: (1 vs 3) is the headline; (1 vs 2) isolates
PureBLAS-vs-OpenBLAS inside our factorization; (2 vs 3) isolates our sparse layer vs
CHOLMOD's on equal dense kernels. That triangle is sufficient.

**Primary gate (non-negotiable, wall-time):** on each gate matrix,

- `median_seconds(PS+PB, numeric refactor) < median_seconds(CHOLMOD+OB)` — lower is
  better, identical input matrix and RHS, and
- the same inequality **also under identical permutations**: both stacks factor under
  `GivenOrdering(p)` with the same `p` (we feed CHOLMOD our AMD permutation and,
  separately, feed PureSparse CHOLMOD's chosen permutation). This same-permutation run is
  *part of the gate*, not a supplementary extra — it isolates factorization throughput
  from ordering quality and closes the loophole where a higher-fill ordering wins a
  throughput metric while losing wall-time.

Reported gate slices: symbolic+numeric (cold), numeric refactor (warm — the
IPM-relevant number), solve. Pass threshold per milestone (see `../ROADMAP.md`).

**Secondary diagnostic (not a gate):** GFlops = own-ordering flop count / median seconds,
reported for all configurations. Explicitly *not* the gate because it is gameable — a
worse ordering inflates flop count and can raise GFlops while wall-time worsens. It stays
in the report purely as a kernel-efficiency lens.

PkgBenchmark.jl reports (commit-to-commit self-regression) supplement the Chairmarks
harness, tracking the same configurations over history.

### 9.4 Test-matrix selection

**CI set** (small, fast, downloaded once — SPD from the SuiteSparse Matrix Collection):
matrices spanning n ∈ [100, 10⁴], mixed domains (structural/power/network), total
download < 20 MB, plus synthetic generators (banded, random sparse SPD, random SQD
KKT-shaped, 2D Laplacian) needing no download. Implementer verifies each named matrix
resolves and is SPD before pinning names in the harness.

**Performance set:** medium and large SPD matrices (structural, circuit, thermal, CFD
domains) up to the classic supernodal stress-test scale, plus a synthetic SQD/KKT set
built from constraint-matrix-shaped generators for the M2 gate.

---

## §10 Milestones

See [`../ROADMAP.md`](../ROADMAP.md) for the canonical, up-to-date milestone status and
task list. Summary:

- **M1** — AMD + symbolic pipeline + supernodal LLᵀ + solve (§2–§4), gated vs
  CHOLMOD+OpenBLAS.
- **M2** — SQD/LDLᵀ + rank-k update/downdate (§5, §7).
- **M3** — GPU backend, in-package CUDA weakdep extension (§8).
- **M4** — Drop-in `cholesky()`/`\` forwarding, stdlib-surface parity.

---

## §11 Clean-room provenance policy

**Prohibited, absolutely:** reading CHOLMOD/SuiteSparse source code, headers, comments,
or commit history — directly or indirectly (including via search-result snippets, LLM
recall of source text, or third-party ports derived from GPL parts of the source). Also
prohibited: reusing SuiteSparse identifier names, struct field names, or
numeric constants where recall of the implementation (rather than a paper or independent
derivation) would be the only plausible origin. Every name and constant in this document
must survive the question "where did this come from?" with an answer that is a paper
citation, a user-guide citation, or an in-document independent derivation. (B1 and B2
were violations of exactly this test and are fixed in this revision.)

**Permitted:** published papers and books; official *user guides* (interface
documentation, not implementation); black-box observation of CHOLMOD's outputs and
performance via SparseArrays; benchmarking against binaries; the source of faer (MIT),
including its AMD and COLAMD, which are ports of SuiteSparse's BSD-3 licensed AMD and
COLAMD. Code derived from faer keeps faer's MIT notice, and code derived from its AMD or
COLAMD also keeps the matching BSD-3 notice.

**Provenance table** (every algorithm, its allowed source):

| Component | Source (papers, books, user guides; faer source for the AMD encoding) |
|---|---|
| AMD | Amestoy–Davis–Duff, SIMAX 1996; AMD package *User Guide* (dense-row default, §2.2 pt 6); node-status encoding and in-place list rewrite (§2.2 pt 7): faer 0.24.4 `src/sparse/linalg/amd.rs` (MIT), a port of SuiteSparse AMD (BSD-3; both notices in `src/ordering/amd.jl`) |
| Elimination tree, postorder | Liu 1986; Davis, *Direct Methods*, ch. 4 |
| Column counts | Gilbert–Ng–Peyton 1994; Davis book ch. 4 (implementation reference, §3.3) |
| Fundamental supernodes | Liu–Ng–Peyton 1993 |
| Relaxed amalgamation | Ashcraft–Grimes 1989; Ng–Peyton 1993 (concept); thresholds: own derivation, §3.5 |
| Left-looking supernodal LLᵀ | Ng–Peyton 1993; Rothberg–Gupta 1993 |
| Simplicial path (§4.5): row structure, up-looking LLᵀ/LDLᵀ | Liu 1990 §3–§4; Davis book ch. 4; QDLDL (Stellato et al.) for the LDLᵀ recurrence; switch-on-a-ratio fact: CHOLMOD *User Guide*; threshold: own measurement |
| SQD / LDLᵀ regularization | Vanderbei 1995; Stellato et al. (OSQP/QDLDL); Clarabel paper |
| Update/downdate | Davis–Hager 1999, 2001 |
| GPU hybrid | Rennich et al. 2016 (concept) |
| Workspace bounds (§3.6) | own derivation from §4.3 schedule |

Full citations for the sources named above (papers only; the PDFs are kept in a local,
gitignored `refs/linear_algebra/` archive that is not distributed):

- P. R. Amestoy, T. A. Davis, I. S. Duff, *An Approximate Minimum Degree Ordering
  Algorithm*, SIAM J. Matrix Anal. Appl. 17(4):886–905, 1996.
- Y. Chen, T. A. Davis, W. W. Hager, S. Rajamanickam, *Algorithm 887: CHOLMOD, Supernodal
  Sparse Cholesky Factorization and Update/Downdate*, ACM TOMS 35(3), 2008
  (`cholmod_toms.pdf`; used as a paper, never its source).
- T. A. Davis, W. W. Hager, *Modifying a Sparse Cholesky Factorization*, SIAM J. Matrix
  Anal. Appl. 20(3):606–627, 1999 (`modify_sparse_cholesky.pdf`).
- T. A. Davis, *Direct Methods for Sparse Linear Systems*, SIAM, 2006.
- J. R. Gilbert, E. G. Ng, B. W. Peyton, *An Efficient Algorithm to Compute Row and
  Column Counts for Sparse Cholesky Factorization*, ORNL/TM-12195, Oak Ridge National
  Laboratory, 1992; SIAM J. Matrix Anal. Appl. 15(4):1075–1091, 1994.
- J. W. H. Liu, *A Compact Row Storage Scheme for Cholesky Factors Using Elimination
  Trees*, ACM Trans. Math. Softw., June 1986, p. 127 ff. (`Supernodal/liu1986.pdf`).
- J. W. H. Liu, *The Role of Elimination Trees in Sparse Factorization*, SIAM J. Matrix
  Anal. Appl. 11(1):134–172, 1990 (`Supernodal/liu1990.pdf`).
- J. W. H. Liu, E. G. Ng, B. W. Peyton, *On Finding Supernodes for Sparse Matrix
  Computations*, SIAM J. Matrix Anal. Appl. 14(1):242–252, 1993.

The remaining sources named in the provenance table (Ashcraft–Grimes 1989, Ng–Peyton 1993,
Rothberg–Gupta 1993, Vanderbei 1995, Stellato et al., the Clarabel paper, Rennich et al.
2016) are cited by author and year only; the archive holds local copies of several of them.
