# PureSparse.jl — Design Document (v2, final)

Pure-Julia supernodal sparse Cholesky/LDLᵀ solver. Clean-room reimplementation of the
CHOLMOD algorithm family (from published papers only) on top of PureBLAS.jl. This is the
canonical implementation document.

Produced by a dedicated design process: **Fable** (model `claude-fable-5`) wrote the first
comprehensive design (v1); **Opus** adversarially reviewed it against the actual AMD paper
(fetched and read), against PureBLAS's real source (signatures/semantics verified
line-by-line), and by hand-tracing the trickiest part of the algorithm (the left-looking
update schedule) — finding 2 BLOCKERs (clean-room provenance issues) and 7 DEFECTs, while
verifying the algorithmic core sound; **Fable** then produced this corrected, final
revision (v2), fixing every BLOCKER/DEFECT. See `§0` below for the full changelog.

For milestone status and the current task list, see [`../ROADMAP.md`](../ROADMAP.md).

---

## §0 Changelog from Fable review v1

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
  under a lock file; zero-allocation gate explicitly required to run in the
  StrictMode-checks-disabled configuration.
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
on. Produced by `simplicial(F::LDLFactor)` (one-time conversion, allocates) or directly by
a simplicial factorization path for small/very-sparse problems.

### 1.3 Module layout

```
src/
  PureSparse.jl        # module, exports, Preferences load
  ordering/amd.jl      # §2.2
  ordering/interface.jl# AbstractOrdering, GivenOrdering, NaturalOrdering
  symbolic/etree.jl    # §3.2
  symbolic/counts.jl   # §3.3
  symbolic/supernodes.jl # §3.4–3.6
  numeric/llt.jl       # §4
  numeric/ldlt.jl      # §5
  numeric/solve.jl     # §4.4, split solves
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
   forming `L_p`, hash each variable in it (hash = sum of its adjacency-list entries mod
   n, per the paper), bucket by hash, and pairwise-compare within buckets;
   indistinguishable variables (identical adjacency once both lie in `L_p`) are merged
   into one supervariable. Mass elimination is **not a separate test**: variables that
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

Complexity: O(nnz · α) expected in practice; worst cases bounded by the compaction
discipline. Output validated in tests against AMD.jl's permutation *quality* (nnz(L)
within a small factor — not equality; tie-breaking differs) and against exact-minimum-
degree on tiny graphs by brute force.

---

## §3 Symbolic analysis

Pipeline: pattern of `A+Aᵀ` (upper triangle, permuted) → etree → postorder → column
counts → fundamental supernodes → relaxed amalgamation → supernodal row patterns and
workspace bounds. All O(nnz + n·α) except pattern union O(nnz).

### 3.1 Pattern setup

Permute A by `perm`, symmetrize pattern (union with transpose), keep the upper triangle
in CSC (equivalently lower in CSR). One allocation pass; done once per `Symbolic`.

### 3.2 Elimination tree

Liu's algorithm (Liu 1986, *The role of elimination trees in sparse factorization*):
single pass over the upper-triangular pattern with a path-compressed `ancestor[]` array.
Then compute a postorder (children-first DFS via first-child/next-sibling arrays built
from `parent`) and relabel `perm`, `parent`, and A's pattern by it. Postordering is
load-bearing: fundamental supernodes require consecutive columns.

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

Bottom-up FIXPOINT pass over the supernodal etree (design §3.5 revision, ROADMAP task
7b'; see the `relaxed_amalgamation` docstring in `src/symbolic/supernodes.jl` for the
full algorithm and the exact union-height row estimate that replaced the original
single-pass proxy): merge child c into parent s if the zero-fraction of the merged
block,

`z = 1 − (true nnz of merged columns) / (merged block cells)`,

is at or below the tier limit for the merged width `nc`:

| merged width `nc` | max z | rationale |
|---|---|---|
| ≤ `amalg_cols[1] = 16` | `amalg_zmax[1] = 0.97` | Blocks in this range are at most 2× PureBLAS's Float64 microkernel register tile (~8 columns); per-supernode fixed costs (kernel dispatch, relmap scatter setup, update-list traversal) dominate regardless of density, so merge nearly always. |
| ≤ `amalg_cols[2] = 64` | `amalg_zmax[2] = 0.35` | Update flops scale ~quadratically in width; a zero-fraction z inflates flops by roughly 1/(1−z)² on the padded block. |
| ≤ `amalg_cols[3] = 128` | `amalg_zmax[3] = 0.08` | Wide panels already run near peak; padding is increasingly pure loss, and panel growth starts pressuring the update-buffer's cache residency, so only near-nesting qualifies. |
| > 128 | no merge | — |

**Empirically recalibrated 2026-07-13 (ROADMAP task 7b')** from the original
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
rows followed by the below-diagonal rows. Built in one pass using a length-n marker
array. `sparent[s]` = supernode containing `parent[last column of s]`.

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
S = symbolic(A; ordering=AMDOrdering())     # analysis, allocates
F = cholesky(A; ordering)                    # symbolic + numeric
cholesky!(F, A2)                             # refactor, same pattern, zero alloc
F = ldlt(A; signs) ; ldlt!(F, A2)
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
7. **Zero-allocation gate:** `@allocated cholesky!(F, A2) == 0` and same for
   `ldlt!`/`solve!` after warmup. **Test-configuration requirement:** this item must run
   in the StrictMode-checks-*disabled* configuration — the runtime checks in layer 2 may
   themselves allocate, and would otherwise fail the very gate they guard. The test
   harness runs two configurations (checks-on for layers 2–5, checks-off for this gate)
   and the CI matrix names them explicitly.

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
recall of source text, or third-party ports that are themselves derived from the
source). Also prohibited: reusing SuiteSparse identifier names, struct field names, or
numeric constants where recall of the implementation (rather than a paper or independent
derivation) would be the only plausible origin. Every name and constant in this document
must survive the question "where did this come from?" with an answer that is a paper
citation, a user-guide citation, or an in-document independent derivation. (B1 and B2
were violations of exactly this test and are fixed in this revision.)

**Permitted:** published papers and books; official *user guides* (interface
documentation, not implementation); black-box observation of CHOLMOD's outputs and
performance via SparseArrays; benchmarking against binaries.

**Provenance table** (every algorithm, its allowed source):

| Component | Source (papers/books/guides only) |
|---|---|
| AMD | Amestoy–Davis–Duff, SIMAX 1996; AMD package *User Guide* (dense-row default, §2.2 pt 6) |
| Elimination tree, postorder | Liu 1986; Davis, *Direct Methods*, ch. 4 |
| Column counts | Gilbert–Ng–Peyton 1994; Davis book ch. 4 (implementation reference, §3.3) |
| Fundamental supernodes | Liu–Ng–Peyton 1993 |
| Relaxed amalgamation | Ashcraft–Grimes 1989; Ng–Peyton 1993 (concept); thresholds: own derivation, §3.5 |
| Left-looking supernodal LLᵀ | Ng–Peyton 1993; Rothberg–Gupta 1993 |
| SQD / LDLᵀ regularization | Vanderbei 1995; Stellato et al. (OSQP/QDLDL); Clarabel paper |
| Update/downdate | Davis–Hager 1999, 2001 |
| GPU hybrid | Rennich et al. 2016 (concept) |
| Workspace bounds (§3.6) | own derivation from §4.3 schedule |

Local reference archive (`refs/linear_algebra/`, gitignored — papers only, never
distributed) contains, among general linear-algebra/FEM/Krylov material:
`cholmod_toms.pdf` (Chen–Davis–Hager–Rajamanickam 2008), `Supernodal/liu1986.pdf`,
`liu1990.pdf`, `liu1993.pdf`, `rothberg1991.pdf`, `ng1993.pdf`, `gilbert1994.pdf`,
`chapter-direct.pdf` (Davis's book), and `modify_sparse_cholesky.pdf` (Davis–Hager).
`biksm.pdf` is unidentified — implementer may check its title/abstract; nothing in this
design depends on it. The AMD paper is not in the local archive but was located at
https://people.engr.tamu.edu/davis/publications_files/An_Approximate_Minimum_Degree_Ordering_Algorithm.pdf
and the CHOLMOD TOMS paper is also mirrored at
https://people.clas.ufl.edu/hager/files/cholmod_alg.pdf.
