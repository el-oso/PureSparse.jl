# Fundamental supernode detection (design.md §3.4) and relaxed amalgamation (§3.5).
# Grounded in Liu–Ng–Peyton 1993 (fundamental supernodes) and Ashcraft–Grimes 1989 /
# Ng–Peyton 1993 (relaxed amalgamation as a concept — the numeric thresholds here are our
# own free tunables, design §0 B2/§1.4, no external provenance).

"""
    fundamental_supernodes(n, parent, colcount; fundamental=true) -> (nsuper, super)

Partition the POSTORDERED columns `1:n` into supernodes. `super` has length `nsuper+1`;
supernode `s` owns columns `super[s]:super[s+1]-1`. Columns j, j+1 belong to the same
supernode iff `parent[j]==j+1`, `colcount[j]==colcount[j+1]+1`, AND (when
`fundamental=true`, the default) `j+1` has exactly one etree child (design §3.4
condition 3 — required for a genuinely *fundamental* partition per Liu–Ng–Peyton; the
first two conditions alone give a valid superset partition but not a fundamental one).

`fundamental=false` (design_qr_m5b.md §A2.2, M5b's own default for its frontal
partition — Cholesky/LDLᵀ callers keep `fundamental=true` unchanged) drops the
only-child condition: SPQR paper §2.3's two-condition supernode test ("for these two
columns to reside in the same *fundamental* supernode, j must also be the only child of
j+1; SuiteSparseQR does not use this restriction"). The two remaining conditions still
guarantee identical column patterns (paper, same sentence), so every pattern-dependent
consumer of `super` (staircase logic, row structure) is unaffected by the flag.
"""
function fundamental_supernodes(n::Int, parent::Vector{Ti}, colcount::Vector{Ti};
        fundamental::Bool = true) where {Ti<:Integer}
    childcount = zeros(Ti, n)
    if fundamental
        @inbounds for j in 1:n
            p = parent[j]
            p != 0 && (childcount[p] += one(Ti))
        end
    end
    super = Ti[1]
    j = 1
    @inbounds while j <= n
        while j < n && parent[j] == j + 1 && colcount[j] == colcount[j + 1] + 1 &&
                (!fundamental || childcount[j + 1] == 1)
            j += 1
        end
        j += 1
        push!(super, Ti(j))
    end
    return length(super) - 1, super
end

"""
    supernode_tree(n, nsuper, super, parent) -> (snode_of, sparent)

`snode_of[j]` = the supernode containing column `j`. `sparent[s]` = the supernode
containing `parent[last column of s]` (`0` if `s`'s last column is an etree root).
"""
function supernode_tree(n::Int, nsuper::Int, super::Vector{Ti}, parent::Vector{Ti}) where {Ti<:Integer}
    snode_of = Vector{Ti}(undef, n)
    @inbounds for s in 1:nsuper, j in super[s]:(super[s + 1] - 1)
        snode_of[j] = Ti(s)
    end
    sparent = Vector{Ti}(undef, nsuper)
    @inbounds for s in 1:nsuper
        p = parent[super[s + 1] - 1]
        sparent[s] = p == 0 ? zero(Ti) : snode_of[p]
    end
    return snode_of, sparent
end

# Tier lookup for the merged-width thresholds in tuning.jl. Returns 0 (never merge) if
# the merged width exceeds every tier.
@inline function _amalg_tier(merged_cols::Int, amalg_cols::NTuple{3,Int})
    merged_cols <= amalg_cols[1] && return 1
    merged_cols <= amalg_cols[2] && return 2
    merged_cols <= amalg_cols[3] && return 3
    return 0
end

# Path-halving owner lookup for dead-supernode redirect: `owner[s]` points from an
# absorbed supernode to the block that absorbed it (`owner[s] == s` while alive). Merges
# only ever point a smaller index at a larger one (etree postorder ⇒ sparent > child), so
# chains are strictly ascending and halving terminates.
@inline function _find_owner(owner::Vector{Ti}, s::Ti) where {Ti<:Integer}
    @inbounds while owner[s] != s
        owner[s] = owner[owner[s]]
        s = owner[s]
    end
    return s
end

"""
    relaxed_amalgamation(n, nsuper, super, parent, colcount;
                          amalg_cols=AMALG_COLS, amalg_zmax=AMALG_ZMAX)
        -> (nsuper2, super2)

Bottom-up fixpoint (design §3.5) over the fundamental supernodal etree: merge supernode
`s` into the alive block `t` containing `parent[endc[s]]` — padding with explicit zeros —
when `s`'s columns CONTIGUOUSLY precede `t`'s (the only case a dense-panel merge can
represent) and the merged block's zero-fraction is within the tier limit for its width
(tuning.jl). Ascending (postorder) passes are repeated until a pass performs no merge:
one ascending pass can only merge each parent's single currently-contiguous child (the
column-contiguity test admits exactly one child at a time, and by the time an accepted
merge extends the parent's range leftward to expose the next sibling, that sibling has
already been passed over), so bushy etrees — 2D grid Laplacians under AMD, most etree
nodes with 2+ children — need one extra pass per additional absorbable sibling.
Termination: every pass either merges nothing (loop exits) or strictly decreases the
alive-block count (bounded below by 1), so at most `nsuper` passes run; in practice the
pass count is the maximum sibling-absorption depth (single digits on the gate set).
Absorbed supernodes redirect to their absorbing block through a path-halved `owner`
array, so later passes find the current alive merge target in near-O(1).

Row count of the merged block — EXACT, not a proxy (independent derivation from the
etree column-inclusion property, `struct(L[:,j]) ∖ {j} ⊆ struct(L[:,parent(j)])`, Liu's
etree theory, design §3.2):

Every block this process forms has exactly ONE "range root" — a column whose etree
parent lies outside the block's column interval — namely its last column `endc`, by
induction: a fundamental supernode is a parent chain (only its last column's parent
leaves the interval), and a merge of child block `[a1,b1]` into target `[a2,b2]`
requires `parent[b1] ∈ [a2,b2]` (that is what `sparent`-with-redirect targets), so `b1`
stops being a range root and the merged block's only root is `b2`. Consequently the
etree path upward from any column of the block exits the interval only through `endc`,
and by repeated column inclusion every below-diagonal row of every member column lands
in `struct(L[:,endc])` — while `endc`'s own pattern is trivially a subset of the block's
rows. The block's below-diagonal row set is therefore exactly
`struct(L[:,endc]) ∖ {endc}` and its stored height is

    height = ncols + colcount[endc] − 1

with no pattern arrays or unions needed. (This also matches what `supernode_rowind`
computes for the final partition, so panel sizes agree with the merge-time estimate.)
The previous single-pass proxy `colcount[start[s]]` equals this only while patterns nest
exactly; under cascaded sibling absorption it underestimates the union height and lets
the zero-fraction gate silently accept over-fat merges (measured on laplacian2d 80×80:
naive fixpoint with the proxy inflated padded cells to ~7× nnzL — see ROADMAP task 7b').

`true_nnz` over a merged block is a contiguous colcount sum, taken O(1) from a prefix
sum. The zero-fraction convention (`z = 1 − true_nnz / (height·width)`, rectangle cells
vs lower-triangular nonzeros) is unchanged from the single-pass version.
"""
function relaxed_amalgamation(
        n::Int, nsuper::Int, super::Vector{Ti}, parent::Vector{Ti}, colcount::Vector{Ti};
        amalg_cols::NTuple{3,Int} = AMALG_COLS, amalg_zmax::NTuple{3,Float64} = AMALG_ZMAX,
) where {Ti<:Integer}
    _, sparent = supernode_tree(n, nsuper, super, parent)
    start = Vector{Ti}(undef, nsuper)
    endc = Vector{Ti}(undef, nsuper)
    @inbounds for s in 1:nsuper
        start[s] = super[s]
        endc[s] = super[s + 1] - one(Ti)
    end
    alive = trues(nsuper)
    owner = collect(Ti, 1:nsuper)

    # ccsum[j+1] = Σ colcount[1:j]: O(1) true-nnz for any contiguous column range.
    ccsum = Vector{Int}(undef, n + 1)
    ccsum[1] = 0
    @inbounds for j in 1:n
        ccsum[j + 1] = ccsum[j] + Int(colcount[j])
    end

    merged_any = true
    @inbounds while merged_any
        merged_any = false
        for s in 1:(nsuper - 1)
            alive[s] || continue
            sparent[s] == 0 && continue        # s's last column is an etree root
            t = _find_owner(owner, sparent[s]) # alive block containing parent[endc[s]]
            t <= s && continue                 # malformed (shouldn't happen: postorder ⇒ ancestors > s)
            endc[s] + 1 == start[t] || continue  # not column-contiguous: not a mergeable pair (design §3.5)

            merged_cols = Int(endc[t] - start[s]) + 1
            tier = _amalg_tier(merged_cols, amalg_cols)
            tier == 0 && continue

            height = merged_cols + Int(colcount[endc[t]]) - 1  # exact (see docstring)
            true_nnz = ccsum[endc[t] + 1] - ccsum[start[s]]
            z = 1.0 - true_nnz / (height * merged_cols)

            if z <= amalg_zmax[tier]
                start[t] = start[s]
                alive[s] = false
                owner[s] = t
                merged_any = true
            end
        end
    end

    super2 = Ti[]
    @inbounds for s in 1:nsuper
        alive[s] && push!(super2, start[s])
    end
    push!(super2, Ti(n + 1))
    return length(super2) - 1, super2
end

"""
    supernode_rowind(n, colptr, rowval, parent, nsuper, super)
        -> (rowind_ptr, rowind, snode_of, sparent, max_update_size, max_extend_rows)

Row structure of L per (post-amalgamation) supernode (design.md §3.6):
`rowind[s]` = `columns_of_s ∪ pattern(A[:,cols(s)]) below s ∪ ⋃_{children c} (rowind[c]
minus c's columns)`. `colptr`/`rowval` is the postordered STRICT UPPER pattern (as
`etree`/`column_counts` consume). Also computes the two independent workspace-size
derivations from the §4.3 update schedule (design §0 B1 — NOT CHOLMOD's
`maxcsize`/`maxesize`): `max_update_size` = `max over (descendant d, ancestor s) pairs of
|R|·|R₁|`, and `max_extend_rows` = the tallest below-diagonal panel.

Algorithm: one ascending (postorder) pass with a per-supernode stamped marker array
(`mark`) for O(1) dedup, avoiding an O(n) clear between supernodes. For supernode `s`:
add its own columns, then the below-diagonal entries of those columns read from the
UPPER-triangle transpose (`hadj`/`hptr` — by symmetry, "row `c` has an entry in column
`i>c`" ⟺ "column `c` has an entry at row `i>c`", exactly the seed pattern needed, and the
same transpose construction `column_counts` uses), then each child's rowind with the
child's OWN columns excluded (they are always the smallest entries of a child's sorted
rowind — every propagated row is an etree-ancestor of the child's diagonal columns, and
postorder gives ancestors strictly increasing column index, so nothing from a
descendant's propagated set can be smaller than the child's own first column).
"""
function supernode_rowind(
        n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}, parent::Vector{Ti},
        nsuper::Int, super::Vector{Ti},
) where {Ti<:Integer}
    # Transpose of the strict-upper pattern (same construction as column_counts): hadj
    # column c holds {i > c : A[c,i] ≠ 0} = the original below-diagonal seed for L's
    # column c (by symmetry of the unfilled pattern).
    nz = Int(colptr[n + 1]) - 1
    cnt = zeros(Ti, n)
    @inbounds for p in 1:nz
        cnt[rowval[p]] += one(Ti)
    end
    hptr = Vector{Ti}(undef, n + 1)
    hptr[1] = one(Ti)
    @inbounds for r in 1:n
        hptr[r + 1] = hptr[r] + cnt[r]
    end
    hadj = Vector{Ti}(undef, nz)
    fill!(cnt, zero(Ti))
    @inbounds for c in 1:n
        for p in colptr[c]:(colptr[c + 1] - 1)
            r = rowval[p]
            hadj[hptr[r] + cnt[r]] = Ti(c)
            cnt[r] += one(Ti)
        end
    end

    snode_of, sparent = supernode_tree(n, nsuper, super, parent)
    child_head = zeros(Ti, nsuper)
    child_next = zeros(Ti, nsuper)
    @inbounds for s in 1:nsuper
        p = sparent[s]
        if p != 0
            child_next[s] = child_head[p]
            child_head[p] = Ti(s)
        end
    end

    mark = zeros(Int, n)
    rowind_ptr = Vector{Ti}(undef, nsuper + 1)
    rowind_ptr[1] = one(Ti)
    rowind = Ti[]
    max_update_size = 0
    max_extend_rows = 0

    @inbounds for s in 1:nsuper
        j0, j1 = super[s], super[s + 1] - one(Ti)
        rows = Ti[]
        for c in j0:j1
            if mark[c] != s
                mark[c] = s
                push!(rows, c)
            end
        end
        for c in j0:j1
            for p in hptr[c]:(hptr[c + 1] - 1)
                i = hadj[p]
                if mark[i] != s
                    mark[i] = s
                    push!(rows, i)
                end
            end
        end
        c = child_head[s]
        while c != 0
            c_last = super[c + 1] - one(Ti)   # c's own last column (fundamental-etree indexing == column index here)
            for p in rowind_ptr[c]:(rowind_ptr[c + 1] - 1)
                i = rowind[p]
                if i > c_last && mark[i] != s
                    mark[i] = s
                    push!(rows, i)
                end
            end
            c = child_next[c]
        end
        sort!(rows)
        append!(rowind, rows)
        rowind_ptr[s + 1] = Ti(length(rowind) + 1)

        nrow_s = length(rows)
        ncol_s = Int(j1 - j0) + 1
        max_extend_rows = max(max_extend_rows, nrow_s - ncol_s)

        # Workspace bound (design §3.6 / §0 B1): for THIS supernode d = s acting as a
        # descendant, walk its row list against the ancestor boundaries it crosses,
        # accumulating |R|*|R1| for each ancestor s' it updates (R = rows of d at/below
        # first(s'), R1 = the subset within s' own columns) — a single pass since rows
        # are sorted and ancestor boundaries are increasing.
        k = ncol_s + 1  # first below-diagonal row's position in `rows` (1-based)
        while k <= nrow_s
            anc = snode_of[rows[k]]
            anc_j1 = super[anc + 1] - one(Ti)
            r1 = 0
            k2 = k
            while k2 <= nrow_s && rows[k2] <= anc_j1
                r1 += 1
                k2 += 1
            end
            rtotal = nrow_s - k + 1   # |R|: all remaining rows from k to the end
            max_update_size = max(max_update_size, rtotal * r1)
            k += r1
        end
    end

    return rowind_ptr, rowind, snode_of, sparent, max_update_size, max_extend_rows
end
