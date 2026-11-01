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
function fundamental_supernodes(
        n::Int, parent::Vector{Ti}, colcount::Vector{Ti};
        fundamental::Bool = true
    ) where {Ti <: Integer}
    # Only "exactly one child" is tested, so the count saturates at 2.
    childcount = zeros(UInt8, n)
    if fundamental
        @inbounds for j in 1:n
            p = parent[j]
            iszero(p) || (childcount[p] = min(childcount[p] + 0x01, 0x02))
        end
    end
    # Column j starts a new supernode unless it continues j-1's; two sweeps, one to count
    # and one to fill, so `super` is allocated at its exact size.
    @inline continues(j) = parent[j - 1] == j && colcount[j - 1] == colcount[j] + 1 &&
        (!fundamental || isone(childcount[j]))
    nsuper = n > 0 ? 1 : 0
    @inbounds for j in 2:n
        continues(j) || (nsuper += 1)
    end
    super = Vector{Ti}(undef, nsuper + 1)
    super[1] = one(Ti)
    s = 1
    @inbounds for j in 2:n
        continues(j) && continue
        s += 1
        super[s] = Ti(j)
    end
    super[nsuper + 1] = Ti(n + 1)
    return nsuper, super
end

"""
    supernode_tree(n, nsuper, super, parent) -> (snode_of, sparent)

`snode_of[j]` = the supernode containing column `j`. `sparent[s]` = the supernode
containing `parent[last column of s]` (`0` if `s`'s last column is an etree root).
"""
function supernode_tree(n::Int, nsuper::Int, super::Vector{Ti}, parent::Vector{Ti}) where {Ti <: Integer}
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
@inline function _amalg_tier(merged_cols::Int, amalg_cols::NTuple{3, Int})
    merged_cols <= amalg_cols[1] && return 1
    merged_cols <= amalg_cols[2] && return 2
    merged_cols <= amalg_cols[3] && return 3
    return 0
end

# Path-halving owner lookup for dead-supernode redirect: `owner[s]` points from an
# absorbed supernode to the block that absorbed it (`owner[s] == s` while alive). Merges
# only ever point a smaller index at a larger one (etree postorder ⇒ sparent > child), so
# chains are strictly ascending and halving terminates.
@inline function _find_owner!(owner::Vector{Ti}, s::Ti) where {Ti <: Integer}
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
pass count is the maximum sibling-absorption depth (10 and 14 on the 2D 300² and 3D 45³
Laplacians). A pass after the first evaluates only the supernodes whose outcome can
have changed since their last evaluation, which gives the same result as a full sweep
at the cost of the merges alone (the argument is in the body). Absorbed supernodes
redirect to their absorbing block through a path-halved `owner` array, so later passes
find the current alive merge target in near-O(1).

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
naive fixpoint with the proxy inflated padded cells to ~7× nnzL).

`true_nnz` over a merged block is a contiguous colcount sum, taken O(1) from a prefix
sum. The zero-fraction convention (`z = 1 − true_nnz / (height·width)`, rectangle cells
vs lower-triangular nonzeros) is unchanged from the single-pass version.
"""
function relaxed_amalgamation(
        n::Int, nsuper::Int, super::Vector{Ti}, parent::Vector{Ti}, colcount::Vector{Ti};
        amalg_cols::NTuple{3, Int} = AMALG_COLS, amalg_zmax::NTuple{3, Float64} = AMALG_ZMAX,
    ) where {Ti <: Integer}
    # Every block spans whole fundamental supernodes: `sfirst[s]` is the fundamental
    # supernode whose first column starts block s, so `super[sfirst[s]]` is the block's
    # first column and `ssum` (below) gives its true nnz in O(1). A block's last column
    # is `super[s + 1] - 1` throughout: merges only extend a block downward.
    sfirst = collect(Ti, 1:nsuper)
    _, sparent = supernode_tree(n, nsuper, super, parent)
    alive = trues(nsuper)
    owner = collect(Ti, 1:nsuper)

    # ssum[s] = Σ colcount[1:super[s]-1].
    ssum = Vector{Int}(undef, nsuper + 1)
    ssum[1] = 0
    @inbounds for s in 1:nsuper
        acc = ssum[s]
        for j in super[s]:(super[s + 1] - 1)
            acc += Int(colcount[j])
        end
        ssum[s + 1] = acc
    end

    # A pass evaluates its candidates in ascending order; the first pass evaluates every
    # supernode. Afterwards only a supernode whose evaluation can have changed is a
    # candidate, which gives exactly the result of re-sweeping all supernodes until no
    # merge happens: only a block's contiguous predecessor can merge into it, so a merge
    # of s into t changes the outcome for just two supernodes. One is t itself, which is
    # evaluated right away: nothing evaluated between positions s and t can touch t's
    # block or its target, so the result is the one a sweep reaching t would get. The
    # other is the block ending just before s's, fundamental supernode `sfirst[s] - 1`,
    # the new contiguous predecessor, evaluated in the next pass since it precedes s.
    # Candidates live in one bit set scanned upward; a bit set during a pass always lies
    # behind the scan, so it is picked up by the next pass.
    queued = trues(nsuper)
    nsuper > 0 && (queued[nsuper] = false)
    s = findfirst(queued)
    @inbounds while !isnothing(s)
        queued[s] = false
        s0 = s
        while alive[s] && !iszero(sparent[s])   # a root's last column has no target
            t = _find_owner!(owner, sparent[s]) # alive block containing s's parent column
            t <= s && break                    # malformed (shouldn't happen: postorder ⇒ ancestors > s)
            sfirst[t] == s + 1 || break        # not column-contiguous: not a mergeable pair

            endt = Int(super[t + 1]) - 1
            merged_cols = endt - Int(super[sfirst[s]]) + 1
            tier = _amalg_tier(merged_cols, amalg_cols)
            iszero(tier) && break

            height = merged_cols + Int(colcount[endt]) - 1  # exact (see docstring)
            true_nnz = ssum[t + 1] - ssum[sfirst[s]]
            z = 1.0 - true_nnz / (height * merged_cols)
            z <= amalg_zmax[tier] || break

            c = sfirst[s] - one(Ti)
            c >= 1 && (queued[c] = true)
            sfirst[t] = sfirst[s]
            alive[s] = false
            owner[s] = t
            s = t
        end
        s = findnext(queued, s0 + 1)
        isnothing(s) && (s = findfirst(queued))   # next pass
    end

    super2 = Vector{Ti}(undef, nsuper + 1)
    nsuper2 = 0
    @inbounds for s in 1:nsuper
        alive[s] || continue
        nsuper2 += 1
        super2[nsuper2] = super[sfirst[s]]
    end
    super2[nsuper2 + 1] = Ti(n + 1)
    resize!(super2, nsuper2 + 1)
    return nsuper2, super2
end

"""
    supernode_rowind(n, colptr, rowval, parent, colcount, nsuper, super)
        -> (rowind_ptr, rowind, snode_of, sparent, max_update_size, max_extend_rows)

Row structure of L per (post-amalgamation) supernode (design.md §3.6): `rowind[s]` =
the columns of `s` followed, in ascending order, by `struct(L[:, last column of s])`
below the block — the union of its member columns' patterns, since every member's
etree path leaves the block only through its last column (the one range root of a
block, see [`relaxed_amalgamation`](@ref)). `colptr`/`rowval` is the postordered STRICT
UPPER pattern (as `etree`/`column_counts` consume), `colcount` its column counts, which
size `rowind` exactly: `|rowind[s]| = ncols(s) + colcount[last column of s] − 1`.
Also computes the two independent workspace-size derivations from the §4.3 update
schedule (design §0 B1 — NOT CHOLMOD's `maxcsize`/`maxesize`): `max_update_size` =
`max over (descendant d, ancestor s) pairs of |R|·|R₁|`, and `max_extend_rows` = the
tallest below-diagonal panel.

Algorithm: Liu's row subtrees (Liu 1990, *The role of elimination trees in sparse
factorization*, sections 3-4) at supernode granularity. Row `i` of `L` is nonzero in
the columns climbed from every row index `j` of column `i` of the upper pattern toward
`i`; each etree path leaves a supernode through its last column, so the climb runs over
the supernodal tree `sparent`, stopping at a supernode already stamped with `i` (the
rest of its path is stamped too) or at `i`'s own supernode. Rows are processed in
ascending order, so each supernode's list comes out sorted, and the stamp array is
indexed by supernode, not column.
"""
function supernode_rowind(
        n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}, parent::Vector{Ti}, colcount::Vector{Ti},
        nsuper::Int, super::Vector{Ti},
    ) where {Ti <: Integer}
    snode_of, sparent = supernode_tree(n, nsuper, super, parent)
    rowind_ptr = Vector{Ti}(undef, nsuper + 1)
    rowind_ptr[1] = one(Ti)
    @inbounds for s in 1:nsuper
        rowind_ptr[s + 1] = rowind_ptr[s] + (super[s + 1] - super[s]) + colcount[super[s + 1] - 1] - one(Ti)
    end
    rowind = Vector{Ti}(undef, Int(rowind_ptr[nsuper + 1]) - 1)
    cursor = Vector{Ti}(undef, nsuper)    # next free slot of each supernode's list
    @inbounds for s in 1:nsuper
        q = rowind_ptr[s]
        for c in super[s]:(super[s + 1] - one(Ti))
            rowind[q] = c
            q += one(Ti)
        end
        cursor[s] = q
    end
    stamp = zeros(Ti, nsuper)             # the last row whose climb passed through s
    @inbounds for i in 1:n
        si = snode_of[i]
        for p in colptr[i]:(colptr[i + 1] - 1)
            t = snode_of[rowval[p]]
            while t != si && stamp[t] != Ti(i)
                # Every climb ends at i's own supernode when parent and pattern agree;
                # reaching the root first means they do not, so stop before indexing 0.
                iszero(t) && throw(ArgumentError(lazy"supernode_rowind: row $i of column $(rowval[p]) climbs to the root before reaching its own supernode; parent and pattern disagree"))
                stamp[t] = Ti(i)
                rowind[cursor[t]] = Ti(i)
                cursor[t] += one(Ti)
                t = sparent[t]
            end
        end
    end
    # Every list must have been filled exactly: the column counts and the climbed
    # structure describe the same L.
    @inbounds for s in 1:nsuper
        cursor[s] == rowind_ptr[s + 1] ||
            throw(ErrorException(lazy"supernode_rowind: supernode $s has $(cursor[s] - rowind_ptr[s]) rows, column counts say $(rowind_ptr[s + 1] - rowind_ptr[s])"))
    end

    max_update_size = 0
    max_extend_rows = 0
    @inbounds for s in 1:nsuper
        rows = view(rowind, Int(rowind_ptr[s]):(Int(rowind_ptr[s + 1]) - 1))
        nrow_s = length(rows)
        ncol_s = Int(super[s + 1] - super[s])
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
