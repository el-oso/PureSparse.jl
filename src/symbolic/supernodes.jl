# Fundamental supernode detection (design.md §3.4) and relaxed amalgamation (§3.5).
# Grounded in Liu–Ng–Peyton 1993 (fundamental supernodes) and Ashcraft–Grimes 1989 /
# Ng–Peyton 1993 (relaxed amalgamation as a concept — the numeric thresholds here are our
# own free tunables, design §0 B2/§1.4, no external provenance).

"""
    fundamental_supernodes(n, parent, colcount) -> (nsuper, super)

Partition the POSTORDERED columns `1:n` into fundamental supernodes. `super` has length
`nsuper+1`; supernode `s` owns columns `super[s]:super[s+1]-1`. Columns j, j+1 belong to
the same fundamental supernode iff `parent[j]==j+1`, `colcount[j]==colcount[j+1]+1`, AND
`j+1` has exactly one etree child (design §3.4 condition 3 — required for a genuinely
*fundamental* partition per Liu–Ng–Peyton; the first two conditions alone give a valid
superset partition but not a fundamental one).
"""
function fundamental_supernodes(n::Int, parent::Vector{Ti}, colcount::Vector{Ti}) where {Ti<:Integer}
    childcount = zeros(Ti, n)
    @inbounds for j in 1:n
        p = parent[j]
        p != 0 && (childcount[p] += one(Ti))
    end
    super = Ti[1]
    j = 1
    @inbounds while j <= n
        while j < n && parent[j] == j + 1 && colcount[j] == colcount[j + 1] + 1 && childcount[j + 1] == 1
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

"""
    relaxed_amalgamation(n, nsuper, super, parent, colcount;
                          amalg_cols=AMALG_COLS, amalg_zmax=AMALG_ZMAX)
        -> (nsuper2, super2)

Bottom-up pass (design §3.5) over the fundamental supernodal etree: merge supernode `s`
into its parent `t = sparent[s]` — padding with explicit zeros — when `s`'s columns
CONTIGUOUSLY precede `t`'s (the only case a dense-panel merge can represent; per etree
postorder, this holds exactly when `s` is `t`'s last-visited child) and the merged
block's zero-fraction is within the tier limit for its width (tuning.jl). Processes
supernodes in ascending (postorder) order so a supernode's own absorptions (always from
smaller indices — etree children always postorder before their parent) are complete
before it is itself considered for merging into its parent; a chain of merges is grown by
sequentially extending the surviving parent's column range leftward, no separate
union-find needed since a merge only ever redirects a smaller index into a larger one.

Row-count estimate: `s` always contiguously precedes `t`, so `start[s] < start[t]` and
`s`'s own top column becomes the merged block's new topmost column — `colcount[start[s]]`
is the row-count proxy used (colcount is non-increasing down an etree path, so the
topmost column bounds the taller columns further down).
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

    @inbounds for s in 1:(nsuper - 1)
        t = sparent[s]
        (t == 0 || t <= s) && continue     # root, or malformed (shouldn't happen: etree postorder ⇒ t > s)
        endc[s] + 1 == start[t] || continue  # not column-contiguous: not a mergeable pair (design §3.5)

        ns = Int(endc[s] - start[s]) + 1
        nt = Int(endc[t] - start[t]) + 1
        merged_cols = ns + nt
        tier = _amalg_tier(merged_cols, amalg_cols)
        tier == 0 && continue

        # start[s] < start[t] always (s contiguously precedes t), so s's own top column
        # becomes the new merged block's topmost column — use ITS colcount, not t's old
        # (pre-merge) top. Using colcount[start[t]] here undercounts the required row
        # height whenever colcount[start[s]] > colcount[start[t]] (the common case, since
        # colcount is non-increasing along an etree path), silently accepting merges that
        # violate their own zmax bound.
        rows_est = Int(colcount[start[s]])
        true_nnz = 0
        for j in start[s]:endc[s]
            true_nnz += Int(colcount[j])
        end
        for j in start[t]:endc[t]
            true_nnz += Int(colcount[j])
        end
        merged_cells = rows_est * merged_cols
        z = merged_cells == 0 ? 0.0 : 1.0 - true_nnz / merged_cells

        if z <= amalg_zmax[tier]
            start[t] = start[s]
            alive[s] = false
        end
    end

    super2 = Ti[]
    @inbounds for s in 1:nsuper
        alive[s] && push!(super2, start[s])
    end
    push!(super2, Ti(n + 1))
    return length(super2) - 1, super2
end
