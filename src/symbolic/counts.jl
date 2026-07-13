# Column counts (design.md §3.3): the Gilbert–Ng–Peyton skeleton-graph leaf-counting
# algorithm (Gilbert, Ng, Peyton 1994, *An efficient algorithm to compute row and column
# counts for sparse Cholesky factorization*), implemented from the full presentation in
# Davis, *Direct Methods for Sparse Linear Systems* (SIAM, 2006), ch. 4, per the design's
# N6 directive (the §3.3 sketch is not self-contained) — never CHOLMOD source
# (CLAUDE.md req 1).
#
# Theory in brief: for i > j, L[i,j] ≠ 0 iff j lies in the "row subtree" T^i — the
# etree-connected union of paths from {d : A[i,d] ≠ 0, d < i} up to i. So
# colcount[j] = 1 + #{row subtrees containing j}. Each T^i is decomposed by its skeleton
# leaves: a leaf contributes +1 at itself and every consecutive-leaf pair contributes -1
# at their least common ancestor; summing these deltas over subtree(j) then yields the
# indicator of j ∈ T^i, and one bottom-up accumulation over the (postordered) etree sums
# all subtrees at once. Skeleton-leaf detection uses the first-descendant array `first_`
# and per-row `maxfirst`; consecutive leaves per row via `prevleaf`; LCAs via a
# path-halving disjoint-set `ancestor` whose links are installed in postorder, so the
# find-root of the previous leaf is exactly the LCA with the current node.

"""
    column_counts(n, colptr, rowval, parent) -> colcount::Vector{Ti}

`colcount[j] = |{i ≥ j : L[i,j] ≠ 0}|` for the Cholesky factor of the symmetric matrix
whose STRICT UPPER triangle (postordered) is given by `colptr`/`rowval` (1-based CSC),
with elimination tree `parent` (also postordered; `parent[j]==0` marks a root). Design.md
§3.3. O(nnz·α(nnz,n)), computed without forming L.
"""
function column_counts(
        n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}, parent::Vector{Ti},
) where {Ti<:Integer}
    # Higher adjacency = transpose of the strict-upper pattern: column j of `hadj` holds
    # {i > j : A[i,j] ≠ 0} (row j of the upper triangle). The main pass needs row
    # structures in increasing j, which the upper CSC can't serve directly; one counting
    # transpose provides them (rows come out sorted ascending since columns are scanned
    # in order).
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

    # First-descendant pass. The pattern is already postordered, so node j's postorder
    # index IS j: `first_[q]` = smallest node in subtree(q). Climb from each j installing
    # `first_` until an ancestor already has one (set by an earlier, smaller subtree
    # node). A node whose `first_` is still unset when its own turn comes is an etree
    # leaf — its diagonal entry seeds `colcount[j] = 1`; non-leaves start at 0 (their
    # diagonal is accounted for by the leaf/LCA deltas plus the parent decrement below).
    first_ = zeros(Ti, n)
    colcount = Vector{Ti}(undef, n)   # holds the deltas until the final accumulation
    @inbounds for j in 1:n
        colcount[j] = (first_[j] == 0) ? one(Ti) : zero(Ti)
        q = Ti(j)
        while q != 0 && first_[q] == 0
            first_[q] = Ti(j)
            q = parent[q]
        end
    end

    # Main pass, j in postorder. `j` is a skeleton leaf of row subtree T^i iff
    # `first_[j] > maxfirst[i]` (its subtree lies strictly beyond every previously seen
    # leaf of T^i — the pruning that makes the count exact, not an overcount). First leaf
    # of T^i: +1 at j. Subsequent leaf: +1 at j, -1 at lca(prevleaf[i], j). The
    # disjoint-set `ancestor` gets the link ancestor[j] = parent[j] only AFTER j is
    # processed, so find-root from prevleaf[i] stops exactly at the lowest
    # not-yet-processed ancestor — the LCA. Every non-root j also decrements its parent's
    # delta, cancelling the diagonal's spill into proper ancestors during accumulation.
    maxfirst = zeros(Ti, n)
    prevleaf = zeros(Ti, n)
    ancestor = Vector{Ti}(undef, n)
    @inbounds for j in 1:n
        ancestor[j] = Ti(j)
    end
    @inbounds for j in 1:n
        pj = parent[j]
        if pj != 0
            colcount[pj] -= one(Ti)
        end
        for p in hptr[j]:(hptr[j + 1] - 1)
            i = hadj[p]                            # i > j by construction
            first_[j] > maxfirst[i] || continue    # not a skeleton leaf of T^i
            maxfirst[i] = first_[j]
            colcount[j] += one(Ti)
            pl = prevleaf[i]
            if pl != 0
                # q = lca(pl, j) by path halving (design §3.3 N6). Iterative, bounded —
                # no recursion (CLAUDE.md req 4).
                q = pl
                while ancestor[q] != q
                    ancestor[q] = ancestor[ancestor[q]]
                    q = ancestor[q]
                end
                colcount[q] -= one(Ti)
            end
            prevleaf[i] = j
        end
        if pj != 0
            ancestor[j] = pj
        end
    end

    # Bottom-up accumulation: colcount[j] = Σ deltas over subtree(j). Postorder
    # guarantees every child index < parent index, so one forward sweep suffices.
    @inbounds for j in 1:n
        pj = parent[j]
        if pj != 0
            colcount[pj] += colcount[j]
        end
    end
    return colcount
end

"""
    nnz_l(colcount) -> total nonzeros of L

`sum(colcount)` — `nnz(L)` including the diagonal (design §3.3).
"""
nnz_l(colcount::Vector{<:Integer}) = sum(colcount)

"""
    chol_flops(colcount) -> Σⱼ colcount[j]²

Exact LLᵀ factorization flop count (design §3.3), the GFlops-diagnostic denominator.
"""
chol_flops(colcount::Vector{<:Integer}) = sum(abs2, colcount)
