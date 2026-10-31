# Pattern setup (design.md §3.1) and elimination tree + postorder (design.md §3.2).
# Grounded in Liu 1986 (the elimination-tree paper cited in design.md §3.2) and the
# presentation in Davis, *Direct Methods for Sparse Linear Systems* (SIAM, 2006), ch. 4 —
# never CHOLMOD source (CONTRIBUTING.md req 1).

# Sort `v[lo:hi]` in place. Pattern columns are short (a handful of entries for the
# matrices this package targets), where an inline insertion sort beats the per-call
# overhead of `sort!` on a view; longer runs fall back to `sort!`.
@inline function _sort_segment!(v::Vector{Ti}, lo::Integer, hi::Integer) where {Ti <: Integer}
    len = hi - lo + 1
    if len > 32
        sort!(view(v, lo:hi))
        return v
    end
    for k in (lo + 1):hi
        x = v[k]
        m = k - 1
        while m >= lo && v[m] > x
            v[m + 1] = v[m]
            m -= 1
        end
        v[m + 1] = x
    end
    return v
end

"""
    symmetrized_upper(n, colptr, rowval, perm, iperm) -> (colptr2, rowval2)

Permute the structurally-symmetric pattern given by `colptr`/`rowval` (CSC, 1-based) by
`perm` (with inverse `iperm`), symmetrize (`pattern(A) ∪ pattern(Aᵀ)`), drop the
diagonal, and return the STRICT UPPER triangle in CSC form: column `j` of the result
holds the sorted rows `i < j` such that `(P A Pᵀ)[i,j] ≠ 0` (equivalently `[j,i] ≠ 0` by
symmetry). This is the input etree/colcount pass needs (design §3.1).
"""
function symmetrized_upper(
        n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}, perm::Vector{Ti}, iperm::Vector{Ti},
    ) where {Ti <: Integer}
    # Pass 1: count, per permuted column j, how many strict-upper entries land there.
    counts = zeros(Ti, n + 1)
    @inbounds for jorig in 1:n
        j = iperm[jorig]
        for p in colptr[jorig]:(colptr[jorig + 1] - 1)
            iorig = rowval[p]
            iorig == jorig && continue           # skip diagonal
            i = iperm[iorig]
            lo, hi = i < j ? (i, j) : (j, i)      # symmetrized: this nonzero contributes
            counts[hi] += one(Ti)                 # an entry at (lo, hi) with lo < hi
        end
    end
    colptr2 = Vector{Ti}(undef, n + 1)
    colptr2[1] = one(Ti)
    @inbounds for j in 1:n
        colptr2[j + 1] = colptr2[j] + counts[j]
    end
    # Pass 2: scatter. Reuse `counts` as a per-column write cursor.
    fill!(counts, zero(Ti))
    rowval2 = Vector{Ti}(undef, colptr2[n + 1] - 1)
    @inbounds for jorig in 1:n
        j = iperm[jorig]
        for p in colptr[jorig]:(colptr[jorig + 1] - 1)
            iorig = rowval[p]
            iorig == jorig && continue
            i = iperm[iorig]
            lo, hi = i < j ? (i, j) : (j, i)
            dest = colptr2[hi] + counts[hi]
            rowval2[dest] = lo
            counts[hi] += one(Ti)
        end
    end
    # Pass 3: sort each column's rows and drop duplicates in place (a nonzero may have
    # appeared from both A(i,j) and A(j,i) before symmetrization).
    write_ptr = Vector{Ti}(undef, n + 1)
    write_ptr[1] = one(Ti)
    @inbounds for j in 1:n
        lo, hi = colptr2[j], colptr2[j + 1] - 1
        _sort_segment!(rowval2, lo, hi)
        w = write_ptr[j]
        prev = zero(Ti)
        first = true
        for k in lo:hi
            v = rowval2[k]
            if first || v != prev
                rowval2[w] = v
                w += one(Ti)
                prev = v
                first = false
            end
        end
        write_ptr[j + 1] = w
    end
    # The loop above already compacted duplicates IN PLACE, column by column, writing
    # each column's deduped run starting exactly where the previous column's run ended
    # (`write_ptr[j]`) — so `rowval2[1:write_ptr[n+1]-1]` is already the fully compacted,
    # CONTIGUOUS final array; no second pass is needed (an earlier version of this
    # function re-read from the original `colptr2` offsets here, which is wrong: Pass 3
    # already overwrote those positions during compaction of later columns).
    resize!(rowval2, write_ptr[n + 1] - 1)
    return write_ptr, rowval2
end

"""
    permuted_lower(n, colptr, rowval, perm, iperm) -> (hptr, hadj)

Below-diagonal pattern of the symmetric permutation of a full, diagonal-free,
duplicate-free pattern (the output of [`full_symmetric_pattern`](@ref)): column `c` of
`hadj` holds the rows `{iperm[i] : i ∈ column perm[c]} ∩ (c, n]`, in arrival order.
Each edge is stored once per endpoint, so it lands in exactly one column and the
result has `length(rowval) ÷ 2` entries. This is the pattern [`column_counts_lower`](@ref)
and [`supernode_rowind_lower`](@ref) read (design.md §3.1).
"""
function permuted_lower(
        n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}, perm::Vector{Ti}, iperm::Vector{Ti},
    ) where {Ti <: Integer}
    hptr = Vector{Ti}(undef, n + 1)
    hadj = Vector{Ti}(undef, length(rowval) ÷ 2)
    w = 1
    hptr[1] = one(Ti)
    @inbounds for c in 1:n
        jorig = perm[c]
        for p in colptr[jorig]:(colptr[jorig + 1] - 1)
            r = iperm[rowval[p]]
            r > c || continue
            hadj[w] = r
            w += 1
        end
        hptr[c + 1] = Ti(w)
    end
    return hptr, hadj
end

"""
    etree(n, colptr, rowval) -> parent::Vector{Ti}
    etree(n, colptr, rowval, perm, iperm) -> parent::Vector{Ti}

Elimination tree of the symmetric pattern given by its STRICT UPPER triangle in CSC form
(as returned by [`symmetrized_upper`](@ref)). Liu's algorithm: one pass with a
path-compressed `ancestor` array. `parent[j]` is `0` for a root. Design.md §3.2.

The five-argument form takes a FULL symmetric pattern (both triangles, as returned by
[`full_symmetric_pattern`](@ref)) together with a permutation and returns the tree of
the permuted matrix `P A Pᵀ` without forming it: column `k` of the permuted matrix is
column `perm[k]`, and the entries with `iperm[i] < k` are its strict-upper part.
"""
function etree(n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}) where {Ti <: Integer}
    parent = zeros(Ti, n)
    ancestor = zeros(Ti, n)
    @inbounds for k in 1:n
        for p in colptr[k]:(colptr[k + 1] - 1)
            i = rowval[p]
            i >= k && continue   # strict upper triangle only holds i < k already; defensive
            _etree_link!(parent, ancestor, i, Ti(k))
        end
    end
    return parent
end

function etree(
        n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}, perm::Vector{Ti}, iperm::Vector{Ti},
    ) where {Ti <: Integer}
    parent = zeros(Ti, n)
    ancestor = zeros(Ti, n)
    @inbounds for k in 1:n
        jorig = perm[k]
        for p in colptr[jorig]:(colptr[jorig + 1] - 1)
            i = iperm[rowval[p]]
            i >= k && continue   # the other endpoint's copy of this edge handles it
            _etree_link!(parent, ancestor, i, Ti(k))
        end
    end
    return parent
end

# Liu's path-compressed climb from i: every node on i's current virtual path gets
# ancestor k, and the path's root becomes a child of k.
@inline function _etree_link!(parent::Vector{Ti}, ancestor::Vector{Ti}, i::Ti, k::Ti) where {Ti}
    @inbounds while !iszero(i) && i != k
        inext = ancestor[i]
        ancestor[i] = k
        if iszero(inext)
            parent[i] = k
            break
        end
        i = inext
    end
    return nothing
end

"""
    postorder(n, parent) -> (post::Vector{Ti}, postinv::Vector{Ti})

Postorder of the forest given by `parent`. `post[k]` is the k-th node visited;
`postinv[j]` is node `j`'s position in the postorder. Iterative (stack-based) — no
recursion whose depth depends on runtime data (CONTRIBUTING.md req 4). Each node's
children are visited in ascending column order (design.md §3.2); any sibling order
gives a valid postorder, and this one needs no data beyond `parent`.
"""
function postorder(n::Int, parent::Vector{Ti}) where {Ti <: Integer}
    # Build child lists via a head/next linked-list (bucketed by parent). Head insertion
    # reverses insertion order, so inserting in DESCENDING visit order yields per-parent
    # lists in ascending visit order, which the DFS then consumes front-to-back.
    head = zeros(Ti, n + 1)     # head[n+1] = roots (parent == 0)
    next = zeros(Ti, n)
    @inbounds for j in n:-1:1
        p = iszero(parent[j]) ? n + 1 : parent[j]
        next[j] = head[p]
        head[p] = j
    end
    post = Vector{Ti}(undef, n)
    postinv = Vector{Ti}(undef, n)
    stack = Vector{Ti}(undef, n + 1)
    sp = 0
    k = 1
    @inbounds for root in _linked_list(head, next, n + 1)
        sp += 1
        stack[sp] = root
        while sp > 0
            v = stack[sp]
            c = head[v]
            if c == 0
                # no unvisited children left: emit v, pop
                post[k] = v
                postinv[v] = k
                k += 1
                sp -= 1
            else
                head[v] = next[c]   # consume child c
                sp += 1
                stack[sp] = c
            end
        end
    end
    return post, postinv
end

# Small helper: iterate the linked list rooted at `head[start]` (does not mutate).
function _linked_list(head::Vector{Ti}, next::Vector{Ti}, start::Int) where {Ti <: Integer}
    out = Ti[]
    v = head[start]
    while v != 0
        push!(out, v)
        v = next[v]
    end
    return out
end

"""
    relabel_parent(n, parent, postinv) -> parent2

Elimination tree in the postordered numbering: `parent2[postinv[j]] = postinv[parent[j]]`
(roots stay `0`). Equal to `etree` of the relabeled pattern, since a symmetric
permutation that is a postorder of the tree leaves the tree itself unchanged
(design.md §3.2).
"""
function relabel_parent(n::Int, parent::Vector{Ti}, postinv::Vector{Ti}) where {Ti <: Integer}
    parent2 = Vector{Ti}(undef, n)
    @inbounds for j in 1:n
        p = parent[j]
        parent2[postinv[j]] = iszero(p) ? zero(Ti) : postinv[p]
    end
    return parent2
end

"""
    relabel_pattern(colptr, rowval, postinv) -> (colptr2, rowval2)

Relabel a strict-upper CSC pattern by the postorder (`postinv[oldlabel] = newlabel`),
returning a new strict-upper CSC pattern in the postordered numbering. Design.md §3.2:
postordering is composed into the final permutation so supernode detection sees
contiguous children.
"""
function relabel_pattern(
        n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}, postinv::Vector{Ti},
    ) where {Ti <: Integer}
    counts = zeros(Ti, n + 1)
    @inbounds for j in 1:n
        newj = postinv[j]
        for p in colptr[j]:(colptr[j + 1] - 1)
            newi = postinv[rowval[p]]
            hi = max(newi, newj)
            counts[hi] += one(Ti)
        end
    end
    colptr2 = Vector{Ti}(undef, n + 1)
    colptr2[1] = one(Ti)
    @inbounds for j in 1:n
        colptr2[j + 1] = colptr2[j] + counts[j]
    end
    fill!(counts, zero(Ti))
    rowval2 = Vector{Ti}(undef, colptr2[n + 1] - 1)
    @inbounds for j in 1:n
        newj = postinv[j]
        for p in colptr[j]:(colptr[j + 1] - 1)
            newi = postinv[rowval[p]]
            lo, hi = newi < newj ? (newi, newj) : (newj, newi)
            dest = colptr2[hi] + counts[hi]
            rowval2[dest] = lo
            counts[hi] += one(Ti)
        end
    end
    @inbounds for j in 1:n
        _sort_segment!(rowval2, colptr2[j], colptr2[j + 1] - 1)
    end
    return colptr2, rowval2
end
