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
    permuted_upper(n, colptr, rowval, iperm) -> (colptr2, rowval2)

[`symmetrized_upper`](@ref) for a pattern that is already symmetric, diagonal-free and
duplicate-free (the output of [`full_symmetric_pattern`](@ref)): each edge is stored
once per endpoint, so the endpoint whose permuted column is the larger one yields the
strict-upper entry and no deduplication pass is needed.
"""
function permuted_upper(
        n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}, iperm::Vector{Ti},
    ) where {Ti <: Integer}
    counts = zeros(Ti, n)
    @inbounds for jorig in 1:n
        j = iperm[jorig]
        for p in colptr[jorig]:(colptr[jorig + 1] - 1)
            iperm[rowval[p]] < j && (counts[j] += one(Ti))
        end
    end
    colptr2 = Vector{Ti}(undef, n + 1)
    colptr2[1] = one(Ti)
    @inbounds for j in 1:n
        colptr2[j + 1] = colptr2[j] + counts[j]
    end
    fill!(counts, zero(Ti))
    rowval2 = Vector{Ti}(undef, colptr2[n + 1] - 1)
    @inbounds for jorig in 1:n
        j = iperm[jorig]
        for p in colptr[jorig]:(colptr[jorig + 1] - 1)
            i = iperm[rowval[p]]
            i < j || continue
            rowval2[colptr2[j] + counts[j]] = i
            counts[j] += one(Ti)
        end
    end
    @inbounds for j in 1:n
        _sort_segment!(rowval2, colptr2[j], colptr2[j + 1] - 1)
    end
    return colptr2, rowval2
end

"""
    etree(n, colptr, rowval) -> parent::Vector{Ti}

Elimination tree of the symmetric pattern given by its STRICT UPPER triangle in CSC form
(as returned by [`symmetrized_upper`](@ref)). Liu's algorithm: one pass with a
path-compressed `ancestor` array. `parent[j] == 0` marks a root. Design.md §3.2.
"""
function etree(n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}) where {Ti <: Integer}
    parent = zeros(Ti, n)
    ancestor = zeros(Ti, n)
    @inbounds for k in 1:n
        for p in colptr[k]:(colptr[k + 1] - 1)
            i = rowval[p]
            i >= k && continue   # strict upper triangle only holds i < k already; defensive
            while i != 0 && i != k
                inext = ancestor[i]
                ancestor[i] = k
                if inext == 0
                    parent[i] = k
                    break
                end
                i = inext
            end
        end
    end
    return parent
end

"""
    postorder(n, parent[, priority]) -> (post::Vector{Ti}, postinv::Vector{Ti})

Postorder of the forest given by `parent`. `post[k]` is the k-th node visited;
`postinv[j]` is node `j`'s position in the postorder. Iterative (stack-based) — no
recursion whose depth depends on runtime data (CONTRIBUTING.md req 4).

Without `priority`, each node's children are visited in ascending column order (for
determinism — design.md §3.2). With `priority` (length `n`), children are visited in
ascending `(priority, column)` order, so the LAST-visited child — the one the postorder
makes numerically contiguous with its parent, and therefore the only sibling eligible
for relaxed amalgamation's column-contiguity requirement (design §3.5) — is the
max-priority child. Any sibling order yields a valid postorder (parents still follow all
descendants), so this changes nothing about etree/colcount correctness downstream, only
which sibling ends up parent-contiguous.

Why colcount is the priority `symbolic()` passes (own derivation, no external recipe):
for an etree child `s` of `t`, `pattern(L[:,s]) \\ {s,t} ⊆ pattern(L[:,t])` (Liu 1986,
as cited in design.md §3.2), hence `colcount[s] ≤ colcount[t] + 1`
with equality exactly when the patterns nest — the fundamental-supernode condition
(Liu–Ng–Peyton 1993, design §3.4). So the max-colcount child is the closest to exact
nesting, i.e. the one whose merge pads the fewest explicit zeros into the block that
relaxed amalgamation's zero-fraction test evaluates (Ashcraft–Grimes 1989 / Ng–Peyton
1993 motivate the merge itself; the sibling-selection rule is our own reasoning on top).
"""
function postorder(
        n::Int, parent::Vector{Ti}, priority::Union{Nothing, Vector{Ti}} = nothing,
    ) where {Ti <: Integer}
    # Build child lists via a head/next linked-list (bucketed by parent). Head insertion
    # reverses insertion order, so inserting in DESCENDING visit order yields per-parent
    # lists in ascending visit order, which the DFS then consumes front-to-back.
    head = zeros(Ti, n + 1)     # head[n+1] = roots (parent == 0)
    next = zeros(Ti, n)
    if isnothing(priority)
        @inbounds for j in n:-1:1
            p = parent[j] == 0 ? n + 1 : parent[j]
            next[j] = head[p]
            head[p] = j
        end
    else
        length(priority) == n ||
            throw(DimensionMismatch("postorder: priority has length $(length(priority)), expected $n"))
        # Descending (priority, column) insertion ⇒ ascending (priority, column) visit
        # order ⇒ last-visited sibling = max priority, ties broken by largest column
        # (identical to the no-priority behavior when priorities are uniform).
        # Counting sort by priority (a column count, so 1 ≤ priority ≤ n), stable in
        # column; walking the sorted order backwards gives the descending insertion.
        pmax = 0
        for j in 1:n
            pmax = max(pmax, Int(priority[j]))
        end
        bucket = zeros(Int, pmax + 2)
        @inbounds for j in 1:n
            bucket[Int(priority[j]) + 2] += 1
        end
        @inbounds for d in 2:(pmax + 2)
            bucket[d] += bucket[d - 1]
        end
        ins = Vector{Ti}(undef, n)
        @inbounds for j in 1:n
            d = Int(priority[j]) + 1
            ins[bucket[d] + 1] = Ti(j)
            bucket[d] += 1
        end
        @inbounds for k in n:-1:1
            j = ins[k]
            p = iszero(parent[j]) ? n + 1 : parent[j]
            next[j] = head[p]
            head[p] = j
        end
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
