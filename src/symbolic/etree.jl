# Pattern setup (design.md §3.1) and elimination tree + postorder (design.md §3.2).
# Grounded in Liu 1986 (`refs/linear_algebra/Supernodal/liu1986.pdf`) and the
# presentation in Davis, *Direct Methods for Sparse Linear Systems* (SIAM, 2006), ch. 4
# (`refs/linear_algebra/chapter-direct.pdf`) — never CHOLMOD source (CLAUDE.md req 1).

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
) where {Ti<:Integer}
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
        seg = view(rowval2, lo:hi)
        sort!(seg)
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
    etree(n, colptr, rowval) -> parent::Vector{Ti}

Elimination tree of the symmetric pattern given by its STRICT UPPER triangle in CSC form
(as returned by [`symmetrized_upper`](@ref)). Liu's algorithm: one pass with a
path-compressed `ancestor` array. `parent[j] == 0` marks a root. Design.md §3.2.
"""
function etree(n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}) where {Ti<:Integer}
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
    postorder(n, parent) -> (post::Vector{Ti}, postinv::Vector{Ti})

Postorder of the forest given by `parent` (children visited in ascending column order,
for determinism — design.md §3.2). `post[k]` is the k-th node visited; `postinv[j]` is
node `j`'s position in the postorder. Iterative (stack-based) — no recursion whose depth
depends on runtime data (CLAUDE.md req 4).
"""
function postorder(n::Int, parent::Vector{Ti}) where {Ti<:Integer}
    # Build child lists via a head/next linked-list (bucketed by parent), children in
    # ascending order so DFS visits them ascending too.
    head = zeros(Ti, n + 1)     # head[n+1] = roots (parent == 0)
    next = zeros(Ti, n)
    @inbounds for j in n:-1:1
        p = parent[j] == 0 ? n + 1 : parent[j]
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
function _linked_list(head::Vector{Ti}, next::Vector{Ti}, start::Int) where {Ti<:Integer}
    out = Ti[]
    v = head[start]
    while v != 0
        push!(out, v)
        v = next[v]
    end
    return out
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
) where {Ti<:Integer}
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
        sort!(view(rowval2, colptr2[j]:(colptr2[j + 1] - 1)))
    end
    return colptr2, rowval2
end
