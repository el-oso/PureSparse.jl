# Exact pattern of L for the simplicial path (design.md §4.5), by columns and by rows.
#
# Source: Liu, *The Role of Elimination Trees in Sparse Factorization* (1990), sections
# 3-4 — the structure of row k of L is the row subtree of the elimination tree: the
# nodes reached by climbing from every row index of column k of the permuted upper
# triangle toward k, stopping at a node already reached for this row. Davis, *Direct
# Methods for Sparse Linear Systems* (2006), chapter 4 gives the same traversal with a
# per-row marker array and a stack that collects the climbed paths. Written from those
# texts only.

"""
    l_pattern(n, parent, colcount, ucolptr, urowval) -> (lcolptr, lrowval, lrowptr, lcolval)

Row indices of `L` by columns (`lcolptr`/`lrowval`, each column's diagonal first, rows
ascending) and the same pattern by rows (`lrowptr`/`lcolval`, each row's diagonal last,
the other columns in an order where every column precedes its etree ancestors) for the
postordered elimination tree `parent`, the column counts `colcount` (which size the
arrays exactly) and the permuted upper triangle of A in CSC form with its diagonal
(`ucolptr`/`urowval`, [`upper_assembly_map`](@ref)).

One pass over the rows: row `k`'s structure is climbed from each entry of column `k`
of the upper triangle, marking with `flag[i] == k`; `k` is appended to each reached
column (rows arrive in order, so every column ends up sorted), and each climbed path —
a descendant-to-ancestor chain — is placed in front of the paths found earlier on a
scratch stack, which is the order the row is then stored in. A later path stops at a
node of an earlier one, so its nodes are descendants of that node and none of the
earlier path's nodes below it are related to them; placing it first keeps every
column before its ancestors.
"""
function l_pattern(
        n::Int, parent::Vector{Ti}, colcount::Vector{Ti}, ucolptr::Vector{Ti}, urowval::Vector{Ti},
    ) where {Ti <: Integer}
    lcolptr = Vector{Ti}(undef, n + 1)
    lcolptr[1] = one(Ti)
    @inbounds for j in 1:n
        lcolptr[j + 1] = lcolptr[j] + colcount[j]
    end
    nnzL = Int(lcolptr[n + 1]) - 1
    lrowval = Vector{Ti}(undef, nnzL)
    lrowptr = Vector{Ti}(undef, n + 1)
    lcolval = Vector{Ti}(undef, nnzL)
    cursor = lcolptr[1:n]
    flag = zeros(Ti, n)
    stk = Vector{Ti}(undef, n)        # paths of the current row: scratch in stk[1:len],
    # collected row in stk[top:n]; together they hold distinct
    # nodes other than k, so top >= len + 2 and the scratch
    # never reaches the collected row
    w = 1                             # next free slot of lcolval
    lrowptr[1] = one(Ti)
    @inbounds for k in 1:n
        flag[k] = Ti(k)
        # the diagonal is column k's first entry: no earlier row can reach column k
        q = Int(cursor[k])
        cursor[k] = Ti(q + 1)
        lrowval[q] = Ti(k)
        top = n + 1
        for p in Int(ucolptr[k]):(Int(ucolptr[k + 1]) - 1)
            i = Int(urowval[p])
            len = 0
            while flag[i] != Ti(k)
                flag[i] = Ti(k)
                len += 1
                stk[len] = Ti(i)
                q = Int(cursor[i])
                cursor[i] = Ti(q + 1)
                lrowval[q] = Ti(k)
                i = Int(parent[i])
            end
            # Move the path below the collected row, last node first: the write
            # position stays above every scratch slot still to be read (top >= len + 2),
            # which a copy from the first node on would not guarantee, and the path
            # keeps its descendant-to-ancestor order.
            for t in len:-1:1
                top -= 1
                stk[top] = stk[t]
            end
        end
        for t in top:n
            lcolval[w] = stk[t]
            w += 1
        end
        lcolval[w] = Ti(k)
        w += 1
        lrowptr[k + 1] = Ti(w)
    end
    return lcolptr, lrowval, lrowptr, lcolval
end
