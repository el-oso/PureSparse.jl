# Column-singleton pre-elimination, design_qr.md §2.3 (M5a task 9). SPQR paper §2.1,
# reimplemented from the paper's description (never its source): a column singleton is
# a column with exactly one LIVE nonzero whose magnitude exceeds a threshold; permute
# it (and its row) to the front, delete both, repeat. Breadth-first peeling on a
# value-aware row-form of A, O(|A|) total.

"""
    _row_form_values(m, n, colptr, rowval, nzval) -> (rowptr, rowidx, rowpos)

Value-aware row-form of `A` (like [`csc_transpose`](@ref), but also returns `rowpos`:
for each entry, its ORIGINAL position in `colptr`/`rowval`/`nzval`, so callers can
fetch `nzval[rowpos[k]]` without a separate value copy). Column-ascending scatter
order keeps each row's entries sorted by column, same as `csc_transpose`.
"""
function _row_form_values(m::Int, n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}) where {Ti<:Integer}
    deg = zeros(Ti, m)
    @inbounds for p in eachindex(rowval)
        deg[rowval[p]] += one(Ti)
    end
    rowptr = Vector{Ti}(undef, m + 1)
    rowptr[1] = one(Ti)
    @inbounds for i in 1:m
        rowptr[i + 1] = rowptr[i] + deg[i]
    end
    cursor = copy(rowptr)
    rowidx = Vector{Ti}(undef, length(rowval))
    rowpos = Vector{Ti}(undef, length(rowval))
    @inbounds for j in 1:n, p in colptr[j]:(colptr[j + 1] - 1)
        i = rowval[p]
        rowidx[cursor[i]] = Ti(j)
        rowpos[cursor[i]] = Ti(p)
        cursor[i] += one(Ti)
    end
    return rowptr, rowidx, rowpos
end

"""
    peel_column_singletons(A::SparseMatrixCSC{T,Ti}, threshold::T) -> (peel_col, peel_row, collive, rowlive)

Breadth-first column-singleton peeling (design_qr.md §2.3): repeatedly find a LIVE
column with exactly one LIVE row entry whose magnitude exceeds `threshold`, peel both
(column and row) to the front, and repeat. `peel_col[k]`/`peel_row[k]` (original
indices) are the `k`-th peeled column/row — this directly gives `R11`/`R12`: row `k`'s
entries are exactly the ORIGINAL row `peel_row[k]`'s stored entries (no numerical work,
no fill — the "reflector" for a length-1 pivot vector is a pure sign/scale, §2.3).
`collive`/`rowlive` mark the surviving (`A22`) columns/rows. A magnitude test failure
(a structural singleton whose value is too small) leaves the column live for the main
symbolic/numeric pipeline to handle — not a bug, §2.3's "values, not just pattern"
policy point.
"""
function peel_column_singletons(A::SparseMatrixCSC{T,Ti}, threshold::T) where {T<:Real,Ti<:Integer}
    m, n = size(A)
    rowptr, rowidx, rowpos = _row_form_values(m, n, A.colptr, A.rowval)

    coldeg = Vector{Ti}(undef, n)
    @inbounds for j in 1:n
        coldeg[j] = A.colptr[j + 1] - A.colptr[j]
    end
    rowlive = trues(m)
    collive = trues(n)
    peel_col = Ti[]
    peel_row = Ti[]

    queue = Ti[j for j in 1:n if coldeg[j] == 1]
    qi = 1
    @inbounds while qi <= length(queue)
        j = queue[qi]
        qi += 1
        (!collive[j] || coldeg[j] != 1) && continue
        r = zero(Ti)
        rp = zero(Ti)
        for p in A.colptr[j]:(A.colptr[j + 1] - 1)
            i = A.rowval[p]
            if rowlive[i]
                r = i
                rp = Ti(p)
                break
            end
        end
        r == 0 && continue                     # bookkeeping says live but none found: skip defensively
        abs(A.nzval[rp]) <= threshold && continue   # §2.3: magnitude test failed, leave it live

        collive[j] = false
        rowlive[r] = false
        push!(peel_col, j)
        push!(peel_row, r)
        for p in rowptr[r]:(rowptr[r + 1] - 1)
            jj = rowidx[p]
            collive[jj] || continue
            coldeg[jj] -= one(Ti)
            coldeg[jj] == one(Ti) && push!(queue, jj)
        end
    end
    return peel_col, peel_row, collive, rowlive
end
