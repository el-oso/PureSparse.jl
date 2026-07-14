# AMDOrdering on the explicit pattern of AᵀA, design_qr.md §2.2.6 (M5a task 2). Kept as
# a first-class alternative to COLAMD, not a placeholder: MA49 orders exactly this way
# (SPQR paper §2.2), and SPQR's own measured default prefers AMD-on-AᵀA for m > 2n
# (SPQR paper §5.4, Table VI).

"""
    csc_transpose(m, n, colptr, rowval) -> (colptr2, rowval2)

Transpose an `m×n` sparse pattern given in CSC form into the `n×m` CSC pattern of its
transpose — equivalently, the "row-form" of the original (design_qr.md §2.2.6/§2.3/
§3.4: column `i` of the result holds the sorted, ascending set of original columns `j`
with a nonzero at original row `i`). Standard counting-sort transpose; each output
column is naturally produced in ascending order since the input is scanned
column-by-column in increasing `j`, so no separate sort pass is needed.
"""
function csc_transpose(m::Int, n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}) where {Ti<:Integer}
    deg = zeros(Ti, m)
    @inbounds for p in eachindex(rowval)
        deg[rowval[p]] += one(Ti)
    end
    colptr2 = Vector{Ti}(undef, m + 1)
    colptr2[1] = one(Ti)
    @inbounds for i in 1:m
        colptr2[i + 1] = colptr2[i] + deg[i]
    end
    cursor = copy(colptr2)
    rowval2 = Vector{Ti}(undef, length(rowval))
    @inbounds for j in 1:n, p in colptr[j]:(colptr[j + 1] - 1)
        i = rowval[p]
        rowval2[cursor[i]] = Ti(j)
        cursor[i] += one(Ti)
    end
    return colptr2, rowval2
end

"""
    ata_pattern(m, n, colptr, rowval) -> (colptr2, rowval2)

Pattern of `AᵀA` (`n×n`, diagonal removed, both triangles stored — the input `order`
expects, design.md §2.1), built WITHOUT ever forming `AᵀA` as a `SparseMatrixCSC`
(design_qr.md §2.2.6): for each row `i` of `A`, every pair of columns `(p,q)` sharing a
nonzero in row `i` contributes an edge `(p,q)` to `pattern(AᵀA)`, since
`(AᵀA)[p,q] = Σ_i A[i,p]·A[i,q]` is structurally nonzero iff some row has both. A
monotonically-tagged marker array (the same "no clearing needed between columns" idiom
as `Workspace.relmap`, design.md §4.3 step 1) dedupes edges seen from more than one
shared row. Cost: worst-case `O(Σ_i nnz(row i)²)` time, `O(|AᵀA|)` memory (paid once
per symbolic — the price COLAMD avoids, design_qr.md §2.2.6).
"""
function ata_pattern(m::Int, n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}) where {Ti<:Integer}
    n == 0 && return (ones(Ti, 1), Ti[])
    rowptr, rowidx = csc_transpose(m, n, colptr, rowval)
    mark = zeros(Ti, n)
    # Pass 1: count nnz per column of pattern(AᵀA).
    deg = zeros(Ti, n)
    @inbounds for j in 1:n
        for p in colptr[j]:(colptr[j + 1] - 1)
            i = rowval[p]                              # a row touching column j
            for pp in rowptr[i]:(rowptr[i + 1] - 1)
                q = rowidx[pp]
                (q == j || mark[q] == j) && continue
                mark[q] = j
                deg[j] += one(Ti)
            end
        end
    end
    colptr2 = Vector{Ti}(undef, n + 1)
    colptr2[1] = one(Ti)
    @inbounds for j in 1:n
        colptr2[j + 1] = colptr2[j] + deg[j]
    end
    # Pass 2: scatter (same tag idiom, reused without re-zeroing between passes since
    # every tag comparison below is again against the current column j).
    rowval2 = Vector{Ti}(undef, colptr2[n + 1] - 1)
    cursor = copy(colptr2)
    fill!(mark, zero(Ti))
    @inbounds for j in 1:n
        for p in colptr[j]:(colptr[j + 1] - 1)
            i = rowval[p]
            for pp in rowptr[i]:(rowptr[i + 1] - 1)
                q = rowidx[pp]
                (q == j || mark[q] == j) && continue
                mark[q] = j
                rowval2[cursor[j]] = q
                cursor[j] += one(Ti)
            end
        end
    end
    @inbounds for j in 1:n
        sort!(view(rowval2, colptr2[j]:(colptr2[j + 1] - 1)))
    end
    return colptr2, rowval2
end

"""
    order_columns(alg::AMDOrdering, m, n, colptr, rowval) -> Vector{Ti}

AMDOrdering's sparse-QR column ordering (design_qr.md §2.2.6): form `pattern(AᵀA)` via
[`ata_pattern`](@ref) and delegate to the untouched symmetric `order`.
"""
function order_columns(alg::AMDOrdering, m::Int, n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}) where {Ti<:Integer}
    colptr2, rowval2 = ata_pattern(m, n, colptr, rowval)
    return order(alg, n, colptr2, rowval2)
end
