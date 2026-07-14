# Sparse QR symbolic analysis, design_qr.md §3 (M5a task 4: star pattern builder).
# Drives the EXISTING etree.jl/counts.jl functions unchanged — no reimplementation
# (design_qr.md §1.5's module-layout note).

"""
    star_pattern(m, n, colptr, rowval, ciperm) -> (colptr2, rowval2)

Build the star matrix `S` (Gilbert–Li–Ng–Peyton 2001, via the survey's §7.1
presentation — the primary paper is unavailable, design_qr.md §3.2) for `A` (`m×n`,
CSC) under the column permutation `ciperm` (`ciperm[jorig]` = permuted position of
original column `jorig`, the `symmetrized_upper`/`full_symmetric_pattern` convention).
Column `k` of `S` (in PERMUTED column space) is the union, over every row `i` of `A`
whose leftmost nonzero in permuted column order is column `k`, of that row's *other*
permuted-column entries — a "star" centered at `k`. `|S| ≤ |A|`.

Returned as a ONE-triangle pattern (only column `k`'s own star edges are stored, not
yet mirrored into row/column `j`'s list) — feed the result through
[`symmetrized_upper`](@ref) with an identity permutation to symmetrize before handing
to [`etree`](@ref)/[`column_counts`](@ref) (design_qr.md §3.2: "feed the strict-upper
part through the existing `symmetrized_upper`-shaped entry points unchanged").

**Correctness (H1, design_qr.md §3.2):** `G(S)` and `G(AᵀA)` have the same FILLED
graph (proved via the fill-path theorem, Rose–Tarjan), so `etree(S)` = the column
elimination tree of `A` and `column_counts(S)` = `rcount` (row sizes of `R`) —
without ever forming `AᵀA`.
"""
function star_pattern(m::Int, n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}, ciperm::Vector{Ti}) where {Ti<:Integer}
    rowptr, rowidx = csc_transpose(m, n, colptr, rowval)  # row-form: rowptr[i]:rowptr[i+1]-1 = row i's original columns, sorted

    # Bucket rows by their star-center kmin via a head/next intrusive linked list (the
    # same idiom `Workspace.head`/`next` uses for descendant lists, design.md §4.3) —
    # REQUIRED, not a style choice: a single marker array tagged by `kmin` is only a
    # valid dedup check while all rows sharing that `kmin` are processed contiguously.
    # Rows are naturally encountered in ROW order (1:m), and unrelated rows sharing a
    # DIFFERENT kmin can touch the same target column in between — that would silently
    # overwrite `mark[j]`'s tag before an earlier kmin group's own dedup check runs
    # again, producing duplicate entries. Grouping by kmin first (this bucketing pass)
    # and then processing star-column k's entire row bucket before moving to k+1 is what
    # keeps the "one shared marker array, no re-zeroing" idiom (`ata_pattern`'s own
    # style) sound here — `ata_pattern` avoids the hazard for a different reason (its
    # outer loop already IS the target column), which does not carry over to a
    # row-outer-loop construction.
    head = zeros(Ti, n)
    next = zeros(Ti, m)
    @inbounds for i in 1:m
        lo, hi = rowptr[i], rowptr[i + 1] - 1
        lo > hi && continue                      # empty row: no star membership
        kmin = typemax(Ti)
        for p in lo:hi
            j = ciperm[rowidx[p]]
            j < kmin && (kmin = j)
        end
        next[i] = head[kmin]
        head[kmin] = Ti(i)
    end

    mark = zeros(Ti, n)
    # Pass 1: count nnz per star-column, one kmin bucket at a time.
    deg = zeros(Ti, n)
    @inbounds for k in 1:n
        i = head[k]
        while i != 0
            lo, hi = rowptr[i], rowptr[i + 1] - 1
            for p in lo:hi
                j = ciperm[rowidx[p]]
                (j == k || mark[j] == k) && continue
                mark[j] = k
                deg[k] += one(Ti)
            end
            i = next[i]
        end
    end
    colptr2 = Vector{Ti}(undef, n + 1)
    colptr2[1] = one(Ti)
    @inbounds for k in 1:n
        colptr2[k + 1] = colptr2[k] + deg[k]
    end
    # Pass 2: scatter (same tag idiom and bucket order; reset once, safe for the same
    # reason as pass 1 — each k's bucket is exhausted before k+1 starts).
    rowval2 = Vector{Ti}(undef, colptr2[n + 1] - 1)
    cursor = copy(colptr2)
    fill!(mark, zero(Ti))
    @inbounds for k in 1:n
        i = head[k]
        while i != 0
            lo, hi = rowptr[i], rowptr[i + 1] - 1
            for p in lo:hi
                j = ciperm[rowidx[p]]
                (j == k || mark[j] == k) && continue
                mark[j] = k
                rowval2[cursor[k]] = j
                cursor[k] += one(Ti)
            end
            i = next[i]
        end
    end
    @inbounds for k in 1:n
        sort!(view(rowval2, colptr2[k]:(colptr2[k + 1] - 1)))
    end
    return colptr2, rowval2
end
