# Symbolic driver (design.md §1.2 "analyze once" split; M1 task list item 5): ties
# ordering -> etree -> postorder -> column counts -> supernode detection/amalgamation ->
# row structure into one `Symbolic`, plus panel offsets (`px`) and the numeric-load
# assembly map (`amap`).
#
# Input convention: `A` is read via its LOWER triangle ONLY (rows i >= columns j,
# diagonal included) — matches `Symmetric(A, :L)` semantics (LinearAlgebra) and this
# package's existing test-oracle convention (test/setups/oracle_setup.jl). Any upper-
# triangle entries A happens to store are ignored, not an error (so a fully-stored
# symmetric SparseMatrixCSC works too, just redundantly).

"""
    full_symmetric_pattern([Tn,] n, colptr, rowval) -> (colptr2, rowval2)

The symmetric pattern of the strictly-lower-triangular entries (`i > j`) of a CSC
pattern, BOTH triangles stored, no diagonal — the input contract [`order`](@ref) expects
(design.md §2.1). Only the lower triangle is read, matching the numeric phase (the
module convention above): any stored upper-triangle entry is ignored. Duplicate entries
are dropped. The result has index type `Tn` (default: the input's), which must hold
`2·nnz + 1`.
"""
full_symmetric_pattern(n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}) where {Ti <: Integer} =
    full_symmetric_pattern(Ti, n, colptr, rowval)

function full_symmetric_pattern(
        ::Type{Tn}, n::Int, colptr::Vector{Ti}, rowval::Vector{Ti},
    ) where {Tn <: Integer, Ti <: Integer}
    # Pass 1: every strictly-lower entry (i, j) lands in both column i and column j.
    # `colptr2[c + 1]` counts column c, so the prefix sum below needs no second array.
    colptr2 = zeros(Tn, n + 1)
    @inbounds for j in 1:n, p in colptr[j]:(colptr[j + 1] - 1)
        i = rowval[p]
        i > Ti(j) || continue
        colptr2[i + 1] += one(Tn)
        colptr2[j + 1] += one(Tn)
    end
    colptr2[1] = one(Tn)
    @inbounds for j in 1:n
        colptr2[j + 1] += colptr2[j]
    end
    # Pass 2: scatter with `colptr2[c]` as column c's write cursor; afterwards it holds
    # column c's end, which is column c+1's start, so one shift restores the pointers.
    rowval2 = Vector{Tn}(undef, colptr2[n + 1] - 1)
    @inbounds for j in 1:n, p in colptr[j]:(colptr[j + 1] - 1)
        i = rowval[p]
        i > Ti(j) || continue
        rowval2[colptr2[i]] = Tn(j)
        colptr2[i] += one(Tn)
        rowval2[colptr2[j]] = Tn(i)
        colptr2[j] += one(Tn)
    end
    @inbounds for c in n:-1:2
        colptr2[c] = colptr2[c - 1]
    end
    colptr2[1] = one(Tn)
    # Pass 3: drop duplicates in place, stamping each kept row with its column. Column j's
    # deduplicated run starts where column j-1's ended, so the result is already
    # contiguous. Rows within a column are left in arrival order: no consumer of this
    # pattern (ordering, etree, permuted_lower) depends on it. The pass is skipped when
    # every input column has strictly increasing rows, which rules out duplicates.
    _strictly_increasing_columns(n, colptr, rowval) && return colptr2, rowval2
    stamp = zeros(Tn, n)
    w = one(Tn)
    @inbounds for j in 1:n
        lo, hi = colptr2[j], colptr2[j + 1] - one(Tn)
        colptr2[j] = w
        for k in lo:hi
            v = rowval2[k]
            stamp[v] == Tn(j) && continue
            stamp[v] = Tn(j)
            rowval2[w] = v
            w += one(Tn)
        end
    end
    colptr2[n + 1] = w
    resize!(rowval2, w - 1)
    return colptr2, rowval2
end

# The column pointers must be nondecreasing, start at 1 and end within `rowval`, and
# every stored row index must lie in 1:n.
function _check_rows(n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}) where {Ti <: Integer}
    length(colptr) == n + 1 || throw(ArgumentError(lazy"symbolic: colptr has length $(length(colptr)), expected $(n + 1)"))
    isone(colptr[1]) || throw(ArgumentError(lazy"symbolic: colptr[1] = $(colptr[1]), expected 1"))
    Int(colptr[n + 1]) - 1 <= length(rowval) ||
        throw(ArgumentError(lazy"symbolic: colptr[n + 1] - 1 = $(colptr[n + 1] - 1) exceeds the $(length(rowval)) stored entries"))
    for j in 1:n
        colptr[j] <= colptr[j + 1] || throw(ArgumentError(lazy"symbolic: colptr decreases at column $j"))
        for p in colptr[j]:(colptr[j + 1] - 1)
            i = rowval[p]
            (1 <= i <= n) || throw(ArgumentError(lazy"symbolic: row index $i at entry $p is outside 1:$n"))
        end
    end
    return nothing
end

# True when the rows of every column are strictly increasing (the usual CSC invariant).
function _strictly_increasing_columns(n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}) where {Ti <: Integer}
    @inbounds for j in 1:n
        prev = zero(Ti)
        for p in colptr[j]:(colptr[j + 1] - 1)
            i = rowval[p]
            i > prev || return false
            prev = i
        end
    end
    return true
end

"""
    assembly_map(A, iperm, nsuper, super, snode_of, rowind_ptr, rowind, px, hptr, hadj, hslot) -> amap::Vector{Ti}

For every stored entry of `A`'s LOWER triangle (`i >= j`), the destination linear offset
in the numeric factor's `x` buffer (design.md §4.2's precomputed scatter map — what makes
`cholesky!` a straight allocation-free replay). Entries in `A`'s upper triangle (`i < j`,
if stored — ignored per this module's input convention) get sentinel `0`; a numeric LOAD
skips those. Sized `nnz(A)` (one slot per A's own stored entry) rather than `nnz(tril(A))`
alone — simpler and equally O(1) to consume, since the sentinel makes "not used" explicit.
The strictly-lower entries are reached through the permuted lower pattern and its slots
(`hptr`/`hadj`/`hslot`, [`permuted_lower`](@ref)): column `c`'s entries all land in
`c`'s supernode, whose row list is scattered once into `relmap`. Diagonal entries are
found by a pass over `A` itself. `relmap` is scratch of length at least `n`,
overwritten here.
"""
function assembly_map(
        A::SparseMatrixCSC, iperm::Vector{Tn},
        nsuper::Int, super::Vector{Tn}, snode_of::Vector{Tn},
        rowind_ptr::Vector{Tn}, rowind::Vector{Tn}, px::Vector{Ti},
        hptr::Vector{Tn}, hadj::Vector{Tn}, hslot::Vector{Tn};
        relmap::Vector{Tn} = Vector{Tn}(undef, size(A, 1)),
    ) where {Ti <: Integer, Tn <: Integer}
    n = size(A, 1)
    length(relmap) >= n || throw(ArgumentError("assembly_map: relmap must have length >= n"))
    colptr, rowvalA = A.colptr, A.rowval
    amap = zeros(Ti, length(rowvalA))
    @inbounds for s in 1:nsuper
        j0 = super[s]
        nrow = Int(rowind_ptr[s + 1] - rowind_ptr[s])
        k = 1
        for r in rowind_ptr[s]:(rowind_ptr[s + 1] - 1)
            relmap[rowind[r]] = Tn(k)
            k += 1
        end
        for c in j0:(super[s + 1] - one(Tn))
            col_off = Int(px[s]) + Int(c - j0) * nrow - 1
            for q in hptr[c]:(hptr[c + 1] - 1)
                amap[hslot[q]] = Ti(col_off + Int(relmap[hadj[q]]))
            end
        end
    end
    # Diagonal entries: row c sits at position c - j0 + 1 of its own supernode's rows.
    @inbounds for j in 1:n, p in colptr[j]:(colptr[j + 1] - 1)
        Int(rowvalA[p]) == j || continue
        c = iperm[j]
        s = snode_of[c]
        nrow = Int(rowind_ptr[s + 1] - rowind_ptr[s])
        amap[p] = Ti(Int(px[s]) + Int(c - super[s]) * (nrow + 1))
    end
    return amap
end

"""
    upper_assembly_map(A, iperm) -> (ucolptr, urowval, amap)

The permuted upper triangle of `A` (diagonal included) as a CSC pattern in factor
order — column `k` holds the rows `i <= k` of `P·A·Pᵀ`, in arrival order — together with
the map from every stored entry of `A` to its slot in that pattern (the simplicial
path's assembly map, design.md §4.5; sentinel `0` for an upper-triangle entry of `A`,
which is never read, as in [`assembly_map`](@ref)). Each stored lower-triangle entry
gets its own slot, so a numeric load fills every slot without clearing.
"""
function upper_assembly_map(A::SparseMatrixCSC, iperm::Vector{Ti}) where {Ti <: Integer}
    n = size(A, 1)
    colptr, rowvalA = A.colptr, A.rowval
    ucolptr = zeros(Ti, n + 1)
    @inbounds for j in 1:n, p in colptr[j]:(colptr[j + 1] - 1)
        i = rowvalA[p]
        i >= j || continue
        ucolptr[max(iperm[i], iperm[j]) + 1] += one(Ti)
    end
    ucolptr[1] = one(Ti)
    @inbounds for k in 1:n
        ucolptr[k + 1] += ucolptr[k]
    end
    next = ucolptr[1:n]
    urowval = Vector{Ti}(undef, Int(ucolptr[n + 1]) - 1)
    amap = zeros(Ti, length(rowvalA))
    @inbounds for j in 1:n, p in colptr[j]:(colptr[j + 1] - 1)
        i = rowvalA[p]
        i >= j || continue
        lo, hi = minmax(iperm[i], iperm[j])
        q = next[hi]
        next[hi] = q + one(Ti)
        urowval[q] = lo
        amap[p] = q
    end
    return ucolptr, urowval, amap
end

"""
    symbolic(A::SparseMatrixCSC; ordering::AbstractOrdering=AMDOrdering(), method=:auto) -> Symbolic

Full symbolic analysis of `A` (read via its lower triangle, design.md driver docstring
above): fill-reducing ordering, elimination tree, column counts, then either supernode
detection/relaxed amalgamation, row structure, panel offsets and the panel assembly map
(supernodal path) or the exact pattern of `L` by columns and by rows and the
permuted-upper assembly map (simplicial path, design.md §4.5). Computed once and shared by reference across
every numeric factorization of a matrix with this sparsity pattern (design.md §1.2).

`method` is `:auto` (simplicial when the mean work per stored entry of `L`,
`flops / nnzL`, is at or below the `simplicial_ratio` preference, see `tuning.jl`),
`:simplicial`, or `:supernodal`.

The analysis runs on `Int32` indices whenever `A`'s index type is wider and `n`,
`nnz(A)` and `nnz(L)` fit (design.md §3.1): every stage works on index arrays, so this
halves the bytes allocated and touched. Only the arrays stored in the `Symbolic` are
widened back to `A`'s index type; results are identical either way.
"""
function symbolic(
        A::SparseMatrixCSC{T, Ti}; ordering::AbstractOrdering = AMDOrdering(), method::Symbol = :auto,
    ) where {T, Ti <: Integer}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("symbolic: A must be square, got $(size(A))"))
    (method === :auto || method === :simplicial || method === :supernodal) ||
        throw(ArgumentError("symbolic: method must be :auto, :simplicial, or :supernodal (got $method)"))
    # Every row index must lie in 1:n: the pattern passes (`full_symmetric_pattern`,
    # `permuted_lower`, `assembly_map`) index their arrays by the stored rows without
    # further checks. The permutation an ordering returns is checked in `_symbolic`.
    _check_rows(n, A.colptr, A.rowval)
    # The full pattern holds up to 2·nnz(A) entries plus n + 1 pointers; `nnz(L)` is
    # checked once the column counts are known.
    if Ti !== Int32 && sizeof(Ti) > 4 && n + 1 <= typemax(Int32) && 2 * nnz(A) + 2 <= typemax(Int32)
        return _symbolic(Int32, A, ordering, method)
    end
    return _symbolic(Ti, A, ordering, method)
end

# The analysis proper, on index type `Tn`; the result carries `A`'s index type.
function _symbolic(
        ::Type{Tn}, A::SparseMatrixCSC{T, Ti}, ordering::AbstractOrdering, method::Symbol,
    ) where {Tn <: Integer, T, Ti <: Integer}
    n = size(A, 1)
    fcolptr, frowval = full_symmetric_pattern(Tn, n, A.colptr, A.rowval)
    amd_perm = order(ordering, n, fcolptr, frowval)
    length(amd_perm) == n || throw(ArgumentError(lazy"symbolic: the ordering returned $(length(amd_perm)) indices for n = $n"))
    # The inverse doubles as the permutation check: every index must be in 1:n and
    # appear exactly once.
    amd_iperm = zeros(Tn, n)
    for (k, p) in enumerate(amd_perm)
        (1 <= p <= n && iszero(amd_iperm[p])) ||
            throw(ArgumentError(lazy"symbolic: the ordering is not a permutation of 1:n (entry $k is $p)"))
        amd_iperm[p] = Tn(k)
    end

    # The etree is read straight off the full pattern through the AMD permutation; the
    # permuted pattern is materialized only once, in its final labeling, below.
    parent0 = etree(n, fcolptr, frowval, amd_perm, amd_iperm)

    # Postorder with children visited in ascending column order (design.md §3.2): the
    # postorder is composed into `perm`, so it changes supernode boundaries but not fill.
    post, postinv = postorder(n, parent0)

    perm = Vector{Tn}(undef, n)     # perm[finallabel] = original index (design.md §3.2 composition)
    @inbounds for orig in 1:n
        perm[postinv[amd_iperm[orig]]] = Tn(orig)
    end
    iperm = Vector{Tn}(undef, n)
    @inbounds for (k, p) in enumerate(perm)
        iperm[p] = Tn(k)
    end

    parent = relabel_parent(n, parent0, postinv)
    # Column counts and supernodal row structure read each column's below-diagonal
    # pattern (design.md §3.1), so that is the only permuted pattern formed.
    hptr, hadj, hslot = permuted_lower(n, A.colptr, A.rowval, perm, iperm)
    # Buffers spent by the steps above serve as the column-count scratch.
    colcount = column_counts_lower(n, parent, hptr, hadj; work = (amd_iperm, parent0, post, postinv))
    nnzL = Int(nnz_l(colcount))
    flops = Float64(chol_flops(colcount))
    # Row-structure pointers run to nnz(L) + 1; a factor too large for Tn is analyzed
    # again on A's own index type.
    if Tn !== Ti && nnzL + 1 > typemax(Tn)
        return _symbolic(Ti, A, ordering, method)
    end

    # Path choice (design.md §4.5) from the column counts alone, before any supernode
    # work; the comparison is written without a division so an empty matrix is well
    # defined.
    if method === :simplicial || (method === :auto && flops <= SIMPLICIAL_RATIO * nnzL)
        iperm_ = convert(Vector{Ti}, iperm)
        parent_ = convert(Vector{Ti}, parent)
        colcount_ = convert(Vector{Ti}, colcount)
        ucolptr, urowval, amap = upper_assembly_map(A, iperm_)
        lcolptr, lrowval, lrowptr, lcolval = l_pattern(n, parent_, colcount_, ucolptr, urowval)
        one_ = Ti[one(Ti)]   # length-1 pointer arrays: `super[nsuper + 1]`-style reads stay valid
        return Symbolic{Ti}(
            n, convert(Vector{Ti}, perm), iperm_, parent_, colcount_,
            0, one_, one_, Ti[], Ti[], Ti[], one_, amap,
            true, lcolptr, lrowval, lrowptr, lcolval, ucolptr, urowval,
            0, 0, nnzL, flops,
        )
    end

    nsuper0, super0 = fundamental_supernodes(n, parent, colcount)
    nsuper, super = relaxed_amalgamation(n, nsuper0, super0, parent, colcount)

    # The row structure is climbed from the upper pattern (column i: its rows j < i),
    # the transpose of the lower one; `lower_pattern` is that transpose.
    uptr, uadj = lower_pattern!(fcolptr, frowval, n, hptr, hadj)   # the full pattern is spent
    rowind_ptr, rowind, snode_of, sparent, max_update_size, max_extend_rows =
        supernode_rowind(n, uptr, uadj, parent, colcount, nsuper, super)

    # Panel offsets and the assembly map address the numeric buffer, so they are built
    # on A's index type directly.
    px = Vector{Ti}(undef, nsuper + 1)
    px[1] = one(Ti)
    @inbounds for s in 1:nsuper
        nrow = Int(rowind_ptr[s + 1] - rowind_ptr[s])
        ncol = Int(super[s + 1] - super[s])
        px[s + 1] = px[s] + Ti(nrow * ncol)
    end

    amap = assembly_map(A, iperm, nsuper, super, snode_of, rowind_ptr, rowind, px, hptr, hadj, hslot; relmap = uptr)

    return Symbolic{Ti}(
        n, convert(Vector{Ti}, perm), convert(Vector{Ti}, iperm), convert(Vector{Ti}, parent),
        convert(Vector{Ti}, colcount),
        nsuper, convert(Vector{Ti}, super), convert(Vector{Ti}, rowind_ptr), convert(Vector{Ti}, rowind),
        convert(Vector{Ti}, snode_of), convert(Vector{Ti}, sparent), px, amap,
        false, Ti[], Ti[], Ti[], Ti[], Ti[], Ti[],
        max_update_size, max_extend_rows,
        nnzL, flops,
    )
end
