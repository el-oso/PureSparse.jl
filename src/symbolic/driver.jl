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
    full_symmetric_pattern(n, colptr, rowval) -> (colptr2, rowval2)

Symmetrize an arbitrary (possibly one-triangle-only, possibly with duplicate or diagonal
entries) sparse pattern into `pattern(A) ∪ pattern(Aᵀ)` with the diagonal removed, BOTH
triangles stored — the input contract [`order`](@ref) expects (design.md §2.1).
"""
function full_symmetric_pattern(n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}) where {Ti<:Integer}
    idperm = collect(Ti, 1:n)
    ucolptr, urowval = symmetrized_upper(n, colptr, rowval, idperm, idperm)
    deg = zeros(Ti, n)
    @inbounds for j in 1:n, p in ucolptr[j]:(ucolptr[j + 1] - 1)
        i = urowval[p]
        deg[i] += one(Ti)
        deg[j] += one(Ti)
    end
    colptr2 = Vector{Ti}(undef, n + 1)
    colptr2[1] = one(Ti)
    @inbounds for j in 1:n
        colptr2[j + 1] = colptr2[j] + deg[j]
    end
    rowval2 = Vector{Ti}(undef, colptr2[n + 1] - 1)
    cursor = copy(colptr2)
    @inbounds for j in 1:n, p in ucolptr[j]:(ucolptr[j + 1] - 1)
        i = urowval[p]
        rowval2[cursor[i]] = Ti(j)
        cursor[i] += one(Ti)
        rowval2[cursor[j]] = i
        cursor[j] += one(Ti)
    end
    @inbounds for j in 1:n
        sort!(view(rowval2, colptr2[j]:(colptr2[j + 1] - 1)))
    end
    return colptr2, rowval2
end

"""
    assembly_map(A, perm, iperm, nsuper, super, snode_of, rowind_ptr, rowind, px) -> amap::Vector{Ti}

For every stored entry of `A`'s LOWER triangle (`i >= j`), the destination linear offset
in the numeric factor's `x` buffer (design.md §4.2's precomputed scatter map — what makes
`cholesky!` a straight allocation-free replay). Entries in `A`'s upper triangle (`i < j`,
if stored — ignored per this module's input convention) get sentinel `0`; a numeric LOAD
skips those. Sized `nnz(A)` (one slot per A's own stored entry) rather than `nnz(tril(A))`
alone — simpler and equally O(1) to consume, since the sentinel makes "not used" explicit.
"""
function assembly_map(
        A::SparseMatrixCSC, perm::Vector{Ti}, iperm::Vector{Ti},
        nsuper::Int, super::Vector{Ti}, snode_of::Vector{Ti},
        rowind_ptr::Vector{Ti}, rowind::Vector{Ti}, px::Vector{Ti},
) where {Ti<:Integer}
    n = size(A, 1)
    colptr, rowvalA = A.colptr, A.rowval
    nz = length(rowvalA)
    lo_arr = zeros(Ti, nz)
    hi_arr = zeros(Ti, nz)
    keep = falses(nz)
    scount = zeros(Ti, nsuper)
    @inbounds for j in 1:n, p in colptr[j]:(colptr[j + 1] - 1)
        i = rowvalA[p]
        i >= j || continue
        ni, nj = iperm[i], iperm[j]
        lo, hi = ni < nj ? (ni, nj) : (nj, ni)
        keep[p] = true
        lo_arr[p] = lo
        hi_arr[p] = hi
        scount[snode_of[lo]] += one(Ti)
    end
    sptr = Vector{Ti}(undef, nsuper + 1)
    sptr[1] = one(Ti)
    @inbounds for s in 1:nsuper
        sptr[s + 1] = sptr[s] + scount[s]
    end
    bucket_p = Vector{Ti}(undef, sptr[nsuper + 1] - 1)
    cursor = copy(sptr)
    @inbounds for j in 1:n, p in colptr[j]:(colptr[j + 1] - 1)
        keep[p] || continue
        s = snode_of[lo_arr[p]]
        bucket_p[cursor[s]] = Ti(p)
        cursor[s] += one(Ti)
    end

    amap = zeros(Ti, nz)
    relmap = zeros(Ti, n)
    @inbounds for s in 1:nsuper
        j0 = super[s]
        nrow = Int(rowind_ptr[s + 1] - rowind_ptr[s])
        k = 1
        for r in rowind_ptr[s]:(rowind_ptr[s + 1] - 1)
            relmap[rowind[r]] = Ti(k)
            k += 1
        end
        for bp in sptr[s]:(sptr[s + 1] - 1)
            p = bucket_p[bp]
            lo, hi = lo_arr[p], hi_arr[p]
            local_row = Int(relmap[hi])
            local_col = Int(lo - j0) + 1
            amap[p] = px[s] + Ti((local_col - 1) * nrow + (local_row - 1))
        end
    end
    return amap
end

"""
    symbolic(A::SparseMatrixCSC; ordering::AbstractOrdering=AMDOrdering()) -> Symbolic

Full symbolic analysis of `A` (read via its lower triangle, design.md driver docstring
above): fill-reducing ordering, elimination tree, column counts, supernode
detection/relaxed amalgamation, row structure, panel offsets, and the numeric-load
assembly map. Computed once and shared by reference across every numeric factorization of
a matrix with this sparsity pattern (design.md §1.2).
"""
function symbolic(A::SparseMatrixCSC{T,Ti}; ordering::AbstractOrdering = AMDOrdering()) where {T,Ti<:Integer}
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("symbolic: A must be square, got $(size(A))"))

    fcolptr, frowval = full_symmetric_pattern(n, A.colptr, A.rowval)
    amd_perm = order(ordering, n, fcolptr, frowval)
    amd_iperm = Vector{Ti}(undef, n)
    @inbounds for (k, p) in enumerate(amd_perm)
        amd_iperm[p] = Ti(k)
    end

    ucolptr, urowval = symmetrized_upper(n, A.colptr, A.rowval, amd_perm, amd_iperm)
    parent0 = etree(n, ucolptr, urowval)

    # Merge-aware postorder (design §3.5): relaxed amalgamation requires column
    # contiguity, which the postorder grants to exactly ONE child per parent (the
    # last-visited sibling) — so pick that sibling by column count (derivation in the
    # `postorder` docstring). column_counts (Gilbert–Ng–Peyton) needs a postordered
    # labeling, so run a preliminary default-order postorder + relabel to get exact
    # counts and map them back to parent0's labels. Column counts are a property of the
    # elimination order, unchanged by etree-postorder relabeling (design §3.2 — the
    # postorder is composed into `perm` precisely because it preserves fill), and here
    # they carry no correctness weight anyway: they are only a sibling-ordering
    # heuristic, and the final `colcount` below is recomputed on the final pattern.
    #
    # MEASURED (2026-07-13, ROADMAP "CURRENT FOCUS"): on the gate matrices this changes
    # WHICH sibling merges but not HOW MANY merge — the zero-fraction test rejects almost
    # no contiguity-eligible pair (2 of 1777 on laplacian2d 80×80), so nsuper is
    # unchanged; the binding constraint is one-contiguous-child-per-parent per
    # amalgamation pass, not sibling choice. Kept because any future multi-pass/cascade
    # amalgamation needs a principled sibling order as its substrate.
    _, postinv_pre = postorder(n, parent0)
    cp_pre, rv_pre = relabel_pattern(n, ucolptr, urowval, postinv_pre)
    cc_pre = column_counts(n, cp_pre, rv_pre, etree(n, cp_pre, rv_pre))
    colcount0 = Vector{Ti}(undef, n)
    @inbounds for j in 1:n
        colcount0[j] = cc_pre[postinv_pre[j]]
    end
    post, postinv = postorder(n, parent0, colcount0)

    perm = Vector{Ti}(undef, n)     # perm[finallabel] = original index (design.md §3.2 composition)
    @inbounds for orig in 1:n
        perm[postinv[amd_iperm[orig]]] = Ti(orig)
    end
    iperm = Vector{Ti}(undef, n)
    @inbounds for (k, p) in enumerate(perm)
        iperm[p] = Ti(k)
    end

    cp2, rv2 = relabel_pattern(n, ucolptr, urowval, postinv)
    parent = etree(n, cp2, rv2)
    colcount = column_counts(n, cp2, rv2, parent)

    nsuper0, super0 = fundamental_supernodes(n, parent, colcount)
    nsuper, super = relaxed_amalgamation(n, nsuper0, super0, parent, colcount)

    rowind_ptr, rowind, snode_of, sparent, max_update_size, max_extend_rows =
        supernode_rowind(n, cp2, rv2, parent, nsuper, super)

    px = Vector{Ti}(undef, nsuper + 1)
    px[1] = one(Ti)
    @inbounds for s in 1:nsuper
        nrow = Int(rowind_ptr[s + 1] - rowind_ptr[s])
        ncol = Int(super[s + 1] - super[s])
        px[s + 1] = px[s] + Ti(nrow * ncol)
    end

    amap = assembly_map(A, perm, iperm, nsuper, super, snode_of, rowind_ptr, rowind, px)

    return Symbolic{Ti}(
        n, perm, iperm, parent, colcount,
        nsuper, super, rowind_ptr, rowind, snode_of, sparent, px, amap,
        max_update_size, max_extend_rows,
        Int(nnz_l(colcount)), Float64(chol_flops(colcount)),
    )
end
