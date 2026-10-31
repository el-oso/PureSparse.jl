# Simplicial (column-by-column) LLᵀ and LDLᵀ over the exact pattern of L, and the
# triangular solves that go with it (design.md §4.5). Taken by every factor whose
# `Symbolic` has `simplicial == true`; the supernodal entry points branch here after
# their argument checks.
#
# Sources: Davis, *Direct Methods for Sparse Linear Systems* (2006), chapter 4 — the
# up-looking Cholesky as one sparse triangular solve per row of L, over that row's
# structure in the elimination tree — and, for the LDLᵀ variant, the way the QDLDL
# solver (Stellato et al.) carries D through the same recurrence. Written from those
# texts only.
#
# Up-looking factorization: row k of L is `L[k, 1:k-1] = (L[1:k-1, 1:k-1]⁻¹ a_k)ᵀ` with
# `a_k` the upper triangle of column k of P·A·Pᵀ, followed by `L[k,k]² = a_kk − ‖L[k,
# 1:k-1]‖²` (LDLᵀ: `d_k = a_kk − Σ L[k,i]² d_i`). The solve runs over the row's
# structure only — `sym.lcolval[sym.lrowptr[k]:...]`, the columns i < k with L[k,i] ≠ 0
# in ascending order, which is a topological order of the etree — scattering into the
# dense vector `x` that is all-zero between rows. Each computed `L[k,i]` is stored
# twice: appended to column i of the by-column values `lx` (rows arrive ascending, so
# the column's slot order is the pattern's) and at its slot in row k of the by-row
# values `ltx`. Per-row work is proportional to the nonzeros it touches; the total is
# `Σⱼ colcount[j]²/2` multiply-adds, as for the supernodal path.
#
# `ltx` exists for the backward solve: with L held by rows, `Lᵀ·y = b` is the same
# scatter sweep as the forward solve with L held by columns, instead of a per-column
# dot product whose accumulator serializes the floating-point adds.
#
# Every sweep below is a kernel that takes its arrays as arguments. Reached through the
# factor's fields instead, the array data pointers are reloaded inside the inner loop
# (the compiler cannot hoist them past the scatter stores), which was measured at 1.5×
# the time of the same loop on the lap2d forward sweep (180 against 116 µs).
#
# This is per-column sparse arithmetic, not a dense block kernel, so it does not go
# through PureBLAS (CONTRIBUTING.md requirement 9 concerns dense kernels only).

# Scatter the stored lower triangle of A into the permuted-upper value buffer through
# `amap` (the design.md §4.2 replay, aimed at `Workspace.ux`). Every slot of `ux` is the
# image of exactly one stored lower-triangle entry, so no clearing is needed. Returns
# the largest magnitude seen, the ‖A‖ scale of the LDLᵀ regularization floor (design.md
# §5.1).
function _assemble_upper!(ux::Vector{T}, amap::Vector{Ti}, nzval::Vector{T}) where {T, Ti <: Integer}
    ascale = zero(T)
    for p in eachindex(nzval, amap)
        m = Int(amap[p])
        iszero(m) && continue
        v = nzval[p]
        ux[m] = v
        a = abs(v)
        a > ascale && (ascale = a)
    end
    return ascale
end

# Per-factorization reset: column cursors at the column starts and the dense scatter
# vector cleared.
function _simplicial_reset!(cursor::Vector{Ti}, x::Vector{T}, lcolptr::Vector{Ti}, n::Int) where {T, Ti <: Integer}
    for j in 1:n
        cursor[j] = lcolptr[j]
        x[j] = zero(T)
    end
    return nothing
end

# The kernels' array arguments come from one `Symbolic` and the `Workspace`/factor
# sized from it; these checks pin the sizes the `@inbounds` loops below rely on. The
# index values inside the pattern arrays (rows in 1:n, slots in 1:nnzL) are the
# invariants `l_pattern`/`upper_assembly_map` establish.
function _check_simplicial_sizes(
        x::AbstractVector, invdiag::Vector, cursor::Vector, lcolptr::Vector, lrowval::Vector,
        lx::Vector, ltx::Vector, lrowptr::Vector, lcolval::Vector, ux::Vector, ucolptr::Vector,
        urowval::Vector, n::Int,
    )
    nnzL = Int(lcolptr[n + 1]) - 1
    (
        length(x) == n && length(invdiag) == n && length(cursor) == n &&
            length(lcolptr) == n + 1 && length(lrowptr) == n + 1 &&
            Int(lrowptr[n + 1]) - 1 == nnzL &&
            length(lrowval) == nnzL && length(lcolval) == nnzL &&
            length(lx) == nnzL && length(ltx) == nnzL &&
            length(ucolptr) == n + 1 && length(ux) == length(urowval) == Int(ucolptr[n + 1]) - 1
    ) || throw(DimensionMismatch("simplicial factor arrays are inconsistent with the symbolic analysis"))
    return nothing
end

# Up-looking LLᵀ over rows 1:n. Returns the first row whose pivot is not positive (a
# NaN fails too), or 0 on success. `@inbounds` on the scatter loop: measured on the
# 2D 100×100 Laplacian (Zen 5c, single core) the checked loop runs the whole
# factorization in 4.96 ms against 4.01 ms without the checks, with the pattern
# indices guaranteed by `l_pattern` and the sizes by `_check_simplicial_sizes`.
function _uplooking_llt!(
        lx::Vector{T}, ltx::Vector{T}, invdiag::Vector{T}, x::Vector{T}, cursor::Vector{Ti},
        lcolptr::Vector{Ti}, lrowval::Vector{Ti}, lrowptr::Vector{Ti}, lcolval::Vector{Ti},
        ux::Vector{T}, ucolptr::Vector{Ti}, urowval::Vector{Ti}, n::Int,
    ) where {T, Ti <: Integer}
    _check_simplicial_sizes(x, invdiag, cursor, lcolptr, lrowval, lx, ltx, lrowptr, lcolval, ux, ucolptr, urowval, n)
    for k in 1:n
        for p in Int(ucolptr[k]):(Int(ucolptr[k + 1]) - 1)
            x[Int(urowval[p])] = ux[p]
        end
        d = x[k]
        x[k] = zero(T)
        tend = Int(lrowptr[k + 1]) - 1         # row k's last slot holds its diagonal
        for t in Int(lrowptr[k]):(tend - 1)
            i = Int(lcolval[t])
            p0 = Int(lcolptr[i])
            lki = x[i] * invdiag[i]           # L[k,i] = x[i] / L[i,i]
            x[i] = zero(T)
            @inbounds for p in (p0 + 1):(Int(cursor[i]) - 1)
                x[Int(lrowval[p])] -= lx[p] * lki
            end
            d -= lki * lki
            q = Int(cursor[i])
            cursor[i] = Ti(q + 1)
            lx[q] = lki
            ltx[t] = lki
        end
        d > zero(T) || return k
        ljj = sqrt(d)
        q = Int(cursor[k])
        cursor[k] = Ti(q + 1)
        lx[q] = ljj
        ltx[tend] = ljj
        invdiag[k] = inv(ljj)
    end
    return 0
end

function _cholesky_simplicial!(F::SupernodalFactor{T, Ti}, A::SparseMatrixCSC{T}) where {T, Ti <: Integer}
    sym = F.sym
    n = sym.n
    ws = F.ws
    _assemble_upper!(ws.ux, sym.amap, A.nzval)
    _simplicial_reset!(ws.cursor, ws.rhs, sym.lcolptr, n)
    F.stats.nnzL = sym.nnzL
    F.stats.flops = sym.flops
    fail = _uplooking_llt!(
        F.lx, F.ltx, ws.invdiag, ws.rhs, ws.cursor,
        sym.lcolptr, sym.lrowval, sym.lrowptr, sym.lcolval,
        ws.ux, sym.ucolptr, sym.urowval, n,
    )
    F.stats.fail_col = fail
    F.ok = iszero(fail)
    F.ok || return F
    check_finite(F.lx, "cholesky!")
    return F
end

# Up-looking LDLᵀ with signed regularization (design.md §5.1). Returns
# `(dmax, dmin)`, the extreme |pivot| magnitudes, for the condition estimate.
function _uplooking_ldlt!(
        lx::Vector{T}, ltx::Vector{T}, dvec::Vector{T}, invdiag::Vector{T}, x::Vector{T},
        cursor::Vector{Ti}, lcolptr::Vector{Ti}, lrowval::Vector{Ti}, lrowptr::Vector{Ti},
        lcolval::Vector{Ti}, ux::Vector{T}, ucolptr::Vector{Ti}, urowval::Vector{Ti},
        signs::Vector{Int8}, stats::FactorStats, delta::T, zeta, n::Int,
    ) where {T, Ti <: Integer}
    _check_simplicial_sizes(x, invdiag, cursor, lcolptr, lrowval, lx, ltx, lrowptr, lcolval, ux, ucolptr, urowval, n)
    dmax = zero(T)
    dmin = T(Inf)
    for k in 1:n
        for p in Int(ucolptr[k]):(Int(ucolptr[k + 1]) - 1)
            x[Int(urowval[p])] = ux[p]
        end
        d = x[k]
        x[k] = zero(T)
        tend = Int(lrowptr[k + 1]) - 1
        for t in Int(lrowptr[k]):(tend - 1)
            i = Int(lcolval[t])
            p0 = Int(lcolptr[i])
            xi = x[i]                         # = L[k,i]·d_i
            x[i] = zero(T)
            lki = xi * invdiag[i]
            @inbounds for p in (p0 + 1):(Int(cursor[i]) - 1)   # same measurement as the LLᵀ loop
                x[Int(lrowval[p])] -= lx[p] * xi
            end
            d -= lki * xi                     # L[k,i]²·d_i
            q = Int(cursor[i])
            cursor[i] = Ti(q + 1)
            lx[q] = lki
            ltx[t] = lki
        end
        d = _signed_pivot!(stats, d, signs[k], delta, zeta, dmax)
        dvec[k] = d
        invdiag[k] = inv(d)
        adf = abs(d)
        adf > dmax && (dmax = adf)
        adf < dmin && (dmin = adf)
        q = Int(cursor[k])
        cursor[k] = Ti(q + 1)
        lx[q] = one(T)
        ltx[tend] = one(T)
    end
    return dmax, dmin
end

function _ldlt_simplicial!(F::LDLFactor{T, Ti}, A::SparseMatrixCSC{T}) where {T, Ti <: Integer}
    sym = F.sym
    n = sym.n
    ws = F.ws
    stats = F.stats
    ascale = _assemble_upper!(ws.ux, sym.amap, A.nzval)
    delta = T(LDLT_DELTA) * (iszero(ascale) ? one(T) : ascale)
    zeta = eps(real(T))
    _simplicial_reset!(ws.cursor, ws.rhs, sym.lcolptr, n)
    stats.n_pos = 0
    stats.n_neg = 0
    stats.n_zero = 0
    stats.n_perturbed = 0
    stats.max_perturbation = 0.0
    dmax, dmin = _uplooking_ldlt!(
        F.lx, F.ltx, F.d, ws.invdiag, ws.rhs, ws.cursor,
        sym.lcolptr, sym.lrowval, sym.lrowptr, sym.lcolval,
        ws.ux, sym.ucolptr, sym.urowval, F.signs, stats, delta, zeta, n,
    )
    stats.nnzL = sym.nnzL
    stats.flops = sym.flops
    stats.rcond_est = dmax > zero(T) ? Float64(dmin / dmax) : Inf
    F.ok = true
    check_finite(F.lx, "ldlt!")
    check_finite(F.d, "ldlt!")
    return F
end

# ---------------------------------------------------------------------------
# Triangular solves, factor order. Both sweeps scatter: the forward one down the
# columns of L, the backward one along its rows (`Lᵀ` by columns). `L` is unit-lower
# for an LDLFactor (`unit`; its stored diagonal 1 is skipped) and carries its true
# diagonal, applied through the stored reciprocal, for a SupernodalFactor. The two
# kernels are one function: a CSC of L and a CSC of Lᵀ have the same shape, and only
# the direction of the column loop differs.
# ---------------------------------------------------------------------------

#
# `@inbounds` on the scatter loop: measured on the 2D 100×100 Laplacian (Zen 5c, single
# core) the checked forward sweep takes 115 µs against 91 µs without the checks, a
# fifth of the whole solve. The sizes are checked here; the row indices are the
# pattern's.
# `unit` and `forward` are compile-time flags (`Val`), so each of the four sweeps is
# its own straight loop; as run-time booleans their per-column selects cost a measured
# 20% on the lap2d sweeps.
function _scatter_sweep!(
        y::AbstractVector{T}, colptr::Vector{Ti}, rowval::Vector{Ti}, vals::Vector{T},
        invdiag::Vector{T}, ::Val{unit}, ::Val{forward}, n::Int,
    ) where {T, Ti <: Integer, unit, forward}
    Base.require_one_based_indexing(y)
    (length(y) == n && length(invdiag) == n && length(colptr) == n + 1 && length(rowval) == length(vals) == Int(colptr[n + 1]) - 1) ||
        throw(DimensionMismatch("solve: vector length $(length(y)) does not match the factor (n = $n)"))
    for jj in 1:n
        j = forward ? jj : n + 1 - jj
        p0 = Int(colptr[j])
        p1 = Int(colptr[j + 1]) - 1
        # the diagonal is the first slot of a column of L, the last of a row
        lo = forward ? p0 + 1 : p0
        hi = forward ? p1 : p1 - 1
        yj = y[j]
        if !unit
            yj *= invdiag[j]
            y[j] = yj
        end
        if forward
            @inbounds for p in lo:hi
                y[Int(rowval[p])] -= vals[p] * yj
            end
        else
            # Walked downward so that, with the rows also visited downward, the pattern
            # and value arrays are read as one descending stream rather than a
            # descending sequence of short ascending runs: measured 119 against 100 µs
            # on the lap2d backward sweep (the order within a row is free here).
            @inbounds for p in hi:-1:lo
                y[Int(rowval[p])] -= vals[p] * yj
            end
        end
    end
    return y
end

_unitdiag(::SupernodalFactor) = Val(false)
_unitdiag(::LDLFactor) = Val(true)

function _solve_L_csc!(y::AbstractVector{T}, F::_PanelFactor{T, Ti}) where {T, Ti <: Integer}
    sym = F.sym
    return _scatter_sweep!(y, sym.lcolptr, sym.lrowval, F.lx, F.ws.invdiag, _unitdiag(F), Val(true), sym.n)
end

function _solve_Lt_csc!(y::AbstractVector{T}, F::_PanelFactor{T, Ti}) where {T, Ti <: Integer}
    sym = F.sym
    return _scatter_sweep!(y, sym.lrowptr, sym.lcolval, F.ltx, F.ws.invdiag, _unitdiag(F), Val(false), sym.n)
end

function _solve_L_csc!(y::AbstractMatrix{T}, F::_PanelFactor{T, Ti}) where {T, Ti <: Integer}
    for c in axes(y, 2)
        _solve_L_csc!(view(y, :, c), F)
    end
    return y
end

function _solve_Lt_csc!(y::AbstractMatrix{T}, F::_PanelFactor{T, Ti}) where {T, Ti <: Integer}
    for c in axes(y, 2)
        _solve_Lt_csc!(view(y, :, c), F)
    end
    return y
end
