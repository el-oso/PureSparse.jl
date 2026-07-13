# Supernodal triangular solves (design.md §4.4). Correctness-first pass; zero-allocation
# hardening is M1 task list item 7 (a deliberately separate, later step — design.md
# ROADMAP.md M1 task breakdown), so this allocates a permuted-RHS scratch buffer per call
# for now rather than retrofitting `Workspace` under time pressure.

"""
    solve!(x::AbstractVecOrMat, F::SupernodalFactor, b::AbstractVecOrMat) -> x

Solve `P·A·Pᵀ·y = P·b` via the supernodal `L`, then unpermute: `x = Pᵀ·y` solves `A·x = b`.
`x` and `b` may alias. Design.md §4.4.
"""
function solve!(x::AbstractVecOrMat{T}, F::SupernodalFactor{T,Ti}, b::AbstractVecOrMat{T}) where {T,Ti<:Integer}
    sym = F.sym
    n = sym.n
    perm = sym.perm
    nrhs = ndims(b) == 1 ? 1 : size(b, 2)
    y = ndims(b) == 1 ? Vector{T}(undef, n) : Matrix{T}(undef, n, nrhs)

    @inbounds for k in 1:n
        if ndims(b) == 1
            y[k] = b[Int(perm[k])]
        else
            for c in 1:nrhs
                y[k, c] = b[Int(perm[k]), c]
            end
        end
    end

    _solve_L!(y, F)
    _solve_Lt!(y, F)

    @inbounds for k in 1:n
        if ndims(b) == 1
            x[Int(perm[k])] = y[k]
        else
            for c in 1:nrhs
                x[Int(perm[k]), c] = y[k, c]
            end
        end
    end
    return x
end

"""
    (F::SupernodalFactor) \\ b -> x

Convenience allocating wrapper over [`solve!`](@ref).
"""
Base.:\(F::SupernodalFactor{T}, b::AbstractVector{T}) where {T} = solve!(similar(b), F, b)
Base.:\(F::SupernodalFactor{T}, b::AbstractMatrix{T}) where {T} = solve!(similar(b), F, b)

"""
    solve_L!(y::AbstractVecOrMat, F::SupernodalFactor)

Forward solve `L·y := y` in place, in FACTOR ordering (`y` already permuted — see
[`solve!`](@ref) for the full `A·x=b` solve). Exported split solve (design.md §4.4/§6).
"""
solve_L!(y::AbstractVecOrMat, F::SupernodalFactor) = _solve_L!(y, F)

"""
    solve_Lt!(y::AbstractVecOrMat, F::SupernodalFactor)

Backward solve `Lᵀ·y := y` in place, in FACTOR ordering. Exported split solve.
"""
solve_Lt!(y::AbstractVecOrMat, F::SupernodalFactor) = _solve_Lt!(y, F)

function _solve_L!(y::Vector{T}, F::SupernodalFactor{T,Ti}) where {T,Ti<:Integer}
    sym = F.sym
    nsuper = sym.nsuper
    super = sym.super
    rowind_ptr = sym.rowind_ptr
    rowind = sym.rowind
    px = sym.px
    x = F.x

    GC.@preserve x y begin
    @inbounds for s in 1:nsuper
        j0 = Int(super[s])
        j1 = Int(super[s + 1]) - 1
        nscol = j1 - j0 + 1
        rp0 = Int(rowind_ptr[s])
        nsrow = Int(rowind_ptr[s + 1]) - rp0
        panel = _panelview(x, Int(px[s]), nsrow, nscol)
        Ldiag = view(panel, 1:nscol, 1:nscol)
        yblk = _panelview(y, j0, nscol, 1)
        trsm!(yblk, Ldiag; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = one(T))

        if nsrow > nscol
            Lbelow = view(panel, (nscol + 1):nsrow, 1:nscol)
            upd = Matrix{T}(undef, nsrow - nscol, 1)
            gemm!(upd, Lbelow, yblk; transA = 'N', transB = 'N', alpha = -one(T), beta = zero(T))
            for a in 1:(nsrow - nscol)
                ra = _row(rowind, rp0, nscol + a)
                y[ra] += upd[a, 1]
            end
        end
    end
    end
    return y
end

function _solve_L!(y::Matrix{T}, F::SupernodalFactor{T,Ti}) where {T,Ti<:Integer}
    sym = F.sym
    nsuper = sym.nsuper
    super = sym.super
    rowind_ptr = sym.rowind_ptr
    rowind = sym.rowind
    px = sym.px
    x = F.x
    nrhs = size(y, 2)

    GC.@preserve x begin
    @inbounds for s in 1:nsuper
        j0 = Int(super[s])
        j1 = Int(super[s + 1]) - 1
        nscol = j1 - j0 + 1
        rp0 = Int(rowind_ptr[s])
        nsrow = Int(rowind_ptr[s + 1]) - rp0
        panel = _panelview(x, Int(px[s]), nsrow, nscol)
        Ldiag = view(panel, 1:nscol, 1:nscol)
        yblk = view(y, j0:j1, :)
        trsm!(yblk, Ldiag; side = 'L', uplo = 'L', transA = 'N', diag = 'N', alpha = one(T))

        if nsrow > nscol
            Lbelow = view(panel, (nscol + 1):nsrow, 1:nscol)
            upd = Matrix{T}(undef, nsrow - nscol, nrhs)
            gemm!(upd, Lbelow, yblk; transA = 'N', transB = 'N', alpha = -one(T), beta = zero(T))
            for a in 1:(nsrow - nscol)
                ra = _row(rowind, rp0, nscol + a)
                for c in 1:nrhs
                    y[ra, c] += upd[a, c]
                end
            end
        end
    end
    end
    return y
end

function _solve_Lt!(y::Vector{T}, F::SupernodalFactor{T,Ti}) where {T,Ti<:Integer}
    sym = F.sym
    nsuper = sym.nsuper
    super = sym.super
    rowind_ptr = sym.rowind_ptr
    rowind = sym.rowind
    px = sym.px
    x = F.x

    GC.@preserve x y begin
    @inbounds for s in nsuper:-1:1
        j0 = Int(super[s])
        j1 = Int(super[s + 1]) - 1
        nscol = j1 - j0 + 1
        rp0 = Int(rowind_ptr[s])
        nsrow = Int(rowind_ptr[s + 1]) - rp0
        panel = _panelview(x, Int(px[s]), nsrow, nscol)
        Ldiag = view(panel, 1:nscol, 1:nscol)
        yblk = _panelview(y, j0, nscol, 1)

        if nsrow > nscol
            Lbelow = view(panel, (nscol + 1):nsrow, 1:nscol)
            gathered = Matrix{T}(undef, nsrow - nscol, 1)
            for a in 1:(nsrow - nscol)
                ra = _row(rowind, rp0, nscol + a)
                gathered[a, 1] = y[ra]
            end
            gemm!(yblk, Lbelow, gathered; transA = 'T', transB = 'N', alpha = -one(T), beta = one(T))
        end
        trsm!(yblk, Ldiag; side = 'L', uplo = 'L', transA = 'T', diag = 'N', alpha = one(T))
    end
    end
    return y
end

function _solve_Lt!(y::Matrix{T}, F::SupernodalFactor{T,Ti}) where {T,Ti<:Integer}
    sym = F.sym
    nsuper = sym.nsuper
    super = sym.super
    rowind_ptr = sym.rowind_ptr
    rowind = sym.rowind
    px = sym.px
    x = F.x
    nrhs = size(y, 2)

    GC.@preserve x begin
    @inbounds for s in nsuper:-1:1
        j0 = Int(super[s])
        j1 = Int(super[s + 1]) - 1
        nscol = j1 - j0 + 1
        rp0 = Int(rowind_ptr[s])
        nsrow = Int(rowind_ptr[s + 1]) - rp0
        panel = _panelview(x, Int(px[s]), nsrow, nscol)
        Ldiag = view(panel, 1:nscol, 1:nscol)
        yblk = view(y, j0:j1, :)

        if nsrow > nscol
            Lbelow = view(panel, (nscol + 1):nsrow, 1:nscol)
            gathered = Matrix{T}(undef, nsrow - nscol, nrhs)
            for a in 1:(nsrow - nscol)
                ra = _row(rowind, rp0, nscol + a)
                for c in 1:nrhs
                    gathered[a, c] = y[ra, c]
                end
            end
            gemm!(yblk, Lbelow, gathered; transA = 'T', transB = 'N', alpha = -one(T), beta = one(T))
        end
        trsm!(yblk, Ldiag; side = 'L', uplo = 'L', transA = 'T', diag = 'N', alpha = one(T))
    end
    end
    return y
end
