# Supernodal triangular solves (design.md §4.4). Correctness-first pass; zero-allocation
# hardening is M1 task list item 7 (a deliberately separate, later step — design.md
# ROADMAP.md M1 task breakdown), so this allocates a permuted-RHS scratch buffer per call
# for now rather than retrofitting `Workspace` under time pressure.
#
# Both supernodal factor types share the identical panel layout (design.md §1.2), so the
# L/Lᵀ sweeps are written once over this union; the only difference is the diagonal:
# LLᵀ panels store L's true diagonal (trsm diag='N'), LDLᵀ panels store unit-lower L
# (trsm diag='U') plus the separate D solved by `solve_D!`.
const _PanelFactor{T,Ti} = Union{SupernodalFactor{T,Ti},LDLFactor{T,Ti}}
_diagchar(::SupernodalFactor) = 'N'
_diagchar(::LDLFactor) = 'U'

"""
    solve!(x::AbstractVecOrMat, F::Union{SupernodalFactor,LDLFactor}, b::AbstractVecOrMat) -> x

Solve `P·A·Pᵀ·y = P·b` via the supernodal factor, then unpermute: `x = Pᵀ·y` solves
`A·x = b`. For an [`LDLFactor`](@ref) the diagonal stage `solve_D!` runs between the two
triangular sweeps (design.md §4.4). `x` and `b` may alias. Zero allocations for the
single-RHS (`b::AbstractVector`) form (the permuted-RHS scratch is `F.ws.rhs`, sized
`n` once in `Workspace`); multi-RHS (`b::AbstractMatrix`) still allocates a fresh
`n×nrhs` scratch per call — `nrhs` is caller-chosen and unbounded, so it cannot be
pre-sized (CLAUDE.md requirement 5 targets the per-factor hot path, which for an
interior-point consumer is single-RHS refactor+solve, not a growing multi-RHS count).
"""
function solve!(x::AbstractVector{T}, F::_PanelFactor{T,Ti}, b::AbstractVector{T}) where {T,Ti<:Integer}
    sym = F.sym
    n = sym.n
    perm = sym.perm
    y = F.ws.rhs

    @inbounds for k in 1:n
        y[k] = b[Int(perm[k])]
    end

    _solve_L!(y, F)
    F isa LDLFactor && _solve_D!(y, F)
    _solve_Lt!(y, F)

    @inbounds for k in 1:n
        x[Int(perm[k])] = y[k]
    end
    return x
end

function solve!(x::AbstractMatrix{T}, F::_PanelFactor{T,Ti}, b::AbstractMatrix{T}) where {T,Ti<:Integer}
    sym = F.sym
    n = sym.n
    perm = sym.perm
    nrhs = size(b, 2)
    y = Matrix{T}(undef, n, nrhs)

    @inbounds for k in 1:n, c in 1:nrhs
        y[k, c] = b[Int(perm[k]), c]
    end

    _solve_L!(y, F)
    F isa LDLFactor && _solve_D!(y, F)
    _solve_Lt!(y, F)

    @inbounds for k in 1:n, c in 1:nrhs
        x[Int(perm[k]), c] = y[k, c]
    end
    return x
end

"""
    (F::Union{SupernodalFactor,LDLFactor}) \\ b -> x

Convenience allocating wrapper over [`solve!`](@ref).
"""
Base.:\(F::_PanelFactor{T}, b::AbstractVector{T}) where {T} = solve!(similar(b), F, b)
Base.:\(F::_PanelFactor{T}, b::AbstractMatrix{T}) where {T} = solve!(similar(b), F, b)

"""
    solve_L!(y::AbstractVecOrMat, F::Union{SupernodalFactor,LDLFactor})

Forward solve `L·y := y` in place, in FACTOR ordering (`y` already permuted — see
[`solve!`](@ref) for the full `A·x=b` solve). Exported split solve (design.md §4.4/§6).
For an `LDLFactor`, `L` is unit-lower.
"""
solve_L!(y::AbstractVecOrMat, F::_PanelFactor) = _solve_L!(y, F)

"""
    solve_Lt!(y::AbstractVecOrMat, F::Union{SupernodalFactor,LDLFactor})

Backward solve `Lᵀ·y := y` in place, in FACTOR ordering. Exported split solve.
"""
solve_Lt!(y::AbstractVecOrMat, F::_PanelFactor) = _solve_Lt!(y, F)

"""
    solve_D!(y::AbstractVecOrMat, F::LDLFactor)

Diagonal solve `D·y := y` in place, in FACTOR ordering (design.md §5.2/§6) — the middle
stage of the LDLᵀ `solve!`, exported for iterative refinement and split-solve consumers.
"""
solve_D!(y::AbstractVecOrMat, F::LDLFactor) = _solve_D!(y, F)

function _solve_D!(y::Vector{T}, F::LDLFactor{T,Ti}) where {T,Ti<:Integer}
    d = F.d
    @inbounds for k in 1:F.sym.n
        y[k] /= d[k]
    end
    return y
end

function _solve_D!(y::Matrix{T}, F::LDLFactor{T,Ti}) where {T,Ti<:Integer}
    d = F.d
    nrhs = size(y, 2)
    @inbounds for k in 1:F.sym.n
        invd = inv(d[k])
        for c in 1:nrhs
            y[k, c] *= invd
        end
    end
    return y
end

function _solve_L!(y::Vector{T}, F::_PanelFactor{T,Ti}) where {T,Ti<:Integer}
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
        trsm!(yblk, Ldiag; side = 'L', uplo = 'L', transA = 'N', diag = _diagchar(F), alpha = one(T))

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

function _solve_L!(y::Matrix{T}, F::_PanelFactor{T,Ti}) where {T,Ti<:Integer}
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
        trsm!(yblk, Ldiag; side = 'L', uplo = 'L', transA = 'N', diag = _diagchar(F), alpha = one(T))

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

function _solve_Lt!(y::Vector{T}, F::_PanelFactor{T,Ti}) where {T,Ti<:Integer}
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
        trsm!(yblk, Ldiag; side = 'L', uplo = 'L', transA = 'T', diag = _diagchar(F), alpha = one(T))
    end
    end
    return y
end

function _solve_Lt!(y::Matrix{T}, F::_PanelFactor{T,Ti}) where {T,Ti<:Integer}
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
        trsm!(yblk, Ldiag; side = 'L', uplo = 'L', transA = 'T', diag = _diagchar(F), alpha = one(T))
    end
    end
    return y
end
