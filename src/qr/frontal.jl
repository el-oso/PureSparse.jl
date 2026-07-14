# M5b multifrontal QR numeric phase (docs/design_qr_m5b.md §A4.3/§A4.4/§A5/§A6).
# Front factorization uses the LAPACK reflector convention (H = I - tau*v*vᵀ, v[pivot]=1
# implicit) throughout — the convention PureBLAS's `wy_t!`/`wy_apply!` (P1a/P1b) expect
# (design_qr_m5b.md §A7.1) and the same one PureBLAS's own `qr_unblocked!`/`geqrf!` use
# internally for the actual reflector math (their OWN *stored* `tau` array afterward uses
# an inverted "faer" convention — irrelevant here since this file forms its own reflectors
# directly, never touching that stored-array convention).

"""
    QRFrontWorkspace{T,Ti<:Integer}

Preallocated scratch for [`QRFrontFactor`](@ref)'s numeric loop, sized once from
[`QRFrontSymbolic`](@ref)'s capacity scalars (design_qr_m5b.md §A4.4) — zero
allocations after construction.
"""
struct QRFrontWorkspace{T,Ti<:Integer}
    g2l::Vector{Ti}
    cg2l::Vector{Ti}
    bucket::Vector{Ti}
    stair::Vector{Ti}
    wy::WYApplyWorkspace{T}
    tau_panel::Vector{T}
    Tm::Matrix{T}
    yqt::Vector{T}
    rhs::Vector{T}
end

function QRFrontWorkspace{T,Ti}(fsym::QRFrontSymbolic{Ti}) where {T,Ti<:Integer}
    nb = length(fsym.base.parent)
    NB = fsym.max_front_cols == 0 ? 1 : qr_block_size(fsym.max_front_rows, fsym.max_front_cols)
    return QRFrontWorkspace{T,Ti}(
        zeros(Ti, nb),
        zeros(Ti, nb),
        Vector{Ti}(undef, max(fsym.max_front_cols + 1, 1)),
        Vector{Ti}(undef, max(fsym.max_front_cols, 1)),
        WYApplyWorkspace{T}(max(fsym.max_front_rows, 1), max(NB, 1), max(fsym.max_front_cols, 1)),
        Vector{T}(undef, max(NB, 1)),
        Matrix{T}(undef, max(NB, 1), max(NB, 1)),
        Vector{T}(undef, max(fsym.max_front_rows, 1)),
        Vector{T}(undef, fsym.base.m),
    )
end

"""
    QRFrontFactor{T<:Real,Ti<:Integer} <: AbstractSparseFactor{T}

Multifrontal sparse QR factor (design_qr_m5b.md §A4.3): dense per-front storage, one
rectangle per front laid contiguously in `fval` at `fsym.fvalptr` offsets. Produced by
[`qr_frontal`](@ref)/refactored in place by [`qr!`](@ref) (zero allocations after the
first call, mirroring [`QRFactor`](@ref)'s own contract).
"""
mutable struct QRFrontFactor{T<:Real,Ti<:Integer} <: AbstractSparseFactor{T}
    fsym::QRFrontSymbolic{Ti}
    fval::Vector{T}
    ftau::Vector{T}
    tauv::Vector{T}
    frowind::Vector{Ti}
    fmincol::Vector{Ti}
    fm::Vector{Ti}
    fr::Vector{Ti}
    fe::Vector{Ti}   # e_f = total reflectors formed (pivotal + non-pivotal) this front;
                      # rows (fe+1):fm are mathematically all-zero residue (more rows
                      # than columns), NOT survivors — pass-up only sets fmincol for
                      # (fr+1):fe, so the child-gather loop must stop at fe, not fm
                      # (reading past it hits garbage assembly-time LOCAL mincols, a
                      # real bug: BoundsError/segfault when misread as a global column).
    fnpanel::Vector{Ti}
    pnrows::Vector{Ti}
    pncols::Vector{Ti}
    pbs::Vector{Ti}
    elimcol::Vector{Ti}   # same indexing/capacity as tauv: elimcol[t] = the FRONT-LOCAL
                          # column of elimination t — needed because a panel's alive
                          # reflectors are not always column-contiguous (a dead-skipped
                          # pivotal column breaks contiguity), so solve-phase replay
                          # cannot recover "which columns" from a simple range
    fpivotrow::Vector{Ti}
    rval::Vector{T}
    rowval::Vector{T}
    ws::QRFrontWorkspace{T,Ti}
    stats::QRStats
    ok::Bool
end

function QRFrontFactor{T,Ti}(fsym::QRFrontSymbolic{Ti}) where {T,Ti<:Integer}
    nb = length(fsym.base.parent)
    nz = length(fsym.rowptr) > 0 ? Int(fsym.rowptr[end] - 1) : 0
    tauv_cap = 0
    @inbounds for f in 1:fsym.nfront
        n_f = Int(fsym.fcolptr[f + 1] - fsym.fcolptr[f])
        tauv_cap += min(Int(fsym.fmmax[f]), n_f)
    end
    return QRFrontFactor{T,Ti}(
        fsym,
        Vector{T}(undef, fsym.nnzVF),
        Vector{T}(undef, Int(fsym.ftauptr[end] - 1)),
        Vector{T}(undef, max(tauv_cap, 1)),
        Vector{Ti}(undef, Int(fsym.frowptr2[end] - 1)),
        Vector{Ti}(undef, Int(fsym.frowptr2[end] - 1)),
        zeros(Ti, fsym.nfront),
        zeros(Ti, fsym.nfront),
        zeros(Ti, fsym.nfront),
        zeros(Ti, fsym.nfront),
        Vector{Ti}(undef, Int(fsym.fpanelptr[end] - 1)),
        Vector{Ti}(undef, Int(fsym.fpanelptr[end] - 1)),
        Vector{Ti}(undef, Int(fsym.fpanelptr[end] - 1)),
        Vector{Ti}(undef, max(tauv_cap, 1)),
        zeros(Ti, nb),
        zeros(T, fsym.nnzRF),
        Vector{T}(undef, nz),
        QRFrontWorkspace{T,Ti}(fsym),
        QRStats(),
        true,
    )
end

# ── column-local helpers ──────────────────────────────────────────────────────────

# LAPACK dlarfg (real): forms the reflector for Ff[lo:hi, jj] in place (v[lo]=1
# implicit, essential below stored at Ff[lo+1:hi,jj], R diagonal at Ff[lo,jj]).
# `xnorm` (the FULL column norm, precomputed by the caller for the dead-pivot test) is
# reused here rather than recomputed. Returns tau (LAPACK convention).
@inline function _front_form_reflector!(Ff::AbstractMatrix{T}, lo::Int, hi::Int, jj::Int, xnorm::T) where {T}
    alpha = Ff[lo, jj]
    s = alpha >= zero(T) ? one(T) : -one(T)   # sign(0) := +1
    beta = -s * xnorm
    tau = (beta - alpha) / beta
    invdenom = one(T) / (alpha - beta)
    @inbounds for i in (lo + 1):hi
        Ff[i, jj] *= invdenom
    end
    Ff[lo, jj] = beta
    return tau
end

# Applies the reflector at column `jj` (v[lo]=1 implicit, Ff[lo+1:hi,jj] essential,
# tau) to column `jcol`, rows lo:hi — the in-panel rank-1 apply (design_qr_m5b.md
# §A5.3's step (a), mirroring PureBLAS's own `_qr_apply1_f64!` structure).
@inline function _front_apply1!(Ff::AbstractMatrix{T}, lo::Int, hi::Int, jj::Int, jcol::Int, tau::T) where {T}
    dot = Ff[lo, jcol]
    @inbounds for i in (lo + 1):hi
        dot += Ff[i, jj] * Ff[i, jcol]
    end
    kk = -tau * dot
    Ff[lo, jcol] += kk
    @inbounds for i in (lo + 1):hi
        Ff[i, jcol] += kk * Ff[i, jj]
    end
    return nothing
end

# Panel column extent: grow from `j` while width < NB, or the staircase jump from the
# panel's own start exceeds max(1, NB÷2) (faer's split rule, qr.rs — cited heuristic,
# design_qr_m5b.md §A5.3, adopted as-is; a free tunable, revisit only on measurement).
@inline function _panel_extent(j::Int, n_f::Int, stair::AbstractVector{Ti}, NB::Int) where {Ti}
    j2 = j
    stair0 = stair[j]
    @inbounds while j2 < n_f
        (j2 - j + 1) >= NB && break
        (stair[j2 + 1] - stair0) >= max(1, NB ÷ 2) && break
        j2 += 1
    end
    return j2 + 1   # exclusive upper bound, matching design's [j, j2) convention
end

@inline n_f_of(fsym::QRFrontSymbolic, f::Int) = Int(fsym.fcolptr[f + 1] - fsym.fcolptr[f])

# Front f's mmax_f × n_f rectangle reshaped from the flat `fval` storage at its
# `fvalptr` offset — shared by assembly, factorization, and solve replay.
@inline function _front_view(F::QRFrontFactor{T,Ti}, f::Int) where {T,Ti<:Integer}
    fsym = F.fsym
    mmax_f = Int(fsym.fmmax[f])
    n_f = n_f_of(fsym, f)
    lo = Int(fsym.fvalptr[f])
    return reshape(view(F.fval, lo:(lo + mmax_f * n_f - 1)), mmax_f, n_f)
end

include("frontal_assemble.jl")
include("frontal_numeric.jl")
include("frontal_solve.jl")
