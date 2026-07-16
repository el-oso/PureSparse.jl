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
    # _assemble_front!'s own scratch (design_qr_m5b.md §A5.2 step 2/3), preallocated
    # to `max_front_rows`/`max_front_cols` capacity so assembly no longer allocates a
    # fresh `push!`-grown Vector per front — a real per-front allocation cost flagged
    # as a known gap since task 16a landed, closed here (task 16e's first lever).
    phys::Vector{Ti}
    mincol::Vector{Ti}
    srcfront::Vector{Ti}
    srcrow::Vector{Ti}
    slotof::Vector{Ti}
    acursor::Vector{Ti}
    # _factorize_front!'s own per-front scratch (same task-16e lever as above)
    elim_col::Vector{Ti}
    # solve!'s own R-space scratch (§A6) — sized once to nb, not per call
    solve_cc::Vector{T}
    # apply_Q!'s own per-front panel-boundary scratch (§A6) — worst case one panel
    # per column, so capacity ncols+1 covers every front.
    row0s::Vector{Int}
    tvbases::Vector{Int}
end

function QRFrontWorkspace{T,Ti}(fsym::QRFrontSymbolic{Ti}) where {T,Ti<:Integer}
    nb = length(fsym.base.parent)
    # Read the ONE NB the symbolic phase committed to (fsym.nb) — do NOT recompute it
    # here. The ftau T-slab was budgeted with exactly this value; sizing Tm / the panel
    # cap from a separately-recomputed NB is what let the two diverge and overflow ftau
    # (see the `nb` field's doc comment on QRFrontSymbolic). `fsym.nb == qr_block_size(
    # max_front_rows, max_front_cols)` by construction, so this is identical in value —
    # but now sourced from a single place, so it cannot drift.
    NB = fsym.nb
    mrows = max(fsym.max_front_rows, 1)
    ncols = max(fsym.max_front_cols, 1)
    return QRFrontWorkspace{T,Ti}(
        zeros(Ti, nb),
        zeros(Ti, nb),
        Vector{Ti}(undef, ncols + 1),
        Vector{Ti}(undef, ncols),
        WYApplyWorkspace{T}(mrows, max(NB, 1), ncols),
        Vector{T}(undef, max(NB, 1)),
        Matrix{T}(undef, max(NB, 1), max(NB, 1)),
        Vector{T}(undef, mrows),
        Vector{T}(undef, fsym.base.m),
        Vector{Ti}(undef, mrows),
        Vector{Ti}(undef, mrows),
        Vector{Ti}(undef, mrows),
        Vector{Ti}(undef, mrows),
        Vector{Ti}(undef, mrows),
        Vector{Ti}(undef, ncols),
        Vector{Ti}(undef, mrows),
        Vector{T}(undef, nb),
        Vector{Int}(undef, ncols + 1),
        Vector{Int}(undef, ncols + 1),
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
    fscalar::Vector{Bool}   # true: this front used the small-front scalar fallback
                            # (tuning.jl QR_FRONTAL_UNBLOCKED_THRESHOLD, faer's own
                            # qr_in_place_unblocked precedent) — no WY panels stored
                            # for it (fnpanel[f]==0), solve replay walks F.elimcol/
                            # F.tauv column-by-column instead of via _gather_panel_V!/
                            # wy_apply!. Distinct from "fnpanel==0 because every column
                            # was dead" (a legitimate blocked-path outcome) — checking
                            # this flag rather than fnpanel==0 avoids that ambiguity.
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
        zeros(Bool, fsym.nfront),
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
    @inbounds @simd for i in (lo + 1):hi
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
    # a dot-product reduction (@simd's textbook case: Julia's @simd permits the
    # float-add reordering this needs to vectorize, the same relaxation any BLAS-1
    # dot-product kernel takes — LLT.jl's own hot loop already establishes this
    # pattern in this codebase, src/numeric/llt.jl:74).
    @inbounds @simd for i in (lo + 1):hi
        dot += Ff[i, jj] * Ff[i, jcol]
    end
    kk = -tau * dot
    Ff[lo, jcol] += kk
    @inbounds @simd for i in (lo + 1):hi
        Ff[i, jcol] += kk * Ff[i, jj]
    end
    return nothing
end

# faer's `recommended_block_size` tier function (faer 0.24.1,
# src/linalg/qr/no_pivoting/factor.rs:91-116), RE-translated here after a first
# over-correction removed it entirely (see `_factorize_front!`'s header). Ground truth
# re-read from faer's actual source (qr.rs:609-613 symbolic-time call site,
# qr.rs:1260-1265 the split-trigger use): faer calls this tier function TWICE, for two
# genuinely different purposes, easy to conflate —
#   1. qr.rs:609-613 (symbolic time): `max_block_size[s] = recommended_block_size(
#      s_row_count, s_col_count)` — a PER-FRONT value from the front's actual row/col
#      counts. THIS is what feeds the split-trigger threshold at qr.rs:1260-1265
#      (`max_block_size / 2`) — i.e. how far a row's min-col may jump before a group
#      ends is itself SIZE-DEPENDENT (a huge front tolerates bigger jumps before
#      splitting). `_factorize_front!` calls this ONCE per front as `NBf`.
#   2. qr.rs:1276-1277 (numeric time, INSIDE the trigger): `bs = recommended_block_size(
#      left.nrows(), left.ncols())` re-derived PER GROUP, clamped to `max_block_size`,
#      and fed ONLY into `qr_in_place`'s own internal recursive dense-kernel blocking
#      (out of scope, §A7.4 — PureBLAS's single-level `wy_t!`/`wy_apply!` substitute
#      for that whole call) — this is the value the first over-correction wrongly
#      conflated with (1) and, in the ORIGINAL buggy draft, wrongly used to sub-split a
#      single stored group into multiple small WY blocks (a real bug, correctly
#      reverted). Only (2) has no counterpart here; (1) does and was wrongly dropped
#      alongside it. See `_factorize_front!`'s header for the full account.
@inline function _qr_faer_block_size(nrows::Int, ncols::Int)
    prod = nrows * ncols
    sz = min(nrows, ncols)
    bs = prod > 8192 * 8192 ? 256 :
        prod > 2048 * 2048 ? 128 :
        prod > 1024 * 1024 ? 64 :
        prod > 512 * 512 ? 48 :
        prod > 128 * 128 ? 32 :
        prod > 32 * 32 ? 8 :
        prod > 16 * 16 ? 4 : 1
    return max(min(bs, sz), 1)
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
