# Multifrontal solve phase (design_qr_m5b.md §A6): replays stored front panels against
# a shared physical-row-space work vector, postorder for Qᵀ, reverse postorder for Q.
#
# faer correspondence (`SupernodalQrRef::apply_Q_transpose_in_place_with_conj`,
# qr.rs:719-819, and `solve_in_place_with_conj`, qr.rs:824-883): the per-front replay
# below walks the SAME stored block-descriptor records the factorization wrote
# (faer's tau_block_size/householder_nrows/householder_ncols triple, qr.rs:786-800 —
# here pbs/pnrows/pncols) in the same order, gathering the front's rows out of the
# shared vector, applying each stored block's V/T, and scattering back
# (qr.rs:761-810). Deviations, each forced by PureSparse machinery faer lacks:
#   - faer reads each block's V in place from the stored front
#     (s_H.submatrix(start, start, nrows, ncols), qr.rs:789); here V is gathered into
#     an explicit-unit copy because (a) PureBLAS's `wy_apply!` contract requires it
#     (wy.jl header) and (b) dead-pivot skipping (which faer has none of) makes a
#     block's live reflector columns non-contiguous — `F.elimcol` recovers them.
#   - faer stages all fronts' rows in an m×k tmp at supernode-start offsets and reads
#     R-space components out of tmp[0:n] at the end (qr.rs:757-818); here `solve!`
#     reads them via `fpivotrow` (the dead-pivot-aware equivalent of that readout).
#   - faer's `solve_in_place` back-substitutes R via blocked supernodal panels of its
#     L=Rᵀ storage (qr.rs:848-882); `solve_R!`/`solve_Rt!` substitute over the padded
#     row storage scalar-wise (rval has no strided column-major panel to hand to a
#     BLAS-3 kernel — a storage-layer difference, §A5.5, not an orchestration one).
#
# A panel's alive reflectors are NOT generally column-contiguous within the front (a
# dead-skipped pivotal column breaks contiguity), so V cannot be recovered as a simple
# view into the front rectangle — `F.elimcol` (front-local column per elimination,
# same indexing as `F.tauv`) is used to GATHER the correct (possibly scattered)
# columns into `ws.wy.V` fresh, exactly mirroring how factorization built it.
# (`_tauv_base` — the per-elimination τ base offset — now lives in `frontal.jl`, shared
# with the numeric factor loop.)

# Gather panel `pnl`'s V (mp×pb, explicit unit diagonal) into `ws.wy.V` from the
# front's stored rectangle, using `F.elimcol`/`F.tauv` (NOT a column-range view — see
# module header). `row0`/`mp` = the panel's row extent (as recorded by factorization).
function _gather_panel_V!(F::QRFrontFactor{T,Ti}, f::Int, pnl::Int, row0::Int, mp::Int, tvbase::Int) where {T,Ti<:Integer}
    fsym = F.fsym
    ws = F.ws
    Ff = _front_view(F, f)
    pb = Int(F.pbs[Int(fsym.fpanelptr[f]) + pnl - 1])
    Vv = view(ws.wy.V, 1:mp, 1:pb)
    fill!(Vv, zero(T))
    @inbounds for c in 1:pb
        jj = Int(F.elimcol[tvbase + c - 1])
        # this reflector's own local row (relative to the FRONT, 1-based) is
        # `row0 + c - 1` (eliminations within a panel are consumed in row order,
        # matching factorization's own k-advance) — its essential data spans
        # Ff[row0+c-1+1 : row0+mp-1, jj] (up to the panel's own row extent).
        krow = row0 + c - 1
        Vv[c, c] = one(T)
        @simd for i in (krow + 1):(row0 + mp - 1)
            Vv[i - row0 + 1, c] = Ff[i, jj]
        end
    end
    return Vv
end

# Scalar-front replay (F.fscalar[f], tuning.jl QR_FRONTAL_UNBLOCKED_THRESHOLD):
# elimination t's reflector applied directly from Ff (never gathered into a V
# matrix — there is no stored T/panel for a scalar front to gather against). Uses
# the FULL m_f row range rather than tracking each column's own row extent: Ff[i,jj]
# for i beyond that column's true structural support is exactly zero (assembly's
# zero-init, `_assemble_front!` step 4, never touched past there by anything else),
# so the reflector's implicit v is correctly zero there too — extending the range to
# m_f changes no arithmetic, only adds provably-zero terms, and needs no extra
# per-elimination row-extent bookkeeping (`stair` is a transient, per-front-during-
# qr! workspace array, stale by solve time — the same reasoning `_gather_panel_V!`
# already relies on `F.pnrows`, not `stair`, for the blocked case). Each H_t is
# self-adjoint (H = I - tau·v·vᵀ, real, symmetric), so the SAME formula applies for
# both apply_Qt! and apply_Q! — only the iteration order over t differs.
@inline function _scalar_apply_to_vec!(yqt::AbstractVector{T}, Ff::AbstractMatrix{T}, t::Int, m_f::Int, jj::Int, local_tau::T) where {T}
    local_tau == zero(T) && return nothing
    dot = yqt[t]
    @inbounds @simd for i in (t + 1):m_f
        dot += Ff[i, jj] * yqt[i]
    end
    kk = -local_tau * dot
    yqt[t] += kk
    @inbounds @simd for i in (t + 1):m_f
        yqt[i] += kk * Ff[i, jj]
    end
    return nothing
end

@inline function _panel_T(F::QRFrontFactor{T,Ti}, f::Int, pnl::Int) where {T,Ti<:Integer}
    fsym = F.fsym
    panello = Int(fsym.fpanelptr[f])
    base = Int(fsym.ftauptr[f])
    @inbounds for p in 1:(pnl - 1)
        pb = Int(F.pbs[panello + p - 1])
        base += pb * pb
    end
    pb = Int(F.pbs[panello + pnl - 1])
    return reshape(view(F.ftau, base:(base + pb * pb - 1)), pb, pb)
end

"""
    apply_Qt!(y::AbstractVector, F::QRFrontFactor) -> y

`y ← Qᵀy` in place (`y` has length `F.fsym.base.mb`, physical row space): fronts in
postorder (ascending), panels within each front forward, `trans='T'` — the mirror of
M5a's own `apply_Qt!` but replaying STORED per-panel `V`/`T` instead of re-walking a
row-subtree (§A6).
"""
function apply_Qt!(y::AbstractVector{T}, F::QRFrontFactor{T,Ti}) where {T,Ti<:Integer}
    fsym = F.fsym
    ws = F.ws
    @inbounds for f in 1:fsym.nfront
        m_f = Int(F.fm[f])
        m_f == 0 && continue
        frowlo = Int(fsym.frowptr2[f])
        yqt = view(ws.yqt, 1:m_f)
        for i in 1:m_f
            yqt[i] = y[F.frowind[frowlo + i - 1]]
        end
        tvbase = _tauv_base(fsym, f)
        if F.fscalar[f]
            Ff = _front_view(F, f)
            e_f = Int(F.fe[f])
            for t in 1:e_f
                jj = Int(F.elimcol[tvbase + t - 1])
                _scalar_apply_to_vec!(yqt, Ff, t, m_f, jj, F.tauv[tvbase + t - 1])
            end
        else
            panello = Int(fsym.fpanelptr[f])
            row0 = 1
            for pnl in 1:Int(F.fnpanel[f])
                mp = Int(F.pnrows[panello + pnl - 1])
                pb = Int(F.pbs[panello + pnl - 1])
                Vp = _gather_panel_V!(F, f, pnl, row0, mp, tvbase)
                Tv = _panel_T(F, f, pnl)
                yb = reshape(view(yqt, row0:(row0 + mp - 1)), mp, 1)
                wy_apply!('T', yb, Vp, Tv, ws.wy)
                row0 += pb
                tvbase += pb
            end
        end
        for i in 1:m_f
            y[F.frowind[frowlo + i - 1]] = yqt[i]
        end
    end
    return y
end

"""
    apply_Q!(y::AbstractVector, F::QRFrontFactor) -> y

`y ← Qy` in place: fronts in REVERSE postorder, panels within each front reverse,
`trans='N'` (§A6, the mirror of `apply_Qt!`).
"""
function apply_Q!(y::AbstractVector{T}, F::QRFrontFactor{T,Ti}) where {T,Ti<:Integer}
    fsym = F.fsym
    ws = F.ws
    @inbounds for f in fsym.nfront:-1:1
        m_f = Int(F.fm[f])
        m_f == 0 && continue
        frowlo = Int(fsym.frowptr2[f])
        yqt = view(ws.yqt, 1:m_f)
        for i in 1:m_f
            yqt[i] = y[F.frowind[frowlo + i - 1]]
        end
        if F.fscalar[f]
            Ff = _front_view(F, f)
            tvbase = _tauv_base(fsym, f)
            e_f = Int(F.fe[f])
            for t in e_f:-1:1
                jj = Int(F.elimcol[tvbase + t - 1])
                _scalar_apply_to_vec!(yqt, Ff, t, m_f, jj, F.tauv[tvbase + t - 1])
            end
        else
            panello = Int(fsym.fpanelptr[f])
            tvbase0 = _tauv_base(fsym, f)
            npanel = Int(F.fnpanel[f])
            row0s = ws.row0s
            tvbases = ws.tvbases
            row0s[1] = 1
            tvbases[1] = tvbase0
            for pnl in 1:npanel
                row0s[pnl + 1] = row0s[pnl] + Int(F.pbs[panello + pnl - 1])
                tvbases[pnl + 1] = tvbases[pnl] + Int(F.pbs[panello + pnl - 1])
            end
            for pnl in npanel:-1:1
                mp = Int(F.pnrows[panello + pnl - 1])
                row0 = row0s[pnl]
                Vp = _gather_panel_V!(F, f, pnl, row0, mp, tvbases[pnl])
                Tv = _panel_T(F, f, pnl)
                yb = reshape(view(yqt, row0:(row0 + mp - 1)), mp, 1)
                wy_apply!('N', yb, Vp, Tv, ws.wy)
            end
        end
        for i in 1:m_f
            y[F.frowind[frowlo + i - 1]] = yqt[i]
        end
    end
    return y
end

"""
    solve_R!(x::AbstractVector, F::QRFrontFactor, c::AbstractVector) -> x

`R·x = c` via back-substitution over the padded rows (design_qr_m5b.md §A6): row `k`'s
entries are `rval[frptr[k]:frptr[k+1]-1]` against implicit columns
`fcolind[fsnode[k]][pos(k):n_f]`; `fpivotrow[k]==0` (dead) forces `x[k]=0`, matching
M5a's basic-solution semantics. `x`/`c` have length `n′`; may alias.
"""
function solve_R!(x::AbstractVector{T}, F::QRFrontFactor{T,Ti}, c::AbstractVector{T}) where {T,Ti<:Integer}
    fsym = F.fsym
    nb = length(fsym.base.parent)
    @inbounds for k in nb:-1:1
        if F.fpivotrow[k] == 0
            x[k] = zero(T)
            continue
        end
        lo = Int(fsym.frptr[k])
        diag = F.rval[lo]
        f = fsym.fsnode[k]
        colslo = Int(fsym.fcolptr[f])
        n_f = n_f_of(fsym, f)
        pos = k - Int(fsym.fsuper[f]) + 1
        s = c[k]
        p = lo + 1
        for jc in (pos + 1):n_f
            gcol = fsym.fcolind[colslo + jc - 1]
            s -= F.rval[p] * x[gcol]
            p += 1
        end
        x[k] = s / diag
    end
    return x
end

"""
    solve_Rt!(x::AbstractVector, F::QRFrontFactor, c::AbstractVector) -> x

`Rᵀ·x = c` via forward substitution (mirror of [`solve_R!`](@ref), design_qr_m5b.md
§A6) — forward scatter since `R` is stored row-wise. `x`/`c` may alias.
"""
function solve_Rt!(x::AbstractVector{T}, F::QRFrontFactor{T,Ti}, c::AbstractVector{T}) where {T,Ti<:Integer}
    fsym = F.fsym
    nb = length(fsym.base.parent)
    x !== c && copyto!(x, 1, c, 1, nb)
    @inbounds for k in 1:nb
        if F.fpivotrow[k] == 0
            x[k] = zero(T)
            continue
        end
        lo = Int(fsym.frptr[k])
        diag = F.rval[lo]
        f = fsym.fsnode[k]
        colslo = Int(fsym.fcolptr[f])
        n_f = n_f_of(fsym, f)
        pos = k - Int(fsym.fsuper[f]) + 1
        xk = x[k] / diag
        x[k] = xk
        p = lo + 1
        for jc in (pos + 1):n_f
            gcol = fsym.fcolind[colslo + jc - 1]
            x[gcol] -= F.rval[p] * xk
            p += 1
        end
    end
    return x
end

"""
    solve!(x::AbstractVector, F::QRFrontFactor, b::AbstractVector) -> x

Least-squares (`m ≥ n`) / basic (rank-deficient or `m < n`, dead columns zero) solve
(design_qr_m5b.md §A6, mirroring M5a's own `solve!`): gather `b` into physical row
space via `rperm`, `apply_Qt!`, `solve_R!`, scatter into `x` via `cperm`. `x`/`b` are
full space (length `n`/`m`); the frontal path never carries singletons (`sym.n1==0`
always, §A1.2), so unlike M5a's own `solve!` there is no singleton-block prepend.
"""
function solve!(x::AbstractVector{T}, F::QRFrontFactor{T,Ti}, b::AbstractVector{T}) where {T,Ti<:Integer}
    fsym = F.fsym
    sym = fsym.base
    nb = length(sym.parent)
    y = view(F.ws.rhs, 1:sym.mb)
    fill!(y, zero(T))
    @inbounds for p in 1:sym.m
        phys = sym.riperm[p]
        phys <= sym.mb && (y[phys] = b[p])
    end
    apply_Qt!(y, F)
    # gather Qᵀb into R's own (block-column) space via fpivotrow (B2's own reasoning,
    # design_qr.md §3.4/§6.2: row k of R lives at physical row fpivotrow[k], not k)
    cc = view(F.ws.solve_cc, 1:nb)
    @inbounds for k in 1:nb
        piv = F.fpivotrow[k]
        cc[k] = piv == 0 ? zero(T) : y[piv]
    end
    solve_R!(cc, F, cc)
    @inbounds for k in 1:nb
        x[sym.cperm[k]] = cc[k]
    end
    fill!(y, zero(T))
    return x
end

"""
    F \\ b -> x

`solve!` allocating its own output (design_qr_m5b.md §A6).
"""
Base.:\(F::QRFrontFactor{T,Ti}, b::AbstractVector{T}) where {T,Ti<:Integer} = solve!(Vector{T}(undef, F.fsym.base.n), F, b)

"""
    ldiv!(x::AbstractVector, F::QRFrontFactor, b::AbstractVector) -> x

Alias for [`solve!`](@ref) (stdlib-compatible spelling, design_qr_m5b.md §A6).
"""
LinearAlgebra.ldiv!(x::AbstractVector{T}, F::QRFrontFactor{T,Ti}, b::AbstractVector{T}) where {T,Ti<:Integer} = solve!(x, F, b)
