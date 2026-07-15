# Dense per-front factorization: staircase-blocked panels (design_qr_m5b.md §A5.3),
# dead-pivot mechanics (§A5.4), harvest + pass-up (§A5.5), and the top-level qr! driver
# (§A5.1). Factorization, R-harvest, and elimination-order tracking (needed for
# pass-up's rewritten min-cols) are done in ONE pass — harvesting/tracking AFTER the
# fact by re-deriving "which column was eliminated when" from already-transformed data
# is both redundant and a real bug magnet (an earlier draft did this and mis-indexed
# the reflector's own row range; fixed by never separating the two).

# Full front factorization + harvest + pass-up bookkeeping. Returns (dropped mass²,
# dead-pivot count) — front-level stats are accumulated by the caller.
function _factorize_front!(F::QRFrontFactor{T,Ti}, f::Int, m_f::Int, tau::T) where {T,Ti<:Integer}
    fsym = F.fsym
    ws = F.ws
    n_f = n_f_of(fsym, f)
    Ff_full = _front_view(F, f)
    Ff = view(Ff_full, 1:m_f, 1:n_f)
    p_f = Int(fsym.fsuper[f + 1] - fsym.fsuper[f])
    stair = ws.stair   # populated by _assemble_front! for this front

    # F.tauv's own per-front base offset (a SEPARATE, much smaller cursor than
    # `fsym.ftauptr`'s NB-scaled T-matrix slab — conflating the two was a real bug:
    # ftaubase must advance one slot per ELIMINATION, capacity Σ min(mmax_f,n_f); the
    # T-slab (ttaucur below) advances one slot per PANEL, capacity Σ NB*min(mmax_f,n_f)).
    ftaubase = 1
    @inbounds for fp in 1:(f - 1)
        ftaubase += min(Int(fsym.fmmax[fp]), n_f_of(fsym, fp))
    end
    panelbase = Int(fsym.fpanelptr[f])
    frowlo = Int(fsym.frowptr2[f])
    colslo = Int(fsym.fcolptr[f])
    NB = size(ws.Tm, 1)

    # elimination-order bookkeeping: local front-column of the t-th elimination
    # (t = 1..e_f), used only by pass-up below — small, front-width-bounded scratch.
    elim_col = ws.elim_col

    k = 1
    j = 1
    npanel = 0
    r_live = 0
    dropped_sq = zero(T)
    n_dead_front = 0
    ttaucur = Int(fsym.ftauptr[f])   # cursor into F.ftau's per-panel T storage
    # deferred-harvest scratch (§A5.5): a pivotal row's off-diagonal R entries at
    # OUT-of-panel columns (j2:n_f) aren't final until the panel's BLOCK trailing
    # apply lands, which only happens once, after the whole column loop below —
    # so those entries are harvested afterward, not inline (see note at the call
    # site; harvesting inline against pre-update Ff was a real bug, caught by
    # comparing this front's R against the M5a column-QR oracle: only each row's
    # OWN diagonal matched, every later-column entry in that row was wrong).
    piv_k = ws.piv_k
    piv_rlo = ws.piv_rlo
    @inbounds while j <= n_f && k <= m_f
        j2 = _panel_extent(j, n_f, stair, NB)
        panel_start_k = k
        mp = min(Int(stair[j2 - 1]), m_f) - panel_start_k + 1
        Vv = view(ws.wy.V, 1:mp, 1:(j2 - j))
        fill!(Vv, zero(T))
        pcount = 0
        npiv = 0
        for jj in j:(j2 - 1)
            hi = min(Int(stair[jj]), m_f)
            hi < k && continue
            xnorm = k > hi ? zero(T) : nrm2(view(Ff, k:hi, jj))
            is_pivotal = jj <= p_f
            if is_pivotal && (xnorm == zero(T) || (tau > zero(T) && xnorm <= tau))
                dropped_sq += xnorm * xnorm
                n_dead_front += 1
                continue   # k does NOT advance; no reflector; column jj contributes nothing
            end
            local_tau = if xnorm == zero(T)
                zero(T)   # B3: trivial identity reflector (non-pivotal exact-zero column)
            else
                _front_form_reflector!(Ff, k, hi, jj, xnorm)
            end
            pcount += 1
            Vv[k - panel_start_k + 1, pcount] = one(T)
            for i in (k + 1):hi
                Vv[i - panel_start_k + 1, pcount] = Ff[i, jj]
            end
            ws.tau_panel[pcount] = local_tau
            F.tauv[ftaubase] = local_tau
            F.elimcol[ftaubase] = Ti(jj)
            ftaubase += 1
            elim_col[k] = Ti(jj)

            # apply this reflector to the REMAINING in-panel columns FIRST — a
            # row's off-diagonal R entries at those columns are only final once
            # this update lands (harvesting before it would read pre-update Ff).
            if local_tau != zero(T)
                for jcol in (jj + 1):(j2 - 1)
                    _front_apply1!(Ff, k, hi, jj, jcol, local_tau)
                end
            end

            if is_pivotal
                r_live += 1
                gk = fsym.fcolind[colslo + jj - 1]
                F.fpivotrow[gk] = F.frowind[frowlo + k - 1]
                rlo = Int(fsym.frptr[gk])
                # harvest the now-final in-panel portion (jj:(j2-1)); OUT-of-panel
                # columns (j2:n_f) still await the panel's BLOCK trailing apply
                # below (fires once, after this whole column loop) — deferred.
                for jc in jj:(j2 - 1)
                    F.rval[rlo] = Ff[k, jc]
                    rlo += 1
                end
                if j2 <= n_f
                    npiv += 1
                    piv_k[npiv] = Ti(k)
                    piv_rlo[npiv] = Ti(rlo)
                end
            end
            # Ff[k,jj] is LEFT holding beta (LAPACK's own in-place dlarfg convention),
            # not restored to an implicit 1: solve-phase V-gather hardcodes the unit
            # diagonal itself (`_gather_panel_V!`, frontal_solve.jl) rather than
            # reading it from Ff, so nothing downstream needs this position to read
            # back as 1 — and for a NON-pivotal column, this is exactly the reduced
            # value the PARENT front's C-block pass-up reads (`_assemble_front!`'s
            # child-gather loop, via `Fc[crow, jc]`) once this row survives to
            # r_live+1:e_f; overwriting it here was a real bug (silently corrupting
            # every survivor row's own mincol entry before it ever reached the parent).
            k += 1
        end
        if pcount > 0
            Vp = view(Vv, 1:mp, 1:pcount)
            Tv = view(ws.Tm, 1:pcount, 1:pcount)
            wy_t!(Tv, Vp, view(ws.tau_panel, 1:pcount), view(ws.wy.G, 1:pcount, 1:pcount))
            if j2 <= n_f
                Ctrail = view(Ff, panel_start_k:(panel_start_k + mp - 1), j2:n_f)
                wy_apply!('T', Ctrail, Vp, Tv, ws.wy)
                # deferred harvest: the block apply just landed columns j2:n_f for
                # every pivotal row in this panel — harvest those now.
                for t in 1:npiv
                    kk = Int(piv_k[t])
                    rlo = Int(piv_rlo[t])
                    for jc in j2:n_f
                        F.rval[rlo] = Ff[kk, jc]
                        rlo += 1
                    end
                end
            end
            npanel += 1
            F.pnrows[panelbase + npanel - 1] = Ti(mp)
            F.pncols[panelbase + npanel - 1] = Ti(pcount)
            F.pbs[panelbase + npanel - 1] = Ti(pcount)
            # persist T for the solve-phase replay (§A5.3: "compact-WY T's are STORED,
            # not rebuilt per solve") — pcount×pcount, packed panel-by-panel into the
            # front's ftau slab (sized NB*min(mmax_f,n_f) >= Σ pcount² since pcount<=NB).
            @inbounds for cc in 1:pcount, rr in 1:pcount
                F.ftau[ttaucur + (cc - 1) * pcount + rr - 1] = Tv[rr, cc]
            end
            ttaucur += pcount * pcount
        end
        j = j2
    end

    e_f = k - 1
    F.fnpanel[f] = Ti(npanel)
    F.fr[f] = Ti(r_live)
    F.fm[f] = Ti(m_f)
    F.fe[f] = Ti(e_f)

    # pass-up: rows r_live+1..e_f are survivors — their rewritten min-col is the front
    # column of the elimination that consumed them (elim_col[r_live+1 .. e_f], the
    # SAME local row indices, since elimination row t's front-local column IS
    # elim_col[t] by construction of the loop above — no re-derivation needed).
    @inbounds for t in (r_live + 1):e_f
        gcol = fsym.fcolind[colslo + Int(elim_col[t]) - 1]
        F.fmincol[frowlo + t - 1] = gcol
    end
    # rows beyond e_f (dead-column residue / structural zero) are dropped: their mass
    # is already counted above (§A5.4); frowind/fmincol slots beyond e_f are simply
    # never read again (pass-up only reads r_live+1:e_f, assembly only reads
    # r_c+1:m_c of a child, i.e. up to fr[c]+1:fm[c] — dropped rows sit in
    # (e_f+1):m_f, outside that range since e_f<=m_f always).

    return dropped_sq, n_dead_front
end

"""
    qr!(F::QRFrontFactor, A::SparseMatrixCSC; tol=nothing) -> QRFrontFactor

Multifrontal refactorize in place (design_qr_m5b.md §A5.1): `A` must share `F.fsym`'s
sparsity pattern. The dense per-front factorization is zero-allocation after warmup;
`_assemble_front!`'s small per-front scratch still allocates (documented there, §A5.2's
own precedent) — a follow-up optimization target, not a correctness gap.
"""
function qr!(F::QRFrontFactor{T,Ti}, A::SparseMatrixCSC{T,Ti}; tol::Union{Nothing,Real} = nothing) where {T,Ti<:Integer}
    fsym = F.fsym
    sym = fsym.base
    τ = _qr_threshold(A, tol)

    fill!(F.fpivotrow, zero(Ti))
    fill!(F.fm, zero(Ti))
    fill!(F.fr, zero(Ti))
    fill!(F.fe, zero(Ti))
    fill!(F.rval, zero(T))

    seq = 0
    @inbounds for k in 1:length(sym.parent)
        origcol = sym.cperm[k]
        for p in A.colptr[origcol]:(A.colptr[origcol + 1] - 1)
            seq += 1
            F.rowval[fsym.atrans[seq]] = A.nzval[p]
        end
    end

    n_dead = 0
    dropped_sq = zero(T)
    rank = 0
    @inbounds for f in 1:fsym.nfront
        m_f = _assemble_front!(F, f)
        dsq, ndf = _factorize_front!(F, f, m_f, τ)
        dropped_sq += dsq
        n_dead += ndf
        rank += Int(F.fr[f])
    end

    F.stats.nnzR = fsym.nnzRF
    F.stats.nnzV = fsym.nnzVF
    F.stats.flops = fsym.fflops
    F.stats.rank = rank
    F.stats.n_dead = n_dead
    F.stats.dropped_norm = Float64(sqrt(dropped_sq))
    F.ok = true
    return F
end

"""
    qr_frontal(A::SparseMatrixCSC; ordering, tol=nothing, fundamental=false) -> QRFrontFactor

One-shot multifrontal sparse QR factorization (design_qr_m5b.md §A1/§A5). Mirrors
[`_qr_block`](@ref)'s role for the `:column` (M5a) path — the frontal path's own
self-contained entry point, always `sym.n1 == 0` (the frontal factorizer never sees
singleton columns directly, §A1.2).
"""
function qr_frontal(A::SparseMatrixCSC{T,Ti}; ordering::AbstractOrdering,
        tol::Union{Nothing,Real} = nothing, fundamental::Bool = false) where {T,Ti<:Integer}
    sym = symbolic_qr(A; ordering)
    fsym = symbolic_qr_frontal(sym, A; fundamental)
    F = QRFrontFactor{T,Ti}(fsym)
    qr!(F, A; tol)
    return F
end
