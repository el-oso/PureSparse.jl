# Dense per-front factorization — a mechanical port of faer 0.24.1's supernodal QR
# numeric core, `factorize_supernodal_numeric_qr_impl`'s per-supernode dense phase
# (src/sparse/linalg/qr.rs:1246-1344), onto PureSparse's own symbolic layer and
# PureBLAS's `wy_t!`/`wy_apply!` kernels (design_qr_m5b.md §A7's boundary: faer's
# dense-kernel INTERNALS are not translated — PureBLAS's proven blocked compact-WY
# kernels substitute for `qr_in_place`/`apply_block_householder_sequence_*`; what is
# translated is faer's ORCHESTRATION: staircase-panel boundaries, per-panel block-size
# re-derivation, block loop, trailing applies, post-factorization R harvest, and the
# pass-up bookkeeping).
#
# Structure map (faer qr.rs line ranges → here):
#   1246-1265  panel split rule: scan rows in min-col-sorted order; a panel ends
#              before the first row whose LOCAL min-col jumps ≥ max(1, NBf÷2) past
#              the panel's reference min-col (or at the row-list sentinel). NOTE
#              this is a COLUMN-index jump — the pre-port code broke panels on a
#              staircase ROW-count jump and capped width at NB, a mis-reading of
#              the same faer heuristic (design_qr_m5b.md §A5.3's transcription);
#              faer panels are uncapped in width and sub-blocked internally.
#   1266-1269  panel extents: nrows = all unconsumed rows before the boundary row;
#              ncols = min(nrows, min-col span). Columns are consumed contiguously
#              from the column cursor regardless of staircase gaps (a gap's columns
#              fold into the next panel; with no rows supporting them they die as
#              zero columns there — faer's dense QR skips them via rank detection,
#              here the pivotal dead test / `hi < k` skip does, same net effect).
#   1278-1279  per-panel block size bs = min(front max NBf, tier(nrows, ncols)) —
#              `_qr_faer_block_size`, frontal.jl.
#   1288-1304  the panel's blocked QR + trailing applies. faer: `qr_in_place(left)`
#              (internally blocked at bs), then one block-sequence apply to `right`
#              (all front columns past the panel), chunked at bs
#              (householder.rs:790-807). Here: per bs-block, scalar reflector loop
#              (rank-policy-coupled, stays in PureSparse per §A7.4) + `wy_t!` +
#              TWO `wy_apply!('T')` calls — (a) remaining panel columns (faer's
#              in-`qr_in_place` trailing update, factor.rs:241-249), (b) the
#              trailing right. Interleaving (b) per block instead of batching all
#              blocks afterward applies the identical operands in the identical
#              order (right is disjoint from the panel, applies commute across
#              disjoint column sets) while letting the gathered V panel be reused —
#              PureBLAS's `wy_apply!` reads an explicit-unit V copy by contract
#              (wy.jl header), where faer reads reflectors in place.
#   1310-1324  R harvest: one post-factorization pass copying the retired rows'
#              upper trapezoid out of the front (faer copies rows 0..min(m,ncols)
#              triangularly into L=Rᵀ; here elimination t's row goes to the padded
#              rval row of its column elim_col[t] — the dead-pivot generalization,
#              since with Heath skipping row t's column is elim_col[t] ≥ t, not t).
#   1325-1344  pass-up min-col rewrite (see the pass-up block below).
#
# Deliberate deviations from faer (each because faer has NO rank detection in this
# layer — it assumes every panel consumes exactly ncols rows):
#   - dead-pivot mechanics (SPQR-paper Heath handling, §A5.4): the row cursor k
#     advances only per LIVE reflector, so `nrows = idx - k` generalizes faer's
#     `idx - current_start`; dropped-mass/n_dead/fpivotrow accounting is ours.
#   - the pass-up min-col rewrite uses elim_col (exact) where faer writes
#     max(row index, original min-col) (qr.rs:1331-1344) — a safe left-conservative
#     approximation that is only exact at full rank; ours is exact under dead
#     pivots too and was already tested.
#   - per-reflector row extents use the staircase (hi = stair[jj]) for formation
#     and the scalar in-block applies — rows below stair[jj] are structurally zero
#     in column jj, so this is bit-identical to faer's full-panel-height reflector
#     with less work; the BLOCKED applies span the full panel height like faer's.

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
    # T-slab (ttaucur below) advances one slot per BLOCK, capacity Σ NB*min(mmax_f,n_f)).
    ftaubase = 1
    @inbounds for fp in 1:(f - 1)
        ftaubase += min(Int(fsym.fmmax[fp]), n_f_of(fsym, fp))
    end
    panelbase = Int(fsym.fpanelptr[f])
    frowlo = Int(fsym.frowptr2[f])
    colslo = Int(fsym.fcolptr[f])
    NB = size(ws.Tm, 1)
    # faer's per-front maximum block size (qr.rs:609-613 computes it at symbolic time
    # from the front's exact row/col counts; here numeric-time from the actual m_f —
    # same tier function), clamped to the symbolic/workspace capacity NB (see
    # `_qr_faer_block_size`'s header note).
    NBf = min(NB, _qr_faer_block_size(m_f, n_f))
    split_jump = max(1, NBf ÷ 2)   # faer qr.rs:1261-1265

    # elimination-order bookkeeping: local front-column of the t-th elimination
    # (t = 1..e_f), used by the post-loop R harvest and pass-up below.
    elim_col = ws.elim_col

    k = 1                  # row cursor: next unconsumed row (faer current_start, row role)
    j = 1                  # column cursor (faer current_start, column role)
    current_min_col = 1    # the panel's reference local min-col (faer qr.rs:1247)
    npanel = 0
    r_live = 0
    dropped_sq = zero(T)
    n_dead_front = 0
    ttaucur = Int(fsym.ftauptr[f])   # cursor into F.ftau's per-block T storage
    @inbounds for idx in 1:(m_f + 1)
        # faer qr.rs:1249-1259: row idx's local min-col, sentinel past the last row.
        # F.fmincol holds assembly's LOCAL min-cols (ascending) until pass-up
        # rewrites the survivor tail to global columns after this loop.
        idx_min_col = idx <= m_f ? Int(F.fmincol[frowlo + idx - 1]) : n_f + 1
        # faer's split condition (qr.rs:1260-1265)
        if !(idx_min_col == n_f + 1 || idx_min_col >= current_min_col + split_jump)
            continue
        end
        nrows = idx - k                              # unconsumed rows k..idx-1 (qr.rs:1266)
        span = idx_min_col - current_min_col
        ncols_grp = min(nrows, span)                 # qr.rs:1268-1269
        current_min_col = idx_min_col                # qr.rs:1305
        ncols_grp <= 0 && continue
        row_hi = idx - 1
        j1 = j + ncols_grp - 1   # ≤ n_f: j ≤ current_min_col_old and ncols_grp ≤ span
                                 # imply j1 ≤ idx_min_col - 1 ≤ n_f (faer's own
                                 # current_start ≤ current_min_col invariant)
        bs = min(NBf, _qr_faer_block_size(nrows, ncols_grp))   # qr.rs:1278-1279
        b0 = j
        while b0 <= j1
            b1 = min(b0 + bs - 1, j1)
            panel_start_k = k
            mp = row_hi - panel_start_k + 1   # block row extent: full panel height
                                              # below the cursor, like faer's `left`
            mp <= 0 && break                  # rows exhausted (dead-pivot deficit)
            Vv = view(ws.wy.V, 1:mp, 1:(b1 - b0 + 1))
            fill!(Vv, zero(T))
            pcount = 0
            for jj in b0:b1
                hi = min(Int(stair[jj]), m_f)
                hi < k && continue   # no unconsumed support (staircase-gap column)
                xnorm = nrm2(view(Ff, k:hi, jj))
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

                # scalar apply to the REMAINING in-BLOCK columns (faer's unblocked
                # leaf role, factor.rs:64-82; block-confined like factor.rs:170-253's
                # recursion width — the block's own trailing columns are handled by
                # the wy_apply! below, exactly geqrf!'s proven single-level shape)
                if local_tau != zero(T)
                    for jcol in (jj + 1):b1
                        _front_apply1!(Ff, k, hi, jj, jcol, local_tau)
                    end
                end

                if is_pivotal
                    r_live += 1
                    gk = fsym.fcolind[colslo + jj - 1]
                    F.fpivotrow[gk] = F.frowind[frowlo + k - 1]
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
                # (a) block → remaining PANEL columns (faer's in-qr_in_place trailing
                #     update, factor.rs:241-249)
                if b1 < j1
                    wy_apply!('T', view(Ff, panel_start_k:row_hi, (b1 + 1):j1), Vp, Tv, ws.wy)
                end
                # (b) block → trailing right, all front columns past the panel
                #     (faer's post-panel sequence apply, qr.rs:1295-1304 chunked at bs
                #     via householder.rs:790-807 — interleaved per block here, same
                #     operands in the same order; see the header's structure map)
                if j1 < n_f
                    wy_apply!('T', view(Ff, panel_start_k:row_hi, (j1 + 1):n_f), Vp, Tv, ws.wy)
                end
                npanel += 1
                # block descriptor triple — faer's tau_block_size/householder_nrows/
                # householder_ncols records (qr.rs:1280-1282), consumed by the solve
                # replay exactly as faer's apply replays them (qr.rs:783-804)
                F.pnrows[panelbase + npanel - 1] = Ti(mp)
                F.pncols[panelbase + npanel - 1] = Ti(pcount)
                F.pbs[panelbase + npanel - 1] = Ti(pcount)
                # persist T for the solve-phase replay (§A5.3: "compact-WY T's are STORED,
                # not rebuilt per solve") — pcount×pcount, packed block-by-block into the
                # front's ftau slab (sized NB*min(mmax_f,n_f) >= Σ pcount² since pcount<=NB).
                for cc in 1:pcount, rr in 1:pcount
                    F.ftau[ttaucur + (cc - 1) * pcount + rr - 1] = Tv[rr, cc]
                end
                ttaucur += pcount * pcount
            end
            b0 = b1 + 1
        end
        j = j1 + 1
    end

    e_f = k - 1

    # R harvest — faer's post-factorization triangular copy-out (qr.rs:1310-1324),
    # elim_col-generalized for dead pivots (see header): elimination t retired front
    # row t at front column elim_col[t]; if that column is pivotal, row t IS the R row
    # of its global column, final in Ff once the loop above completed (every trailing
    # apply that touches it has landed — the pre-port inline+deferred harvest split
    # existed only because harvest ran inside the panel loop).
    @inbounds for t in 1:e_f
        jj = Int(elim_col[t])
        jj <= p_f || continue
        gk = fsym.fcolind[colslo + jj - 1]
        rlo = Int(fsym.frptr[gk])
        for jc in jj:n_f
            F.rval[rlo] = Ff[t, jc]
            rlo += 1
        end
    end
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
sparsity pattern. Zero-allocation after construction (gated in
qr_frontal_numeric_tests.jl).
"""
function qr!(F::QRFrontFactor{T,Ti}, A::SparseMatrixCSC{T,Ti}; tol::Union{Nothing,Real} = nothing) where {T,Ti<:Integer}
    fsym = F.fsym
    sym = fsym.base
    τ = _qr_threshold(A, tol)

    # NOT translating faer's one-shot `householder_val.fill(zero())` (qr.rs:1064-1066)
    # here: faer's per-supernode capacity is EXACT (no rank detection), so zeroing all
    # of it costs exactly what factorization touches. PureSparse's `fmmax_f` is a
    # rank-deficiency-aware UPPER BOUND (design_qr_m5b.md §A3.2) that can exceed the
    # actual `m_f` substantially deep in the tree — zeroing the full capacity here
    # measurably regressed qr! (~2x on several gate matrices, Chairmarks median).
    # Zeroing stays per-front, used-extent-only, in `_assemble_front!` step 4
    # (task 16e's own zero-alloc-pass precedent) — a deliberate deviation from faer's
    # structure, forced by a storage-model difference this port does not touch.
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
