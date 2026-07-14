# Front assembly (design_qr_m5b.md §A5.2): gathers a front's A-rows (from the block A's
# row-form) and its children's contribution-block rows (already stored, sorted, in the
# children's own front rectangles) into a staircase-sorted, dense front rectangle.
#
# `F.frowind[slot]` is a PHYSICAL ROW LABEL (1..mb), not a data source — for an A-row it
# is that row's own physical number; for a pass-up (child C-)row it is INHERITED
# verbatim from the child's own `frowind` entry (ultimately tracing back to whichever
# physical row was first assigned it, possibly several fronts down the tree). Labels
# are what the solve phase's shared `y` (length mb) vector is gathered/scattered by
# (§A6) — they carry no information for assembly's own VALUE scatter, which instead
# reads directly from either the block's row-form (A-rows) or the child's OWN stored
# front rectangle (child rows), tracked separately below during the gather.

function _assemble_front!(F::QRFrontFactor{T,Ti}, f::Int) where {T,Ti<:Integer}
    fsym = F.fsym
    sym = fsym.base
    ws = F.ws
    n_f = Int(fsym.fcolptr[f + 1] - fsym.fcolptr[f])
    colslo = Int(fsym.fcolptr[f])
    Ff_full = _front_view(F, f)

    # step 1: g2l (global block-column -> local front-column, 1-based)
    @inbounds for (i, gc) in enumerate(view(fsym.fcolind, colslo:(colslo + n_f - 1)))
        ws.g2l[gc] = Ti(i)
    end

    # step 2: gather incoming rows — A-rows (ascending pivotal column, ascending
    # physical row) then children (ascending front order; each child's own survivor
    # rows are ALREADY sorted and CONTIGUOUS in gather order, tracked via
    # childrange below so step 4 can build cg2l once per child).
    mmax_f = Int(fsym.fmmax[f])
    phys = Vector{Ti}(undef, 0); sizehint!(phys, mmax_f)
    mincol = Vector{Ti}(undef, 0); sizehint!(mincol, mmax_f)
    srcfront = Vector{Ti}(undef, 0); sizehint!(srcfront, mmax_f)   # 0 = A-row
    srcrow = Vector{Ti}(undef, 0); sizehint!(srcrow, mmax_f)        # original row (A-row) or child-local row
    @inbounds for k in fsym.fsuper[f]:(fsym.fsuper[f + 1] - 1)
        for physp in fsym.arowptr[k]:(fsym.arowptr[k + 1] - 1)
            push!(phys, Ti(physp))              # PHYSICAL row label (1..mb) — what
                                                 # solve's shared y vector is indexed by
            push!(mincol, ws.g2l[k])
            push!(srcfront, zero(Ti))
            push!(srcrow, sym.rperm[physp])     # ORIGINAL row — for the row-form value lookup
        end
    end
    @inbounds for cp in fsym.fchildptr[f]:(fsym.fchildptr[f + 1] - 1)
        c = fsym.fchildren[cp]
        r_c = Int(F.fr[c])
        e_c = Int(F.fe[c])   # NOT F.fm[c]: rows (e_c+1):fm(c) are all-zero residue
        crowlo = Int(fsym.frowptr2[c])
        for t in (r_c + 1):e_c
            push!(phys, F.frowind[crowlo + t - 1])
            push!(mincol, ws.g2l[F.fmincol[crowlo + t - 1]])
            push!(srcfront, Ti(c))
            push!(srcrow, Ti(t))
        end
    end
    m_f = length(phys)

    # step 3: staircase counting sort by local min-col (stable within each bucket,
    # since we scatter in gather order — A-rows before children, ascending within
    # each, exactly the order pushed above). `slotof[t]` = t's final sorted slot.
    # `count[j]` (reusing ws.bucket's first n_f slots) starts as the raw per-column
    # count, then becomes the CUMULATIVE count through column j (= the staircase
    # `stair[j]` directly — no off-by-one bucket-index trick, which an earlier draft
    # got wrong: bucket[mc+1] vs bucket[mc] indexing mismatch between the counting
    # and the prefix-sum/cursor-start steps silently dropped rows).
    count = ws.bucket
    @inbounds for j in 1:n_f
        count[j] = zero(Ti)
    end
    @inbounds for mc in mincol
        count[mc] += one(Ti)
    end
    @inbounds for j in 2:n_f
        count[j] += count[j - 1]
    end
    @inbounds for j in 1:n_f
        ws.stair[j] = count[j]
    end
    cursor = Vector{Ti}(undef, n_f)
    cursor[1] = one(Ti)
    @inbounds for j in 2:n_f
        cursor[j] = count[j - 1] + one(Ti)
    end
    rowlo = Int(fsym.frowptr2[f])
    slotof = Vector{Ti}(undef, m_f)
    @inbounds for t in 1:m_f
        mc = Int(mincol[t])
        slot = cursor[mc]
        F.frowind[rowlo + slot - 1] = phys[t]
        F.fmincol[rowlo + slot - 1] = Ti(mc)
        slotof[t] = slot
        cursor[mc] += one(Ti)
    end

    # step 4: zero the USED extent, then scatter values in GATHER order (grouped by
    # source: A-rows, then each child's contiguous block — cg2l is built once per
    # child, not once per row).
    Fused = view(Ff_full, 1:m_f, 1:n_f)
    @inbounds for jcol in 1:n_f, i in 1:m_f
        Fused[i, jcol] = zero(T)
    end

    t = 1
    @inbounds while t <= m_f && srcfront[t] == 0
        r = srcrow[t]
        slot = Int(slotof[t])
        for q in fsym.rowptr[r]:(fsym.rowptr[r + 1] - 1)
            col = fsym.rowcol[q]
            lc = ws.g2l[col]
            lc != 0 && (Fused[slot, lc] = F.rowval[q])
        end
        t += 1
    end
    @inbounds while t <= m_f
        c = Int(srcfront[t])
        n_c = Int(fsym.fcolptr[c + 1] - fsym.fcolptr[c])
        cslo = Int(fsym.fcolptr[c])
        for jc in 1:n_c
            ws.cg2l[fsym.fcolind[cslo + jc - 1]] = Ti(jc)
        end
        Fc = _front_view(F, c)
        while t <= m_f && srcfront[t] == c
            crow = Int(srcrow[t])
            slot = Int(slotof[t])
            gmincol = fsym.fcolind[cslo + _child_local_mincol(F, c, crow) - 1]
            jc0 = ws.cg2l[gmincol]
            for jc in jc0:n_c
                gc = fsym.fcolind[cslo + jc - 1]
                lc = ws.g2l[gc]
                lc != 0 && (Fused[slot, lc] = Fc[crow, jc])
            end
            t += 1
        end
        for jc in 1:n_c
            ws.cg2l[fsym.fcolind[cslo + jc - 1]] = zero(Ti)
        end
    end

    # step 5: un-set g2l
    @inbounds for gc in view(fsym.fcolind, colslo:(colslo + n_f - 1))
        ws.g2l[gc] = zero(Ti)
    end

    return m_f
end

# The child-local column position (1-based, within the child's OWN front) of physical
# row `crow`'s (rewritten, post-triangularization) min-col — recovered from the
# child's OWN fmincol (a global column id) via a direct scan of the child's OWN column
# list. `n_c` is small (front width), so a linear scan is acceptable here; revisit
# with a stored per-row local-mincol cache only if profiling shows this matters.
@inline function _child_local_mincol(F::QRFrontFactor{T,Ti}, c::Int, crow::Int) where {T,Ti<:Integer}
    fsym = F.fsym
    crowlo = Int(fsym.frowptr2[c])
    gcol = F.fmincol[crowlo + crow - 1]
    cslo = Int(fsym.fcolptr[c])
    n_c = Int(fsym.fcolptr[c + 1] - fsym.fcolptr[c])
    @inbounds for jc in 1:n_c
        fsym.fcolind[cslo + jc - 1] == gcol && return jc
    end
    return n_c + 1   # defensive: shouldn't happen (gcol is always one of the child's own columns)
end
