# M5b multifrontal QR symbolic layer (docs/design_qr_m5b.md §A2-§A4). Builds on the
# EXISTING M5a symbolic (qr/symbolic.jl's `symbolic_qr`) and the EXISTING supernode
# machinery (symbolic/supernodes.jl), unchanged except `fundamental_supernodes`'s new
# `fundamental::Bool` keyword (§A2.2). Operates entirely in BLOCK space (the frontal
# path is what `_qr_block` factorizes on, exactly like M5a's own numeric loop — the
# caller ensures `sym.n1 == 0` by construction, §A1.2).

"""
    QRFrontSymbolic{Ti<:Integer}

Front partition, tree, column structure, and per-front storage capacities for M5b's
multifrontal numeric phase (design_qr_m5b.md §A4.2). Composes the M5a symbolic
(`base`) by reference — every array below is BLOCK space (columns `1:n′`, physical
rows `1:mb`, `n′ = length(base.parent)`), matching `QRSymbolic`'s own D5 convention.
"""
struct QRFrontSymbolic{Ti<:Integer}
    base::QRSymbolic{Ti}
    # --- front partition & tree (post-amalgamation) ---
    nfront::Int
    fsuper::Vector{Ti}
    fsnode::Vector{Ti}
    fparent::Vector{Ti}
    fchildptr::Vector{Ti}
    fchildren::Vector{Ti}
    # --- front column structure (supernode_rowind output) ---
    fcolptr::Vector{Ti}
    fcolind::Vector{Ti}
    # --- A-row assignment & row-form access ---
    arowptr::Vector{Ti}
    rowptr::Vector{Ti}
    rowcol::Vector{Ti}
    atrans::Vector{Ti}
    # --- capacities from the assembly simulation (τ-robust upper bounds) ---
    fmmax::Vector{Ti}
    fcrmax::Vector{Ti}
    fvalptr::Vector{Ti}
    frowptr2::Vector{Ti}
    ftauptr::Vector{Ti}
    fpanelptr::Vector{Ti}
    frptr::Vector{Ti}
    # --- scalars ---
    nnzVF::Int
    nnzRF::Int
    max_front_rows::Int
    max_front_cols::Int
    # Global compact-WY panel block size = qr_block_size(max_front_rows, max_front_cols).
    # THE single source of truth for NB, shared by BOTH `ftauptr`'s per-front T-slab
    # budget (sized `nb * min(mmax_f, n_f)` below) AND `QRFrontWorkspace.Tm`'s dimension
    # / the numeric loop's panel-width cap (frontal.jl, `size(ws.Tm, 1)`). These two MUST
    # use the identical value: the numeric loop packs each panel's `pcount×pcount` T
    # (pcount ≤ nb) into the slab, so a slab budgeted with a SMALLER nb than the panel
    # cap overflows `ftau` — a genuine out-of-bounds write. That is exactly what happened
    # when the slab used `qr_block_size(0, 0)` (=8) while the workspace used
    # `qr_block_size(800, 169)` (=16): `qr_block_size` is dimension-dependent, so the
    # "one query with (0,0) suffices" assumption was wrong. The bug was invisible on
    # Zen3 (the stray write landed in benign adjacent heap) and a hard SIGSEGV on Zen5
    # (unmapped page) — a portability-contract violation traced via `--check-bounds=yes`
    # turning the segfault into a clean BoundsError at frontal_numeric.jl:284.
    nb::Int
    fflops::Float64
end

"""
    _front_children(nfront, fparent) -> (fchildptr, fchildren)

CSC-form children lists of the front tree, ascending front order (§A2.3) — the same
head/next-then-flatten idiom `relaxed_amalgamation`'s own `supernode_tree` construction
uses internally, materialized flat here since it is symbolic (built once).
"""
function _front_children(nfront::Int, fparent::Vector{Ti}) where {Ti<:Integer}
    cnt = zeros(Ti, nfront)
    @inbounds for s in 1:nfront
        p = fparent[s]
        p != 0 && (cnt[p] += one(Ti))
    end
    fchildptr = Vector{Ti}(undef, nfront + 1)
    fchildptr[1] = one(Ti)
    @inbounds for s in 1:nfront
        fchildptr[s + 1] = fchildptr[s] + cnt[s]
    end
    fchildren = Vector{Ti}(undef, fchildptr[nfront + 1] - 1)
    cursor = copy(fchildptr)
    @inbounds for s in 1:nfront
        p = fparent[s]
        if p != 0
            fchildren[cursor[p]] = Ti(s)
            cursor[p] += one(Ti)
        end
    end
    return fchildptr, fchildren
end

"""
    _block_row_form(m, n, colptr, rowval, ciperm) -> (rowptr, rowcol, atrans)

Row-form of the block `A` in PERMUTED column order, indexed by ORIGINAL row (not yet
restricted to physical rows — the caller relabels via `riperm`, §A4.1) — each row's
column list sorted ascending in final column order. `atrans[seq]` (`seq` = the 1-based
position in the cperm-column-order walk `qr!`'s own step 1 already performs) gives the
destination slot in the row-form's value buffer — the `amap` idiom (design.md §4.2),
QR-shaped: filling numeric values from a fresh `A2` in `qr!` is then one O(nnz) pass
with no per-entry searching.
"""
function _block_row_form(m::Int, n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}, cperm::Vector{Ti}, ciperm::Vector{Ti}) where {Ti<:Integer}
    nz = Int(colptr[n + 1]) - 1
    cnt = zeros(Ti, m)
    @inbounds for p in 1:nz
        cnt[rowval[p]] += one(Ti)
    end
    rowptr = Vector{Ti}(undef, m + 1)
    rowptr[1] = one(Ti)
    @inbounds for r in 1:m
        rowptr[r + 1] = rowptr[r] + cnt[r]
    end
    rowcol = Vector{Ti}(undef, nz)
    atrans = Vector{Ti}(undef, nz)
    cursor = copy(rowptr)
    seq = 0
    @inbounds for k in 1:n
        origcol = cperm[k]
        for p in colptr[origcol]:(colptr[origcol + 1] - 1)
            seq += 1
            r = rowval[p]
            slot = cursor[r]
            rowcol[slot] = ciperm[origcol]     # == k, permuted column id
            atrans[seq] = slot
            cursor[r] += one(Ti)
        end
    end
    # each row's entries were scattered in ASCENDING k (cperm-column-walk) order, which
    # IS ascending final-column order already (k runs 1:n) — no re-sort needed.
    return rowptr, rowcol, atrans
end

"""
    symbolic_qr_frontal(sym::QRSymbolic, A::SparseMatrixCSC; fundamental=false) -> QRFrontSymbolic

Build the M5b frontal symbolic layer on top of an already-built M5a `sym` (`sym.n1 ==
0` required — the frontal path factorizes the non-singleton block only, §A1.2) and its
originating block matrix `A` (`size(A) == (sym.m, sym.n)`). `fundamental=false` is
M5b's own default (SPQR paper's two-condition supernode test, §A2.2); pass `true` only
to compare against the fundamental (3-condition) partition.
"""
function symbolic_qr_frontal(sym::QRSymbolic{Ti}, A::SparseMatrixCSC{T,Ti};
        fundamental::Bool = false, amalg_cols::NTuple{3,Int} = AMALG_COLS,
        amalg_zmax::NTuple{3,Float64} = AMALG_ZMAX) where {T,Ti<:Integer}
    sym.n1 == 0 || throw(ArgumentError("symbolic_qr_frontal: sym.n1 = $(sym.n1) > 0 — the frontal path factorizes the non-singleton block only"))
    m, n = size(A)
    (m == sym.m && n == sym.n) || throw(DimensionMismatch("symbolic_qr_frontal: size(A) = ($m,$n), expected ($(sym.m),$(sym.n))"))
    nb = length(sym.parent)

    nsuper, super = fundamental_supernodes(nb, sym.parent, sym.rcount; fundamental)
    nfront, fsuper = relaxed_amalgamation(nb, nsuper, super, sym.parent, sym.rcount; amalg_cols, amalg_zmax)
    fsnode, fparent = supernode_tree(nb, nfront, fsuper, sym.parent)
    fchildptr, fchildren = _front_children(nfront, fparent)
    fcolptr, fcolind, _, _, _, _ = supernode_rowind(nb, sym.sptr, sym.sind, sym.parent, nfront, fsuper)

    # arowptr: A-row assignment (§A3.1) — recomputed from leftcol in O(m), matching
    # qr_row_structure's own internal `aptr` (M5a drops it; the frontal builder needs
    # it, §A4.1). leftcol is recomputed here (not stored on QRSymbolic) via the same
    # row_leftcol helper symbolic_qr itself uses.
    _, _, leftcol = row_leftcol(m, n, A.colptr, A.rowval, sym.ciperm)
    a = zeros(Ti, nb)
    @inbounds for r in 1:m
        k = leftcol[r]
        k != 0 && (a[k] += one(Ti))
    end
    arowptr = Vector{Ti}(undef, nb + 1)
    arowptr[1] = one(Ti)
    @inbounds for k in 1:nb
        arowptr[k + 1] = arowptr[k] + a[k]
    end

    rowptr, rowcol, atrans = _block_row_form(m, n, A.colptr, A.rowval, sym.cperm, sym.ciperm)

    # Assembly simulation (§A3.4): one ascending (postorder) pass. c_f = n_f - p_f
    # (non-pivotal front columns); mmax_f/crmax_f are the rank-aware capacity bounds
    # (§A3.2, own derivation — the trapezoid clamp).
    fmmax = zeros(Ti, nfront)
    fcrmax = zeros(Ti, nfront)
    fvalptr = Vector{Ti}(undef, nfront + 1); fvalptr[1] = one(Ti)
    frowptr2 = Vector{Ti}(undef, nfront + 1); frowptr2[1] = one(Ti)
    ftauptr = Vector{Ti}(undef, nfront + 1); ftauptr[1] = one(Ti)
    fpanelptr = Vector{Ti}(undef, nfront + 1); fpanelptr[1] = one(Ti)
    max_front_rows = 0
    max_front_cols = 0
    fflops = 0.0
    @inbounds for f in 1:nfront
        p_f = Int(fsuper[f + 1] - fsuper[f])
        n_f = Int(fcolptr[f + 1] - fcolptr[f])
        c_f = n_f - p_f
        a_f = Int(arowptr[fsuper[f + 1]] - arowptr[fsuper[f]])
        mmax_f = a_f
        for cp in fchildptr[f]:(fchildptr[f + 1] - 1)
            mmax_f += Int(fcrmax[fchildren[cp]])
        end
        crmax_f = min(mmax_f, c_f)
        fmmax[f] = Ti(mmax_f)
        fcrmax[f] = Ti(crmax_f)
        max_front_rows = max(max_front_rows, mmax_f)
        max_front_cols = max(max_front_cols, n_f)
        fvalptr[f + 1] = fvalptr[f] + Ti(mmax_f * n_f)
        frowptr2[f + 1] = frowptr2[f] + Ti(mmax_f)
        fpanelptr[f + 1] = fpanelptr[f] + Ti(n_f)   # capacity: at most n_f panels
    end

    # ftau T-slab budget — MUST use the SAME global NB the numeric workspace will
    # (frontal.jl's `QRFrontWorkspace`, `qr_block_size(max_front_rows, max_front_cols)`),
    # so it can only be computed here, after the loop above has found the max front dims.
    # See the `nb` field's doc comment on the struct for why an inconsistent NB here is a
    # real OOB, not a mere over/under-allocation. `min(mmax_f, n_f)` per front bounds the
    # number of eliminations (= Σ pcount over that front's panels); with each pcount ≤ nb,
    # Σ pcount² ≤ nb · Σ pcount ≤ nb · min(mmax_f, n_f), so this budget is exactly right.
    nb_global = max_front_cols == 0 ? 1 : qr_block_size(max_front_rows, max_front_cols)
    @inbounds for f in 1:nfront
        n_f = Int(fcolptr[f + 1] - fcolptr[f])
        ftauptr[f + 1] = ftauptr[f] + Ti(nb_global * min(Int(fmmax[f]), n_f))
    end

    # Padded R row pointers (§A5.5): row k (global/final column index) owns
    # n_f - pos(k) + 1 slots, where pos(k) is k's 1-based position in its front's
    # column list (front columns are pivotal-first, so a pivotal column k at front
    # position pos owns the tail fcolind[f][pos:n_f]).
    frptr = Vector{Ti}(undef, nb + 1)
    frptr[1] = one(Ti)
    @inbounds for k in 1:nb
        f = fsnode[k]
        n_f = Int(fcolptr[f + 1] - fcolptr[f])
        pos = k - Int(fsuper[f]) + 1   # pivotal columns are fcolind's first p_f entries,
                                       # stored in the same order as fsuper's own range
        frptr[k + 1] = frptr[k] + Ti(n_f - pos + 1)
    end

    # Flops estimate (§A3.5): exact-mode diagnostic only (never a gate, design.md §9.3
    # D2) — accumulated per front from its own rcount range (rcount[k] IS the exact
    # per-column stair height already computed by column_counts, so this is a direct
    # sum, not a re-simulation of the staircase).
    @inbounds for k in 1:nb
        f = fsnode[k]
        n_f = Int(fcolptr[f + 1] - fcolptr[f])
        pos = k - Int(fsuper[f]) + 1
        ell = Int(sym.rcount[k])
        fflops += 3.0 * ell + 4.0 * ell * (n_f - pos)
    end

    nnzVF = Int(fvalptr[nfront + 1] - 1)
    nnzRF = Int(frptr[nb + 1] - 1)

    return QRFrontSymbolic{Ti}(
        sym, nfront, fsuper, fsnode, fparent, fchildptr, fchildren,
        fcolptr, fcolind, arowptr, rowptr, rowcol, atrans,
        fmmax, fcrmax, fvalptr, frowptr2, ftauptr, fpanelptr, frptr,
        nnzVF, nnzRF, max_front_rows, max_front_cols, nb_global, fflops,
    )
end
