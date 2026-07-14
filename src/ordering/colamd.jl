# Column Approximate Minimum Degree ordering (design_qr.md §2.2) — the `order_columns`
# method for the `COLAMDOrdering` struct declared in `ordering/interface.jl`.
#
# Grounded EXCLUSIVELY in the two published sources (CLAUDE.md requirement 1 /
# design_qr.md §11 — the COLAMD C library, CHOLMOD and SuiteSparse source never read,
# in any form):
#
#   [P] Davis, Gilbert, Larimore, Ng, "A Column Approximate Minimum Degree Ordering
#       Algorithm", ACM TOMS 30(3):353-376, 2004.
#   [T] Larimore, "An Approximate Minimum Degree Column Ordering Algorithm",
#       MS thesis, University of Florida, 1998 (chapters 3-4).
#
# Specifically implemented from the sources (both read in full for this implementation):
#   * [P] §3, Algorithms 1-2 — row-merge symbolic LU: per-row column-pattern sets
#     `R_i`, per-column row-reference sets `C_j`, pivot-row formation
#     `R_r = (⋃_{i∈C_c} R_i) \ c` with regular row absorption, the symbolic update
#     `C_j = (C_j \ C_c) ∪ {r}`, and eq. (2)'s represented-row count `l_k` with the
#     `l_k = 0 ⇒ R_k := ∅, K := ∅` discard branch (design_qr.md §2.2 pt 2, D9).
#   * [P] §4.8's recommended variant (design_qr.md §2.2 pt 3, adopted verbatim):
#     initial metric = the COLMMD-style loose bound (eq. (3)); metric during the
#     elimination = the AMD-style approximate external row degree bound (eq. (4)),
#     computed with Algorithm 3's tag-array bookkeeping (after the first pass over the
#     pivot row's columns, `w[i] − t = ‖R_i \ R_r‖`); NO initial aggressive absorption;
#     aggressive row absorption during elimination ON ([P] §4.7); super-columns and
#     mass elimination ON ([P] §4, [T] §3.3).
#   * [T] §3.3 Algorithm 1 — the implementation-precision form this code follows:
#     uniform row ids for original and merged pivot rows (no bold/plain distinction),
#     the per-row `−1` in the initial degree `d_j = Σ_{i∈C_j}(|R_i| − 1)`, the
#     set-difference decrement by super-column thickness, inline aggressive absorption
#     when the difference hits zero, "further mass elimination" of columns whose
#     difference-sum is zero (distinct from row-level aggressive absorption), and the
#     final score `d_j = d_j + |R_r| − |j|` computed only AFTER super-column detection.
#   * [T] §4.1-4.2 — the single index array of size 2·nnz + n_col holding the column
#     form then the row form, with merged pivot rows appended at the free tail and
#     compacting garbage collection when the tail is exhausted ([T] §4.2.5: live rows
#     marked by storing the complement of their id over their first pattern entry, the
#     displaced entry stashed aside; columns never move, so they compact by index);
#     dense/null row and column pre-elimination with newly-null column detection
#     ([T] §4.2.3); the super-column hash `(Σ_{i∈C_j} i) mod n_col` ([T] §4.2.4);
#     natural-order tie-breaking via degree-list insertion order ([T] §4.2.3); the
#     pivot row reusing the id of the first row in the pivot column ([T] §4.2.4).
#
# Deliberate deviations from the sources, all named (H6 review aid):
#   * Dense thresholds (design_qr.md §2.2 pt 5, D1): absolute-count-with-√-scaling
#     (`max(COLAMD_DENSE_FLOOR, mult·√dim)`, the AMD-shaped heuristic this project
#     already uses) instead of [P]'s 50%-density default, which the paper itself calls
#     "probably too high for most matrices" (p. 362).
#   * Scores additionally tightened by `min(d_j, nactive_rem − |j|)` — the count of all
#     other remaining columns is a valid upper bound on an external column degree; the
#     same own-derivation tightening `ordering/amd.jl` applies (design.md §0 N4). It
#     also bounds every score below n, sizing the degree-list head array.
#   * Members of a mass-eliminated super-column are emitted principal-first in
#     absorption-chain order (the `svnext` chain idiom from `ordering/amd.jl`) instead
#     of via [T] §4.2.7's parent-tree `order_children` walk — the order among columns
#     with identical patterns is arbitrary for fill purposes.
#   * The generation-tagged mark/`w` arrays replace [P] Algorithm 3's explicit
#     `t = t + max(maxᵢ‖R_i‖, maxᵢ‖A_i‖)` update with the O(1) bump-by-(n+1) idiom
#     already used by `ordering/amd.jl` (`w` values this step lie in `[t, t + n]`
#     since a row degree never exceeds n, so bumping by n + 1 invalidates them all).
#
# Storage-layout point requiring care (own engineering, documented once here): the
# merged pivot row takes the id of the first live row of the pivot column ([T] §4.2.4).
# At formation time, other columns' `C_j` lists may still hold references to the OLD row
# under that id (they are pruned lazily). Within the formation step, ANY occurrence of
# the reused id in a `C_j` is therefore stale — the NEW pivot row is only appended to
# the `C_j` lists at final scoring, after all pruning passes. The phase-1 scan below
# hence treats `i == r` exactly like a dead row. Conversely, after the step completes,
# no stale reference can survive: every column referencing an absorbed row lies in
# `R_r ∪ {c}` (if `i ∈ C_c` then each `j ∈ R_i` satisfies `j ∈ R_r` or `j = c`; if `i`
# was aggressively absorbed then `R_i ⊆ R_r`), and phases 1-2 prune all of those.

# Column lifecycle. Rows need only a live/dead flag (`rowlive`).
const _COLAMD_LIVE = Int8(0)     # live principal (super)column
const _COLAMD_MERGED = Int8(1)   # non-principal: absorbed into a super-column
const _COLAMD_ORDERED = Int8(2)  # eliminated (pivot / mass / dense / null), position assigned

"""
    order_columns(alg::COLAMDOrdering, m, n, colptr, rowval) -> Vector{Ti}

Column Approximate Minimum Degree ordering (Davis–Gilbert–Larimore–Ng 2004,
design_qr.md §2.2) of the `m×n` pattern of `A` given in 1-based CSC form — computed
directly on `A`'s pattern, never forming `AᵀA` (the paper's recommended variant, §4.8:
COLMMD initial metric, AMD-style approximate external row degree during elimination,
aggressive row absorption during elimination, super-rows and super-columns).

Returns the permutation `p` with `p[k]` = the original column eliminated at step `k`
(`perm[new_position] = old_index`, the `order_columns` contract in
`ordering/interface.jl`). Rows with more than `max(COLAMD_DENSE_FLOOR,
alg.dense_row_mult·√n)` entries are withheld from the ordering entirely; columns with
more than `max(COLAMD_DENSE_FLOOR, alg.dense_col_mult·√m)` entries are withheld and
ordered last (before columns that are empty, which come very last; columns emptied BY
the dense-row withholding — "newly null", thesis §4.2.3 — come just before the dense
block), each block in ascending original index (design_qr.md §2.2 pt 5).
"""
function order_columns(alg::COLAMDOrdering, m::Int, n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}) where {Ti<:Integer}
    n == 0 && return Ti[]
    nnzA = Int(colptr[n + 1]) - 1
    perm = Vector{Ti}(undef, n)

    # ---- init_rows_cols ([T] §4.2.2): index array = [column form | row form | elbow] ----
    cap = 2 * nnzA + n + 1
    iw = Vector{Int}(undef, cap)
    cstart = Vector{Int}(undef, n)
    clen = Vector{Int}(undef, n)
    @inbounds for j in 1:n
        cstart[j] = Int(colptr[j])
        clen[j] = Int(colptr[j + 1] - colptr[j])
    end
    @inbounds for p in 1:nnzA
        iw[p] = Int(rowval[p])
    end
    rdeg = zeros(Int, m)                     # |R_i|, super-column-thickness-weighted
    @inbounds for p in 1:nnzA
        rdeg[iw[p]] += 1
    end
    rstart = Vector{Int}(undef, m)
    pos = nnzA + 1
    @inbounds for i in 1:m
        rstart[i] = pos
        pos += rdeg[i]
    end
    cursor = copy(rstart)
    @inbounds for j in 1:n, p in cstart[j]:(cstart[j] + clen[j] - 1)
        i = iw[p]
        iw[cursor[i]] = j
        cursor[i] += 1
    end
    rlen = copy(rdeg)
    free = 2 * nnzA + 1                      # first slot of the elbow-room tail

    colstate = fill(_COLAMD_LIVE, n)
    rowlive = fill(true, m)
    thick = ones(Int, n)                     # |j|: original-column count of super-column j
    svnext = zeros(Int, n)                   # super-column member chains (output expansion)
    svtail = collect(1:n)

    # ---- init_scoring ([T] §4.2.3): dense/null pre-elimination, initial scores ----
    # Output layout, back to front: [active | newly-null | dense | null], each block in
    # ascending original index (see docstring). Thresholds: design_qr.md §2.2 pt 5 (D1).
    col_thresh = max(COLAMD_DENSE_FLOOR, alg.dense_col_mult * sqrt(m))
    row_thresh = max(COLAMD_DENSE_FLOOR, alg.dense_row_mult * sqrt(n))
    nullcols = Int[]
    densecols = Int[]
    for j in 1:n
        if clen[j] == 0
            push!(nullcols, j)
        elseif clen[j] > col_thresh
            push!(densecols, j)
        end
    end
    for (b, j) in enumerate(nullcols)
        perm[n - length(nullcols) + b] = Ti(j)
        colstate[j] = _COLAMD_ORDERED
    end
    dbase = n - length(nullcols) - length(densecols)
    @inbounds for (b, j) in enumerate(densecols)
        perm[dbase + b] = Ti(j)
        colstate[j] = _COLAMD_ORDERED
        for p in cstart[j]:(cstart[j] + clen[j] - 1)
            rdeg[iw[p]] -= 1                 # [T] §4.2.3: dense columns reduce row degrees
        end
    end
    # dense and null rows removed AFTER dense-column removal ([T] §4.2.3 sequence),
    # judged on the updated degrees
    @inbounds for i in 1:m
        (rdeg[i] == 0 || rdeg[i] > row_thresh) && (rowlive[i] = false)
    end
    # initial COLMMD scores d_j = Σ_{i∈C_j}(|R_i| − 1) ([P] eq. (3); [T] §3.3 — the
    # per-row −1 is the super-column thickness, 1 during initial scoring), pruning dead
    # rows from every C_j in the same pass; columns left with no live rows are the
    # "newly null" ones, ordered as late as possible
    newlynull = Int[]
    score = zeros(Int, n)
    nactive = 0
    @inbounds for j in 1:n
        colstate[j] == _COLAMD_LIVE || continue
        s = 0
        wpos = cstart[j]
        for p in cstart[j]:(cstart[j] + clen[j] - 1)
            i = iw[p]
            rowlive[i] || continue
            iw[wpos] = i
            wpos += 1
            s += rdeg[i] - 1
        end
        clen[j] = wpos - cstart[j]
        if clen[j] == 0
            push!(newlynull, j)
            colstate[j] = _COLAMD_ORDERED
        else
            score[j] = s
            nactive += 1
        end
    end
    for (b, j) in enumerate(newlynull)
        perm[nactive + b] = Ti(j)
    end

    # ---- degree lists ([T] §4.1.4/§4.2.3): reverse-index insertion at head keeps the
    # head of every bucket at the smallest original index — natural-order tie-breaking ----
    dhead = zeros(Int, n)
    dnext = zeros(Int, n)
    dprev = zeros(Int, n)
    mindeg = n
    @inbounds for j in n:-1:1
        colstate[j] == _COLAMD_LIVE || continue
        score[j] = min(score[j], nactive - 1)    # own tightening (header); sizes dhead
        _amd_dl_insert!(dhead, dnext, dprev, score[j], j)
        mindeg = min(mindeg, score[j])
    end

    # ---- find_ordering ([T] §4.2.4) ----
    nrep = ones(Int, m)                      # l of [P] eq. (2): represented candidate rows
    w = zeros(Int, m)                        # Algorithm 3's tag array (w[i] − t = ‖R_i \ R_r‖)
    wflg = 1
    mk = zeros(Int, n)                       # column marks (pivot-row merge dedup)
    tag = 0
    rmk = zeros(Int, m)                      # row marks (super-column pattern compare)
    rtag = 0
    hhead = zeros(Int, n)                    # super-column hash buckets
    hnext = zeros(Int, n)
    hbuckets = Int[]
    psc = zeros(Int, n)                      # phase-2 difference sums, pending final score
    fcol = Vector{Int}(undef, m)             # garbage collection: displaced first entries
    ncolslive = nactive                      # live principal columns (GC space bound)
    nactive_rem = nactive                    # un-ordered original columns (score cap)
    kout = 0

    while kout < nactive
        # ---- pivot selection: minimum score, head of bucket ----
        d = mindeg
        @inbounds while dhead[d + 1] == 0
            d += 1
        end
        c = dhead[d + 1]
        _amd_dl_remove!(dhead, dnext, dprev, d, c)
        mindeg = d

        # ---- ensure tail room for the merged pivot row ([T] §4.2.4/§4.2.5) ----
        need = 0
        @inbounds for p in cstart[c]:(cstart[c] + clen[c] - 1)
            i = iw[p]
            rowlive[i] && (need += rlen[i])
        end
        need = min(need, ncolslive)
        if cap - free + 1 < need
            free = _colamd_gc!(iw, n, m, cstart, clen, colstate, rstart, rlen, rowlive, fcol, free)
            if cap - free + 1 < need
                # Unreachable by the storage bound ([P] §3: each C_j update never grows,
                # each pivot row is smaller than the rows it absorbs, so live storage
                # never exceeds 2·nnz and the post-GC tail is ≥ n ≥ need); defensive.
                cap = free + need + n
                resize!(iw, cap)
            end
        end

        # ---- mass elimination: order super-column c ([T] §4.2.4 first section) ----
        colstate[c] = _COLAMD_ORDERED
        ncolslive -= 1
        nactive_rem -= thick[c]
        v = c
        while v != 0
            kout += 1
            perm[kout] = Ti(v)
            v = svnext[v]
        end

        # ---- merged pivot row R_r = (⋃_{i∈C_c} R_i) \ c ([P] Alg 2; [T] §4.2.4),
        # reusing the id of the first live row; regular row absorption frees the rest ----
        tag = _amd_bump_tag!(mk, tag)
        r = 0
        rpos = free
        rd = 0                               # |R_r|, thickness-weighted
        lsum = 0
        @inbounds for p in cstart[c]:(cstart[c] + clen[c] - 1)
            i = iw[p]
            rowlive[i] || continue
            r == 0 && (r = i)
            lsum += nrep[i]
            for q in rstart[i]:(rstart[i] + rlen[i] - 1)
                j = iw[q]
                colstate[j] == _COLAMD_LIVE || continue   # skips c itself and stale entries
                mk[j] == tag && continue
                mk[j] = tag
                iw[rpos] = j
                rpos += 1
                rd += thick[j]
            end
            rowlive[i] = false
        end
        # [P] eq. (2): mass-eliminating the |c| columns of super-column c consumes one
        # candidate row per column (own reconciliation of eq. (2)'s "− 1" with [T]'s
        # super-column mass elimination; saturating at 0 for the non-strong-Hall case)
        lr = max(lsum - thick[c], 0)
        rl = rpos - free
        if r == 0 || rl == 0
            # no live candidate rows, or a pivot row with an empty pattern: nothing can
            # ever reference it — discard ([P] p. 361, "R_k and C_k can be discarded")
            continue
        end
        rstart[r] = free
        rlen[r] = rl
        free = rpos
        rowlive[r] = true
        rdeg[r] = rd
        nrep[r] = lr

        # ---- phase 1 ([P] Alg 3 first pass; [T] §4.2.4): prune C_j := C_j \ C_c
        # (dead rows and the stale reused id — header note), compute the set
        # differences w[i] − t = ‖R_i \ R_r‖ by subtracting each containing column's
        # thickness, and aggressively absorb rows whose difference hits zero ([P] §4.7) ----
        wflg = _amd_bump_wflg!(w, wflg, n)
        t = wflg
        @inbounds for p in rstart[r]:(rstart[r] + rlen[r] - 1)
            j = iw[p]
            _amd_dl_remove!(dhead, dnext, dprev, score[j], j)   # re-bucketed at final scoring
            wpos = cstart[j]
            for q in cstart[j]:(cstart[j] + clen[j] - 1)
                i = iw[q]
                (i == r || !rowlive[i]) && continue
                iw[wpos] = i
                wpos += 1
                if w[i] < t
                    w[i] = rdeg[i] + t
                end
                w[i] -= thick[j]
                if w[i] == t
                    # ‖R_i \ R_r‖ = 0: R_i ⊆ R_r — aggressive row absorption, even
                    # though i ∉ C_c ([P] §4.7; [T] §4.2.4)
                    rowlive[i] = false
                    wpos -= 1
                end
            end
            clen[j] = wpos - cstart[j]
        end

        # ---- l_k = 0 discard branch ([P] Alg 2/3 pp. 361/365, verbatim timing:
        # "K := {k}; if l_k = 0 then R_k := ∅; K := ∅" sits BETWEEN the set-difference
        # pass above and the degree-summing pass below; l_k is eq. (2)'s value computed
        # once at pivot-row formation and never modified afterward — design_qr.md §2.2
        # pt 2, D9): a pivot row representing no non-pivotal rows is discarded entirely.
        # Phase 2 never runs for it, {r} is never added to any C_j, and the surviving
        # columns keep their previous scores verbatim (Algorithm 3's final loop is
        # skipped when R_k := ∅); phase 1's prune and aggressive absorption stand, as
        # they do in Algorithm 3. ----
        if lr == 0
            rowlive[r] = false
            @inbounds for p in rstart[r]:(rstart[r] + rlen[r] - 1)
                j = iw[p]
                _amd_dl_insert!(dhead, dnext, dprev, score[j], j)
                mindeg = min(mindeg, score[j])
            end
            continue
        end

        # ---- phase 2 ([P] Alg 3 second pass; [T] §4.2.4): sum the differences into
        # d_j, prune rows absorbed later in phase 1, hash for super-column detection,
        # and further-mass-eliminate columns whose pattern collapsed onto the pivot's ----
        wpos_r = rstart[r]
        @inbounds for p in rstart[r]:(rstart[r] + rlen[r] - 1)
            j = iw[p]
            s = 0
            hsum = 0
            cw = cstart[j]
            for q in cstart[j]:(cstart[j] + clen[j] - 1)
                i = iw[q]
                rowlive[i] || continue
                iw[cw] = i
                cw += 1
                s += w[i] - t
                hsum += i
            end
            clen[j] = cw - cstart[j]
            if s == 0
                # further mass elimination ([T] §4.2.4 — column-level, distinct from
                # phase 1's row-level aggressive absorption): C_j has no live row left,
                # so j is indistinguishable from the pivot; order it now and shrink the
                # pivot row's degree by its thickness (l_k is NOT touched — [P] eq. (2)
                # computes it once at formation and Alg 3 never modifies it)
                colstate[j] = _COLAMD_ORDERED
                ncolslive -= 1
                nactive_rem -= thick[j]
                v = j
                while v != 0
                    kout += 1
                    perm[kout] = Ti(v)
                    v = svnext[v]
                end
                rdeg[r] -= thick[j]
            else
                iw[wpos_r] = j
                wpos_r += 1
                psc[j] = s
                h = mod(hsum, n) + 1         # [T] §4.2.4: (Σ_{i∈C_j} i) mod n_col
                hhead[h] == 0 && push!(hbuckets, h)
                hnext[j] = hhead[h]
                hhead[h] = j
            end
        end
        rlen[r] = wpos_r - rstart[r]

        # Every column in R_r collapsed onto the pivot (further mass elimination took
        # the s == 0 branch for all of them, so nothing was hashed and hbuckets is
        # empty): the pivot row's pattern is empty and nothing can ever reference it —
        # discard it, same reasoning as the rl == 0 discard at formation.
        if rlen[r] == 0
            rowlive[r] = false
            continue
        end

        # ---- detect_super_cols ([T] §4.2.6): pairwise compare within hash buckets;
        # C_i = C_j (live rows, both lists pruned by phase 2 and both excluding the
        # not-yet-appended pivot row) ⇒ merge j into i. Quotient-graph
        # indistinguishability may miss elimination-graph-indistinguishable pairs
        # ([T] §3.3) — a missed merge is a quality miss, never a false one. ----
        for h in hbuckets
            i = hhead[h]
            while i != 0
                if colstate[i] != _COLAMD_LIVE
                    i = hnext[i]
                    continue
                end
                previ = i
                j = hnext[i]
                while j != 0
                    jn = hnext[j]
                    same = colstate[j] == _COLAMD_LIVE && clen[j] == clen[i]
                    if same
                        rtag = _amd_bump_tag!(rmk, rtag)
                        @inbounds for q in cstart[i]:(cstart[i] + clen[i] - 1)
                            rmk[iw[q]] = rtag
                        end
                        @inbounds for q in cstart[j]:(cstart[j] + clen[j] - 1)
                            if rmk[iw[q]] != rtag
                                same = false
                                break
                            end
                        end
                    end
                    if same
                        thick[i] += thick[j]
                        colstate[j] = _COLAMD_MERGED
                        clen[j] = 0
                        ncolslive -= 1
                        svnext[svtail[i]] = j
                        svtail[i] = svtail[j]
                        hnext[previ] = jn    # unlink j from the bucket
                    else
                        previ = j
                    end
                    j = jn
                end
                i = hnext[i]
            end
            hhead[h] = 0
        end
        empty!(hbuckets)

        # ---- final scoring ([T] §4.2.4 last section, AFTER super-column detection so
        # |R_r| reflects further mass elimination and |j| reflects grown thicknesses):
        # d_j = Σ‖R_i \ R_r‖ + |R_r| − |j|, then C_j := C_j ∪ {r} — the append slot is
        # guaranteed: every j ∈ R_r lost at least one C_c row to phase 1's prune this
        # step, and garbage collection (which packs C_j exactly) only runs before it ----
        wpos_r = rstart[r]
        @inbounds for p in rstart[r]:(rstart[r] + rlen[r] - 1)
            j = iw[p]
            colstate[j] == _COLAMD_LIVE || continue    # drop merged-away members
            iw[wpos_r] = j
            wpos_r += 1
            sc = psc[j] + rdeg[r] - thick[j]
            sc = min(sc, nactive_rem - thick[j])       # own tightening (header)
            score[j] = max(sc, 0)
            iw[cstart[j] + clen[j]] = r
            clen[j] += 1
            _amd_dl_insert!(dhead, dnext, dprev, score[j], j)
            mindeg = min(mindeg, score[j])
        end
        rlen[r] = wpos_r - rstart[r]
    end

    @assert kout == nactive
    return perm
end

# Compacting garbage collection ([T] §4.2.5): columns never move during the ordering
# (their storage only shrinks in place), so live columns compact front-to-back by
# index; live rows are found positionally by the complement marker written over their
# first pattern entry (the displaced entry stashed in `fcol` — [T]'s `first_column`;
# `-i` here plays the role of the thesis's ones' complement, adapted to 1-based ids).
# Dead column entries are pruned from row patterns while copying (their thickness
# already lives with a principal column that co-occurs in the same rows, so `rdeg`
# stays exact). Returns the new free-tail start.
function _colamd_gc!(iw::Vector{Int}, n::Int, m::Int, cstart::Vector{Int}, clen::Vector{Int},
        colstate::Vector{Int8}, rstart::Vector{Int}, rlen::Vector{Int},
        rowlive::Vector{Bool}, fcol::Vector{Int}, free::Int)
    @inbounds for i in 1:m
        rowlive[i] || continue
        if rlen[i] == 0
            rowlive[i] = false               # an empty live row can never be referenced
            continue
        end
        fcol[i] = iw[rstart[i]]
        iw[rstart[i]] = -i
    end
    # columns (all row region markers lie strictly after every column slice, so this
    # forward copy cannot touch them)
    wpos = 1
    @inbounds for j in 1:n
        colstate[j] == _COLAMD_LIVE || continue
        src = cstart[j]
        len = clen[j]
        cstart[j] = wpos
        for k in 0:(len - 1)
            iw[wpos + k] = iw[src + k]
        end
        wpos += len
    end
    # rows, in physical order (positive values between markers are dead-slice garbage —
    # always former column data or unmarked dead rows, both hold positive ids only)
    p = wpos
    wdst = wpos
    @inbounds while p < free
        v = iw[p]
        if v >= 0
            p += 1
            continue
        end
        i = -v
        src = rstart[i]
        len = rlen[i]
        rstart[i] = wdst
        newlen = 0
        e = fcol[i]                          # the marker displaced this entry
        if colstate[e] == _COLAMD_LIVE
            iw[wdst] = e
            newlen = 1
        end
        for k in 1:(len - 1)
            e = iw[src + k]
            if colstate[e] == _COLAMD_LIVE
                iw[wdst + newlen] = e
                newlen += 1
            end
        end
        rlen[i] = newlen
        newlen == 0 && (rowlive[i] = false)
        wdst += newlen
        p = src + len
    end
    return wdst
end
