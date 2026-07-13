# Approximate Minimum Degree ordering (design.md §2.2) — the `order` method for the
# `AMDOrdering` struct declared in `ordering/interface.jl`.
#
# Grounded EXCLUSIVELY in the published paper (CLAUDE.md requirement 1 — CHOLMOD /
# SuiteSparse / reference-AMD source never read, in any form):
#
#   Amestoy, Davis, Duff, "An Approximate Minimum Degree Ordering Algorithm",
#   SIAM J. Matrix Anal. Appl. 17(4):886-905, Dec. 1996.
#
# Specifically implemented from the paper (fetched and read for this implementation):
#   * Algorithm 1 — the quotient-graph minimum degree method: pivot selection, pivot
#     element formation `L_p = (A_p ∪ ⋃_{e∈E_p} L_e) \ p̄`, redundant-entry removal
#     `A_i = (A_i \ L_p) \ p̄`, element absorption `E_i = (E_i \ E_p) ∪ {p}`,
#     supervariable detection/merging, conversion of the pivot to an element.
#   * Algorithm 2 + eq. (5) — the single-scan `w[]` workspace computation of
#     `|L_e \ L_p|` for every element adjacent to `L_p`.
#   * Eq. (4) — the three-term approximate external degree bound (see `_amd_order!`).
#   * §5 — the supervariable hash `Hash(i) = ((Σ_{j∈A_i} j + Σ_{e∈E_i} e) mod (n−1)) + 1`
#     and aggressive element absorption (`|L_e \ L_p| = 0` ⇒ absorb `e` into `p` even
#     when `e ∉ E_p`), plus the one-workspace-with-elbow-room storage discipline
#     ("garbage collection occurs if the elbow room is exhausted").
#
# Mass elimination is NOT a separate pass (design.md §0 D4): variables that become
# indistinguishable are merged into one supervariable by the hash detection below, and a
# merged supervariable is eliminated in one pivot step when selected — mass elimination
# falls out of that.
#
# The dense-row stripping (threshold `max(AMD_DENSE_FLOOR, dense_mult·√n)`) is the AMD
# *package's* documented user-guide default treatment, not part of the 1996 paper's
# algorithm text (design.md §0 N5); stripped variables are appended last.
#
# Storage layout, status encoding, compaction procedure, and generation-tagged mark /
# `w` handling below are our own independent engineering — the paper prescribes only
# the quotient-graph sets and the elbow-room/garbage-collection discipline.

# Node lifecycle states. Every node id 1:n starts as a live variable; a pivot becomes an
# element; merged variables and absorbed elements die (their workspace slice is freed).
const _AMD_VAR = Int8(0)       # live principal (super)variable
const _AMD_DEADVAR = Int8(1)   # variable merged into another supervariable
const _AMD_ELT = Int8(2)       # live element (an eliminated pivot)
const _AMD_DEADELT = Int8(3)   # absorbed element
const _AMD_DENSE = Int8(4)     # dense-stripped variable, ordered last

"""
    order(alg::AMDOrdering, n, colptr, rowval) -> Vector{Ti}

Approximate Minimum Degree ordering (Amestoy–Davis–Duff 1996, design.md §2.2) of the
FULL symmetric pattern (both triangles, no diagonal) given in 1-based CSC form.

Returns the permutation `p` with `p[k]` = the original index eliminated at step `k`
(`perm[new_position] = old_index`, the same convention as `NaturalOrdering` /
`GivenOrdering` — `symbolic`'s `iperm[old] = new` is its inverse). Rows/columns whose
degree exceeds `max(AMD_DENSE_FLOOR, alg.dense_mult·√n)` are stripped before the main
loop and appended last, in ascending index order (design.md §2.2 pt 6).
"""
function order(alg::AMDOrdering, n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}) where {Ti<:Integer}
    n == 0 && return Ti[]

    # ---- dense-row stripping (design §2.2 pt 6) ----
    threshold = max(AMD_DENSE_FLOOR, alg.dense_mult * sqrt(n))
    deg0 = zeros(Int, n)
    @inbounds for j in 1:n, q in colptr[j]:(colptr[j + 1] - 1)
        Int(rowval[q]) == j && continue      # contract says no diagonal; tolerate one
        deg0[j] += 1
    end
    state = fill(_AMD_VAR, n)
    ndense = 0
    for j in 1:n
        if deg0[j] > threshold
            state[j] = _AMD_DENSE
            ndense += 1
        end
    end

    # ---- quotient-graph storage (design §2.2 pt 1) ----
    # One shared integer workspace `qw`: every live node owns one contiguous slice.
    # For a live VARIABLE i the slice is its element list E_i (`ne[i]` entries) followed
    # by its variable list A_i (`na[i]` entries). For a live ELEMENT e the slice is L_e
    # (`na[e]` entries; `ne[e]` unused). Updated slices only ever shrink in place — the
    # paper's storage argument (§3): eliminating p frees at least one slot in every
    # updated list, so total live storage never exceeds the initial pattern. New pivot
    # elements L_p are written to the free tail; `_amd_compact!` reclaims fragmentation
    # when the tail is exhausted (paper §5: "garbage collection occurs if the elbow
    # room is exhausted ... in practice, elbow room of size n is sufficient").
    ptr = zeros(Int, n)
    ne = zeros(Int, n)
    na = zeros(Int, n)
    nv = ones(Int, n)          # |i|: original-variable count of supervariable i
    lvars = zeros(Int, n)      # |L_e| in ORIGINAL variables, for elements (Σ nv at creation)
    deg = zeros(Int, n)        # approximate external degree bound d̄ᵢ (variables)

    nzkeep = 0
    @inbounds for j in 1:n, q in colptr[j]:(colptr[j + 1] - 1)
        i = Int(rowval[q])
        (i == j || state[j] == _AMD_DENSE || state[i] == _AMD_DENSE) && continue
        nzkeep += 1
    end
    cap = nzkeep + n + 1                     # initial pattern + elbow room of n
    qw = Vector{Int}(undef, cap)
    free = 1
    @inbounds for j in 1:n
        state[j] == _AMD_DENSE && continue
        ptr[j] = free
        for q in colptr[j]:(colptr[j + 1] - 1)
            i = Int(rowval[q])
            (i == j || state[i] == _AMD_DENSE) && continue
            qw[free] = i
            free += 1
        end
        na[j] = free - ptr[j]
        deg[j] = na[j]                        # all supervariables are singletons initially
    end

    # ---- degree-list buckets (design §2.2 pt 5): O(1) insert/remove, min-scan extract ----
    dhead = zeros(Int, n)                    # dhead[d+1] heads the bucket for degree d
    dnext = zeros(Int, n)
    dprev = zeros(Int, n)
    mindeg = n
    @inbounds for j in 1:n
        state[j] == _AMD_DENSE && continue
        _amd_dl_insert!(dhead, dnext, dprev, deg[j], j)
        mindeg = min(mindeg, deg[j])
    end

    # ---- scratch state ----
    mk = zeros(Int, n)                       # generation-tagged mark array
    tag = 0
    w = zeros(Int, n)                        # Algorithm 2's w[] (offset by wflg per pivot)
    wflg = 1
    hhead = zeros(Int, n)                    # hash buckets for supervariable detection
    hnext = zeros(Int, n)
    hbuckets = Int[]                         # buckets touched this pivot (reset after use)
    svnext = zeros(Int, n)                   # supervariable member chains (output expansion)
    svtail = collect(1:n)
    stash = Vector{Int}(undef, n)            # A_i copy during the in-place list rewrite
    hmod = max(n - 1, 1)

    perm = Vector{Ti}(undef, n)
    kout = 0
    nelim = 0
    nondense = n - ndense

    while nelim < nondense
        # ---- select variable p minimizing d̄_p (Algorithm 1) ----
        d = mindeg
        @inbounds while dhead[d + 1] == 0
            d += 1
        end
        p = dhead[d + 1]
        _amd_dl_remove!(dhead, dnext, dprev, d, p)
        mindeg = d
        nvp = nv[p]

        # ---- ensure tail space for L_p (≤ one entry per live supervariable) ----
        need = nondense - nelim
        if cap - free + 1 < need
            free = _amd_compact!(qw, ptr, ne, na, state, n, free)
            if cap - free + 1 < need
                # Unreachable by the storage invariant (live ≤ initial pattern, so the
                # post-compaction tail is ≥ n+1 ≥ need); defensive fallback only.
                cap = free + need + n
                resize!(qw, cap)
            end
        end

        # ---- form L_p = (A_p ∪ ⋃_{e∈E_p} L_e) \ p̄, absorbing E_p (Algorithm 1) ----
        tag = _amd_bump_tag!(mk, tag)
        mk[p] = tag                          # excludes p̄ (dead members never appear live)
        lp = free
        lplen = 0
        degme = 0                            # |L_p| in original variables
        pp = ptr[p]
        @inbounds for q in (pp + ne[p]):(pp + ne[p] + na[p] - 1)
            v = qw[q]
            state[v] == _AMD_VAR || continue # skip stale dead entries
            mk[v] == tag && continue
            mk[v] = tag
            qw[lp + lplen] = v
            lplen += 1
            degme += nv[v]
        end
        @inbounds for q in pp:(pp + ne[p] - 1)
            e = qw[q]
            state[e] == _AMD_ELT || continue
            for r in ptr[e]:(ptr[e] + na[e] - 1)
                v = qw[r]
                state[v] == _AMD_VAR || continue
                mk[v] == tag && continue
                mk[v] = tag
                qw[lp + lplen] = v
                lplen += 1
                degme += nv[v]
            end
            state[e] = _AMD_DEADELT          # natural absorption: e ∈ E_p dies into p
            ptr[e] = 0
        end

        # convert variable p to element p (its old slice becomes garbage)
        state[p] = _AMD_ELT
        ptr[p] = lp
        ne[p] = 0
        na[p] = lplen
        lvars[p] = degme
        free = lp + lplen
        nelim += nvp
        nleft = nondense - nelim             # uneliminated non-dense original variables

        # ---- Algorithm 2: w[e] − wflg = |L_e \ L_p| for every element touching L_p ----
        # Scanning each i ∈ L_p and each e ∈ E_i decrements w[e] (initialized to |L_e|)
        # once per member variable of L_e ∩ L_p, so w[e] ends at |L_e| − |L_e ∩ L_p|.
        # The wflg offset replaces the paper's "assume w(k) < 0" reset with a
        # generation counter (no O(n) clear per pivot).
        wflg = _amd_bump_wflg!(w, wflg, n)
        @inbounds for q in lp:(lp + lplen - 1)
            i = qw[q]
            _amd_dl_remove!(dhead, dnext, dprev, deg[i], i)   # re-bucketed below
            for r in ptr[i]:(ptr[i] + ne[i] - 1)
                e = qw[r]
                state[e] == _AMD_ELT || continue
                if w[e] < wflg
                    w[e] = lvars[e] + wflg
                end
                w[e] -= nv[i]
            end
        end

        # ---- degree update (eq. (4)), list pruning, hashing ----
        @inbounds for q in lp:(lp + lplen - 1)
            i = qw[q]
            base = ptr[i]
            elen = ne[i]
            alen = na[i]
            # Stash A_i first: appending p to the pruned E_i below can overwrite A_i's
            # first slot before it is read (exactly the case where the slot freed by
            # this pivot is p ∈ A_i rather than an absorbed element of E_i).
            for t in 1:alen
                stash[t] = qw[base + elen + t - 1]
            end
            # E_i := (E_i \ absorbed) ∪ {p}  — element absorption, Algorithm 1
            wpos = base
            esum = 0                          # Σ_{e ∈ E_i \ {p}} |L_e \ L_p|
            hsum = 0
            for r in base:(base + elen - 1)
                e = qw[r]
                state[e] == _AMD_ELT || continue
                # eq. (5): every live e ∈ E_i was scanned above, so w[e] ≥ wflg holds;
                # the |L_e| fallback is defensive only.
                ext = w[e] >= wflg ? w[e] - wflg : lvars[e]
                if ext == 0 && alg.aggressive
                    # aggressive absorption (§5, design §2.2 pt 4): L_e ⊆ L_p makes e
                    # redundant even though e ∉ E_p.
                    state[e] = _AMD_DEADELT
                    ptr[e] = 0
                    continue
                end
                esum += ext
                qw[wpos] = e
                wpos += 1
                hsum += e
            end
            qw[wpos] = p
            wpos += 1
            hsum += p
            ne[i] = wpos - base
            # A_i := (A_i \ L_p) \ p̄  — redundant-entry removal, Algorithm 1
            asum = 0                          # |A_i \ i| in original variables
            for t in 1:alen
                v = stash[t]
                state[v] == _AMD_VAR || continue
                mk[v] == tag && continue      # ∈ L_p ∪ {p}: covered by element p now
                qw[wpos] = v
                wpos += 1
                asum += nv[v]
                hsum += v
            end
            na[i] = wpos - base - ne[i]
            # eq. (4), with the set differences subtracting the WHOLE supervariable 𝐢
            # (design §0 N4):
            #   d̄ᵢ = min( n − k,
            #             d̄ᵢ_prev + |L_p \ 𝐢|,
            #             |A_i \ 𝐢| + |L_p \ 𝐢| + Σ_{e ∈ E_i \ {p}} |L_e \ L_p| )
            # `nleft − nv[i]` (all other remaining variables) tightens the paper's
            # first term n − k and is still a valid upper bound on external degree.
            lpext = degme - nv[i]             # |L_p \ 𝐢|  (i ∈ L_p, so subtract nv[i])
            di = min(nleft - nv[i], deg[i] + lpext, asum + lpext + esum)
            deg[i] = max(di, 0)
            # Hash(i) = ((Σ_{j∈A_i} j + Σ_{e∈E_i} e) mod (n−1)) + 1   (paper §5)
            h = mod(hsum, hmod) + 1
            if hhead[h] == 0
                push!(hbuckets, h)
            end
            hnext[i] = hhead[h]
            hhead[h] = i
        end

        # ---- supervariable detection (design §2.2 pt 3) ----
        # Pairwise compare within hash buckets; indistinguishable in the quotient graph
        # means identical E and A lists (as sets) once both lie in L_p. Comparison is on
        # the stored lists — a stale dead element retained in one list but pruned from
        # the other can only cause a MISSED merge (a quality miss the paper also
        # accepts, §3.2), never a false one.
        for h in hbuckets
            i = hhead[h]
            while i != 0
                prevj = i
                j = hnext[i]
                while j != 0
                    same = ne[i] == ne[j] && na[i] == na[j]
                    if same
                        tag = _amd_bump_tag!(mk, tag)
                        @inbounds for r in ptr[i]:(ptr[i] + ne[i] + na[i] - 1)
                            mk[qw[r]] = tag
                        end
                        @inbounds for r in ptr[j]:(ptr[j] + ne[j] + na[j] - 1)
                            if mk[qw[r]] != tag
                                same = false
                                break
                            end
                        end
                    end
                    jn = hnext[j]
                    if same
                        # merge: i = i ∪ j, d̄ᵢ −= |j| (Algorithm 1)
                        deg[i] -= nv[j]
                        nv[i] += nv[j]
                        nv[j] = 0
                        state[j] = _AMD_DEADVAR
                        ptr[j] = 0
                        ne[j] = 0
                        na[j] = 0
                        svnext[svtail[i]] = j
                        svtail[i] = svtail[j]
                        hnext[prevj] = jn     # unlink j from the bucket
                    else
                        prevj = j
                    end
                    j = jn
                end
                i = hnext[i]
            end
            hhead[h] = 0
        end
        empty!(hbuckets)

        # ---- compact L_p of merged-away members; re-bucket the survivors ----
        # (lvars[p] = degme stays exact: a merge moves nv between two members of L_p.)
        wpos = lp
        @inbounds for q in lp:(lp + lplen - 1)
            v = qw[q]
            state[v] == _AMD_VAR || continue
            qw[wpos] = v
            wpos += 1
            _amd_dl_insert!(dhead, dnext, dprev, deg[v], v)
            mindeg = min(mindeg, deg[v])
        end
        na[p] = wpos - lp
        free = wpos                           # reclaim L_p's tail slack immediately

        # ---- emit p̄: the pivot and every variable mass-eliminated with it ----
        v = p
        while v != 0
            kout += 1
            perm[kout] = Ti(v)
            v = svnext[v]
        end
    end

    # ---- dense-stripped variables ordered last, ascending (design §2.2 pt 6) ----
    @inbounds for j in 1:n
        if state[j] == _AMD_DENSE
            kout += 1
            perm[kout] = Ti(j)
        end
    end
    @assert kout == n
    return perm
end

# Doubly-linked degree-bucket helpers (design §2.2 pt 5). `d` must be the degree under
# which `i` is currently filed.
@inline function _amd_dl_insert!(dhead::Vector{Int}, dnext::Vector{Int}, dprev::Vector{Int}, d::Int, i::Int)
    nxt = dhead[d + 1]
    dnext[i] = nxt
    dprev[i] = 0
    nxt != 0 && (dprev[nxt] = i)
    dhead[d + 1] = i
    return nothing
end

@inline function _amd_dl_remove!(dhead::Vector{Int}, dnext::Vector{Int}, dprev::Vector{Int}, d::Int, i::Int)
    prv = dprev[i]
    nxt = dnext[i]
    if prv == 0
        dhead[d + 1] = nxt
    else
        dnext[prv] = nxt
    end
    nxt != 0 && (dprev[nxt] = prv)
    return nothing
end

# Generation counters: `mk[v] == tag` means "marked this round"; bumping the tag
# invalidates all marks in O(1). The wraparound reset never fires in practice.
function _amd_bump_tag!(mk::Vector{Int}, tag::Int)
    if tag >= typemax(Int) - 1
        fill!(mk, 0)
        tag = 0
    end
    return tag + 1
end

# w[] generation: values from a previous pivot satisfy w[e] < wflg after the bump
# (per-pivot values never exceed wflg + n), replacing the paper's "assume w(k) < 0".
function _amd_bump_wflg!(w::Vector{Int}, wflg::Int, n::Int)
    if wflg >= typemax(Int) - 2 * (n + 1)
        fill!(w, 0)
        wflg = 1
    end
    return wflg + n + 1
end

# In-place garbage compaction (design §2.2 pt 1; paper §5's "garbage collection"):
# slide every live slice to the front of the workspace, in address order, and return
# the new free-tail start. Slices only ever shrink, so the move is always leftward and
# a forward copy is safe.
function _amd_compact!(qw::Vector{Int}, ptr::Vector{Int}, ne::Vector{Int}, na::Vector{Int},
        state::Vector{Int8}, n::Int, free::Int)
    live = Int[]
    for v in 1:n
        ptr[v] > 0 || continue
        (state[v] == _AMD_VAR || state[v] == _AMD_ELT) || continue
        push!(live, v)
    end
    sort!(live; by = v -> ptr[v])
    wpos = 1
    for v in live
        len = state[v] == _AMD_VAR ? ne[v] + na[v] : na[v]
        src = ptr[v]
        ptr[v] = wpos
        if src != wpos
            @inbounds for t in 0:(len - 1)
                qw[wpos + t] = qw[src + t]
            end
        end
        wpos += len
    end
    return wpos
end
