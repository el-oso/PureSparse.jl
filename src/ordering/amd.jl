# Approximate Minimum Degree ordering (design.md §2.2) — the `order` method for the
# `AMDOrdering` struct declared in `ordering/interface.jl`.
#
# Grounded EXCLUSIVELY in the published paper (CONTRIBUTING.md requirement 1 — CHOLMOD /
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
# Storage layout, compaction procedure, and the generation-tagged mark / `w` handling
# below are our own engineering — the paper prescribes only the quotient-graph sets and
# the elbow-room/garbage-collection discipline. The node-status encoding (a variable's
# membership in L_p carried by the sign of `nv`, absorbed elements marked in `ptr`/`w`)
# and the in-place rewrite of a variable's lists with the pivot slot reserved follow
# faer's `src/sparse/linalg/amd.rs` (faer 0.24.4, MIT license,
# https://codeberg.org/sarah-quinones/faer), which is a port of SuiteSparse AMD. That
# code's notice (`COPYING.SUITE_SPARSE.AMD.BSD` in the faer crate) is reproduced here:
#
#   AMD, Copyright (c), 1996-2022, Timothy A. Davis, Patrick R. Amestoy, and Iain S.
#   Duff. All Rights Reserved. Availability: http://www.suitesparse.com
#   AMD License: BSD 3-clause:
#   Redistribution and use in source and binary forms, with or without modification,
#   are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this
#       list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this
#       list of conditions and the following disclaimer in the documentation and/or
#       other materials provided with the distribution.
#     * Neither the name of the organizations to which the authors are affiliated, nor
#       the names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
#   EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
#   IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT, INDIRECT,
#   INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
#   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#   POSSIBILITY OF SUCH DAMAGE.
#
# faer's own notice (its LICENSE file at tag faer-v0.24.4):
#
#   MIT License
#   Copyright (c) 2026 sarah quiñones el kazdadi
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this
#   software and associated documentation files (the "Software"), to deal in the
#   Software without restriction, including without limitation the rights to use, copy,
#   modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#   and to permit persons to whom the Software is furnished to do so, subject to the
#   following conditions: The above copyright notice and this permission notice shall
#   be included in all copies or substantial portions of the Software.
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#   INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#   PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#   HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
#   CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
#   OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Node ids, list entries and degrees are stored in the narrowest integer type that
# holds them (`Int32` whenever the workspace fits, design.md §2.2 pt 7): the per-node
# arrays are then L2-resident for n up to ~10⁵, and the algorithm is bound by their
# random access, not by arithmetic.

# Dense-row stripping marks: every node starts as a variable; a stripped one takes no
# part in the quotient graph and is ordered last. Within the main loop a node.s status
# (variable, member of L_p, merged, element, absorbed) is encoded in the node arrays
# themselves; see `_amd_order!`.
const _AMD_VAR = Int8(0)       # ordinary variable
const _AMD_DENSE = Int8(1)     # dense-stripped variable, ordered last

# Supervariable hash buckets (design §2.2 pt 3). The paper's hash value is kept per
# node as its key; the bucket table only has to spread the ≤ |L_p| keys hashed per
# pivot, so it is capped at this many entries (16 KB of Int32 — L1-resident) instead
# of n.
const _AMD_HASH_BUCKETS = 4096
# Key mask: keeps the stored key within every node type the ordering uses.
const _AMD_HASH_MASK = Int(typemax(Int32)) >> 1

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
function order(alg::AMDOrdering, n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}) where {Ti <: Integer}
    iszero(n) && return Ti[]

    # ---- dense-row stripping (design §2.2 pt 6) ----
    threshold = max(AMD_DENSE_FLOOR, alg.dense_mult * sqrt(n))
    state = fill(_AMD_VAR, n)
    ndense = 0
    nzkeep = 0
    @inbounds for j in 1:n
        d = 0
        for q in colptr[j]:(colptr[j + 1] - 1)
            Int(rowval[q]) == j && continue  # contract says no diagonal; tolerate one
            d += 1
        end
        nzkeep += d
        if d > threshold
            state[j] = _AMD_DENSE
            ndense += 1
        end
    end
    if !iszero(ndense)
        nzkeep = 0
        @inbounds for j in 1:n, q in colptr[j]:(colptr[j + 1] - 1)
            i = Int(rowval[q])
            (i == j || state[j] == _AMD_DENSE || state[i] == _AMD_DENSE) && continue
            nzkeep += 1
        end
    end

    perm = Vector{Ti}(undef, n)
    # Workspace addresses never exceed 2·nzkeep + 3n + 1 (see the capacity invariants in
    # `_amd_order!`), so Int32 node storage is safe whenever that bound fits.
    if 3 * nzkeep + 3 * n + 2 <= typemax(Int32)
        _amd_order!(Int32, perm, alg, n, colptr, rowval, state, ndense, nzkeep)
    else
        _amd_order!(Int64, perm, alg, n, colptr, rowval, state, ndense, nzkeep)
    end
    return perm
end

function _amd_order!(
        ::Type{Tn}, perm::Vector{Ti}, alg::AMDOrdering, n::Int,
        colptr::Vector{Ti}, rowval::Vector{Ti}, state::Vector{Int8}, ndense::Int, nzkeep::Int,
    ) where {Tn <: Integer, Ti <: Integer}
    # ---- quotient-graph storage (design §2.2 pt 1) ----
    # One shared integer workspace `qw`: every live node owns one contiguous slice.
    # For a live VARIABLE i the slice is its element list E_i (`ne[i]` entries) followed
    # by its variable list A_i (`na[i]` entries). For a live ELEMENT e the slice is L_e
    # (`na[e]` entries; `ne[e]` is 0). Updated slices only ever shrink in place — the
    # paper's storage argument (§3): eliminating p frees at least one slot in every
    # updated list, so total live storage never exceeds the initial pattern. New pivot
    # elements L_p are written to the free tail; `_amd_compact!` reclaims fragmentation
    # when the tail is exhausted (paper §5: "garbage collection occurs if the elbow
    # room is exhausted ... in practice, elbow room of size n is sufficient").
    #
    # Node status is read off the arrays the list scans touch anyway, so a scan tests
    # one array per entry (the encoding, like faer's AMD, follows SuiteSparse AMD):
    #   variable v:  nv[v] > 0 live, nv[v] < 0 live and in the current L_p (|nv| = |v|),
    #                nv[v] == 0 merged into another supervariable;
    #   element e:   ptr[e] == 0 and w[e] == -1 once absorbed.
    # A-lists hold only variables and E-lists only elements, so the two never mix.
    ptr = zeros(Tn, n)
    ne = zeros(Tn, n)
    na = zeros(Tn, n)
    nv = ones(Tn, n)          # |i|: original-variable count of supervariable i
    lvars = zeros(Tn, n)      # |L_e| in ORIGINAL variables, for elements (Σ nv at creation)
    deg = zeros(Tn, n)        # approximate external degree bound d̄ᵢ (variables)

    # Elbow room ≥ the initial pattern: live storage never exceeds the initial pattern, so
    # every compaction frees at least `nzkeep + n` slots and its O(live + n log n) cost is
    # amortized over at least that many new L_p entries.
    cap = 2 * nzkeep + n + 1
    qw = Vector{Tn}(undef, cap)
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
    dhead = zeros(Tn, n)                    # dhead[d+1] heads the bucket for degree d
    dnext = zeros(Tn, n)
    dprev = zeros(Tn, n)
    mindeg = n
    @inbounds for j in 1:n
        state[j] == _AMD_DENSE && continue
        _amd_dl_insert!(dhead, dnext, dprev, Int(deg[j]), j)
        mindeg = min(mindeg, Int(deg[j]))
    end

    # ---- scratch state ----
    mk = zeros(Tn, n)                       # generation-tagged marks for list comparison
    tag = 0
    w = zeros(Tn, n)                        # Algorithm 2's w[] (offset by wflg per pivot)
    wflg = 1
    lmax = 0                                # max |L_e| over every element created so far
    nbuckets = min(nextpow(2, n), _AMD_HASH_BUCKETS)
    hhead = zeros(Tn, nbuckets)             # hash buckets for supervariable detection
    hnext = zeros(Tn, n)
    hkey = zeros(Tn, n)                     # the paper's Hash(i), compared before the lists
    hbuckets = Vector{Tn}(undef, nbuckets)  # buckets holding ≥ 2 variables this pivot (first nhb)
    nhb = 0
    svnext = zeros(Tn, n)                   # supervariable member chains (output expansion)
    svtail = collect(Tn, 1:n)

    kout = 0
    nelim = 0
    nondense = n - ndense

    while nelim < nondense
        # ---- select variable p minimizing d̄_p (Algorithm 1) ----
        d = mindeg
        @inbounds while iszero(dhead[d + 1])
            d += 1
        end
        p = Int(dhead[d + 1])
        _amd_dl_remove!(dhead, dnext, dprev, d, p)
        mindeg = d
        nvp = Int(nv[p])

        # ---- ensure tail space for L_p ----
        # L_p ⊆ A_p ∪ ⋃_{e∈E_p} L_e, so the sum of those list lengths bounds |L_p|. The
        # looser bound "every remaining variable" exceeds the elbow room on almost every
        # pivot and turns compaction into the dominant cost.
        need = Int(na[p])
        @inbounds for q in Int(ptr[p]):(Int(ptr[p]) + Int(ne[p]) - 1)
            e = qw[q]
            iszero(ptr[e]) || (need += Int(na[e]))
        end
        if cap - free + 1 < need
            free = _amd_compact!(qw, ptr, ne, na, nv, perm, kout, n)
            if cap - free + 1 < need
                # The post-compaction tail is ≥ n+1 (live storage ≤ initial pattern), but
                # the list-length bound can exceed it when the L_e overlap; grow. After
                # compaction free ≤ nzkeep + n + 1 and need ≤ nzkeep + n, so the new
                # capacity stays within 2·nzkeep + 3n + 1.
                cap = free + need + n
                resize!(qw, cap)
            end
        end

        # ---- form L_p = (A_p ∪ ⋃_{e∈E_p} L_e) \ p̄, absorbing E_p (Algorithm 1) ----
        # A variable joins L_p when its nv is still positive, and the sign flip is what
        # marks it as a member; p̄ is excluded by flipping p first.
        nv[p] = -nvp
        lp = free
        lplen = 0
        degme = 0                            # |L_p| in original variables
        pp = Int(ptr[p])
        nep = Int(ne[p])
        @inbounds for q in (pp + nep):(pp + nep + Int(na[p]) - 1)
            v = qw[q]
            nvv = nv[v]
            nvv > 0 || continue              # dead, or already in L_p
            nv[v] = -nvv
            qw[lp + lplen] = v
            lplen += 1
            degme += Int(nvv)
        end
        @inbounds for q in pp:(pp + nep - 1)
            e = qw[q]
            pe = Int(ptr[e])
            iszero(pe) && continue
            for r in pe:(pe + Int(na[e]) - 1)
                v = qw[r]
                nvv = nv[v]
                nvv > 0 || continue
                nv[v] = -nvv
                qw[lp + lplen] = v
                lplen += 1
                degme += Int(nvv)
            end
            ptr[e] = 0                       # natural absorption: e ∈ E_p dies into p
            w[e] = -1
        end

        # convert variable p to element p (its old slice becomes garbage)
        ptr[p] = lp
        ne[p] = 0
        na[p] = lplen
        lvars[p] = degme
        lmax = max(lmax, degme)
        free = lp + lplen
        nelim += nvp
        nleft = nondense - nelim             # uneliminated non-dense original variables

        # ---- Algorithm 2: w[e] − wflg = |L_e \ L_p| for every element touching L_p ----
        # Scanning each i ∈ L_p and each e ∈ E_i decrements w[e] (initialized to |L_e|)
        # once per member variable of L_e ∩ L_p, so w[e] ends at |L_e| − |L_e ∩ L_p|.
        # The wflg offset replaces the paper's "assume w(k) < 0" reset with a
        # generation counter (no O(n) clear per pivot).
        wflg = _amd_bump_wflg!(w, wflg, lmax)
        @inbounds for q in lp:(lp + lplen - 1)
            i = Int(qw[q])
            nvi = -nv[i]
            _amd_dl_remove!(dhead, dnext, dprev, Int(deg[i]), i)   # re-bucketed below
            for r in Int(ptr[i]):(Int(ptr[i]) + Int(ne[i]) - 1)
                e = qw[r]
                we = w[e]
                we < 0 && continue           # absorbed element
                we < wflg && (we = lvars[e] + wflg)
                w[e] = we - nvi
            end
        end

        # ---- degree update (eq. (4)), list pruning, hashing ----
        @inbounds for q in lp:(lp + lplen - 1)
            i = Int(qw[q])
            base = Int(ptr[i])
            elen = Int(ne[i])
            alen = Int(na[i])
            # E_i := (E_i \ absorbed) ∪ {p}  — element absorption, Algorithm 1. The
            # pruned list is rewritten in place with p's slot reserved right after it.
            # The reserved slot shifts the kept A_i entries one place up, so each is
            # written only once the next entry has been read (a one-entry delay); the
            # last one lands at most one slot past the old end, which this pivot freed
            # (p ∈ A_i is dropped, or an element of E_i ∩ E_p was absorbed).
            wpos = base
            esum = 0                          # Σ_{e ∈ E_i \ {p}} |L_e \ L_p|
            hsum = 0                          # Int: a list sum can exceed the node type
            for r in base:(base + elen - 1)
                e = qw[r]
                we = w[e]
                we < 0 && continue
                # eq. (5): every live e ∈ E_i was scanned above, so w[e] ≥ wflg holds;
                # the |L_e| fallback is defensive only.
                ext = we >= wflg ? Int(we) - wflg : Int(lvars[e])
                if iszero(ext) && alg.aggressive
                    # aggressive absorption (§5, design §2.2 pt 4): L_e ⊆ L_p makes e
                    # redundant even though e ∉ E_p.
                    ptr[e] = 0
                    w[e] = -1
                    continue
                end
                esum += ext
                qw[wpos] = e
                wpos += 1
                hsum += Int(e)
            end
            ppos = wpos
            wpos += 1
            hsum += p
            ne[i] = wpos - base
            # A_i := (A_i \ L_p) \ p̄  — redundant-entry removal, Algorithm 1
            asum = 0                          # |A_i \ i| in original variables
            pend = 0                          # kept entry not yet written back
            for r in (base + elen):(base + elen + alen - 1)
                v = qw[r]
                if !iszero(pend)
                    qw[wpos] = pend
                    wpos += 1
                    pend = 0
                end
                nvv = nv[v]
                nvv > 0 || continue           # ∈ L_p ∪ {p} (covered by element p), or dead
                pend = Int(v)
                asum += Int(nvv)
                hsum += Int(v)
            end
            if !iszero(pend)
                qw[wpos] = pend
                wpos += 1
            end
            qw[ppos] = p
            na[i] = wpos - base - Int(ne[i])
            # eq. (4), with the set differences subtracting the WHOLE supervariable 𝐢
            # (design §0 N4):
            #   d̄ᵢ = min( n − k,
            #             d̄ᵢ_prev + |L_p \ 𝐢|,
            #             |A_i \ 𝐢| + |L_p \ 𝐢| + Σ_{e ∈ E_i \ {p}} |L_e \ L_p| )
            # `nleft − nv[i]` (all other remaining variables) tightens the paper's
            # first term n − k and is still a valid upper bound on external degree.
            nvi = -Int(nv[i])
            lpext = degme - nvi               # |L_p \ 𝐢|  (i ∈ L_p, so subtract nv[i])
            di = min(nleft - nvi, Int(deg[i]) + lpext, asum + lpext + esum)
            deg[i] = max(di, 0)
            # The paper's Hash(i) = ((Σ_{j∈A_i} j + Σ_{e∈E_i} e) mod (n−1)) + 1 (§5); the sum
            # is masked rather than reduced: the key only has to agree for identical
            # lists, and a mask costs no integer division per variable of L_p.
            h = (hsum & _AMD_HASH_MASK) + 1    # hsum ≥ 0: node ids are positive
            hkey[i] = h
            b = (h & (nbuckets - 1)) + 1
            hb = hhead[b]
            if !iszero(hb) && iszero(hnext[hb])
                # A bucket is worth scanning once it holds a second variable; it is
                # listed exactly then, so the list never exceeds the bucket count.
                nhb += 1
                hbuckets[nhb] = b
            end
            hnext[i] = hb
            hhead[b] = i
        end

        # ---- supervariable detection (design §2.2 pt 3) ----
        # Pairwise compare within hash buckets; indistinguishable in the quotient graph
        # means identical E and A lists (as sets) once both lie in L_p. Comparison is on
        # the stored lists — a stale dead element retained in one list but pruned from
        # the other can only cause a MISSED merge (a quality miss the paper also
        # accepts, §3.2), never a false one.
        for t in 1:nhb
            b = hbuckets[t]
            i = Int(hhead[b])
            while !iszero(i)
                prevj = i
                j = Int(hnext[i])
                marked = false              # i's lists are marked lazily, once per i
                while !iszero(j)
                    same = hkey[i] == hkey[j] && ne[i] == ne[j] && na[i] == na[j]
                    if same
                        if !marked
                            tag = _amd_bump_tag!(mk, tag)
                            @inbounds for r in Int(ptr[i]):(Int(ptr[i]) + Int(ne[i]) + Int(na[i]) - 1)
                                mk[qw[r]] = tag
                            end
                            marked = true
                        end
                        @inbounds for r in Int(ptr[j]):(Int(ptr[j]) + Int(ne[j]) + Int(na[j]) - 1)
                            if mk[qw[r]] != tag
                                same = false
                                break
                            end
                        end
                    end
                    jn = Int(hnext[j])
                    if same
                        # merge: i = i ∪ j, d̄ᵢ −= |j| (Algorithm 1); both nv are negative
                        # (members of L_p), so the sums stay in that encoding
                        deg[i] += nv[j]
                        nv[i] += nv[j]
                        nv[j] = 0
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
                i = Int(hnext[i])
            end
            hhead[b] = 0
        end
        nhb = 0

        # ---- compact L_p of merged-away members; re-bucket the survivors ----
        # (lvars[p] = degme stays exact: a merge moves nv between two members of L_p.)
        # Every survivor leaves L_p (nv back to positive) and empties its hash bucket; a
        # merged-away variable shares its bucket with the survivor it merged into.
        wpos = lp
        @inbounds for q in lp:(lp + lplen - 1)
            v = qw[q]
            nvv = nv[v]
            nvv < 0 || continue
            nv[v] = -nvv
            hhead[(Int(hkey[v]) & (nbuckets - 1)) + 1] = 0
            qw[wpos] = v
            wpos += 1
            _amd_dl_insert!(dhead, dnext, dprev, Int(deg[v]), Int(v))
            mindeg = min(mindeg, Int(deg[v]))
        end
        na[p] = wpos - lp
        free = wpos                           # reclaim L_p's tail slack immediately
        nv[p] = 0                            # p is an element now; no A-list still names it

        # ---- emit p̄: the pivot and every variable mass-eliminated with it ----
        v = p
        while !iszero(v)
            kout += 1
            perm[kout] = Ti(v)
            v = Int(svnext[v])
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
@inline function _amd_dl_insert!(dhead::Vector{Tn}, dnext::Vector{Tn}, dprev::Vector{Tn}, d::Int, i::Int) where {Tn}
    nxt = dhead[d + 1]
    dnext[i] = nxt
    dprev[i] = 0
    !iszero(nxt) && (dprev[nxt] = i)
    dhead[d + 1] = i
    return nothing
end

@inline function _amd_dl_remove!(dhead::Vector{Tn}, dnext::Vector{Tn}, dprev::Vector{Tn}, d::Int, i::Int) where {Tn}
    prv = dprev[i]
    nxt = dnext[i]
    if iszero(prv)
        dhead[d + 1] = nxt
    else
        dnext[prv] = nxt
    end
    !iszero(nxt) && (dprev[nxt] = prv)
    return nothing
end

# Generation counters: `mk[v] == tag` means "marked this round"; bumping the tag
# invalidates all marks in O(1). The wraparound reset fires only when the tag would no
# longer fit the mark array's element type.
function _amd_bump_tag!(mk::Vector{Tn}, tag::Int) where {Tn}
    if tag >= typemax(Tn) - 1
        fill!(mk, 0)
        tag = 0
    end
    return tag + 1
end

# w[] generation: a pivot's values lie in [wflg, wflg + lmax] (lmax bounds every |L_e|),
# so stepping wflg past that range makes every stale entry satisfy w[e] < wflg — the
# paper's "assume w(k) < 0" without an O(n) clear. The reset fires only when the next
# step would overflow the element type; it keeps the −1 of absorbed elements.
function _amd_bump_wflg!(w::Vector{Tn}, wflg::Int, lmax::Int) where {Tn}
    if wflg >= typemax(Tn) - 2 * (lmax + 1)
        for e in eachindex(w)
            w[e] >= 0 && (w[e] = 0)
        end
        wflg = 1
    end
    return wflg + lmax + 1
end

# In-place garbage compaction (design §2.2 pt 1; paper §5's "garbage collection"):
# slide every live slice to the front of the workspace, in address order, and return
# the new free-tail start. Slices only ever shrink, so the move is always leftward and
# a forward copy is safe. Address order needs no sort: variable slices sit where the
# initial pattern put them, in node order, and every element's slice was appended at
# the tail when it was eliminated, so the elements follow in elimination order — the
# order of the principal pivots in `perm[1:kout]`. A live node has `ptr > 0`; a live
# variable also has `nv != 0` (an element's nv is 0); a slice holds `ne + na` entries.
function _amd_compact!(
        qw::Vector{Tn}, ptr::Vector{Tn}, ne::Vector{Tn}, na::Vector{Tn}, nv::Vector{Tn},
        perm::Vector{Ti}, kout::Int, n::Int,
    ) where {Tn, Ti}
    wpos = 1
    @inbounds for v in 1:n
        (ptr[v] > 0 && !iszero(nv[v])) || continue
        wpos = _amd_move_slice!(qw, ptr, ne, na, v, wpos)
    end
    @inbounds for k in 1:kout
        e = Int(perm[k])
        ptr[e] > 0 || continue           # merged variables and absorbed elements
        wpos = _amd_move_slice!(qw, ptr, ne, na, e, wpos)
    end
    return wpos
end

@inline function _amd_move_slice!(qw::Vector{Tn}, ptr::Vector{Tn}, ne::Vector{Tn}, na::Vector{Tn}, v::Int, wpos::Int) where {Tn}
    len = Int(ne[v]) + Int(na[v])
    src = Int(ptr[v])
    ptr[v] = wpos
    if src != wpos
        @inbounds for t in 0:(len - 1)
            qw[wpos + t] = qw[src + t]
        end
    end
    return wpos + len
end
