# Simplicial LDLᵀ conversion, Davis–Hager rank-1 update/downdate, and simplicial split
# solves (design.md §7, §6/§0 N7). Sole algorithmic source: Davis & Hager, *Modifying a
# Sparse Cholesky Factorization*, SIAM J. Matrix Anal. Appl. 20(3), pp. 606–627, 1999
# (`refs/linear_algebra/modify_sparse_cholesky.pdf`), read directly — never CHOLMOD
# source (CLAUDE.md req 1 / design.md §11). Section/equation references below are to
# that paper. Multiple-rank is sequenced single-rank per design.md §7's explicit v1
# scope (the 2001 batched follow-up is a listed extension, not implemented here).

"""
    simplicial(F::LDLFactor{T,Ti}; grow = SIMPLICIAL_GROW) -> SimplicialLDLFactor{T,Ti}

One-time conversion of a supernodal [`LDLFactor`](@ref) into the per-column simplicial
representation that [`updowndate!`](@ref) operates on (design.md §1.2/§6). **Allocates**
— this is a cold/setup path; the subsequent `updowndate!`/split-solve calls on the
result do not allocate.

Each column is stored with slack capacity `min(n - j, max(len, ceil(grow*(len + 1))))`
(`len` = the column's initial strictly-lower count) so that update fill can be absorbed
in place; `grow` defaults to the `simplicial_grow` Preference (see `tuning.jl` for the
derivation of the sizing rule).

# Pattern provenance note

The per-column pattern extracted here is the column's slice of its supernode's
`rowind` — i.e. the supernodal pattern *including* relaxed-amalgamation padding, whose
entries hold exact zeros (they are assembled as zero and every update term touching
them is structurally zero, so they stay exactly zero in floating point). This superset
is used instead of the true per-column L-pattern because it is (a) directly available
without a symbolic recomputation and (b) **closed** in the sense the Davis–Hager walk
requires (paper eq. (3.1)/Prop. 3.1: entries of a column's pattern at or below its
parent must appear in the parent's pattern): within a supernode's diagonal block the
next stored row after `j` is `j+1` and the two slices nest trivially, and across
supernodes the below-diagonal rows of `s` at or past `first(ancestor)` are contained in
the ancestor's `rowind` — the same §4.3/§9.1 superset invariant the numeric supernodal
loop already relies on. `parent` is therefore derived from the *stored* pattern
(`min` of it, paper §2), not copied from `sym.parent`; walking a padded column is a
harmless no-op in the numeric recurrence (paper §5.2: iterations with `w_j = 0` change
nothing).
"""
function simplicial(F::LDLFactor{T,Ti}; grow::Real = SIMPLICIAL_GROW) where {T,Ti<:Integer}
    grow >= 0 || throw(ArgumentError("simplicial: grow must be nonnegative (got $grow)"))
    sym = F.sym
    n = sym.n
    super = sym.super
    rowind_ptr = sym.rowind_ptr
    rowind = sym.rowind

    # slot layout: capacity per column from its strictly-lower supernodal count
    colptr = Vector{Ti}(undef, n + 1)
    colptr[1] = one(Ti)
    @inbounds for s in 1:sym.nsuper
        j0 = Int(super[s])
        j1 = Int(super[s + 1]) - 1
        nsrow = Int(rowind_ptr[s + 1]) - Int(rowind_ptr[s])
        for j in j0:j1
            len = nsrow - (j - j0 + 1)          # rows strictly below j in rowind(s)
            cap = min(n - j, max(len, ceil(Int, grow * (len + 1))))
            colptr[j + 1] = colptr[j] + Ti(cap)
        end
    end

    nslots = Int(colptr[n + 1]) - 1
    rowval = Vector{Ti}(undef, nslots)
    nzval = Vector{T}(undef, nslots)
    colnnz = Vector{Ti}(undef, n)
    parent = Vector{Ti}(undef, n)
    used = 0

    @inbounds for s in 1:sym.nsuper
        j0 = Int(super[s])
        j1 = Int(super[s + 1]) - 1
        nscol = j1 - j0 + 1
        rp0 = Int(rowind_ptr[s])
        nsrow = Int(rowind_ptr[s + 1]) - rp0
        panel = F.panels[s]
        for j in j0:j1
            c = j - j0 + 1                       # local column, also j's position in rowind(s)
            len = nsrow - c
            p = Int(colptr[j]) - 1
            for k in 1:len
                rowval[p + k] = rowind[rp0 + c + k - 1]
                nzval[p + k] = panel[c + k, c]
            end
            colnnz[j] = Ti(len)
            parent[j] = len > 0 ? rowval[p + 1] : zero(Ti)
            used += len
        end
    end

    stats = FactorStats()
    stats.nnzL = used + n                        # + n implicit unit diagonals
    G = SimplicialLDLFactor{T,Ti}(
        sym, colptr, colnnz, rowval, nzval, copy(F.d), parent,
        zeros(T, n), Vector{Ti}(undef, n), stats, F.ok,
    )
    return G
end

"""
    updowndate!(G::SimplicialLDLFactor{T}, w::AbstractVector{T}, sigma::Integer) -> Symbol

Rank-1 modification of the factored matrix: after a `:ok` return, `G` is the LDLᵀ
factor of `A + sigma·w·wᵀ` (`sigma = +1` update, `sigma = -1` downdate), where `A` is
whatever `G` factored before the call. `w` is given in `A`'s **original** row order
(like `ldlt`'s inputs); it is permuted internally. In-place, **zero allocations**, and
O(nnz of the changed columns): only the columns on the elimination-tree path from
`min(support(w))` to the root are touched — Davis & Hager 1999, Theorem 5.2 (downdate:
the pattern of `v = L⁻¹w` is exactly the path `P(k)` in the old tree, `k` per their
eq. (5.1)) and Corollary 5.3 (update: the path `P̄(k)` in the new tree).

Returns one of:

- `:ok` — modification applied; `G` now factors the modified matrix.
- `:refactor_required` — an update's new fill did not fit some column's slack
  capacity (design.md §7's documented overflow contract: no reallocation ever
  happens). `G.ok` is set `false` — columns earlier on the path were already
  modified, so the factor contents are no longer meaningful; rebuild via
  `simplicial(ldlt(...))`. Fill discovery follows the paper's symbolic update
  (Theorem 4.1 / Algorithm 3 Case 1, generalized in §6 Algorithm 6a: the support of
  `w` is merged into the path columns' patterns; a downdate never *removes* stored
  entries here — they are kept as explicit zeros, a valid closed superset, rather
  than running the paper's multiset-subtraction downdate phase, Algorithm 4/6b).
- `:not_definite` — the recurrence signal `ᾱ ≤ 0` fired at some pivot (recorded in
  `G.stats.fail_col`; `G.ok` set `false`). In Algorithm 5 (§5.1, p. 617) the new
  pivot is `d̄ⱼ = (ᾱ/α)·dⱼ` with `α > 0` maintained from the start (`α = 1`), so
  `ᾱ ≤ 0` at column j means the modified matrix's j-th pivot would vanish or flip
  sign: for a positive-definite factor this is exactly "the downdate destroys
  positive-definiteness" (design.md §7), detected *inside* the recurrence before the
  bad pivot is committed, not by inspecting `d` afterwards; for an SQD factor (mixed
  pivot signs) it equally catches an update against a negative pivot — either way
  the modification would change the factor's inertia and is refused.

The numeric recurrence is Algorithm 5 verbatim (their sparse "Method C1, modified" of
Gill et al. — the paper notes it is equivalent, after diagonal scaling, to Pan's
stable orthogonal method): per path column `j`, with `σ = ±1`,

    ᾱ = α + σ·wⱼ²/dⱼ;  γ = wⱼ/(ᾱ·dⱼ);  d̄ⱼ = (ᾱ/α)·dⱼ;  α = ᾱ
    w[i]  -= wⱼ·L[i,j]          (i in column j's pattern)
    L[i,j] += σ·γ·w[i]          (using the just-updated w[i])

and columns with `wⱼ = 0` are skipped (§5.2). Path columns whose parent changes get
`G.parent[j]` rewritten to `min` of the merged pattern (Algorithm 3: `π̄(j) =
min L̄ⱼ \\ {j}`); Theorem 4.1 guarantees no off-path parent changes.
"""
function updowndate!(G::SimplicialLDLFactor{T,Ti}, w::AbstractVector{T}, sigma::Integer) where {T,Ti<:Integer}
    (sigma == 1 || sigma == -1) || throw(ArgumentError("updowndate!: sigma must be +1 or -1 (got $sigma)"))
    n = G.sym.n
    length(w) == n || throw(DimensionMismatch("updowndate!: length(w) = $(length(w)), need n = $n"))
    G.ok || throw(ArgumentError("updowndate!: factor is invalid (ok = false); rebuild it via simplicial(ldlt(...))"))

    perm = G.sym.perm
    colptr = G.colptr
    colnnz = G.colnnz
    rowval = G.rowval
    nzval = G.nzval
    dvec = G.d
    parent = G.parent
    wval = G.wval
    wpat = G.wpat

    # Scatter w into factor order and collect its sorted support (scanning in factor
    # order yields sorted indices for free). O(n) on the input vector — unavoidable for
    # a dense w; the factor-modification work below is O(changed nnz).
    m = 0
    @inbounds for k in 1:n
        v = w[Int(perm[k])]
        iszero(v) && continue
        m += 1
        wpat[m] = Ti(k)
        wval[k] = v
    end
    m == 0 && return :ok

    sig = T(sigma)
    alpha = one(T)

    # Walk the etree path from k = min(support(w)) (Thm 5.2 / Cor 5.3). Candidate fill
    # for the CURRENT column = previous path column's (already-merged) pattern minus
    # its head — by eq. (3.1) closure those are exactly the still-live w entries; for
    # the first column the candidates are the support of w itself (Alg 3 Case 1).
    cand = wpat                                  # current candidate array (sorted)
    clo, chi = 2, m                              # live candidate range within it
    j = Int(wpat[1])

    @inbounds while true
        p0 = Int(colptr[j])
        len = Int(colnnz[j])
        cap = Int(colptr[j + 1]) - p0

        # ---- symbolic step: merge candidates into column j's pattern (Alg 3/6a) ----
        if chi >= clo
            # count candidates not already present (two-pointer over sorted lists)
            extra = 0
            ai = p0
            aend = p0 + len
            for t in clo:chi
                c = Int(cand[t])
                while ai < aend && Int(rowval[ai]) < c
                    ai += 1
                end
                if !(ai < aend && Int(rowval[ai]) == c)
                    extra += 1
                end
            end
            if len + extra > cap
                # design.md §7 overflow contract: no reallocation, caller refactors.
                # Live wval entries here are ⊆ {j} ∪ candidates (see cleanup note
                # below) — zero them so wval's all-zero invariant survives.
                wval[j] = zero(T)
                for t in clo:chi
                    wval[Int(cand[t])] = zero(T)
                end
                G.ok = false
                G.stats.fail_col = j
                return :refactor_required
            end
            if extra > 0
                # backward in-place sorted merge; new slots enter with L value 0
                wp = p0 + len + extra - 1
                ai = p0 + len - 1
                bi = chi
                while bi >= clo
                    c = Int(cand[bi])
                    if ai >= p0 && Int(rowval[ai]) > c
                        rowval[wp] = rowval[ai]
                        nzval[wp] = nzval[ai]
                        ai -= 1
                    elseif ai >= p0 && Int(rowval[ai]) == c
                        rowval[wp] = rowval[ai]
                        nzval[wp] = nzval[ai]
                        ai -= 1
                        bi -= 1
                    else
                        rowval[wp] = Ti(c)
                        nzval[wp] = zero(T)
                        bi -= 1
                    end
                    wp -= 1
                end
                len += extra
                colnnz[j] = Ti(len)
                G.stats.nnzL += extra
            end
        end

        # ---- numeric step: Algorithm 5 (§5.1), skipping w_j = 0 columns (§5.2) ----
        wj = wval[j]
        wval[j] = zero(T)                        # consumed; keeps wval all-zero on exit
        if !iszero(wj)
            dj = dvec[j]
            abar = alpha + sig * wj * wj / dj
            if !(abar > zero(T))
                # d̄_j = (ᾱ/α)d_j with α > 0 would vanish or flip sign: inertia change.
                # Live wval entries are ⊆ column j's merged pattern — zero them.
                for p in p0:(p0 + len - 1)
                    wval[Int(rowval[p])] = zero(T)
                end
                G.ok = false
                G.stats.fail_col = j
                return :not_definite
            end
            gamma = wj / (abar * dj)
            dvec[j] = (abar / alpha) * dj
            alpha = abar
            for p in p0:(p0 + len - 1)
                i = Int(rowval[p])
                wi = wval[i] - wj * nzval[p]
                wval[i] = wi
                nzval[p] += sig * gamma * wi
            end
        end

        # ---- next path node: π̄(j) = min L̄_j \ {j} (Alg 3), 0 pattern-empty = root ----
        newpar = len > 0 ? Int(rowval[p0]) : 0
        parent[j] = Ti(newpar)
        newpar == 0 && break
        cand = rowval
        clo = p0 + 1                             # merged pattern minus its head
        chi = p0 + len - 1
        j = newpar
    end

    G.stats.fail_col = 0
    return :ok
end

"""
    updowndate!(G::SimplicialLDLFactor{T}, W::AbstractMatrix{T}, sigma::Integer) -> Symbol

Rank-k modification `A + sigma·W·Wᵀ` as `size(W, 2)` sequenced rank-1 calls — design.md
§7's explicit v1 scope (the Davis–Hager 2001 batched multiple-rank variant is a listed
extension, not implemented). Stops at the first non-`:ok` column and returns its
status; the factor is then invalid (`G.ok == false`) exactly as for the rank-1 case.
"""
function updowndate!(G::SimplicialLDLFactor{T,Ti}, W::AbstractMatrix{T}, sigma::Integer) where {T,Ti<:Integer}
    for c in axes(W, 2)
        status = updowndate!(G, view(W, :, c), sigma)
        status === :ok || return status
    end
    return :ok
end

# ---------------------------------------------------------------------------
# Split solves (design.md §6/§0 N7): plain CSC column loops, no PureBLAS. These are the
# post-update/downdate solve path — iterative refinement against the just-modified
# factor must not require a supernodal refactorization.
# ---------------------------------------------------------------------------

"""
    solve!(x::AbstractVecOrMat, G::SimplicialLDLFactor, b::AbstractVecOrMat) -> x

Solve `A·x = b` through the simplicial factor: permute, forward `L`, diagonal `D`,
backward `Lᵀ`, unpermute — the same staging as the supernodal [`solve!`](@ref)
(design.md §4.4/§6). Like the supernodal version this allocates a permuted-RHS scratch
buffer per call (correctness-first; the zero-allocation contract applies to
[`updowndate!`](@ref), not to the solves). `x` and `b` may alias.
"""
function solve!(x::AbstractVector{T}, G::SimplicialLDLFactor{T,Ti}, b::AbstractVector{T}) where {T,Ti<:Integer}
    n = G.sym.n
    perm = G.sym.perm
    y = Vector{T}(undef, n)
    @inbounds for k in 1:n
        y[k] = b[Int(perm[k])]
    end
    _solve_L!(y, G)
    _solve_D!(y, G)
    _solve_Lt!(y, G)
    @inbounds for k in 1:n
        x[Int(perm[k])] = y[k]
    end
    return x
end

function solve!(x::AbstractMatrix{T}, G::SimplicialLDLFactor{T,Ti}, b::AbstractMatrix{T}) where {T,Ti<:Integer}
    for c in axes(b, 2)
        solve!(view(x, :, c), G, view(b, :, c))
    end
    return x
end

Base.:\(G::SimplicialLDLFactor{T}, b::AbstractVector{T}) where {T} = solve!(similar(b), G, b)
Base.:\(G::SimplicialLDLFactor{T}, b::AbstractMatrix{T}) where {T} = solve!(similar(b), G, b)

"""
    solve_L!(y::AbstractVecOrMat, G::SimplicialLDLFactor)

Forward solve `L·y := y` in place, in FACTOR ordering (`L` unit-lower; see the
supernodal [`solve_L!`](@ref) for the permutation convention). Exported split solve
(design.md §6/§0 N7).
"""
solve_L!(y::AbstractVector, G::SimplicialLDLFactor) = _solve_L!(y, G)

"""
    solve_D!(y::AbstractVecOrMat, G::SimplicialLDLFactor)

Diagonal solve `D·y := y` in place, in FACTOR ordering. Exported split solve.
"""
solve_D!(y::AbstractVector, G::SimplicialLDLFactor) = _solve_D!(y, G)

"""
    solve_Lt!(y::AbstractVecOrMat, G::SimplicialLDLFactor)

Backward solve `Lᵀ·y := y` in place, in FACTOR ordering. Exported split solve.
"""
solve_Lt!(y::AbstractVector, G::SimplicialLDLFactor) = _solve_Lt!(y, G)

solve_L!(y::AbstractMatrix, G::SimplicialLDLFactor) = (_eachcol_solve!(_solve_L!, y, G); y)
solve_D!(y::AbstractMatrix, G::SimplicialLDLFactor) = (_eachcol_solve!(_solve_D!, y, G); y)
solve_Lt!(y::AbstractMatrix, G::SimplicialLDLFactor) = (_eachcol_solve!(_solve_Lt!, y, G); y)

function _eachcol_solve!(f::F, y::AbstractMatrix, G::SimplicialLDLFactor) where {F}
    for c in axes(y, 2)
        f(view(y, :, c), G)
    end
    return y
end

function _solve_L!(y::AbstractVector{T}, G::SimplicialLDLFactor{T,Ti}) where {T,Ti<:Integer}
    colptr = G.colptr
    colnnz = G.colnnz
    rowval = G.rowval
    nzval = G.nzval
    @inbounds for j in 1:G.sym.n
        yj = y[j]
        iszero(yj) && continue
        p0 = Int(colptr[j])
        for p in p0:(p0 + Int(colnnz[j]) - 1)
            y[Int(rowval[p])] -= nzval[p] * yj
        end
    end
    return y
end

function _solve_D!(y::AbstractVector{T}, G::SimplicialLDLFactor{T,Ti}) where {T,Ti<:Integer}
    d = G.d
    @inbounds for j in 1:G.sym.n
        y[j] /= d[j]
    end
    return y
end

function _solve_Lt!(y::AbstractVector{T}, G::SimplicialLDLFactor{T,Ti}) where {T,Ti<:Integer}
    colptr = G.colptr
    colnnz = G.colnnz
    rowval = G.rowval
    nzval = G.nzval
    @inbounds for j in G.sym.n:-1:1
        acc = y[j]
        p0 = Int(colptr[j])
        for p in p0:(p0 + Int(colnnz[j]) - 1)
            acc -= nzval[p] * y[Int(rowval[p])]
        end
        y[j] = acc
    end
    return y
end
