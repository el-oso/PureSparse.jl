# Sparse QR numeric factorization, design_qr.md §4 (M5a task 6): left-looking column
# Householder, direct sparse transcription of the survey §7.3 `qr_left_householder`
# algorithm (row-subtree traversal in place of the dense `for i=1:k-1` loop).

"""
    _apply_reflector!(y, F, k)

`y[pattern(V_k)] -= (F.beta[k] * (F.vval[V_k] ⋅ y[pattern(V_k)])) * F.vval[V_k]`, i.e.
`y ← H_k·y` restricted to `pattern(V_k)` (`H_k = I` elsewhere since `V_k` is zero
outside its pattern) — the sparse-indexed level-1 kernel shared by the numeric loop's
own apply step (§4.1 step 3) and the solve phase's `apply_Q!`/`apply_Qt!` (§6.1),
which is exactly the same operation applied in different column orders. Caller must
check `F.beta[k] != 0` first (dead/trivial reflectors are a no-op, cheaper to skip
entirely than to call this and multiply by zero).
"""
function _apply_reflector!(y::AbstractVector{T}, F::QRFactor{T,Ti}, k::Integer) where {T,Ti<:Integer}
    sym = F.sym
    vlo, vhi = sym.vptr[k], sym.vptr[k + 1] - 1
    w = zero(T)
    @inbounds for pp in vlo:vhi
        w += F.vval[pp] * y[sym.vrowind[pp]]
    end
    w *= F.beta[k]
    @inbounds for pp in vlo:vhi
        r = sym.vrowind[pp]
        y[r] -= w * F.vval[pp]
    end
    return y
end

"""
    _qr_threshold(A, tol) -> τ

Rank-detection threshold (design_qr.md §5.1/§5.3): `τ = qr_tol_mult · max(m,n) ·
eps(T) · max_j ‖A[:,j]‖₂` when `tol === nothing` (rank detection ON by default — the
safe choice, since an unguarded near-singular pivot produces a `beta` that blows up
toward `Inf`, not just a numerically-suspect answer, design_qr.md §4.4/§5.1). An
explicit `tol` is used directly; `tol ≤ 0` disables rank detection entirely (§5.3) —
only the B3 exact-zero guard (design_qr.md §4.4, unconditional) still applies.
"""
function _qr_threshold(A::SparseMatrixCSC{T}, tol) where {T<:Real}
    isnothing(tol) || return T(tol)
    m, n = size(A)
    maxcolnorm = zero(T)
    @inbounds for j in 1:n
        s = zero(T)
        for p in A.colptr[j]:(A.colptr[j + 1] - 1)
            s += A.nzval[p] * A.nzval[p]
        end
        maxcolnorm = max(maxcolnorm, sqrt(s))
    end
    return T(QR_TOL_MULT) * T(max(m, n)) * eps(T) * maxcolnorm
end

"""
    _qr_block(A::SparseMatrixCSC; ordering, tol) -> QRFactor

Build a fresh, fully-populated `QRFactor` for `A` treated as a self-contained block
(`sym.n1 == 0` always — no singleton awareness). This is what [`qr`](@ref) delegates
to once singletons (if any) have been peeled off and `A` has been restricted to the
surviving `A22` submatrix (design_qr.md §2.3); it is also exactly what `qr(A)` reduces
to when no singletons are found.
"""
function _qr_block(A::SparseMatrixCSC{T,Ti}; ordering::AbstractOrdering, tol::Union{Nothing,Real} = nothing) where {T,Ti<:Integer}
    sym = symbolic_qr(A; ordering)
    rcolind = Vector{Ti}(undef, sym.nnzR)
    rval = zeros(T, sym.nnzR)              # zero-initialized: dead-row slots (D9) stay
                                            # readable-as-zero without extra bookkeeping
    vval = zeros(T, sym.nnzV)
    beta = zeros(T, length(sym.parent))
    ws = QRWorkspace{T,Ti}(sym)
    r1ptr = ones(Ti, 1)                    # n1==0: length-1 sentinel (r1ptr[1]=1, no rows)
    F = QRFactor{T,Ti}(sym, rcolind, rval, vval, beta, r1ptr, Ti[], T[], ws, QRStats(), true)
    qr!(F, A; tol)
    return F
end

"""
    qr(A::SparseMatrixCSC; ordering::AbstractOrdering, tol=nothing, singletons=true, method=:column) -> QRFactor | QRFrontFactor

One-shot sparse QR factorization (design_qr.md §4.3): column-singleton
pre-elimination (§2.3) + [`symbolic_qr`](@ref) + numeric factorization. No default
`ordering` yet (§2.1's stated default, `COLAMDOrdering()`, lands in a later task) —
pass one explicitly, e.g. `AMDOrdering()`. `tol` is the rank-detection threshold
(§5.1/§5.3); see [`_qr_threshold`](@ref) for its default and the `tol ≤ 0` disable
convention — singleton detection's OWN threshold is `qr_singleton_mult × τ` (§1.6/§2.3,
using the SAME `τ`, so the two move together). **`tol ≤ 0` does NOT disable singleton
peeling** — it only relaxes the magnitude test to "any nonzero value passes" (since the
threshold itself becomes `qr_singleton_mult × 0 = 0`); a genuinely structural
singleton (a column with exactly one nonzero entry, which is extremely common —
diagonal-shaped, LP-shaped, and many hand-built test matrices all have them) is still
peeled. `singletons=false` is the actual peeling on/off switch (a coordinator-directed
addition beyond the original design text, added once this interaction surprised
several tests written before this task assumed `sym.n1 == 0` unconditionally — genuinely
useful independent of testing too, e.g. to isolate the core pipeline's own behavior or
compare timings with/without the optimization).

`method` selects the factorization ARCHITECTURE (design_qr_m5b.md §A5.6):
- `:column` (default) — M5a's left-looking column-Householder path (this function's
  own body below), returns a `QRFactor`; generic over `T<:Real`, singleton-aware.
- `:frontal` — M5b's multifrontal path ([`qr_frontal`](@ref)), returns a
  `QRFrontFactor`; Float64-tuned (routes through PureBLAS's `wy_t!`/`wy_apply!`).
  Non-Float64 `T` silently falls back to `:column` until P2's generic kernels land
  (§A7.3/§A5.6 — P2 only gates uniformity of the frontal path, never blocks Float64).
  `singletons` has no effect here: the frontal path never carries singletons
  (`sym.n1 == 0` always, §A1.2).
- `:auto` — dispatches on `sym.flops / sym.nnzR` (both already computed by
  [`symbolic_qr`](@ref), no extra numeric work), `:frontal` when the ratio exceeds
  [`QR_AUTO_METHOD_RATIO`](@ref) (task 16e, `tuning.jl` — measured on the M5 gate
  set, not guessed: every gate matrix where `:column` won sat at ratio ≤ 7, every
  matrix where `:frontal` won sat at ratio ≥ 863, a wide margin). Non-Float64 `T`
  always uses `:column` regardless of the ratio (P2 not yet landed).

Singletons are exploited ONLY in the `:column` path, never in [`symbolic_qr`](@ref)/
[`qr!`](@ref)'s reuse path (§2.3: "a singleton set chosen for A's values is invalid
for A2's" — a genuinely different matrix sharing this pattern could have different
magnitudes at the same entries). The resulting `QRFactor` (`sym.n1 > 0`) is a
terminal, one-shot object — do not call `qr!` on it (§2.3's own restriction; `qr!`
asserts `sym.n1 == 0`).
"""
function qr(A::SparseMatrixCSC{T,Ti}; ordering::AbstractOrdering, tol::Union{Nothing,Real} = nothing,
        singletons::Bool = true, method::Symbol = :column) where {T,Ti<:Integer}
    method in (:column, :frontal, :auto) ||
        throw(ArgumentError("qr: method must be :column, :frontal, or :auto, got :$method"))
    if method === :auto && T === Float64
        sym = symbolic_qr(A; ordering)
        ratio = sym.flops / max(sym.nnzR, 1)
        method = ratio > QR_AUTO_METHOD_RATIO ? :frontal : :column
    end
    if method === :frontal && T === Float64
        return qr_frontal(A; ordering, tol)
    end
    singletons || return _qr_block(A; ordering, tol)
    m, n = size(A)
    tau = _qr_threshold(A, tol)
    singleton_threshold = T(QR_SINGLETON_MULT) * tau
    peel_col, peel_row, collive, rowlive = peel_column_singletons(A, singleton_threshold)
    n1 = length(peel_col)
    n1 == 0 && return _qr_block(A; ordering, tol)
    return _qr_compose_singletons(A, peel_col, peel_row, collive, rowlive, ordering, tol)
end

"""
    qr!(F::QRFactor, A::SparseMatrixCSC; tol=nothing) -> QRFactor

Refactorize in place: `A` must share `F.sym`'s sparsity pattern (design_qr.md §4.3).
Implements the survey §7.3 left-looking loop exactly (design_qr.md §4.1), with the
Foster–Davis-style dead-column drop (§5.2) folded into steps 3/4:

1. **Scatter** column `k` of `A` (rows permuted via `riperm`) into the dense work
   vector `x`.
2. **Row subtree.** Gather `T^k = {i < k : R[i,k] ≠ 0}` by climbing `parent[]` from
   every `j ∈ pattern(S[:,k])` (`sym.sptr`/`sym.sind`), stamped to avoid revisiting a
   node twice within the same column's climb; emit `T^k` in ascending order (prior
   reflectors do not commute).
3. **Apply** each `i ∈ T^k` (skipping `beta[i] == 0`, dead or numerically trivial —
   §4.4's B3 fix makes this one check cover both). For a NUMERICALLY dead `i`
   (`pivotslot[i] != 0` but `beta[i] == 0`), `x[pivotslot[i]]` is dropped mass, not a
   real `R[i,k]` entry (§5.2: row `i` no longer really exists as a pivot, so whatever
   the current column would have written against it is thrown away) — accumulated
   into the running dropped-norm sum of squares and zeroed. A STRUCTURALLY dead `i`
   (`pivotslot[i] == 0`, `vcount[i]==0`) has nothing to harvest or drop at all.
   Otherwise harvest `R[i,k] = x[pivotslot[i]]` normally.
4. **Form reflector `k`** (§4.4) from `x` on `pattern(V_k)`. `xnorm ≤ τ` (§5.1 Heath's
   threshold test, `τ` from [`_qr_threshold`](@ref)) triggers the SAME dead-column
   handling as the B3 exact-zero guard — the detection-time tail `xnorm` itself is
   also dropped mass (§5.2, N2: bounded by `τ`; the LATER per-column discards in step
   3 above are not themselves τ-bounded and are the bulk of `dropped_norm`).

`F.sym.n1 > 0` (a factor produced by [`qr`](@ref)'s singleton pre-elimination path,
§2.3) is REJECTED — singletons are exploited only in the one-shot `qr(A)` path, never
across a refactor (§2.3: "a singleton set chosen for A's values is invalid for A2's").
"""
function qr!(F::QRFactor{T,Ti}, A::SparseMatrixCSC{T,Ti}; tol::Union{Nothing,Real} = nothing) where {T,Ti<:Integer}
    sym = F.sym
    sym.n1 == 0 || throw(ArgumentError(
        "qr!: F has sym.n1 = $(sym.n1) > 0 (produced by qr(A)'s singleton pre-elimination, " *
        "design_qr.md §2.3) — singletons are not exploited across a refactor; build a " *
        "fresh factor via qr(A2) instead of refactoring this one",
    ))
    check_refactor_shape(A, sym.m, sym.n, "qr!")
    ws = F.ws
    nb = length(sym.parent)
    x = ws.x
    stamp = ws.stamp
    tsub = ws.tsub
    pack = ws.pack
    rcursor = ws.rcursor
    tau = _qr_threshold(A, tol)

    fill!(stamp, zero(Ti))
    @inbounds for k in 1:nb
        rcursor[k] = sym.rptr[k]
    end
    fill!(F.beta, zero(T))
    n_dead = 0
    dropped_sq = zero(T)

    @inbounds for k in 1:nb
        # --- Step 1: scatter ---
        origcol = sym.cperm[k]
        for p in A.colptr[origcol]:(A.colptr[origcol + 1] - 1)
            physrow = sym.riperm[A.rowval[p]]
            physrow > sym.mb && continue          # a null row: never in any S_k
            x[physrow] = A.nzval[p]
        end

        # --- Step 2: row subtree ---
        len = 0
        for pp in sym.sptr[k]:(sym.sptr[k + 1] - 1)
            node = sym.sind[pp]
            while node != k && stamp[node] != k
                len += 1
                tsub[len] = node
                stamp[node] = Ti(k)
                node = sym.parent[node]
            end
        end
        sort!(view(tsub, 1:len))

        # --- Step 3: apply prior reflectors ---
        for t in 1:len
            i = tsub[t]
            if F.beta[i] == zero(T)
                piv = sym.pivotslot[i]
                if piv != 0                        # numerically (not structurally) dead:
                    v = x[piv]                      # this k's contribution to dead row i
                    dropped_sq += v * v             # is dropped mass, §5.2
                    x[piv] = zero(T)
                end
                continue
            end
            _apply_reflector!(x, F, i)
            piv = sym.pivotslot[i]
            c = rcursor[i]
            F.rcolind[c] = Ti(k)
            F.rval[c] = x[piv]
            rcursor[i] = c + one(Ti)
            x[piv] = zero(T)
        end

        # --- Step 4: form reflector k ---
        vlo, vhi = sym.vptr[k], sym.vptr[k + 1] - 1
        vlen = vhi - vlo + 1
        c = rcursor[k]
        if vlen == 0
            # Structurally dead column (vcount[k]==0, §3.4): beta[k] already 0 from the
            # fill! above; R[k,k] has an allocated slot (rcount[k] >= 1 always, every
            # column structurally owns its own diagonal) but no value to write.
            F.rcolind[c] = Ti(k)
            F.rval[c] = zero(T)
            rcursor[k] = c + one(Ti)
        else
            for t in 1:vlen
                pack[t] = x[sym.vrowind[vlo + t - 1]]
            end
            xnorm = T(nrm2(view(pack, 1:vlen)))
            if xnorm == zero(T) || (tau > zero(T) && xnorm <= tau)
                # B3 fix (exact zero, unconditional) / §5.1 Heath threshold test
                # (xnorm <= τ, only when rank detection is on): numerically dead,
                # live pattern. Distinct from the vlen==0 structural case above.
                F.rcolind[c] = Ti(k)
                F.rval[c] = zero(T)
                rcursor[k] = c + one(Ti)
                dropped_sq += xnorm * xnorm          # detection-time tail, §5.2 N2
                n_dead += 1
                for t in 1:vlen
                    x[sym.vrowind[vlo + t - 1]] = zero(T)
                end
            else
                xpiv = pack[1]                     # x[pivotslot[k]], vrowind's first entry
                s = xpiv >= zero(T) ? one(T) : -one(T)   # sign(0) := +1 (Opus N4)
                Rkk = -s * xnorm
                v1 = xpiv + s * xnorm
                denom = v1 * v1
                for t in 2:vlen
                    denom += pack[t] * pack[t]
                end
                F.beta[k] = denom == zero(T) ? zero(T) : (T(2) / denom)
                F.vval[vlo] = v1
                for t in 2:vlen
                    F.vval[vlo + t - 1] = pack[t]
                end
                F.rcolind[c] = Ti(k)
                F.rval[c] = Rkk
                rcursor[k] = c + one(Ti)
                for t in 1:vlen
                    x[sym.vrowind[vlo + t - 1]] = zero(T)
                end
            end
        end
    end
    F.stats.nnzR = sym.nnzR
    F.stats.nnzV = sym.nnzV
    F.stats.flops = sym.flops
    F.stats.rank = nb - n_dead
    F.stats.n_dead = n_dead
    F.stats.dropped_norm = Float64(sqrt(dropped_sq))
    F.ok = true
    check_finite(F.rval, "qr!")
    check_finite(F.vval, "qr!")
    return F
end
