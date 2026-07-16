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
    F = QRFactor{T,Ti}(
        sym, rcolind, rval, vval, beta, r1ptr, Ti[], T[],
        sym,                                                  # bsym: n1==0, the block IS the factor
        SparseMatrixCSC(0, 0, Ti[one(Ti)], Ti[], T[]), Ti[], Ti[],  # a22buf/a22map/r1srcpos: unused
        ws, QRStats(), true,
    )
    qr!(F, A; tol)
    return F
end

"""
    qr(A::SparseMatrixCSC; ordering::AbstractOrdering, tol=nothing, singletons=true, method=:column) -> QRFactor | QRFrontFactor

One-shot sparse QR factorization (`design_qr.md` §4.3): column-singleton
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

Singletons are exploited only in the `:column` path (never `:frontal`, §A1.2). The
resulting `QRFactor` (`sym.n1 > 0`) is fully warm-refactorable via [`qr!`](@ref)
(design_qr.md §2.3, warm-refactor update): the peel set's STRUCTURAL half ("exactly
one live nonzero") is pattern-only and therefore refactor-invariant; only the
magnitude test is value-dependent, and `qr!` re-checks it per pivot against the new
values (see its docstring for the drop semantics when a pivot goes numerically small).
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

Refactorize in place: `A` must share `F.sym`'s sparsity pattern (`design_qr.md` §4.3).
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
§2.3) is SUPPORTED (design_qr.md §2.3, warm-refactor update — the original one-shot
restriction is lifted): the structural peel set is refactor-invariant ("exactly one
live nonzero" is a pattern-only property, and a refactor shares the pattern by
contract), so the warm call (1) refreshes the pre-allocated `A22` buffer's values from
`A` through the compose-time `a22map`, (2) re-harvests `R11`/`R12` from `A`'s peeled
rows through `r1srcpos` — with a per-pivot MAGNITUDE guard, since only the magnitude
half of the singleton test is value-dependent: a peeled pivot whose new magnitude
drops to ≤ the singleton threshold (`QR_SINGLETON_MULT · τ`, the same criterion the
cold peel used) is no longer a valid pivot and is dropped into the existing
`n_dead`/`dropped_norm` accounting (its whole `R11`/`R12` row is zeroed and counted as
dropped mass; `solve!` returns the basic solution with that `x` entry 0, exactly like
any other dead column) — and (3) runs the standard block numeric loop on the refreshed
`A22` buffer, writing into the factor's own (shared, block-sized) arrays. All three
steps are zero-allocation (CLAUDE.md req 5).
"""
function qr!(F::QRFactor{T,Ti}, A::SparseMatrixCSC{T,Ti}; tol::Union{Nothing,Real} = nothing) where {T,Ti<:Integer}
    sym = F.sym
    check_refactor_shape(A, sym.m, sym.n, "qr!")
    n1 = sym.n1
    if n1 == 0
        n_dead, dropped_sq = _qr_block_numeric!(F, A, _qr_threshold(A, tol))
    else
        # --- (1) refresh the A22 buffer's values from A (pattern-invariant map) ---
        anz = A.nzval
        buf = F.a22buf.nzval
        amap = F.a22map
        @inbounds for k in eachindex(amap)
            buf[k] = anz[amap[k]]
        end
        # --- (2) re-harvest R11/R12 with the per-pivot magnitude guard (§2.3):
        # r1colind is structural (fixed, already sorted from compose time); only the
        # values change. Row k's diagonal is its FIRST stored entry (same invariant
        # solve! already relies on): every entry of peeled row k lands at final
        # column >= k, with equality exactly at the pivot. The guard reuses the cold
        # peel's own criterion (QR_SINGLETON_MULT × τ, τ recomputed from A's current
        # values) — a pivot that would no longer be peeled is dropped Foster–Davis-
        # style into n_dead/dropped_norm rather than divided by at solve time. ---
        sthr = T(QR_SINGLETON_MULT) * _qr_threshold(A, tol)
        n_dead = 0
        dropped_sq = zero(T)
        @inbounds for k in 1:n1
            lo = Int(F.r1ptr[k])
            hi = Int(F.r1ptr[k + 1]) - 1
            if abs(anz[F.r1srcpos[lo]]) <= sthr
                # Dead singleton pivot: A2's value at a structurally-peeled pivot went
                # numerically small. Q = I on the singleton block (§2.3), so dropping
                # row k of [R11 R12] discards exactly A's peeled row — that whole row
                # is the dropped mass (§5.2's certificate, singleton flavor).
                for p in lo:hi
                    v = anz[F.r1srcpos[p]]
                    dropped_sq += v * v
                    F.r1val[p] = zero(T)
                end
                n_dead += 1
            else
                for p in lo:hi
                    F.r1val[p] = anz[F.r1srcpos[p]]
                end
            end
        end
        # --- (3) warm-refactor the A22 block into the factor's own arrays. τ comes
        # from the refreshed A22 (matching the cold path, where the block's qr! also
        # computed its threshold from A22, not the full A). ---
        nd, dq = _qr_block_numeric!(F, F.a22buf, _qr_threshold(F.a22buf, tol))
        n_dead += nd
        dropped_sq += dq
    end
    F.stats.nnzR = sym.nnzR + length(F.r1colind)   # sym.nnzR is block-only; r1 adds the rest
    F.stats.nnzV = sym.nnzV
    F.stats.flops = sym.flops
    F.stats.rank = length(sym.parent) + n1 - n_dead
    F.stats.n_dead = n_dead
    F.stats.dropped_norm = Float64(sqrt(dropped_sq))
    F.ok = true
    check_finite(F.rval, "qr!")
    check_finite(F.vval, "qr!")
    return F
end

"""
    _qr_block_numeric!(F::QRFactor, A::SparseMatrixCSC, tau) -> (n_dead, dropped_sq)

The left-looking numeric loop (steps 1–4 of [`qr!`](@ref)'s docstring) over the BLOCK:
`A` is the block matrix (`A` itself when `F.sym.n1 == 0`, the refreshed `F.a22buf`
when `n1 > 0`) and all indexing goes through `F.bsym`, the block's OWN symbolic
(block-LOCAL `cperm`/`riperm`; `=== F.sym` when `n1 == 0`, and sharing every other
field with it by reference when `n1 > 0` — see `_qr_compose_singletons`). Writes
`rval`/`rcolind`/`vval`/`beta`/`ws` in place; the caller owns `stats`/`ok`/finiteness.
"""
function _qr_block_numeric!(F::QRFactor{T,Ti}, A::SparseMatrixCSC{T,Ti}, tau::T) where {T,Ti<:Integer}
    sym = F.bsym
    ws = F.ws
    nb = length(sym.parent)
    x = ws.x
    stamp = ws.stamp
    tsub = ws.tsub
    pack = ws.pack
    rcursor = ws.rcursor

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
        # `alg=InsertionSort` pins a zero-alloc algorithm (design's zero-alloc gate,
        # CLAUDE.md req 5): Base's default `sort!` picks RadixSort for `Int` arrays
        # above a size heuristic, and RadixSort allocates scratch buffers
        # (`Base.Sort.make_scratch`) — since `len` here is a per-column row-subtree
        # size (small, bounded by tree depth, not array length), it crosses that
        # threshold on some columns of some matrices but not others, causing the
        # allocation to be silently data-dependent (found via Profile.Allocs, not
        # visible from a single small test-suite-sized matrix). InsertionSort is
        # O(len²) but len is small here by construction, and it never allocates at
        # any size (verified directly).
        sort!(view(tsub, 1:len); alg = InsertionSort)

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
    return n_dead, dropped_sq
end
