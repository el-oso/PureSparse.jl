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
    qr(A::SparseMatrixCSC; ordering::AbstractOrdering) -> QRFactor

One-shot sparse QR factorization: [`symbolic_qr`](@ref) + numeric factorization
(design_qr.md §4.3). No default `ordering` yet (§2.1's stated default,
`COLAMDOrdering()`, lands in a later task) — pass one explicitly, e.g.
`AMDOrdering()`.
"""
function qr(A::SparseMatrixCSC{T,Ti}; ordering::AbstractOrdering) where {T,Ti<:Integer}
    sym = symbolic_qr(A; ordering)
    rcolind = Vector{Ti}(undef, sym.nnzR)
    rval = zeros(T, sym.nnzR)              # zero-initialized: dead-row slots (D9) stay
                                            # readable-as-zero without extra bookkeeping
    vval = zeros(T, sym.nnzV)
    beta = zeros(T, length(sym.parent))
    ws = QRWorkspace{T,Ti}(sym)
    F = QRFactor{T,Ti}(sym, rcolind, rval, vval, beta, ws, QRStats(), true)
    qr!(F, A)
    return F
end

"""
    qr!(F::QRFactor, A::SparseMatrixCSC) -> QRFactor

Refactorize in place: `A` must share `F.sym`'s sparsity pattern (design_qr.md §4.3).
Implements the survey §7.3 left-looking loop exactly (design_qr.md §4.1):

1. **Scatter** column `k` of `A` (rows permuted via `riperm`) into the dense work
   vector `x`.
2. **Row subtree.** Gather `T^k = {i < k : R[i,k] ≠ 0}` by climbing `parent[]` from
   every `j ∈ pattern(S[:,k])` (`sym.sptr`/`sym.sind`), stamped to avoid revisiting a
   node twice within the same column's climb; emit `T^k` in ascending order (prior
   reflectors do not commute).
3. **Apply** each `i ∈ T^k` (skipping `beta[i] == 0`, dead or numerically trivial —
   §4.4's B3 fix makes this one check cover both), then harvest `R[i,k] = x[pivotslot[i]]`
   into row `i`'s cursor and zero that slot.
4. **Form reflector `k`** (§4.4) from `x` on `pattern(V_k)`, including the B3
   zero-norm guard, then zero `x` on that pattern.
"""
function qr!(F::QRFactor{T,Ti}, A::SparseMatrixCSC{T,Ti}) where {T,Ti<:Integer}
    sym = F.sym
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
            F.beta[i] == zero(T) && continue
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
            if xnorm == zero(T)
                # B3 fix: numerically-zero live pattern, unconditional on rank-detection
                # settings (design_qr.md §4.4) — distinct from the vlen==0 structural
                # case above (here the PATTERN is live, only the VALUES are zero).
                F.rcolind[c] = Ti(k)
                F.rval[c] = zero(T)
                rcursor[k] = c + one(Ti)
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
    F.stats.rank = nb                      # no rank-deficiency handling yet (M5a task 8)
    F.stats.n_dead = 0
    F.stats.dropped_norm = 0.0
    F.ok = true
    return F
end
