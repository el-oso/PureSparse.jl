# Sparse QR solve phase, design_qr.md §6 (M5a task 7).

"""
    apply_Qt!(y::AbstractVector, F::QRFactor) -> y

`y ← Qᵀy` in place (`y` has length `F.sym.mb`, physical row space): apply reflectors
`k = 1..n` ASCENDING (`Qᵀ = H_n·…·H_1`, so `H_1` applies first, design_qr.md §6.1).
"""
function apply_Qt!(y::AbstractVector{T}, F::QRFactor{T,Ti}) where {T,Ti<:Integer}
    nb = length(F.sym.parent)
    @inbounds for k in 1:nb
        F.beta[k] == zero(T) && continue
        _apply_reflector!(y, F, k)
    end
    return y
end

"""
    apply_Q!(y::AbstractVector, F::QRFactor) -> y

`y ← Qy` in place: apply reflectors `k = n..1` DESCENDING (`Q = H_1·…·H_n`, so `H_n`
applies first, design_qr.md §6.1).
"""
function apply_Q!(y::AbstractVector{T}, F::QRFactor{T,Ti}) where {T,Ti<:Integer}
    nb = length(F.sym.parent)
    @inbounds for k in nb:-1:1
        F.beta[k] == zero(T) && continue
        _apply_reflector!(y, F, k)
    end
    return y
end

"""
    solve_R!(x::AbstractVector, F::QRFactor, c::AbstractVector) -> x

`R·x = c` via back-substitution over rows of `R` DESCENDING (design_qr.md §6.1): row
`k`'s diagonal (`F.rval[F.sym.rptr[k]]`, always the first stored entry of row `k`,
§4.1) is `0` exactly when column `k` is dead (structurally or the B3 numeric case) —
`x[k] = 0` for those, matching the design's "dead ⇒ x[k] = 0". `x`/`c` have length
`n - n1`; `x`/`c` may alias.
"""
function solve_R!(x::AbstractVector{T}, F::QRFactor{T,Ti}, c::AbstractVector{T}) where {T,Ti<:Integer}
    sym = F.sym
    nb = length(sym.parent)
    ws = F.ws
    @inbounds for k in nb:-1:1
        lo = sym.rptr[k]
        hi = ws.rcursor[k] - 1
        diag = F.rval[lo]
        if diag == zero(T)
            x[k] = zero(T)
            continue
        end
        s = c[k]
        for p in (lo + 1):hi
            s -= F.rval[p] * x[F.rcolind[p]]
        end
        x[k] = s / diag
    end
    return x
end

"""
    solve_Rt!(x::AbstractVector, F::QRFactor, c::AbstractVector) -> x

`Rᵀ·x = c` via forward substitution over rows of `R` ASCENDING (the mirror of
[`solve_R!`](@ref), design_qr.md §6.1) — `R` is stored row-wise, so `Rᵀ`'s forward
solve is done by a forward SCATTER instead of a column gather: once `x[k]` is
finalized, its contribution is scattered immediately into every later `x[j]` that row
`k`'s own stored entries touch (`R[k,j]`, `j > k`), which is exactly the set `Rᵀ`'s
row `j` needs from `k`. Dead columns (`diag == 0`) get `x[k] = 0` and scatter nothing.
`x`/`c` may alias.
"""
function solve_Rt!(x::AbstractVector{T}, F::QRFactor{T,Ti}, c::AbstractVector{T}) where {T,Ti<:Integer}
    sym = F.sym
    nb = length(sym.parent)
    ws = F.ws
    x !== c && copyto!(x, 1, c, 1, nb)
    @inbounds for k in 1:nb
        lo = sym.rptr[k]
        hi = ws.rcursor[k] - 1
        diag = F.rval[lo]
        if diag == zero(T)
            x[k] = zero(T)
            continue
        end
        xk = x[k] / diag
        x[k] = xk
        for p in (lo + 1):hi
            j = F.rcolind[p]
            x[j] -= F.rval[p] * xk
        end
    end
    return x
end

"""
    solve!(x::AbstractVector, F::QRFactor, b::AbstractVector) -> x

Least-squares (`m ≥ n`) / basic (rank-deficient or `m < n`, dead columns zero) solve
(design_qr.md §6.2), `SPQR` paper §3.3's `x = P·(R \\ (Qᵀb))`: gather `b` into the
physical row space via `rperm`, `apply_Qt!`, `solve_R!` on the leading `n - n1`
entries, scatter into `x` via `cperm`. `x`/`b` are FULL space (length `m`/`n`); `x`
and `b` may alias (the physical-space working copy is scratch, `F.ws.x`).

**Row-k-of-R lives at physical row `pivotslot[k]`, not physical row `k`** (design_qr.md
§3.4's B2 fix — the whole reason `pivotslot` exists): after `apply_Qt!`, the value
`solve_R!` needs as `c[k]` is `y[pivotslot[k]]`, gathered explicitly below, not `y[k]`
directly (those coincide only when `mb == n` and every column happens to retire its
own-index row, which is not the general case). A dead column (`pivotslot[k] == 0`) gets
an arbitrary `c[k]` — `solve_R!` ignores it and forces `x[k] = 0` regardless. `mb < n`
is handled the same way as any other dead-column case, not as a special path (design's
own "no special-casing" pattern, mirroring §3.4's B2 argument).
"""
function solve!(x::AbstractVector{T}, F::QRFactor{T,Ti}, b::AbstractVector{T}) where {T,Ti<:Integer}
    sym = F.sym
    m, n = sym.m, sym.n
    nb = length(sym.parent)
    y = F.ws.x                             # length mb, reused as physical-space scratch
    fill!(y, zero(T))
    @inbounds for p in 1:m
        phys = sym.riperm[p]
        phys <= sym.mb && (y[phys] = b[p])
    end
    apply_Qt!(y, F)
    c = Vector{T}(undef, nb)               # correctness-first; zero-alloc is task 10
    @inbounds for k in 1:nb
        piv = sym.pivotslot[k]
        c[k] = piv == 0 ? zero(T) : y[piv]
    end
    xb = Vector{T}(undef, nb)
    solve_R!(xb, F, c)
    @inbounds for k in 1:nb
        x[sym.cperm[k]] = xb[k]
    end
    @inbounds for k in (nb + 1):n
        x[sym.cperm[k]] = zero(T)           # dead beyond the block (n1==0 for now, task 9)
    end
    fill!(y, zero(T))                      # restore F.ws.x's "all-zero between columns"
                                            # invariant (design.md §4.1) before the next qr!
    return x
end

"""
    F \\ b -> x

`solve!` allocating its own output (design_qr.md §6.4).
"""
Base.:\(F::QRFactor{T,Ti}, b::AbstractVector{T}) where {T,Ti<:Integer} = solve!(Vector{T}(undef, F.sym.n), F, b)

"""
    ldiv!(x::AbstractVector, F::QRFactor, b::AbstractVector) -> x

Alias for [`solve!`](@ref) (design_qr.md §6.4's stdlib-compatible spelling).
"""
LinearAlgebra.ldiv!(x::AbstractVector{T}, F::QRFactor{T,Ti}, b::AbstractVector{T}) where {T,Ti<:Integer} = solve!(x, F, b)

"""
    solve_minnorm!(x::AbstractVector, F::QRFactor, b::AbstractVector) -> x

Minimum-norm solve of `A·x = b` for `m < n` (design_qr.md §6.3, George–Heath–Ng 1984 /
SPQR paper §5.1 method (2)): `F` must be the QR factorization of **`Aᵀ`** (`A = P·Rᵀ·Qᵀ`
from `Aᵀ·P = QR`), not of `A` itself. Solves `Rᵀ·z = Pᵀb` forward (`solve_Rt!`), then
`x = Q·[z; 0]` (`apply_Q!`). `b` has length `m` (`Aᵀ`'s column count = `F.sym.n`); `x`
has length `n` (`Aᵀ`'s row count = `F.sym.m`, matching `A`'s column count).

`solve_Rt!` operates entirely in R's own abstract row/column space (`z`, length
`n-n1`) — `z` does NOT overlay physical row space directly, so it is a separate
buffer, not a `view` of `F.ws.x` (an earlier version of this function aliased the two,
which is wrong for exactly the same reason `solve!` needed the `pivotslot` gather:
`[z; 0]`'s embedding into the physical space `apply_Q!` operates on must place `z[k]`
at physical row `pivotslot[k]`, not at physical row `k`).
"""
function solve_minnorm!(x::AbstractVector{T}, F::QRFactor{T,Ti}, b::AbstractVector{T}) where {T,Ti<:Integer}
    sym = F.sym
    nb = length(sym.parent)
    z = Vector{T}(undef, nb)               # correctness-first; zero-alloc is task 10
    @inbounds for k in 1:nb
        z[k] = b[sym.cperm[k]]              # Pᵀb: cperm is Aᵀ's OWN column permutation
    end
    solve_Rt!(z, F, z)
    y = F.ws.x
    fill!(y, zero(T))
    @inbounds for k in 1:nb
        piv = sym.pivotslot[k]
        piv != 0 && (y[piv] = z[k])
    end
    apply_Q!(y, F)
    @inbounds for p in 1:sym.m
        phys = sym.riperm[p]
        x[p] = phys <= sym.mb ? y[phys] : zero(T)
    end
    fill!(y, zero(T))                      # restore F.ws.x's invariant before the next qr!
    return x
end
