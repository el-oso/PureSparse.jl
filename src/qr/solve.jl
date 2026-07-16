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
    n1 = sym.n1
    y = F.ws.x                             # length mb, reused as physical-space scratch
    fill!(y, zero(T))
    # Block physical rows occupy riperm values n1+1 .. n1+mb (design_qr.md §1.4: rperm
    # places the n1 singleton rows first, then the block's own staircase permutation) —
    # singleton/null rows (riperm <= n1, or > n1+mb) never scatter into the block scratch.
    @inbounds for p in 1:m
        phys = sym.riperm[p]
        (phys > n1 && phys - n1 <= sym.mb) && (y[phys - n1] = b[p])
    end
    apply_Qt!(y, F)
    c = F.ws.rblk                          # length nb, zero-alloc (task 10)
    @inbounds for k in 1:nb
        piv = sym.pivotslot[k]
        c[k] = piv == 0 ? zero(T) : y[piv]
    end
    solve_R!(c, F, c)                      # in place: solve_R!'s x/c args may alias
    @inbounds for k in 1:nb
        x[sym.cperm[n1 + k]] = c[k]
    end

    if n1 > 0
        # Singleton block back-substitution (design_qr.md §6.2/§2.3): R11*x1 + R12*x2 = b1,
        # R11 upper triangular n1×n1 — descending so each row's LATER entries (both later
        # singleton columns and all of x2) are already resolved before it's used. R11/R12
        # need no Q transformation (Q=I there, §2.3/task 9 own derivation) — b1[k] is
        # simply b at the k-th singleton's ORIGINAL row, rperm[k].
        x1 = F.ws.n1a                      # length n1, zero-alloc (task 10)
        @inbounds for k in n1:-1:1
            # diag == 0 ⇔ a warm qr! (§2.3 warm-refactor update) dropped this
            # singleton's pivot as numerically dead — basic-solution convention,
            # x1[k] = 0, exactly like solve_R!'s dead-column branch. A cold-composed
            # factor never has a zero diag here (the peel's magnitude test guarantees
            # it), so this branch costs one compare on the common path.
            diag = F.r1val[F.r1ptr[k]]
            if diag == zero(T)
                x1[k] = zero(T)
                continue
            end
            s = b[sym.rperm[k]]
            for p in (F.r1ptr[k] + 1):(F.r1ptr[k + 1] - 1)
                jcol = F.r1colind[p]
                v = F.r1val[p]
                s -= v * (jcol <= n1 ? x1[jcol] : c[jcol - n1])
            end
            x1[k] = s / diag
        end
        @inbounds for k in 1:n1
            x[sym.cperm[k]] = x1[k]
        end
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

**Requires `F.stats.n_dead == 0`** (found via testing, not anticipated in the design
text): the minimum-norm formula assumes `Aᵀ` was factored to FULL rank (every one of
its columns retired a pivot). Rank detection (§5, task 8) is ON by default in `qr()`
and silently drops a numerically-near-singular column — `solve_minnorm!`'s math has
no way to account for a dropped column (there is no "basic solution" concept for a
minimum-norm solve the way §6.2 has one for least squares), so it would silently
return a wrong answer rather than error. Pass `tol=0` to the `qr(Aᵀ; ...)` call that
produced `F` if you need this guarantee on a matrix you haven't separately verified is
well-conditioned.
"""
function solve_minnorm!(x::AbstractVector{T}, F::QRFactor{T,Ti}, b::AbstractVector{T}) where {T,Ti<:Integer}
    sym = F.sym
    F.stats.n_dead == 0 || throw(ArgumentError(
        "solve_minnorm!: F has $(F.stats.n_dead) rank-detected dead column(s) — the " *
        "minimum-norm formula requires Aᵀ to be factored at full rank; pass tol=0 to " *
        "qr(Aᵀ; ...) if Aᵀ is well-conditioned, or this system is genuinely rank-" *
        "deficient and solve_minnorm! does not support that case",
    ))
    nb = length(sym.parent)
    n1 = sym.n1
    c2 = F.ws.rblk                          # length nb, zero-alloc (task 10)
    @inbounds for k in 1:nb
        c2[k] = b[sym.cperm[n1 + k]]
    end

    z1 = F.ws.n1b                           # length n1, zero-alloc (task 10)
    if n1 > 0
        # Rᵀz=Pᵀb, with R=[R11 R12; 0 Rblock] (§2.3/task 9): Rᵀ=[R11ᵀ 0; R12ᵀ Rblockᵀ]
        # is block lower-triangular, so z1 solves FIRST via R11ᵀz1=c1 (forward,
        # ascending — R11 is upper-triangular row-wise stored, so this is the SAME
        # forward-scatter idiom as solve_Rt!, generalized to also push R12's
        # contribution into c2 before the block's own solve_Rt! runs).
        c1 = F.ws.n1a                       # length n1, zero-alloc (task 10)
        @inbounds for k in 1:n1
            c1[k] = b[sym.cperm[k]]
        end
        @inbounds for k in 1:n1
            diag = F.r1val[F.r1ptr[k]]
            zk = c1[k] / diag
            z1[k] = zk
            for p in (F.r1ptr[k] + 1):(F.r1ptr[k + 1] - 1)
                jcol = F.r1colind[p]
                v = F.r1val[p]
                if jcol <= n1
                    c1[jcol] -= v * zk
                else
                    c2[jcol - n1] -= v * zk
                end
            end
        end
    end

    solve_Rt!(c2, F, c2)                   # in place: solve_Rt!'s x/c args may alias
    y = F.ws.x
    fill!(y, zero(T))
    @inbounds for k in 1:nb
        piv = sym.pivotslot[k]
        piv != 0 && (y[piv] = c2[k])
    end
    apply_Q!(y, F)
    @inbounds for p in 1:sym.m
        phys = sym.riperm[p]
        if phys <= n1
            x[p] = z1[phys]
        elseif phys - n1 <= sym.mb
            x[p] = y[phys - n1]
        else
            x[p] = zero(T)
        end
    end
    fill!(y, zero(T))                      # restore F.ws.x's invariant before the next qr!
    return x
end
