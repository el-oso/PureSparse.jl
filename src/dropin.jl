# Drop-in stdlib surface (design.md §10 M4). Only `include`d when `DROPIN_ACTIVE`
# (tuning.jl) — see that const's comment for why this must be a compile-time gate, not
# a runtime one. Scope: `LinearAlgebra.cholesky`/`ldlt` for real (non-complex) sparse
# input — PureSparse itself doesn't support complex element types yet (design.md §1.1
# non-goals), so complex input is deliberately NOT intercepted here and falls through
# to CHOLMOD's own method untouched (verified: our signatures below are a strict
# subset of CHOLMOD's own `Union{...} where T<:Real` signatures — Julia picks the more
# specific match for real `T`, and only CHOLMOD's method applies to `Complex{T}` since
# ours doesn't mention it at all).

using SparseArrays: SparseMatrixCSC
using LinearAlgebra: Symmetric, Hermitian, PosDefException, I

const _RealSparseArg{T<:Real} = Union{
    SparseMatrixCSC{T},
    Symmetric{T,<:SparseMatrixCSC{T}},
    Hermitian{T,<:SparseMatrixCSC{T}},
}

"""
    LinearAlgebra.cholesky(A::Union{SparseMatrixCSC,Symmetric,Hermitian}; check=true, perm=nothing, shift=0) -> SupernodalFactor

**Drop-in override, active only when [`activate!`](@ref) has been called and Julia
restarted** (design.md §10 M4). Matches `SparseArrays.CHOLMOD`'s own `cholesky` kwarg
surface for real element types: `shift` factors `A + shift·I`; `perm` forces that
permutation as the ELIMINATION ORDER instead of running AMD (`GivenOrdering`);
`check=true` (default) throws `PosDefException` on a non-SPD pivot, matching stdlib;
`check=false` returns the factor with `issuccess(F) == false` instead (PureSparse's
own, non-drop-in `cholesky` never throws — this wrapper adds the throw only to match
stdlib's documented behavior). Returns a [`SupernodalFactor`](@ref), NOT a
`CHOLMOD.Factor` — but one that supports the common surface downstream code expects:
`\\`, `issuccess`, `.p` (permutation), `.L` (sparse, factor-ordered, matching
`L·Lᵀ ≈ P·A·Pᵀ` — CHOLMOD's own convention, verified directly), `logdet`, `det` (all
added by this file).

**Known, documented deviation from CHOLMOD's exact `perm=` contract** (verified by
direct comparison, not assumed): CHOLMOD guarantees `F.p == perm` exactly when `perm`
is given. PureSparse cannot offer that guarantee — `symbolic()` always composes ANY
ordering (AMD or given) with a postorder relabeling step, required internally so
supernode detection sees contiguous children (design.md §3.2/§3.4) — so `F.p` reflects
`perm`'s ELIMINATION ORDER (which column is factored before which) but not necessarily
its exact final numbering. The factorization is still mathematically correct for
whatever `F.p` ends up being (`L·Lᵀ ≈ (A[F.p,F.p])`, verified); code that only reads
`F.p` to permute/solve consistently (the overwhelming common case) is unaffected —
only code asserting `F.p == perm` literally would observe the difference.
"""
function LinearAlgebra.cholesky(A::_RealSparseArg{T}; check::Bool = true,
        perm::Union{Nothing,AbstractVector{<:Integer}} = nothing, shift::Real = zero(T)) where {T<:Real}
    Afull = SparseMatrixCSC(sparse(A))
    if !iszero(shift)
        Afull = Afull + T(shift) * I
    end
    ordering = isnothing(perm) ? AMDOrdering() : GivenOrdering(collect(Int, perm))
    F = cholesky(Afull; ordering)
    if check && !issuccess(F)
        throw(PosDefException(F.stats.fail_col))
    end
    return F
end

"""
    LinearAlgebra.ldlt(A::Union{SparseMatrixCSC,Symmetric,Hermitian}; check=true, perm=nothing, shift=0) -> LDLFactor

**Drop-in override, active only when [`activate!`](@ref) has been called and Julia
restarted** (design.md §10 M4). Same kwarg surface and `perm=`/allocation caveats as
[`cholesky`](@ref)'s drop-in above (read that docstring first). Unlike CHOLMOD's own
`ldlt`, which does dynamic pivoting for general symmetric indefinite matrices,
PureSparse's `ldlt` is fixed-pivot with signed regularization (design.md §5.1,
CLAUDE.md requirement 8 — deliberate scope, not a missing feature). This entry point
has no way to receive expected pivot signs (stdlib's `ldlt` doesn't take a `signs`/
`n_pos`/`n_neg` kwarg — verified directly against its actual method signature), so it
always factors with `signs = nothing` — free signs, magnitude-floor-only
regularization (never a forced sign flip). `check=true` throwing on `!issuccess(F)` is
therefore reachable only via the (rare) magnitude-floor path, since a free-sign
factorization can't fail on "wrong sign." Callers who know their matrix's expected
inertia (the common SQD/KKT case) get materially better regularization behavior from
`PureSparse.ldlt(A; n_pos, n_neg)` directly than from this drop-in entry point — this
is an intentional, documented gap in drop-in fidelity, not an oversight.
"""
function LinearAlgebra.ldlt(A::_RealSparseArg{T}; check::Bool = true,
        perm::Union{Nothing,AbstractVector{<:Integer}} = nothing, shift::Real = zero(T)) where {T<:Real}
    Afull = SparseMatrixCSC(sparse(A))
    if !iszero(shift)
        Afull = Afull + T(shift) * I
    end
    ordering = isnothing(perm) ? AMDOrdering() : GivenOrdering(collect(Int, perm))
    F = ldlt(Afull; ordering)
    if check && !issuccess(F)
        throw(PosDefException(F.stats.fail_col))
    end
    return F
end

# --- stdlib surface parity on the returned SupernodalFactor/LDLFactor ---

const _DropinFactor{T,Ti} = Union{SupernodalFactor{T,Ti},LDLFactor{T,Ti}}

"""
    F.p

Fill-reducing permutation (factor order — `perm[k]` = original index at new position
`k`), matching `CHOLMOD.Factor`'s own `.p` convention exactly (`L·Lᵀ ≈ (P·A·Pᵀ)` for
`p = F.p`, verified directly against CHOLMOD's output). Alias for `F.sym.perm`, added
via `getproperty` so existing internal field access (`F.sym`, `F.x`, `F.panels`, ...)
is untouched — this only ADDS `:p`/`:L`, everything else falls through to `getfield`.
"""
function Base.getproperty(F::_DropinFactor, s::Symbol)
    s === :p && return getfield(F, :sym).perm
    s === :L && return sparse_L(F)
    return getfield(F, s)
end
Base.propertynames(F::_DropinFactor) = (fieldnames(typeof(F))..., :p, :L)

"""
    sparse_L(F::Union{SupernodalFactor,LDLFactor}) -> SparseMatrixCSC

Materialize the factor's `L` (lower-triangular, factor-ordered — unit-diagonal for an
[`LDLFactor`](@ref), true diagonal for a [`SupernodalFactor`](@ref); both are read
directly from panel storage, which already stores a literal `1` on the LDLᵀ diagonal —
see `numeric/ldlt.jl`'s base-case loop) as a `SparseMatrixCSC`. Allocates — this is a
one-time extraction, not a hot-path operation; also reachable as `F.L`.
"""
function sparse_L(F::_DropinFactor{T,Ti}) where {T,Ti<:Integer}
    sym = F.sym
    n = sym.n
    I = Ti[]; J = Ti[]; V = T[]
    @inbounds for s in 1:sym.nsuper
        j0 = Int(sym.super[s]); j1 = Int(sym.super[s + 1]) - 1
        nscol = j1 - j0 + 1
        rp0 = Int(sym.rowind_ptr[s])
        nsrow = Int(sym.rowind_ptr[s + 1]) - rp0
        panel = F.panels[s]
        for c in 1:nscol, k in c:nsrow   # k<c: strictly-upper of the diagonal block, unstored
            push!(I, sym.rowind[rp0 + k - 1])
            push!(J, Ti(j0 + c - 1))
            push!(V, panel[k, c])
        end
    end
    return sparse(I, J, V, n, n)
end

"""
    logdet(F::SupernodalFactor) -> Real
    det(F::SupernodalFactor) -> Real

`logdet(P·A·Pᵀ) = 2·Σ log(L_jj)` (permutation-invariant: `det(P) = ±1`, squared away).
Reads the diagonal directly off each supernode's cached panel — no extraction needed.
"""
function LinearAlgebra.logdet(F::SupernodalFactor{T}) where {T}
    sym = F.sym
    acc = zero(T)
    @inbounds for s in 1:sym.nsuper
        nscol = Int(sym.super[s + 1]) - Int(sym.super[s])
        panel = F.panels[s]
        for c in 1:nscol
            acc += log(panel[c, c])
        end
    end
    return 2acc
end
LinearAlgebra.det(F::SupernodalFactor) = exp(LinearAlgebra.logdet(F))

"""
    logdet(F::LDLFactor) -> Real
    det(F::LDLFactor) -> Real

`|det(P·A·Pᵀ)| = |det(D)|` (`L` is unit-lower, `det(L)=1`; `det(P)² = 1` regardless of
`P`'s sign, so `|det(D)|` is exactly `|det(A)|`). Returns the ABSOLUTE VALUE of the
pivot product — my first attempt here returned the signed product and asserted
(wrongly, without actually checking the negative-determinant case) that this matched
CHOLMOD; it doesn't. Checked directly: CHOLMOD's `det`/`logdet` on an indefinite
`ldlt` factor with `diag(2,-3,5)` (true determinant `-30`) returns `det(F) == 30`,
`logdet(F) == log(30)` — the abs-value convention, not the signed product (plausibly
because CHOLMOD's general indefinite `ldlt` does dynamic Bunch–Kaufman-style pivoting
with possible 2×2 blocks internally, where a per-pivot sign isn't as simple to
attribute; not independently confirmed, only the observed input/output behavior is).
Matching that observed behavior exactly (not the mathematically-cleaner signed
version) is what "drop-in" means here — `logdet` is therefore always real, never
throwing `DomainError`/promoting to `Complex`, for any nonsingular factor.
"""
LinearAlgebra.det(F::LDLFactor) = abs(prod(F.d))
LinearAlgebra.logdet(F::LDLFactor) = log(LinearAlgebra.det(F))
