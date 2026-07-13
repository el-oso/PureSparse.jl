# Iterative refinement (design.md §5.2), recommended whenever a factor's
# `stats.n_perturbed > 0` (LDLᵀ signed regularization forced a pivot, so the exact
# factored matrix differs from `A`) — classical residual-correction iteration, no
# paper-specific algorithm beyond textbook Newton-refinement-for-linear-solves.

"""
    refine!(x::AbstractVecOrMat, F::AbstractSparseFactor, A::SparseMatrixCSC, b::AbstractVecOrMat;
            iters::Int = 2) -> x

Iterative refinement: `x = F \\ b`, then `iters` rounds of `x += F \\ (b - A·x)`.
Recommended whenever `F.stats.n_perturbed > 0` (design.md §5.2) — regularization means
`F` exactly factors a perturbed matrix, not `A` itself, so refinement against the true
`A` recovers accuracy. `A` is read via its LOWER triangle only, matching every other
entry point in this package (`symbolic`'s documented convention) — the residual is
computed as `b - Symmetric(A, :L)·x`. Works for any [`AbstractSparseFactor`](@ref)
(`SupernodalFactor`, `LDLFactor`, `SimplicialLDLFactor`) since it only calls the
generic `solve!`. Allocates (residual/correction scratch); not on any zero-allocation
hot path.
"""
function refine!(
        x::AbstractVecOrMat{T}, F::AbstractSparseFactor{T}, A::SparseMatrixCSC{T}, b::AbstractVecOrMat{T};
        iters::Int = 2,
) where {T}
    solve!(x, F, b)
    iters <= 0 && return x
    Asym = LinearAlgebra.Symmetric(A, :L)
    r = similar(b)
    dx = similar(x)
    for _ in 1:iters
        r .= b
        LinearAlgebra.mul!(r, Asym, x, -one(T), one(T))   # r = b - A*x
        solve!(dx, F, r)
        x .+= dx
    end
    return x
end
