# Fill-reducing ordering interface, design.md §2.1. AMD (`ordering/amd.jl`) is the only
# nontrivial ordering shipped in v1; nested dissection / METIS-style / other orderings
# plug in later by subtyping `AbstractOrdering` and implementing `order` — nothing else
# in the symbolic/numeric pipeline depends on which ordering ran.

"""
    AbstractOrdering

Supertype for fill-reducing ordering algorithms. Implement
[`order`](@ref)`(alg, n, colptr, rowval)` to add a new ordering.
"""
abstract type AbstractOrdering end

"""
    order(alg::AbstractOrdering, n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}) where {Ti<:Integer} -> Vector{Ti}

Compute a fill-reducing permutation `p` of `1:n` for the structurally-symmetric pattern
given in CSC form (`colptr`/`rowval` — the caller has already symmetrized to
`pattern(A) ∪ pattern(Aᵀ)` with the diagonal removed). Factoring `P·A·Pᵀ` (with `P` the
permutation matrix for `p`) should have low fill-in.

Contract (design.md §2.1): pure function (no mutation of `colptr`/`rowval`), returns a
valid permutation of `1:n`. Ordering is a cold path — clarity is prioritized over
allocation-freedom.
"""
function order end

"""
    AMDOrdering(; dense_mult=AMD_DENSE_MULT, aggressive=true) <: AbstractOrdering

Approximate Minimum Degree ordering (Amestoy, Davis, Duff 1996). `dense_mult` sets the
dense-row-stripping multiplier (design.md §2.2 pt 6); `aggressive` toggles aggressive
element absorption (design.md §2.2 pt 4).
"""
struct AMDOrdering <: AbstractOrdering
    dense_mult::Float64
    aggressive::Bool
end
AMDOrdering(; dense_mult::Real = AMD_DENSE_MULT, aggressive::Bool = true) =
    AMDOrdering(Float64(dense_mult), aggressive)

"""
    NaturalOrdering() <: AbstractOrdering

Identity permutation — no reordering.
"""
struct NaturalOrdering <: AbstractOrdering end
order(::NaturalOrdering, n::Int, ::Vector{Ti}, ::Vector{Ti}) where {Ti<:Integer} =
    collect(Ti, 1:n)

"""
    GivenOrdering(perm::Vector{<:Integer}) <: AbstractOrdering

Use a caller-supplied permutation directly. The escape hatch for external orderings
(nested dissection, METIS.jl) and the mechanism for the same-permutation benchmark gate
(design.md §9.3).
"""
struct GivenOrdering{Ti<:Integer} <: AbstractOrdering
    perm::Vector{Ti}
end
function order(alg::GivenOrdering{Ti}, n::Int, ::Vector{Ti}, ::Vector{Ti}) where {Ti<:Integer}
    length(alg.perm) == n || throw(DimensionMismatch(
        "GivenOrdering: permutation has length $(length(alg.perm)), expected $n"))
    return alg.perm
end
