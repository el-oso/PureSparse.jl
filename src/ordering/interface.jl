# Fill-reducing ordering interface, design.md §2.1 / design_qr.md §2.1. AMD
# (`ordering/amd.jl`) and COLAMD (`ordering/colamd.jl`) are the nontrivial orderings
# shipped in v1; nested dissection / METIS-style / other orderings plug in later by
# subtyping `AbstractOrdering` and implementing `order`/`order_columns` — nothing else
# in the symbolic/numeric pipeline depends on which ordering ran.

"""
    AbstractOrdering

Supertype for fill-reducing ordering algorithms. Implement [`order`](@ref)`(alg, n,
colptr, rowval)` for symmetric (Cholesky/LDLᵀ) orderings and
[`order_columns`](@ref)`(alg, m, n, colptr, rowval)` for rectangular (sparse QR)
column orderings to add a new ordering.
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
    order_columns(alg::AbstractOrdering, m::Int, n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}) where {Ti<:Integer} -> Vector{Ti}

Compute a fill-reducing column permutation `p` of `1:n` for the (generally
rectangular, generally non-symmetric) pattern of `A` given in CSC form
(design_qr.md §2.1) — the column-ordering counterpart of [`order`](@ref) for sparse
QR, since QR orders columns of a rectangular `A` directly rather than a symmetric
graph. Contract: pure function (no mutation of `colptr`/`rowval`), returns a valid
permutation of `1:n`.
"""
function order_columns end

"""
    AMDOrdering(; dense_mult=AMD_DENSE_MULT, aggressive=true) <: AbstractOrdering

Approximate Minimum Degree ordering (Amestoy, Davis, Duff 1996). `dense_mult` sets the
dense-row-stripping multiplier (design.md §2.2 pt 6); `aggressive` toggles aggressive
element absorption (design.md §2.2 pt 4). `order_columns` (design_qr.md §2.2.6) forms
`pattern(AᵀA)` and delegates to `order` — see `ordering/ata.jl`.
"""
struct AMDOrdering <: AbstractOrdering
    dense_mult::Float64
    aggressive::Bool
end
AMDOrdering(; dense_mult::Real = AMD_DENSE_MULT, aggressive::Bool = true) =
    AMDOrdering(Float64(dense_mult), aggressive)

"""
    COLAMDOrdering(; dense_row_mult=COLAMD_DENSE_ROW_MULT, dense_col_mult=COLAMD_DENSE_COL_MULT) <: AbstractOrdering

Column Approximate Minimum Degree ordering (Davis, Gilbert, Larimore, Ng 2004) — the
sparse QR default (design_qr.md §2.2). `order_columns` operates directly on the
pattern of `A`, never forming `AᵀA`. `dense_row_mult`/`dense_col_mult` set the
dense-row/column withholding multipliers (design_qr.md §2.2 pt 5). Implemented in
`ordering/colamd.jl`; has no symmetric `order` (QR-only ordering).
"""
struct COLAMDOrdering <: AbstractOrdering
    dense_row_mult::Float64
    dense_col_mult::Float64
end
COLAMDOrdering(; dense_row_mult::Real = COLAMD_DENSE_ROW_MULT, dense_col_mult::Real = COLAMD_DENSE_COL_MULT) =
    COLAMDOrdering(Float64(dense_row_mult), Float64(dense_col_mult))

"""
    NaturalOrdering() <: AbstractOrdering

Identity permutation — no reordering.
"""
struct NaturalOrdering <: AbstractOrdering end
order(::NaturalOrdering, n::Int, ::Vector{Ti}, ::Vector{Ti}) where {Ti<:Integer} =
    collect(Ti, 1:n)
order_columns(::NaturalOrdering, m::Int, n::Int, ::Vector{Ti}, ::Vector{Ti}) where {Ti<:Integer} =
    collect(Ti, 1:n)

"""
    GivenOrdering(perm::Vector{<:Integer}) <: AbstractOrdering

Use a caller-supplied permutation directly. The escape hatch for external orderings
(nested dissection, METIS.jl) and the mechanism for the same-permutation benchmark gate
(design.md §9.3 / design_qr.md §9.3). The same permutation vector is reused (unchecked
for symmetric- vs. column-space meaning) by both `order` and `order_columns` — the
caller is responsible for supplying one that fits the call site.
"""
struct GivenOrdering{Ti<:Integer} <: AbstractOrdering
    perm::Vector{Ti}
end
function order(alg::GivenOrdering{Ti}, n::Int, ::Vector{Ti}, ::Vector{Ti}) where {Ti<:Integer}
    length(alg.perm) == n || throw(DimensionMismatch(
        "GivenOrdering: permutation has length $(length(alg.perm)), expected $n"))
    return alg.perm
end
function order_columns(alg::GivenOrdering{Ti}, m::Int, n::Int, ::Vector{Ti}, ::Vector{Ti}) where {Ti<:Integer}
    length(alg.perm) == n || throw(DimensionMismatch(
        "GivenOrdering: permutation has length $(length(alg.perm)), expected $n"))
    return alg.perm
end
