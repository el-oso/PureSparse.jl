# Compile-time interface contracts (TypeContracts.jl) — design.md §9.1 layer 1. These are
# PRECOMPILE-TIME assertions (method surface + inferred return types), eliminated by the
# trimmer, with NO runtime cost and NO runtime failure mode. A violation is a precompile
# error, never a thrown exception. Runtime pre/postconditions are a separate StrictMode
# layer (`strict.jl`) — do not conflate the two (design.md §9.1 D6 / CLAUDE.md req 6).
#
# TypeContracts requires its target to be an abstract type (`@contract` docs: "Abstract
# types only"), so contracts here are declared on `AbstractOrdering` and
# `AbstractSparseFactor{T}` rather than on individual free functions.

using TypeContracts

@contract AbstractOrdering "Every subtype implements `order` (symmetric permutation, design.md §2.1) and `order_columns` (rectangular column permutation for sparse QR, design_qr.md §2.1)." begin
    order(::Self, ::Int, ::Vector, ::Vector)::Vector
    order_columns(::Self, ::Int, ::Int, ::Vector, ::Vector)::Vector
end

@contract AbstractSparseFactor{T} "Every subtype implements `solve!` in place and reports success via `issuccess`." begin
    solve!(::Self, ::AbstractVecOrMat{T}, ::AbstractVecOrMat{T})::AbstractVecOrMat{T}
    issuccess(::Self)::Bool
end
