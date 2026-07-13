module PureSparse

import LinearAlgebra  # NOT `using` — our own `cholesky`/`cholesky!`/`ldlt`/`ldlt!` must not
                       # silently extend LinearAlgebra's/SparseArrays' same-named stdlib
                       # methods before the deliberate, opt-in M4 drop-in (design.md §8) —
                       # `import` keeps LinearAlgebra.* qualified-only, so our definitions
                       # create a genuinely separate function, not a stdlib method-table
                       # overwrite. Verified: SparseArrays.cholesky(::SparseMatrixCSC)
                       # exists (unwrapped, not just Symmetric-wrapped) and a bare `using`
                       # here would silently replace it the moment PureSparse loads.
using SparseArrays
using PureBLAS: potrf!, trsm!, syrk!, syr2k!, gemm!

include("tuning.jl")
include("types.jl")
include("ordering/interface.jl")
include("ordering/amd.jl")
include("symbolic/etree.jl")
include("symbolic/counts.jl")
include("symbolic/supernodes.jl")
include("symbolic/driver.jl")
include("numeric/llt.jl")
include("numeric/solve.jl")
include("contracts.jl")

export symbolic, cholesky, cholesky!, solve!, solve_L!, solve_Lt!, issuccess
# `ldlt`/`ldlt!` (design.md §5, M2 — SQD/LDLᵀ) are not implemented yet; re-add to this
# export list when numeric/ldlt.jl lands rather than exporting a name that throws
# UndefVarError today.
export AbstractOrdering, AMDOrdering, NaturalOrdering, GivenOrdering
export Symbolic, SupernodalFactor, LDLFactor, FactorStats

end # module PureSparse
