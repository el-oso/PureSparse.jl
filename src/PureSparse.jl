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
using PureBLAS: potrf!, trsm!, syrk!, syr2k!, gemm!, ger!

include("tuning.jl")
include("types.jl")
include("ordering/interface.jl")
include("ordering/amd.jl")
include("symbolic/etree.jl")
include("symbolic/counts.jl")
include("symbolic/supernodes.jl")
include("symbolic/driver.jl")
include("numeric/llt.jl")
include("numeric/ldlt.jl")
include("numeric/solve.jl")
include("simplicial/updown.jl")
include("refine.jl")
include("contracts.jl")

export symbolic, cholesky, cholesky!, ldlt, ldlt!, issuccess
export simplicial, updowndate!
export solve!, solve_L!, solve_D!, solve_Lt!, refine!
export AbstractOrdering, AMDOrdering, NaturalOrdering, GivenOrdering
export Symbolic, SupernodalFactor, LDLFactor, SimplicialLDLFactor, FactorStats

end # module PureSparse
