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
using PureBLAS: potrf!, trsm!, syrk!, syr2k!, gemm!, ger!, nrm2
using Preferences: Preferences

include("tuning.jl")
include("types.jl")
include("qr/types.jl")
include("ordering/interface.jl")
include("ordering/amd.jl")
include("ordering/ata.jl")
include("ordering/colamd.jl")
include("symbolic/etree.jl")
include("symbolic/counts.jl")
include("symbolic/supernodes.jl")
include("symbolic/driver.jl")
include("qr/symbolic.jl")
include("qr/numeric.jl")
include("qr/solve.jl")
include("numeric/llt.jl")
include("numeric/ldlt.jl")
include("numeric/solve.jl")
include("simplicial/updown.jl")
include("refine.jl")
include("dropin_toggle.jl")   # activate!/deactivate! — always available, see tuning.jl's DROPIN_ACTIVE
DROPIN_ACTIVE && include("dropin.jl")
include("contracts.jl")

export symbolic, cholesky, cholesky!, ldlt, ldlt!, issuccess
export symbolic_qr, qr, qr!
export apply_Q!, apply_Qt!, solve_R!, solve_Rt!, solve_minnorm!
export simplicial, updowndate!
export solve!, solve_L!, solve_D!, solve_Lt!, refine!
export AbstractOrdering, AMDOrdering, COLAMDOrdering, NaturalOrdering, GivenOrdering
export Symbolic, SupernodalFactor, LDLFactor, SimplicialLDLFactor, FactorStats
export QRSymbolic, QRFactor, QRStats
export activate!, deactivate!
DROPIN_ACTIVE && export sparse_L

end # module PureSparse
