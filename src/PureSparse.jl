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
using PureBLAS: potrf!, trsm!, syrk!, syr2k!, gemm!, gemv!, ger!, nrm2
using PureBLAS: wy_t!, wy_apply!, WYApplyWorkspace, qr_block_size
using Preferences: Preferences

include("tuning.jl")
include("strict.jl")
include("types.jl")
include("qr/types.jl")
# Ordering and symbolic analysis allocate their outputs (zero allocation is required only
# after `symbolic()`); they are still held to type stability and no boxing.
const _ANALYSIS = "symbolic analysis allocates its outputs"
@strict_exempt :noalloc reason = _ANALYSIS include("ordering/interface.jl")
@strict_exempt :noalloc reason = _ANALYSIS include("ordering/amd.jl")
@strict_exempt :noalloc reason = _ANALYSIS include("ordering/ata.jl")
@strict_exempt :noalloc reason = _ANALYSIS include("ordering/colamd.jl")
@strict_exempt :noalloc reason = _ANALYSIS include("symbolic/etree.jl")
@strict_exempt :noalloc reason = _ANALYSIS include("symbolic/counts.jl")
@strict_exempt :noalloc reason = _ANALYSIS include("symbolic/supernodes.jl")
@strict_exempt :noalloc reason = _ANALYSIS include("symbolic/driver.jl")
@strict_exempt :noalloc reason = _ANALYSIS include("qr/symbolic.jl")
@strict_exempt :noalloc reason = _ANALYSIS include("qr/frontal_symbolic.jl")  # M5b multifrontal symbolic layer (design_qr_m5b.md §A4)
@strict_exempt :noalloc reason = _ANALYSIS include("qr/singletons.jl")
include("qr/numeric.jl")
include("qr/solve.jl")
include("qr/frontal.jl")  # M5b multifrontal numeric+solve (design_qr_m5b.md §A4-§A6)
include("numeric/llt.jl")
include("numeric/ldlt.jl")
include("numeric/solve.jl")
include("simplicial/updown.jl")
include("refine.jl")
@strict_exempt reason = "cold: writes a preference" include("dropin_toggle.jl")   # activate!/deactivate! — always available, see tuning.jl's DROPIN_ACTIVE
DROPIN_ACTIVE && @strict_exempt :noalloc reason = "stdlib-compatible allocating entry points" include("dropin.jl")
@strict_exempt reason = "TypeContracts-generated contract metadata" include("contracts.jl")

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

# The allocating constructors build a new factor; `cholesky!`/`ldlt!`/`qr!`/`solve!` are the
# zero-allocation path.
@strict_exempt :noalloc reason = "allocates a new factor" cholesky
@strict_exempt :noalloc reason = "allocates a new factor" ldlt
@strict_exempt :noalloc reason = "allocates a new factor" qr_frontal
@strict_exempt :noalloc reason = "allocates a new factor" simplicial
@strict_exempt :noalloc reason = "allocates a new factor" _build_panels
@strict_exempt :noalloc reason = "allocates a new factor" _qr_block
# `method = :auto` chooses the column or the multifrontal factor at run time.
@strict_exempt :noalloc :typestable reason = "returns QRFactor or QRFrontFactor" qr
@strict_exempt :typestable reason = "always throws" _strict_fail

# Last in the module, so it checks every definition above it.
@strict_policy :default

end # module PureSparse
