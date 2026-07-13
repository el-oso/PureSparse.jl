# OpenBLAS-backed drop-in replacements for PureBLAS's 5 supernodal kernels
# (potrf!/trsm!/syrk!/syr2k!/gemm!), signature-for-signature identical to PureBLAS's
# (verified against PureBLAS's src/lapack.jl:900, src/level3.jl:1614/2657/3100,
# src/gemm.jl:1950). Lives ONLY under benchmark/ — CLAUDE.md requirement 2 forbids calling
# OpenBLAS/LAPACK from src/; this module exists purely so `benchmark/gate.jl` can build the
# "PureSparse+OpenBLAS" kernel-attribution arm (design.md §9.3 config 2) by re-`include`ing
# the unmodified `src/numeric/llt.jl`/`src/numeric/solve.jl` under a different kernel
# binding — no algorithmic duplication, just a different `using`.
module OpenBLASBackend

using LinearAlgebra: LinearAlgebra, BLAS, LAPACK, PosDefException

export potrf!, trsm!, syrk!, syr2k!, gemm!

function potrf!(A::AbstractMatrix; uplo::Char = 'L')
    _, info = LAPACK.potrf!(uplo, A)
    info == 0 || throw(PosDefException(info))
    return A
end

function trsm!(B::AbstractMatrix, A::AbstractMatrix; side::Char = 'L', uplo::Char = 'U',
        transA::Char = 'N', diag::Char = 'N', alpha::Number = true)
    BLAS.trsm!(side, uplo, transA, diag, convert(eltype(B), alpha), A, B)
    return B
end

function syrk!(C::AbstractMatrix, A::AbstractMatrix; uplo::Char = 'U', trans::Char = 'N',
        alpha::Number = true, beta::Number = false)
    BLAS.syrk!(uplo, trans, convert(eltype(C), alpha), A, convert(eltype(C), beta), C)
    return C
end

function syr2k!(C::AbstractMatrix, A::AbstractMatrix, Bm::AbstractMatrix; uplo::Char = 'U',
        trans::Char = 'N', alpha::Number = true, beta::Number = false)
    BLAS.syr2k!(uplo, trans, convert(eltype(C), alpha), A, Bm, convert(eltype(C), beta), C)
    return C
end

function gemm!(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix;
        alpha = one(eltype(C)), beta = zero(eltype(C)), transA::Char = 'N', transB::Char = 'N')
    BLAS.gemm!(transA, transB, convert(eltype(C), alpha), A, B, convert(eltype(C), beta), C)
    return C
end

end # module OpenBLASBackend

# `PureSparseOB`: `numeric/llt.jl` `include`d VERBATIM (same file, not a copy) with
# `potrf!`/`trsm!`/`syrk!`/`syr2k!`/`gemm!` resolving to OpenBLASBackend instead of
# PureBLAS — a genuine kernel swap, not a reimplementation. `Symbolic`/`SupernodalFactor`/
# `Workspace` are IMPORTED (not redefined) from PureSparse so `symbolic(A)` output (which
# only PureSparse itself produces — symbolic analysis calls no dense kernels) plugs straight
# into this module's `cholesky`/`cholesky!`.
#
# Deliberately NOT including `numeric/solve.jl` here: it defines `Base.:\(F::SupernodalFactor,
# ...)`, and re-including it would redefine that SAME method (SupernodalFactor is imported,
# not redefined, so it's the identical type) — Julia allows the overwrite but it is type
# piracy (neither Base nor SupernodalFactor is owned by this module) that would silently
# rebind `\` for EVERY `PureSparseOB.SupernodalFactor` process-wide, corrupting the
# PureSparse+PureBLAS arm's own solve calls. The gate (design.md §9.3) is numeric-refactor
# wall-time, which doesn't need solve; the solve wall-time slice is reported for arms 1 vs 3
# only (PureSparse+PureBLAS's own `solve!`, unaffected).
module PureSparseOB

import LinearAlgebra
using SparseArrays
using ..OpenBLASBackend: potrf!, trsm!, syrk!, syr2k!, gemm!
import PureSparse
import PureSparse: Symbolic, FactorStats, Workspace, AbstractSparseFactor, SupernodalFactor, LDLFactor, issuccess
import PureSparse: AbstractOrdering, AMDOrdering, symbolic

include(joinpath(pkgdir(PureSparse), "src", "numeric", "llt.jl"))

end # module PureSparseOB
