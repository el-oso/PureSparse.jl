# Trim-safety check on the factor-and-solve path — exactly what `julia juliac/build.jl`
# compiles into the puresparse_smoke executable (the M1 "juliac --trim smoke build
# succeeds" gate, design.md §10). TrimCheck @validate runs the same reachability analysis
# as juliac --trim=safe, so a dynamic-dispatch regression on these roots REDs here in the
# ordinary test suite instead of only in the (slow, manual) juliac build. Mirrors
# PureBLAS.jl/test/trim_tests.jl (signatures given by argument TYPES, not values).
# ponytail: validates reachability only; the real end-to-end build stays manual via
# juliac/build.jl (minutes-long, unsuitable per-test-run).

@testitem "TrimCheck trim-safety (factor-and-solve roots)" begin
    using TrimCheck
    @validate(
        init = begin
            using PureSparse
            using SparseArrays
        end,
        # symbolic analysis (AMD ordering default; kwarg-default path included)
        PureSparse.symbolic(SparseMatrixCSC{Float64, Int64}),
        # LLᵀ: allocate-and-factor, in-place refactor, solve
        PureSparse.cholesky(PureSparse.Symbolic{Int64}, SparseMatrixCSC{Float64, Int64}),
        PureSparse.cholesky!(PureSparse.SupernodalFactor{Float64, Int64}, SparseMatrixCSC{Float64, Int64}),
        PureSparse.solve!(Vector{Float64}, PureSparse.SupernodalFactor{Float64, Int64}, Vector{Float64}),
        # LDLᵀ: same trio
        PureSparse.ldlt(PureSparse.Symbolic{Int64}, SparseMatrixCSC{Float64, Int64}),
        PureSparse.ldlt!(PureSparse.LDLFactor{Float64, Int64}, SparseMatrixCSC{Float64, Int64}),
        PureSparse.solve!(Vector{Float64}, PureSparse.LDLFactor{Float64, Int64}, Vector{Float64}),
    )
end
