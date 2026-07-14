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
        # QR (M5a task 10): in-place refactor (n1==0 only, §2.3) + least-squares solve.
        # `symbolic_qr`/`qr` both take a MANDATORY `ordering` keyword (no default, §2.1
        # — COLAMDOrdering() as the default lands in a later task) — TrimCheck's
        # `@validate` only supports positional-argument-type roots (`validate_function`
        # calls `Base.return_types(func, args)` with purely positional types), so those
        # two can't be given as roots here; `qr!`/`solve!` cover the actual numeric hot
        # path this gate exists for.
        PureSparse.qr!(PureSparse.QRFactor{Float64, Int64}, SparseMatrixCSC{Float64, Int64}),
        PureSparse.solve!(Vector{Float64}, PureSparse.QRFactor{Float64, Int64}, Vector{Float64}),
    )
end
