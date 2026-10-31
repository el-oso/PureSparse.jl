# StrictMode guarantee gates on the hottest paths. StrictModeTest's `@test_*` macros prove
# their guarantee (AllocCheck over every path, inference plus JET) and fail the test on a
# violation. StrictModeTest refuses to load with checks disabled, so these cannot pass
# vacuously.
#
# Asserted:
#   - `@assert_concurrency_safe cholesky/ldlt(sym, A)`: the allocating factor treats its
#     `Symbolic` argument as read-only, so one `Symbolic` can be shared by reference across
#     concurrent factorizations (CONTRIBUTING.md req 7).
#   - `@test_typestable solve!(...)` for all three factor kinds (req 3).
#   - `@test_noalloc solve!(xq, Fq, b)`: the QR solve allocates on no path.
#
# Not asserted: `@test_noalloc` on the Cholesky and LDLᵀ `solve!`. AllocCheck finds an
# allocating path there that the `@allocated == 0` gate does not exercise (see ROADMAP.md,
# "Cross-cutting work").

@testitem "StrictMode guarantees: shared Symbolic is read-only; solve! is type-stable" begin
    using StrictModeTest, SparseArrays, Random
    using PureSparse.StrictMode: @assert_concurrency_safe, assert_enabled
    import PureSparse: symbolic, cholesky, cholesky!, ldlt, ldlt!, qr, solve!, AMDOrdering

    @test assert_enabled()

    n = 40
    A = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(4.0, n), 1 => fill(-1.0, n - 1))
    # both paths of the Cholesky/LDLᵀ factors (design.md §4.5), forced explicitly
    sym = symbolic(A; method = :supernodal); F = cholesky(sym, A)
    symL = symbolic(A; method = :supernodal); FL = ldlt(symL, A)
    symS = symbolic(A; method = :simplicial); FS = cholesky(symS, A); FLS = ldlt(symS, A)
    b = randn(MersenneTwister(1), n); x = similar(b); xL = similar(b)
    cholesky!(F, A); solve!(x, F, b)
    ldlt!(FL, A); solve!(xL, FL, b)
    cholesky!(FS, A); solve!(x, FS, b)
    ldlt!(FLS, A); solve!(xL, FLS, b)
    Fq = qr(A; ordering = AMDOrdering(), tol = 0, singletons = false); xq = zeros(n)
    solve!(xq, Fq, b)

    @test (@assert_concurrency_safe(cholesky(sym, A)); true)
    @test (@assert_concurrency_safe(ldlt(symL, A)); true)
    @test (@assert_concurrency_safe(cholesky(symS, A)); true)
    @test (@assert_concurrency_safe(ldlt(symS, A)); true)

    @test_typestable solve!(x, F, b)
    @test_typestable solve!(xL, FL, b)
    @test_typestable solve!(x, FS, b)
    @test_typestable solve!(xL, FLS, b)
    @test_typestable cholesky!(FS, A)
    @test_typestable ldlt!(FLS, A)
    @test_typestable symbolic(A)
    @test_typestable solve!(xq, Fq, b)

    @test_noalloc solve!(xq, Fq, b)
end
