@testitem "check_refactor_shape: throws on dimension mismatch, passes on match" begin
    using SparseArrays
    A = sparse([1.0 0.0; 0.0 1.0])
    @test PureSparse._check_refactor_shape_impl(A, 2, 2, "test") === nothing
    @test_throws PureSparse.StrictMode.StrictViolation PureSparse._check_refactor_shape_impl(A, 3, 2, "test")
    @test_throws PureSparse.StrictMode.StrictViolation PureSparse._check_refactor_shape_impl(A, 2, 3, "test")
end

@testitem "check_refactor_nnz: throws on nnz mismatch, passes on match" begin
    using SparseArrays
    A = sparse([1.0 0.0; 0.0 1.0])   # nnz == 2
    @test PureSparse._check_refactor_nnz_impl(A, 2, "test") === nothing
    @test_throws PureSparse.StrictMode.StrictViolation PureSparse._check_refactor_nnz_impl(A, 3, "test")
end

@testitem "check_finite: throws on NaN/Inf, passes on finite values" begin
    @test PureSparse._check_finite_impl([1.0, 2.0, -3.5], "test") === nothing
    @test_throws PureSparse.StrictMode.StrictViolation PureSparse._check_finite_impl([1.0, NaN, 3.0], "test")
    @test_throws PureSparse.StrictMode.StrictViolation PureSparse._check_finite_impl([1.0, Inf, 3.0], "test")
end

@testitem "checks_enabled() gate: public entry points are no-ops when disabled (the default)" begin
    using SparseArrays
    @test !PureSparse.StrictMode.checks_enabled()   # default build has checks off
    A = sparse([1.0 0.0; 0.0 1.0])
    # A shape/nnz mismatch that would throw via the _impl must be silently skipped here.
    @test PureSparse.check_refactor_shape(A, 99, 99, "test") === nothing
    @test PureSparse.check_refactor_nnz(A, 99, "test") === nothing
    @test PureSparse.check_finite([NaN], "test") === nothing
end
