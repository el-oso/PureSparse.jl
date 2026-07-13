# Downstream-consumer smoke suite (design.md §10 M4 gate: "a downstream
# SparseArrays-dependent smoke test suite passes unmodified"). Written exactly as a
# package with ZERO knowledge of PureSparse would write its own sparse-Cholesky tests:
# stdlib names only (`LinearAlgebra.cholesky`/`ldlt`, `\`, `logdet`, `det`,
# `issuccess`, `F.L`/`F.U`/`F.p`), no PureSparse import, no PureSparse type mentioned
# anywhere. No sibling Pure-ecosystem package has a sparse-Cholesky test suite to
# borrow (PureBLAS is dense-only, PureFFT is FFT), so this synthetic stand-in is the
# downstream consumer — kept stack-agnostic on purpose: it must pass BOTH against plain
# CHOLMOD (no drop-in) and with PureSparse's drop-in active (verified both ways by
# test/dropin_tests.jl; the CHOLMOD run is what proves it encodes no PureSparse-shaped
# expectations). Notably `F.U` is only exercised through `\`, because CHOLMOD's own
# lazy `.U` cannot be materialized (`sparse(F.U)` throws in stdlib 1.12.6, observed) —
# a downstream test using `sparse(F.U)` would never have passed on CHOLMOD either.
using Test, LinearAlgebra, SparseArrays, Random

@testset "downstream sparse solver consumer" begin
    # 2-D Dirichlet Laplacian, the way an FEM/graph package would assemble one
    nx = 12
    T1 = spdiagm(-1 => fill(-1.0, nx - 1), 0 => fill(2.0, nx), 1 => fill(-1.0, nx - 1))
    Id = sparse(1.0I, nx, nx)
    A = kron(Id, T1) + kron(T1, Id) + 0.01I   # SPD
    n = size(A, 1)
    rng = MersenneTwister(42)
    b = randn(rng, n)
    Ad = Matrix(A)

    @testset "cholesky: solve, logdet, factors" begin
        F = cholesky(Symmetric(A))
        @test issuccess(F)
        x = F \ b
        @test norm(A * x - b) / norm(b) < 1e-10
        @test logdet(F) ≈ logdet(Ad) rtol = 1e-8
        @test det(F) ≈ det(Ad) rtol = 1e-6

        L = sparse(F.L)
        p = F.p
        @test sort(p) == collect(1:n)
        @test norm(Matrix(L * L') - Ad[p, p]) / norm(Ad) < 1e-10
        @test F.U \ b ≈ L' \ b rtol = 1e-10   # U behaves as Lᵀ

        # plain SparseMatrixCSC entry point (no Symmetric wrapper)
        F2 = cholesky(A)
        @test norm(A * (F2 \ b) - b) / norm(b) < 1e-10

        # caller-supplied ordering: solves must still be exact
        F3 = cholesky(Symmetric(A); perm = collect(n:-1:1))
        @test F3 \ b ≈ x rtol = 1e-8

        # shift: factors A + c·I
        F4 = cholesky(Symmetric(A); shift = 2.0)
        @test norm((A + 2.0I) * (F4 \ b) - b) / norm(b) < 1e-10

        # Int32 column/row indices
        F5 = cholesky(SparseMatrixCSC{Float64,Int32}(A))
        @test norm(A * (F5 \ b) - b) / norm(b) < 1e-10
    end

    @testset "cholesky: failure reporting" begin
        @test_throws PosDefException cholesky(-A)
        Fbad = cholesky(-A; check = false)
        @test !issuccess(Fbad)
    end

    @testset "ldlt: indefinite solve and determinant" begin
        # symmetric indefinite (saddle-point-like) system
        m = 40
        B = A[1:m, 1:m]
        K = [B sparse(1.0I, m, m); sparse(1.0I, m, m) -B]
        kb = randn(rng, 2m)
        Fk = ldlt(K)
        @test issuccess(Fk)
        xk = Fk \ kb
        @test norm(K * xk - kb) / norm(kb) < 1e-8
        Kd = Matrix(K)
        @test det(Fk) ≈ abs(det(Kd)) rtol = 1e-4
        @test logdet(Fk) ≈ log(abs(det(Kd))) rtol = 1e-6
        @test logdet(Fk) isa Real
    end
end
