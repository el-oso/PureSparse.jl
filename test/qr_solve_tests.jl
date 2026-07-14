@testsetup module QRSolveOracle
using Random, SparseArrays, LinearAlgebra
export random_rect_qr3, well_conditioned

random_rect_qr3(rng, m::Int, n::Int, density::Float64) = sprand(rng, m, n, density)

# Rank-deficiency handling (Heath's threshold test, dead-column dropping) is M5a task
# 8's scope, not yet implemented — an unpivoted Householder reflector on a numerically
# near-singular column produces a huge `beta` (division by a near-zero norm) with no
# guard against it yet (the B3 fix only catches EXACTLY zero, design_qr.md §4.4).
# Solve-phase tests here are restricted to well-conditioned matrices; the near-singular
# case is explicitly left to task 8's own test suite, not silently masked.
function well_conditioned(A::SparseMatrixCSC; tol = 1.0e-8)
    n = size(A, 2)
    n == 0 && return true
    sv = svdvals(Matrix(A))
    return length(sv) >= n && sv[end] >= tol * max(sv[1], 1.0)
end
end

@testitem "apply_Q!/apply_Qt! are genuine inverses (QᵀQ = I)" setup = [QRSolveOracle] begin
    using Random, SparseArrays
    rng = MersenneTwister(9)
    for _ in 1:300
        m = rand(rng, 1:14)
        n = rand(rng, 1:min(m, 12))
        A = random_rect_qr3(rng, m, n, rand(rng, (0.15, 0.3, 0.5)))
        F = PureSparse.qr(A; ordering = PureSparse.AMDOrdering())
        mb = F.sym.mb
        y0 = randn(rng, mb)
        y = copy(y0)
        PureSparse.apply_Qt!(y, F)
        PureSparse.apply_Q!(y, F)
        @test isapprox(y, y0, atol = 1.0e-8)
        # and the other order
        y2 = copy(y0)
        PureSparse.apply_Q!(y2, F)
        PureSparse.apply_Qt!(y2, F)
        @test isapprox(y2, y0, atol = 1.0e-8)
    end
end

@testitem "solve!/\\: least-squares residual is orthogonal to range(A)" setup = [QRSolveOracle] begin
    using Random, SparseArrays, LinearAlgebra
    rng = MersenneTwister(20)
    ntested = 0
    for _ in 1:400
        m = rand(rng, 1:16)
        n = rand(rng, 1:min(m, 14))
        A = random_rect_qr3(rng, m, n, rand(rng, (0.1, 0.15, 0.3, 0.5, 0.7)))
        b = randn(rng, m)
        well_conditioned(A) || continue
        F = PureSparse.qr(A; ordering = PureSparse.AMDOrdering())
        x = F \ b
        ntested += 1
        r = b - A * x
        @test norm(A' * r) < 1.0e-6 * max(1.0, norm(b))
    end
    @test ntested > 100
end

@testitem "solve!: agrees with SparseArrays.qr's own solution (black-box, well-conditioned)" setup = [QRSolveOracle] begin
    using Random, SparseArrays, LinearAlgebra
    rng = MersenneTwister(21)
    ntested = 0
    for _ in 1:200
        m = rand(rng, 4:14)
        n = rand(rng, 1:min(m, 10))
        A = random_rect_qr3(rng, m, n, rand(rng, (0.2, 0.35, 0.5)))
        b = randn(rng, m)
        well_conditioned(A) || continue
        F = PureSparse.qr(A; ordering = PureSparse.AMDOrdering())
        x = F \ b
        xref = Matrix(A) \ b   # dense LS reference (m>=n well-conditioned => unique)
        ntested += 1
        @test isapprox(x, xref, atol = 1.0e-6, rtol = 1.0e-6)
    end
    @test ntested > 50
end

@testitem "solve!: qr! refactor gives consistent solve results" begin
    using Random, SparseArrays, LinearAlgebra
    rng = MersenneTwister(303)
    A = sprand(rng, 10, 6, 0.5)
    F = PureSparse.qr(A; ordering = PureSparse.AMDOrdering())
    A2 = SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, A.nzval .* 2.0 .+ 0.5)
    PureSparse.qr!(F, A2)
    b = randn(rng, 10)
    x = F \ b
    r = b - A2 * x
    @test norm(A2' * r) < 1.0e-6 * max(1.0, norm(b))
end

@testitem "solve_minnorm!: residual + minimum-norm (orthogonal to null(A))" setup = [QRSolveOracle] begin
    using Random, SparseArrays, LinearAlgebra
    rng = MersenneTwister(50)
    ntested = 0
    for _ in 1:300
        m = rand(rng, 2:10)
        n = rand(rng, (m + 1):(m + 8))
        A = random_rect_qr3(rng, m, n, rand(rng, (0.15, 0.3, 0.5)))
        b = randn(rng, m)
        At = sparse(A')
        well_conditioned(At) || continue
        F = PureSparse.qr(At; ordering = PureSparse.AMDOrdering())
        x = PureSparse.solve_minnorm!(zeros(n), F, b)
        ntested += 1
        @test norm(A * x - b) < 1.0e-6 * max(1.0, norm(b))
        Nu = nullspace(Matrix(A))
        if size(Nu, 2) > 0
            @test norm(Nu' * x) < 1.0e-5 * max(1.0, norm(x))
        end
    end
    @test ntested > 60
end

@testitem "solve_R!/solve_Rt! edge cases: dead column forces x[k]=0" begin
    using SparseArrays
    # A = [1 1 1] (design_qr.md's own worked example): columns 2,3 dead.
    A = sparse([1, 1, 1], [1, 2, 3], [1.0, 1.0, 1.0], 1, 3)
    F = PureSparse.qr(A; ordering = PureSparse.NaturalOrdering())
    c = [1.0, 2.0, 3.0]
    x = PureSparse.solve_R!(zeros(3), F, c)
    @test x[2] == 0.0
    @test x[3] == 0.0
    xt = PureSparse.solve_Rt!(zeros(3), F, c)
    @test xt[2] == 0.0
    @test xt[3] == 0.0
end
