@testsetup module QRSingletonComposeOracle
using Random, SparseArrays
export random_rect_compose, dense_R_full

random_rect_compose(rng, m::Int, n::Int, density::Float64) = sprand(rng, m, n, density)

# Full R (n x n), correctly composing R11/R12 (already full-final-column-indexed) with
# the block's own rval/rcolind (BLOCK-LOCAL column indices -- offset by n1 here; an
# earlier version of the ad hoc verification for this task forgot this offset and
# produced a spurious failure that looked like a real composition bug until traced).
function dense_R_full(F)
    n1 = F.sym.n1
    n = F.sym.n
    R = zeros(n, n)
    for k in 1:n1
        for c in F.r1ptr[k]:(F.r1ptr[k + 1] - 1)
            R[k, F.r1colind[c]] = F.r1val[c]
        end
    end
    nb = length(F.sym.parent)
    for k in 1:nb
        for c in F.sym.rptr[k]:(F.ws.rcursor[k] - 1)
            R[n1 + k, n1 + F.rcolind[c]] = F.rval[c]
        end
    end
    return R
end
end

@testitem "singleton composition: RᵀR = AᵀA holds with the full (R11/R12+block) R" setup = [QRSingletonComposeOracle] begin
    using Random, SparseArrays, LinearAlgebra
    rng = MersenneTwister(1)
    ntested_with_singletons = 0
    for _ in 1:300
        m = rand(rng, 3:16)
        n = rand(rng, 1:min(m, 14))
        A = random_rect_compose(rng, m, n, rand(rng, (0.05, 0.1, 0.2)))
        F = PureSparse.qr(A; ordering = PureSparse.AMDOrdering(), tol = 0)
        F.sym.n1 > 0 && (ntested_with_singletons += 1)
        R = dense_R_full(F)
        Ad = Matrix(A)[:, F.sym.cperm]
        @test isapprox(R' * R, Ad' * Ad, atol = 1.0e-8)
    end
    @test ntested_with_singletons > 100   # sanity: singletons are genuinely exercised
end

@testitem "singleton composition: solve! least-squares residual orthogonality (n1>0)" setup = [QRSingletonComposeOracle] begin
    using Random, SparseArrays, LinearAlgebra
    rng = MersenneTwister(1)
    ntested_with_singletons = 0
    for _ in 1:300
        m = rand(rng, 3:16)
        n = rand(rng, 1:min(m, 14))
        A = random_rect_compose(rng, m, n, rand(rng, (0.05, 0.1, 0.2)))
        b = randn(rng, m)
        F = PureSparse.qr(A; ordering = PureSparse.AMDOrdering(), tol = 0)
        F.sym.n1 > 0 && (ntested_with_singletons += 1)
        x = F \ b
        r = b - A * x
        @test norm(A' * r) < 1.0e-6 * max(1.0, norm(b))
    end
    @test ntested_with_singletons > 100
end

@testitem "singleton composition: qr! rejects n1>0 factors with a clear error" begin
    using SparseArrays
    A = sparse([1.0 0.0 1.0; 0.0 1.0 1.0; 0.0 0.0 1.0])   # 3 columns, all become singletons
    F = PureSparse.qr(A; ordering = PureSparse.AMDOrdering())
    @test F.sym.n1 > 0
    @test_throws ArgumentError PureSparse.qr!(F, A)
end

@testitem "singleton composition: exact worked example (design_qr.md §2.3 shape)" begin
    using SparseArrays, LinearAlgebra
    # column 1's only entry is row 1 (an initial singleton); columns 2/3 each start at
    # degree 3, dropping to degree 2 once row 1 dies -- NOT a cascade, exactly one
    # singleton (empirically confirmed, not assumed: a naive "column 1 has 1 entry,
    # the rest don't" reading of a DIFFERENT matrix with a genuinely cascading
    # structure gave n1=3, not 1, when actually checked).
    A = sparse([1.0 0.5 0.7; 0.0 0.9 0.3; 0.0 0.2 0.5])
    F = PureSparse.qr(A; ordering = PureSparse.AMDOrdering(), tol = 0)
    @test F.sym.n1 == 1
    @test F.r1val[F.r1ptr[1]] == A[1, 1]   # no numerical work: raw copy (own derivation)
    b = [1.0, 2.0, 3.0]
    x = F \ b
    r = b - A * x
    @test norm(A' * r) < 1.0e-8
end

@testitem "singleton composition: fully-singleton (diagonal-like) matrix" begin
    using SparseArrays, LinearAlgebra
    A = sparse(1:4, 1:4, [2.0, 3.0, 4.0, 5.0], 4, 4)
    F = PureSparse.qr(A; ordering = PureSparse.AMDOrdering(), tol = 0)
    @test F.sym.n1 == 4   # every column is immediately a singleton
    @test length(F.sym.parent) == 0   # no block remains at all
    b = [1.0, 2.0, 3.0, 4.0]
    x = F \ b
    @test isapprox(A * x, b, atol = 1.0e-10)
end

@testitem "singleton composition: rank detection interacts sanely with n1>0" begin
    using Random, SparseArrays, LinearAlgebra
    rng = MersenneTwister(9)
    ntested = 0
    for _ in 1:150
        m = rand(rng, 3:12)
        n = rand(rng, 1:min(m, 10))
        A = sprand(rng, m, n, rand(rng, (0.05, 0.1, 0.2)))
        F = PureSparse.qr(A; ordering = PureSparse.AMDOrdering())   # default tol
        ntested += 1
        @test F.stats.rank + F.stats.n_dead == F.sym.n   # rank includes n1's live pivots
        @test F.stats.dropped_norm >= 0.0
        @test isfinite(F.stats.dropped_norm)
    end
    @test ntested > 0
end
