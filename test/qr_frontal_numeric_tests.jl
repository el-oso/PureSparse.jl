@testsetup module QRFrontalNumericHelpers
using Random, SparseArrays, LinearAlgebra
export random_rect_frontal2, dense_R_frontal, normal_eq_resid

random_rect_frontal2(rng, m::Int, n::Int, density::Float64) = sprand(rng, m, n, density)

# Dense n×n R (abstract row-of-R layout, indexed by GLOBAL pivotal column) built from
# a QRFrontFactor's row-wise rval/frptr/fcolind storage — the frontal-path analogue of
# qr_numeric_tests.jl's own `dense_R` for QRFactor.
function dense_R_frontal(F)
    fsym = F.fsym
    nb = length(fsym.base.parent)
    R = zeros(eltype(F.rval), nb, nb)
    for k in 1:nb
        F.fpivotrow[k] == 0 && continue
        f = fsym.fsnode[k]
        colslo = Int(fsym.fcolptr[f])
        n_f = Int(fsym.fcolptr[f + 1] - fsym.fcolptr[f])
        pos = k - Int(fsym.fsuper[f]) + 1
        p = Int(fsym.frptr[k])
        for jc in pos:n_f
            gcol = fsym.fcolind[colslo + jc - 1]
            R[k, gcol] = F.rval[p]
            p += 1
        end
    end
    return R
end

normal_eq_resid(A, x, b) = norm(A' * (b - A * x))
end

@testitem "qr_frontal numeric: design_qr_m5b.md §A3.3 worked example (exact RᵀR/solve)" setup=[QRFrontalNumericHelpers] begin
    using SparseArrays, LinearAlgebra, Random

    I = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7]
    J = [1, 2, 4, 1, 2, 5, 2, 4, 3, 4, 3, 5, 4, 5, 4]
    rng = MersenneTwister(1)
    A = sparse(I, J, randn(rng, length(I)) .+ 3.0, 7, 5)

    F = PureSparse.qr_frontal(A; ordering = PureSparse.NaturalOrdering())
    @test F.stats.rank == 5
    @test F.stats.n_dead == 0

    R = dense_R_frontal(F)
    Ad = Matrix(A)[:, F.fsym.base.cperm]
    @test maximum(abs.(R' * R .- Ad' * Ad)) < 1e-10

    b = Float64.(1:7)
    x = PureSparse.solve!(zeros(5), F, b)
    @test normal_eq_resid(A, x, b) < 1e-10
end

@testitem "qr_frontal numeric: random sweep vs M5a :column oracle (RᵀR, normal eqns, x agreement at full rank)" setup=[QRFrontalNumericHelpers] begin
    using SparseArrays, LinearAlgebra, Random

    for seed in 1:60
        rng = MersenneTwister(seed)
        m = rand(rng, 5:40)
        n = rand(rng, 3:min(m, 25))
        dens = rand(rng, 0.1:0.05:0.5)
        A = random_rect_frontal2(rng, m, n, dens)

        F = PureSparse.qr_frontal(A; ordering = PureSparse.NaturalOrdering())
        R = dense_R_frontal(F)
        nb = length(F.fsym.base.parent)
        Ad = Matrix(A)[:, F.fsym.base.cperm]
        @test maximum(abs.(R' * R .- Ad' * Ad)) < 1e-8 * max(1, norm(Ad)^2)

        b = randn(MersenneTwister(seed + 999), m)
        x = PureSparse.solve!(zeros(n), F, b)
        @test normal_eq_resid(A, x, b) < 1e-6 * max(1, norm(b))

        # x-agreement with M5a's own (already-gated) column path is only a valid
        # cross-check at full rank: rank-deficient basic solutions aren't unique
        # (the two paths can break dead-column ties differently), so that case is
        # covered by the normal-equations check above instead (§A9.2's own oracle
        # requirement is agreement on the WELL-POSED case).
        if F.stats.n_dead == 0
            Fcol = PureSparse.qr(A; ordering = PureSparse.NaturalOrdering(), singletons = false)
            xc = PureSparse.solve!(zeros(n), Fcol, b)
            @test norm(x - xc) < 1e-6 * max(1, norm(xc))
        end
    end
end

@testitem "qr_frontal numeric: rank-deficient (duplicate column) still hits the normal-equations oracle" setup=[QRFrontalNumericHelpers] begin
    using SparseArrays, LinearAlgebra, Random

    rng = MersenneTwister(777)
    A = random_rect_frontal2(rng, 20, 10, 0.3)
    A[:, 5] = A[:, 3]

    F = PureSparse.qr_frontal(A; ordering = PureSparse.NaturalOrdering(), tol = 1e-10)
    @test F.stats.n_dead >= 1
    @test F.stats.rank == 9

    b = randn(MersenneTwister(777 + 999), 20)
    x = PureSparse.solve!(zeros(10), F, b)
    @test normal_eq_resid(A, x, b) < 1e-8
end
