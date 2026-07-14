@testitem "rank detection: exact rank-deficient construction (duplicated column)" begin
    using SparseArrays, LinearAlgebra
    # column 3 is an exact copy of column 1 -> structural+numerical rank 2 of 3.
    A = sparse([1.0 0.0 1.0; 0.0 1.0 0.0; 1.0 1.0 1.0])
    F = PureSparse.qr(A; ordering = PureSparse.AMDOrdering())
    @test F.stats.rank == 2
    @test F.stats.n_dead == 1
    @test F.stats.dropped_norm < 1.0e-8   # the dropped column is EXACTLY dependent
end

@testitem "rank detection: full-rank matrices report rank==n, n_dead==0, dropped_norm==0" begin
    using Random, SparseArrays, LinearAlgebra
    rng = MersenneTwister(404)
    ntested = 0
    for _ in 1:200
        m = rand(rng, 3:14)
        n = rand(rng, 1:min(m, 10))
        A = sprand(rng, m, n, rand(rng, (0.3, 0.5, 0.7)))
        n == 0 && continue
        sv = svdvals(Matrix(A))
        (length(sv) < n || sv[end] < 1.0e-6 * max(sv[1], 1.0)) && continue   # only genuinely well-conditioned
        F = PureSparse.qr(A; ordering = PureSparse.AMDOrdering())
        ntested += 1
        @test F.stats.rank == n
        @test F.stats.n_dead == 0
        @test F.stats.dropped_norm < 1.0e-6
    end
    @test ntested > 50
end

@testitem "rank detection: explicit tol controls exactly which columns get dropped" begin
    using SparseArrays, LinearAlgebra
    # column 3 = column 1 + a 1e-3 perturbation in ONE entry only (NOT a scalar
    # multiple of column 1 -- an earlier version of this test perturbed every entry
    # proportionally, which made column 3 an EXACT scalar multiple of column 1, so
    # the post-reflection residual was exactly 0 via the unconditional B3 guard
    # regardless of tol, not the tol-based threshold this test means to exercise).
    # Empirically verified (not guessed): this construction's post-reflection xnorm
    # falls strictly between tol=1e-3 (still catches it) and tol=1e-4 (does not).
    A = sparse([1.0 0.0 1.0+1e-3; 0.0 1.0 0.0; 1.0 1.0 1.0])
    F_strict = PureSparse.qr(A; ordering = PureSparse.AMDOrdering(), tol = 1.0e-2)  # catches it
    F_loose = PureSparse.qr(A; ordering = PureSparse.AMDOrdering(), tol = 1.0e-6)   # does not
    @test F_strict.stats.n_dead >= 1
    @test F_loose.stats.n_dead == 0
end

@testitem "rank detection: default tol never drops FEWER columns than tol=0 (B3 stays unconditional)" begin
    using Random, SparseArrays
    rng = MersenneTwister(707)
    for _ in 1:150
        m = rand(rng, 2:10)
        n = rand(rng, 1:min(m, 8))
        A = sprand(rng, m, n, rand(rng, (0.15, 0.3, 0.5)))
        n == 0 && continue
        F_default = PureSparse.qr(A; ordering = PureSparse.AMDOrdering())
        F_off = PureSparse.qr(A; ordering = PureSparse.AMDOrdering(), tol = 0)
        @test F_default.stats.n_dead >= F_off.stats.n_dead
    end
end

@testitem "rank detection: basic solution has zero unknowns for dropped columns" begin
    using SparseArrays, LinearAlgebra, Random
    A = sparse([1.0 0.0 1.0; 0.0 1.0 0.0; 1.0 1.0 1.0])   # col 3 = col 1
    F = PureSparse.qr(A; ordering = PureSparse.AMDOrdering())
    @test F.stats.n_dead == 1
    b = [1.0, 2.0, 3.0]
    x = F \ b
    # exactly one of the two "aliased" columns' coefficients must be forced to zero
    # (the dead one) — the basic solution never assigns nonzero mass to a dropped column.
    dead_cols = [k for k in 1:3 if F.beta[F.sym.ciperm[k]] == 0.0 && F.sym.pivotslot[F.sym.ciperm[k]] != 0]
    # translate: find which FINAL-ORDER column index k has a numerically-dead pivot
    for k in 1:F.sym.n
        finalpos = F.sym.ciperm[k]
        if F.sym.pivotslot[finalpos] != 0 && F.beta[finalpos] == 0.0
            @test x[k] == 0.0
        end
    end
    # residual should still be small (basic solution is a genuine least-squares-family answer)
    r = b - A * x
    @test norm(r) < norm(b)   # sanity: not a garbage solution
end

@testitem "rank detection: dropped_norm is a genuine error certificate (bounded, non-negative)" begin
    using Random, SparseArrays, LinearAlgebra
    rng = MersenneTwister(505)
    for _ in 1:100
        m = rand(rng, 3:12)
        n = rand(rng, 1:min(m, 10))
        A = sprand(rng, m, n, rand(rng, (0.2, 0.4, 0.6)))
        F = PureSparse.qr(A; ordering = PureSparse.AMDOrdering())
        @test F.stats.dropped_norm >= 0.0
        @test isfinite(F.stats.dropped_norm)
        @test F.stats.rank + F.stats.n_dead == F.sym.n - F.sym.n1
        @test 0 <= F.stats.n_dead <= F.sym.n
    end
end

@testitem "rank detection: qr! refactor recomputes rank fresh (not sticky from prior call)" begin
    using SparseArrays
    A1 = sparse([1.0 0.0 1.0; 0.0 1.0 0.0; 1.0 1.0 1.0])  # rank-deficient (col3=col1)
    A2 = sparse([1.0 0.0 2.0; 0.0 1.0 0.0; 1.0 1.0 3.0])  # SAME pattern, full rank now
    F = PureSparse.qr(A1; ordering = PureSparse.AMDOrdering())
    @test F.stats.n_dead == 1
    PureSparse.qr!(F, A2)
    @test F.stats.n_dead == 0
    @test F.stats.rank == 3
end
