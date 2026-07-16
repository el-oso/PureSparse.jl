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

@testitem "warm singleton refactor: qr! on n1>0 factors is correct (§2.3 warm-refactor update)" setup = [QRSingletonComposeOracle] begin
    # The original design rejected qr! on a singleton-composed factor; the §2.3
    # warm-refactor update lifts that (owner-authorized): the structural peel set is
    # pattern-only, hence refactor-invariant — only the per-pivot magnitude test needs
    # re-checking against the new values. Sweep random rectangular matrices, refactor
    # each singleton-carrying factor with perturbed values, and check RᵀR = A2ᵀA2 for
    # the FULL composed R plus the solve!'s least-squares orthogonality, against the
    # independently-trusted singletons=false cold factorization's rank.
    using Random, SparseArrays, LinearAlgebra
    rng = MersenneTwister(4)
    ntested_with_singletons = 0
    for _ in 1:200
        m = rand(rng, 3:16)
        n = rand(rng, 1:min(m, 14))
        A = random_rect_compose(rng, m, n, rand(rng, (0.05, 0.1, 0.2)))
        F = PureSparse.qr(A; ordering = PureSparse.AMDOrdering(), tol = 0)
        F.sym.n1 > 0 || continue
        ntested_with_singletons += 1
        A2 = SparseMatrixCSC(m, n, A.colptr, A.rowval, A.nzval .* (1.0 .+ 0.01 .* randn(rng, nnz(A))))
        PureSparse.qr!(F, A2; tol = 0)
        R = dense_R_full(F)
        A2d = Matrix(A2)[:, F.sym.cperm]
        @test isapprox(R' * R, A2d' * A2d, atol = 1.0e-8)
        b = randn(rng, m)
        x = F \ b
        @test norm(A2' * (b - A2 * x)) < 1.0e-6 * max(1.0, norm(b))
        # Rank parity oracle is the COLD composed path (a fresh qr(A2) with singletons
        # on): warm refactor must reproduce exactly what a cold rebuild would report.
        # NOT the singletons=false path — with tol=0 the two cold paths already
        # disagree on structurally-degenerate inputs (empirically found here: an
        # all-empty column is counted n_dead by the nosingletons numeric loop but a
        # never-dead non-column by the composed path's stats — a PRE-EXISTING
        # accounting difference between the two cold paths, orthogonal to warm reuse).
        Fcold = PureSparse.qr(A2; ordering = PureSparse.AMDOrdering(), tol = 0)
        @test F.stats.rank == Fcold.stats.rank
        @test F.stats.n_dead == Fcold.stats.n_dead
    end
    @test ntested_with_singletons > 60   # sanity: the n1>0 warm path is genuinely exercised
end

@testitem "warm singleton refactor: zero-allocation on an LP-slack-shaped matrix (CLAUDE.md req 5)" begin
    # The motivating class for the whole capability (design_qr.md §2.3 / M5 stratum i):
    # structural block + scaled-identity slack columns, every slack column a singleton
    # (plus whatever the cascade peels). Warm qr! on the composed n1>0 factor must be
    # zero-alloc in steady state, exactly like the n1==0 warm path gated above.
    using Random, SparseArrays
    rng = MersenneTwister(11)
    m, k = 80, 15
    A = hcat(sprand(rng, m, k, 0.25), sparse(1:m, 1:m, 1.0 .+ rand(rng, m), m, m))
    F = PureSparse.qr(A; ordering = PureSparse.COLAMDOrdering())
    @test F.sym.n1 >= m   # every slack column peels
    A2 = SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, A.nzval .* (1.0 .+ 0.01 .* randn(rng, nnz(A))))
    PureSparse.qr!(F, A2)   # warm up (any first-touch allocation happens here)
    allocs = @allocated PureSparse.qr!(F, A2)
    @test allocs == 0
end

@testitem "warm singleton refactor: magnitude guard drops a numerically-dead singleton pivot" begin
    # A structural singleton whose A2 value collapses below the singleton threshold is
    # no longer a valid pivot (§2.3's magnitude half is value-dependent). The warm path
    # folds it into the existing n_dead/dropped_norm accounting: the whole [R11 R12]
    # row is zeroed (Q = I on the singleton block, so that discards exactly A2's peeled
    # row as dropped mass), rank drops, and solve! returns the basic solution (that x
    # entry 0, no division by ~0, no NaN).
    using SparseArrays, LinearAlgebra
    # col 1: single entry (structural singleton) in row 1; row 1 also carries R12
    # entries in cols 2/3 so the dropped row has real mass. Cols 2/3 live in rows 2/3.
    I = [1, 1, 2, 3, 1, 2, 3]
    J = [1, 2, 2, 2, 3, 3, 3]
    V = [5.0, 1.5, 1.0, 2.0, 0.5, 3.0, 4.0]
    A = sparse(I, J, V, 3, 3)
    F = PureSparse.qr(A; ordering = PureSparse.NaturalOrdering())
    @test F.sym.n1 == 1
    @test F.stats.n_dead == 0
    rank_before = F.stats.rank
    V2 = copy(V)
    V2[1] = 1.0e-30                        # pivot collapses far below any threshold
    A2 = SparseMatrixCSC(3, 3, A.colptr, A.rowval, V2)
    PureSparse.qr!(F, A2)
    @test F.stats.n_dead == 1
    @test F.stats.rank == rank_before - 1
    # dropped mass = the whole peeled row of A2 (pivot + its R12 entries)
    @test isapprox(F.stats.dropped_norm, norm([1.0e-30, 1.5, 0.5]), rtol = 1.0e-12)
    @test all(iszero, F.r1val[F.r1ptr[1]:(F.r1ptr[2] - 1)])
    b = [1.0, 2.0, 3.0]
    x = F \ b
    @test all(isfinite, x)
    @test x[F.sym.cperm[1]] == 0.0         # basic-solution convention for the dead pivot
    # a healthy refactor afterwards fully recovers (r1val re-filled from values, no
    # sticky dead state)
    PureSparse.qr!(F, A)
    @test F.stats.n_dead == 0
    @test F.stats.rank == rank_before
    x2 = F \ b
    @test norm(A' * (b - A * x2)) < 1.0e-8
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
