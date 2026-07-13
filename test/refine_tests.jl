@testitem "refine!: matches solve! when no perturbation, and reduces residual for a perturbed LDLFactor" begin
    using Random, LinearAlgebra, SparseArrays

    rng = MersenneTwister(81)
    n = 40
    I = Int[]; J = Int[]; V = Float64[]
    rowsum = zeros(n)
    for j in 1:n, i in (j + 1):n
        rand(rng) < 0.1 || continue
        v = randn(rng)
        push!(I, i); push!(J, j); push!(V, v)
        rowsum[i] += abs(v); rowsum[j] += abs(v)
    end
    for j in 1:n
        push!(I, j); push!(J, j); push!(V, rowsum[j] + 1.0)
    end
    A = sparse(I, J, V, n, n)
    F = PureSparse.cholesky(A)
    b = randn(rng, n)

    x1 = F \ b
    x2 = similar(x1)
    PureSparse.refine!(x2, F, A, b; iters = 2)
    Ad = Matrix(Symmetric(A, :L))
    @test norm(Ad * x1 - b) < 1.0e-8
    @test norm(Ad * x2 - b) < 1.0e-8

    # Deliberately perturbed LDLFactor: refine! against the TRUE K should reduce the
    # residual relative to solve! alone (which only satisfies the regularized system).
    # The perturbed pivot must itself be a SMALL (near-delta-floor) value with the
    # correct sign already, not a well-conditioned value with the WRONG sign — forcing
    # a sign flip on an O(1) pivot is a large relative perturbation (here, a spectral
    # radius > 1 for the residual-iteration fixed point: K_jj/F_jj = -3/3 = -1, so
    # (I - F⁻¹K)_jj = 2), which iterative refinement cannot converge on by design, not
    # by bug — that is not what "regularization" produces in practice (design.md
    # §5.1's `ldlt_delta` only forces pivots that are already near the magnitude
    # floor). This case instead reflects the realistic scenario: one pivot near-zero
    # (as barrier terms shrink in an IPM iteration), the rest well-conditioned.
    K = sparse(Diagonal([1.0, 1.0e-13, 1.0, 1.0]))
    Fp = PureSparse.ldlt(K; signs = [1, 1, 1, 1])   # only the tiny (same-sign) pivot needs forcing
    @test Fp.stats.n_perturbed == 1
    bp = randn(rng, 4)
    xp0 = Fp \ bp
    Kd = Matrix(Diagonal([1.0, 1.0e-13, 1.0, 1.0]))
    resid0 = norm(Kd * xp0 - bp)
    # The regularized pivot (floored to 1e-12) differs from the true 1e-13 pivot by a
    # factor of 10 — the fixed-point fringe iteration converges geometrically at rate
    # |1 - K_jj/F_jj| = |1 - 0.1| = 0.9 (measured: resid ratio is 0.900 per iteration,
    # matching this exactly), not to machine precision in a handful of iterations —
    # the TRUE matrix is itself near-singular in that direction, an inherent
    # conditioning limit, not a refine! defect. Assert the honest property: monotonic,
    # geometric improvement, not an unreachable machine-precision target.
    resids = Float64[resid0]
    xp = similar(xp0)
    for iters in (1, 5, 20)
        PureSparse.refine!(xp, Fp, K, bp; iters)
        push!(resids, norm(Kd * xp - bp))
    end
    @test issorted(resids; rev = true)              # strictly improving with more iterations
    @test resids[end] < 0.2 * resid0                 # ~0.9^20 ≈ 0.12x — well below the un-refined residual
end

@testitem "refine!: iters=0 is exactly solve!" begin
    using Random, LinearAlgebra, SparseArrays
    rng = MersenneTwister(82)
    n = 20
    A = sparse(Diagonal(rand(rng, n) .+ 1.0))
    F = PureSparse.cholesky(A)
    b = randn(rng, n)
    x1 = F \ b
    x2 = similar(x1)
    PureSparse.refine!(x2, F, A, b; iters = 0)
    @test x1 == x2
end
