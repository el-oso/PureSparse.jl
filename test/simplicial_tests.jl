@testsetup module SimplicialHelpers
using SparseArrays, LinearAlgebra, Random
export dense_unit_L_simplicial, dense_sym, reconstruct_original, relerr, random_spd,
    random_sqd_kkt_s

# Dense n x n unit-lower L from a SimplicialLDLFactor's per-column storage (factor
# order). Slack slots (beyond colnnz) are never read.
function dense_unit_L_simplicial(G)
    n = G.sym.n
    L = Matrix{Float64}(I, n, n)
    for j in 1:n
        p0 = Int(G.colptr[j])
        for p in p0:(p0 + Int(G.colnnz[j]) - 1)
            L[Int(G.rowval[p]), j] = G.nzval[p]
        end
    end
    return L
end

# Dense symmetric matrix from a SparseMatrixCSC read via its lower triangle (the
# symbolic driver's documented input convention — same as ldlt_tests.jl helpers).
function dense_sym(K::SparseMatrixCSC)
    n = size(K, 1)
    Kd = zeros(n, n)
    for j in 1:n, p in K.colptr[j]:(K.colptr[j + 1] - 1)
        i = K.rowval[p]
        i >= j || continue
        Kd[i, j] = K.nzval[p]
        Kd[j, i] = K.nzval[p]
    end
    return Kd
end

# Pᵀ·(L·D·Lᵀ)·P — the factored matrix mapped back to ORIGINAL order, the
# permutation-independent oracle quantity.
function reconstruct_original(G)
    L = dense_unit_L_simplicial(G)
    R = L * Diagonal(G.d) * L'
    ip = invperm(Int.(G.sym.perm))
    return R[ip, ip]
end

relerr(a, b) = norm(a .- b) / max(norm(b), eps(real(float(eltype(b)))))

function random_spd(rng, n::Int, density::Float64)
    A = sprandn(rng, n, n, density)
    return sparse(Symmetric(A * A' + n * I, :L))
end

# Same construction as ldlt_tests.jl's random_sqd_kkt (kept local: testsetups are
# self-contained modules).
function random_sqd_kkt_s(rng, npos::Int, nneg::Int, density::Float64)
    n = npos + nneg
    I_ = Int[]; J_ = Int[]; V_ = Float64[]
    rowsum = zeros(n)
    addsym!(i, j, v) = (push!(I_, i); push!(J_, j); push!(V_, v);
        i != j && (push!(I_, j); push!(J_, i); push!(V_, v));
        rowsum[i] += abs(v); i != j && (rowsum[j] += abs(v)))
    for j in 1:npos, i in (j + 1):npos
        rand(rng) < density && addsym!(i, j, randn(rng))
    end
    for j in 1:nneg, i in (j + 1):nneg
        rand(rng) < density && addsym!(npos + i, npos + j, randn(rng))
    end
    for j in 1:npos, i in 1:nneg
        rand(rng) < density && addsym!(npos + i, j, randn(rng))
    end
    for j in 1:n
        v = (rowsum[j] + 1.0) * (j <= npos ? 1.0 : -1.0)
        push!(I_, j); push!(J_, j); push!(V_, v)
    end
    return sparse(I_, J_, V_, n, n)
end
end

@testitem "simplicial: conversion reproduces the supernodal LDLᵀ factor" setup = [SimplicialHelpers] begin
    using Random, LinearAlgebra, SparseArrays
    rng = MersenneTwister(81)
    for n in (1, 5, 30, 80), density in (0.02, 0.15)
        K = random_spd(rng, n, density)
        F = PureSparse.ldlt(K)
        G = simplicial(F)
        @test G isa SimplicialLDLFactor{Float64,Int}
        @test PureSparse.issuccess(G)
        @test G.d == F.d
        @test relerr(reconstruct_original(G), dense_sym(K)) < 1.0e-12
        # stored pattern is sorted, in-capacity, and parent = min(pattern)
        for j in 1:n
            len = Int(G.colnnz[j])
            @test len <= Int(G.colptr[j + 1]) - Int(G.colptr[j])
            pat = Int.(G.rowval[Int(G.colptr[j]):(Int(G.colptr[j]) + len - 1)])
            @test issorted(pat) && allunique(pat)
            @test all(>(j), pat)
            @test Int(G.parent[j]) == (len > 0 ? pat[1] : 0)
        end
        # simplicial solve agrees with the supernodal solve on the same factor
        b = randn(rng, n)
        @test relerr(G \ b, F \ b) < 1.0e-12
    end
end

@testitem "updowndate!: update matches an independent refactorization of K + wwᵀ" setup = [SimplicialHelpers] begin
    using Random, LinearAlgebra, SparseArrays
    rng = MersenneTwister(82)
    for n in (10, 40, 90), density in (0.03, 0.12), nnzw in (1, 3, 8)
        K = random_spd(rng, n, density)
        G = simplicial(PureSparse.ldlt(K); grow = Float64(n))
        w = zeros(n)
        for i in randperm(rng, n)[1:min(nnzw, n)]
            w[i] = randn(rng)
        end
        @test updowndate!(G, w, +1) === :ok
        @test PureSparse.issuccess(G)
        K2 = sparse(Symmetric(dense_sym(K) + w * w', :L))
        # oracle 1: the updated factor reconstructs K + wwᵀ (completely independent
        # arithmetic path from the recurrence)
        @test relerr(reconstruct_original(G), Matrix(Symmetric(K2, :L))) < 1.0e-12
        # oracle 2: a from-scratch symbolic+numeric refactorization of K + wwᵀ
        # reconstructs the same matrix — two independent computations of one object
        F2 = PureSparse.ldlt(K2)
        G2 = simplicial(F2)
        @test relerr(reconstruct_original(G), reconstruct_original(G2)) < 1.0e-12
    end
end

@testitem "updowndate!: SQD update in the H block preserves inertia and matches refactorization" setup = [SimplicialHelpers] begin
    using Random, LinearAlgebra, SparseArrays
    rng = MersenneTwister(83)
    npos, nneg = 25, 15
    n = npos + nneg
    K = random_sqd_kkt_s(rng, npos, nneg, 0.1)
    F = PureSparse.ldlt(K; n_pos = npos, n_neg = nneg)
    @test F.stats.n_perturbed == 0
    G = simplicial(F)
    w = zeros(n)                          # support in the H block: [H+wwᵀ Aᵀ; A -C] stays SQD
    w[3] = 1.3; w[11] = -0.7; w[20] = 0.4
    @test updowndate!(G, w, +1) === :ok
    K2d = dense_sym(K) + w * w'
    @test relerr(reconstruct_original(G), K2d) < 1.0e-12
    @test count(<(0), G.d) == nneg        # inertia preserved (Vanderbei: SQD ⇒ (n₊,n₋,0))
    @test count(>(0), G.d) == npos
end

@testitem "updowndate!: update-then-downdate round-trip returns the original factor" setup = [SimplicialHelpers] begin
    using Random, LinearAlgebra, SparseArrays
    rng = MersenneTwister(84)
    for n in (12, 50, 100), density in (0.05, 0.15)
        K = random_spd(rng, n, density)
        G = simplicial(PureSparse.ldlt(K); grow = Float64(n))
        L0 = dense_unit_L_simplicial(G)
        d0 = copy(G.d)
        w = zeros(n)
        for i in randperm(rng, n)[1:5]
            w[i] = randn(rng)
        end
        @test updowndate!(G, w, +1) === :ok
        @test updowndate!(G, w, -1) === :ok
        # design.md M2 gate target: round-trip error ≤ 100·eps·n
        tol = 100 * eps(Float64) * n
        @test relerr(dense_unit_L_simplicial(G), L0) < tol
        @test relerr(G.d, d0) < tol
        # pattern may have grown (grown entries return to ~0 numerically but stay
        # stored); the parent map must stay consistent with the stored patterns
        for j in 1:n
            len = Int(G.colnnz[j])
            @test Int(G.parent[j]) == (len > 0 ? Int(G.rowval[Int(G.colptr[j])]) : 0)
        end
        # workspace invariant: wval left all-zero (O(changed nnz) cleanup discipline)
        @test all(iszero, G.wval)
    end
end

@testitem "updowndate!: rank-k matrix wrapper == sequenced rank-1, matches refactorization" setup = [SimplicialHelpers] begin
    using Random, LinearAlgebra, SparseArrays
    rng = MersenneTwister(85)
    n = 60
    K = random_spd(rng, n, 0.08)
    W = zeros(n, 3)
    for c in 1:3, i in randperm(rng, n)[1:4]
        W[i, c] = randn(rng)
    end
    Ga = simplicial(PureSparse.ldlt(K); grow = Float64(n))
    @test updowndate!(Ga, W, +1) === :ok
    Gb = simplicial(PureSparse.ldlt(K); grow = Float64(n))
    for c in 1:3
        @test updowndate!(Gb, W[:, c], +1) === :ok
    end
    @test dense_unit_L_simplicial(Ga) == dense_unit_L_simplicial(Gb)   # identical arithmetic
    @test Ga.d == Gb.d
    K2 = dense_sym(K) + W * W'
    @test relerr(reconstruct_original(Ga), K2) < 1.0e-12
    # and back down, rank-k
    @test updowndate!(Ga, W, -1) === :ok
    @test relerr(reconstruct_original(Ga), dense_sym(K)) < 100 * eps(Float64) * n
end

@testitem "updowndate!: downdate instability is detected via the recurrence (ᾱ ≤ 0)" setup = [SimplicialHelpers] begin
    using Random, LinearAlgebra, SparseArrays
    rng = MersenneTwister(86)
    # downdating more than the matrix "has": I₄ - 4·e₂e₂ᵀ is indefinite
    G = simplicial(PureSparse.ldlt(sparse(1.0I, 4, 4)))
    w = zeros(4); w[2] = 2.0
    @test updowndate!(G, w, -1) === :not_definite
    @test !PureSparse.issuccess(G)
    @test G.stats.fail_col > 0
    @test all(iszero, G.wval)             # workspace invariant survives the early exit
    # a failed factor refuses further modification
    @test_throws ArgumentError updowndate!(G, w, +1)

    # a random SPD case pushed past its smallest eigenvalue along a random direction
    K = random_spd(rng, 30, 0.1)
    Kd = dense_sym(K)
    u = randn(rng, 30); u ./= norm(u)
    lam = eigvals(Symmetric(Kd))[1]
    G2 = simplicial(PureSparse.ldlt(K); grow = 30.0)
    wbad = u .* sqrt(2 * abs(lam) + 2 * (u' * Kd * u))   # uᵀ(K - wwᵀ)u < 0 guaranteed
    @test updowndate!(G2, wbad, -1) === :not_definite
    @test !PureSparse.issuccess(G2)
    # ...while a SAFE downdate on the same matrix succeeds
    G3 = simplicial(PureSparse.ldlt(K); grow = 30.0)
    wok = u .* (0.5 * sqrt(abs(lam)))
    @test updowndate!(G3, wok, -1) === :ok
    @test relerr(reconstruct_original(G3), Kd - wok * wok') < 1.0e-11

    # SQD flavor: an UPDATE against a negative pivot can also change inertia —
    # diag(1, -1) + 4·e₂e₂ᵀ flips the second pivot's sign (ᾱ = 1 - 4 ≤ 0)
    Kq = spdiagm(0 => [1.0, -1.0])
    G4 = simplicial(PureSparse.ldlt(Kq; signs = [1, -1]))
    wq = [0.0, 2.0]
    @test updowndate!(G4, wq, +1) === :not_definite
end

@testitem "updowndate!: pattern growth beyond slack returns :refactor_required" setup = [SimplicialHelpers] begin
    using LinearAlgebra, SparseArrays
    K = spdiagm(0 => fill(2.0, 6))
    w = zeros(6); w[1] = 1.0; w[4] = 1.0
    # grow = 0.0: capacity == initial length, zero slack anywhere — the fill at
    # (4, col 1) cannot be stored
    G0 = simplicial(PureSparse.ldlt(K; ordering = NaturalOrdering()); grow = 0.0)
    @test all(iszero, G0.colnnz)          # diagonal K: no off-diagonal entries at all
    @test updowndate!(G0, w, +1) === :refactor_required
    @test !PureSparse.issuccess(G0)
    @test G0.stats.fail_col > 0
    @test all(iszero, G0.wval)            # workspace invariant survives the early exit
    @test_throws ArgumentError updowndate!(G0, w, +1)   # failed factor refuses reuse

    # identical update with default slack succeeds and is correct — the failure above
    # is the slack policy, not the update itself
    G1 = simplicial(PureSparse.ldlt(K; ordering = NaturalOrdering()))
    @test updowndate!(G1, w, +1) === :ok
    Kd = Matrix(Diagonal(fill(2.0, 6))) + w * w'
    L = dense_unit_L_simplicial(G1)
    ip = invperm(Int.(G1.sym.perm))
    R = (L * Diagonal(G1.d) * L')[ip, ip]
    @test norm(R - Kd) / norm(Kd) < 1.0e-14
end

@testitem "simplicial solves: residual gate after update WITHOUT supernodal refactorization" setup = [SimplicialHelpers] begin
    using Random, LinearAlgebra, SparseArrays
    rng = MersenneTwister(87)
    for n in (20, 60, 120), density in (0.05, 0.12)
        K = random_spd(rng, n, density)
        G = simplicial(PureSparse.ldlt(K); grow = Float64(n))
        w = zeros(n)
        for i in randperm(rng, n)[1:6]
            w[i] = randn(rng)
        end
        @test updowndate!(G, w, +1) === :ok
        K2d = dense_sym(K) + w * w'
        b = randn(rng, n)
        x = G \ b                          # only simplicial solves — F is stale, never refactored
        @test norm(K2d * x - b) / (norm(K2d) * norm(x) + eps()) < 1.0e-12
        # multi-RHS
        B = randn(rng, n, 3)
        X = G \ B
        for c in 1:3
            @test relerr(X[:, c], G \ B[:, c]) < 1.0e-13
        end
        # split solves compose to the full solve (factor order)
        yfull = b[Int.(G.sym.perm)]
        solve_L!(yfull, G); solve_D!(yfull, G); solve_Lt!(yfull, G)
        xsplit = similar(b)
        xsplit[Int.(G.sym.perm)] = yfull
        @test xsplit == x                  # identical arithmetic path
    end
end

@testitem "updowndate!: zero allocations after warmup" setup = [SimplicialHelpers] begin
    using Random, LinearAlgebra, SparseArrays
    rng = MersenneTwister(88)
    n = 50
    K = random_spd(rng, n, 0.1)
    G = simplicial(PureSparse.ldlt(K))
    w = zeros(n); w[7] = 0.3; w[22] = -0.8; w[41] = 0.5
    updowndate!(G, w, +1)                  # warmup (also grows the pattern once)
    updowndate!(G, w, -1)
    @test @allocated(updowndate!(G, w, +1)) == 0
    @test @allocated(updowndate!(G, w, -1)) == 0
end

@testitem "updowndate!: argument validation" setup = [SimplicialHelpers] begin
    using Random, LinearAlgebra, SparseArrays
    rng = MersenneTwister(89)
    K = random_spd(rng, 8, 0.2)
    G = simplicial(PureSparse.ldlt(K))
    @test_throws ArgumentError updowndate!(G, zeros(8), 2)
    @test_throws ArgumentError updowndate!(G, zeros(8), 0)
    @test_throws DimensionMismatch updowndate!(G, zeros(7), 1)
    @test_throws ArgumentError simplicial(PureSparse.ldlt(K); grow = -1.0)
    @test updowndate!(G, zeros(8), 1) === :ok          # empty support: no-op
    @test updowndate!(G, spzeros(8), 1) === :ok        # SparseVector input accepted
end
