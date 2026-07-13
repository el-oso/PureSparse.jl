@testsetup module AMDOracle
using Random
export random_adj, adj_to_full_csc, elimination_fill, exact_mindeg_fill, nedges,
    path_adj, star_adj, random_tree_adj, clique!

# Random structurally-symmetric adjacency (no diagonal), Set-based — same construction
# as EtreeOracle/CountsOracle (duplicated to keep this testsetup self-contained).
function random_adj(rng, n::Int, density::Float64)
    adj = [Set{Int}() for _ in 1:n]
    for j in 1:n, i in 1:(j - 1)
        if rand(rng) < density
            push!(adj[i], j)
            push!(adj[j], i)
        end
    end
    return adj
end

# FULL symmetric CSC (both triangles, no diagonal, sorted rows) — the input contract of
# `order` per ordering/interface.jl (NOT the strict-upper form etree takes).
function adj_to_full_csc(n::Int, adj; Ti::Type = Int)
    colptr = Vector{Ti}(undef, n + 1)
    rowval = Ti[]
    colptr[1] = 1
    for j in 1:n
        append!(rowval, sort!(collect(Ti, adj[j])))
        colptr[j + 1] = length(rowval) + 1
    end
    return colptr, rowval
end

# nnz(L) for a GIVEN elimination order, via the same "elimination game" fill simulator
# as EtreeOracle/CountsOracle: eliminating v costs 1 (diagonal) + its surviving
# neighbors, and connects those neighbors into a clique (the fill).
function elimination_fill(n::Int, adj, perm)
    adj = deepcopy(adj)
    eliminated = falses(n)
    nnzL = 0
    for k in 1:n
        v = perm[k]
        nbrs = [u for u in adj[v] if !eliminated[u]]
        nnzL += 1 + length(nbrs)
        for a in nbrs, b in nbrs
            a == b || push!(adj[a], b)
        end
        eliminated[v] = true
    end
    return nnzL
end

# EXACT greedy minimum degree oracle: at each step eliminate the variable with the
# fewest surviving neighbors (ties -> lowest index), tracking fill exactly. Returns its
# nnz(L). AMD approximates this heuristic; the two need not match, but AMD should never
# be far worse.
function exact_mindeg_fill(n::Int, adj)
    adj = deepcopy(adj)
    eliminated = falses(n)
    nnzL = 0
    for _ in 1:n
        best, bestd = 0, typemax(Int)
        for v in 1:n
            eliminated[v] && continue
            d = count(u -> !eliminated[u], adj[v])
            if d < bestd
                best, bestd = v, d
            end
        end
        v = best
        nbrs = [u for u in adj[v] if !eliminated[u]]
        nnzL += 1 + length(nbrs)
        for a in nbrs, b in nbrs
            a == b || push!(adj[a], b)
        end
        eliminated[v] = true
    end
    return nnzL
end

nedges(adj) = sum(length, adj; init = 0) ÷ 2
# Zero-fill invariant: nnz(L) = n + #edges holds iff the ordering produces NO fill
# (every original edge contributes exactly one below-diagonal entry, once, when its
# first endpoint is eliminated).

function path_adj(n::Int)
    adj = [Set{Int}() for _ in 1:n]
    for v in 1:(n - 1)
        push!(adj[v], v + 1)
        push!(adj[v + 1], v)
    end
    return adj
end

function star_adj(n::Int)
    adj = [Set{Int}() for _ in 1:n]
    for v in 2:n
        push!(adj[1], v)
        push!(adj[v], 1)
    end
    return adj
end

function random_tree_adj(rng, n::Int)
    adj = [Set{Int}() for _ in 1:n]
    for v in 2:n
        u = rand(rng, 1:(v - 1))
        push!(adj[u], v)
        push!(adj[v], u)
    end
    return adj
end

function clique!(adj, verts)
    for a in verts, b in verts
        a != b && push!(adj[a], b)
    end
    return adj
end
end

@testitem "AMD returns a valid permutation on random patterns" setup = [AMDOracle] begin
    using Random
    rng = MersenneTwister(41)
    for n in (1, 2, 3, 5, 8, 13, 20, 50, 120, 200), density in (0.0, 0.02, 0.1, 0.3, 0.8)
        adj = random_adj(rng, n, density)
        colptr, rowval = adj_to_full_csc(n, adj)
        # default; aggressive off; low dense_mult (forces dense-row stripping on the
        # denser patterns, up to stripping nearly every variable)
        for alg in (AMDOrdering(), AMDOrdering(aggressive = false),
                AMDOrdering(dense_mult = 0.1))
            p = PureSparse.order(alg, n, colptr, rowval)
            @test p isa Vector{Int}
            @test sort(p) == 1:n
        end
    end
    # n = 0 and n = 1 edge cases
    @test PureSparse.order(AMDOrdering(), 0, [1], Int[]) == Int[]
    @test PureSparse.order(AMDOrdering(), 1, [1, 1], Int[]) == [1]
    # Ti = Int32: returns Vector{Int32}
    adj = random_adj(rng, 30, 0.2)
    colptr32, rowval32 = adj_to_full_csc(30, adj; Ti = Int32)
    p32 = PureSparse.order(AMDOrdering(), 30, colptr32, rowval32)
    @test p32 isa Vector{Int32}
    @test sort(p32) == 1:30
end

@testitem "AMD fill is close to exact minimum degree on tiny graphs" setup = [AMDOracle] begin
    using Random
    rng = MersenneTwister(42)
    # Approximate vs exact minimum degree are different heuristics — equality is not
    # required (and exact-greedy is itself not optimal, so AMD can even win). The 2x
    # gate catches a badly-broken implementation, not tie-breaking differences.
    for n in 2:8, density in (0.1, 0.3, 0.5, 0.8), _ in 1:20
        adj = random_adj(rng, n, density)
        colptr, rowval = adj_to_full_csc(n, adj)
        p = PureSparse.order(AMDOrdering(), n, colptr, rowval)
        famd = elimination_fill(n, adj, p)
        fexact = exact_mindeg_fill(n, adj)
        @test famd <= 2 * fexact
    end
    # Same gate on a medium size, where the quotient-graph machinery (element
    # absorption, supervariables, compaction) is actually exercised.
    for density in (0.05, 0.15), _ in 1:5
        n = 60
        adj = random_adj(rng, n, density)
        colptr, rowval = adj_to_full_csc(n, adj)
        p = PureSparse.order(AMDOrdering(), n, colptr, rowval)
        @test elimination_fill(n, adj, p) <= 2 * exact_mindeg_fill(n, adj)
    end
end

@testitem "AMD finds zero-fill orderings when they exist" setup = [AMDOracle] begin
    using Random
    rng = MersenneTwister(43)
    # Trees: a leaf always has degree 1, eliminating it adds no fill, and the graph
    # stays a tree — so exact minimum degree is provably zero-fill, and AMD's bound is
    # exact for degree-<=1 variables (paper Thm 4.1: exact whenever |E_i| <= 2).
    for n in (2, 5, 10, 40, 100), _ in 1:5
        adj = random_tree_adj(rng, n)
        colptr, rowval = adj_to_full_csc(n, adj)
        for alg in (AMDOrdering(), AMDOrdering(aggressive = false))
            p = PureSparse.order(alg, n, colptr, rowval)
            @test elimination_fill(n, adj, p) == n + nedges(adj)
        end
    end
    # Path and star (the star exercises repeated element absorption at the hub).
    for n in (8, 30)
        for adj in (path_adj(n), star_adj(n))
            colptr, rowval = adj_to_full_csc(n, adj)
            p = PureSparse.order(AMDOrdering(), n, colptr, rowval)
            @test elimination_fill(n, adj, p) == n + nedges(adj)
        end
    end
    # Already-chordal clique tree: two K5s sharing one vertex. Every minimum-degree
    # vertex is simplicial here, so zero fill is achievable and expected.
    n = 9
    adj = clique!(clique!([Set{Int}() for _ in 1:n], 1:5), 5:9)
    colptr, rowval = adj_to_full_csc(n, adj)
    p = PureSparse.order(AMDOrdering(), n, colptr, rowval)
    @test elimination_fill(n, adj, p) == n + nedges(adj)
end

@testitem "AMD structured cases: dense graph, dense row, disconnected cliques" setup = [AMDOracle] begin
    using Random
    # Fully dense: any ordering gives identical (zero) fill — validity + termination.
    n = 12
    adj = clique!([Set{Int}() for _ in 1:n], 1:n)
    colptr, rowval = adj_to_full_csc(n, adj)
    p = PureSparse.order(AMDOrdering(), n, colptr, rowval)
    @test sort(p) == 1:n
    @test elimination_fill(n, adj, p) == n + nedges(adj)
    # (Supervariable detection should mass-eliminate a clique in very few pivots; the
    # zero-fill identity above is the observable consequence.)

    # Explicit dense row: sparse background + one hub adjacent to everything.
    # dense_mult = 1.0 on n = 100 puts the threshold at max(16, 10) = 16, so only the
    # hub (degree 99) is stripped; it must appear exactly once, ordered last.
    rng = MersenneTwister(44)
    n = 100
    adj = random_adj(rng, n, 0.03)
    hub = 7
    for v in 1:n
        v == hub && continue
        push!(adj[hub], v)
        push!(adj[v], hub)
    end
    colptr, rowval = adj_to_full_csc(n, adj)
    alg = AMDOrdering(dense_mult = 1.0)
    p = PureSparse.order(alg, n, colptr, rowval)
    @test sort(p) == 1:n                       # hub appears exactly once (bijection)
    threshold = max(PureSparse.AMD_DENSE_FLOOR, alg.dense_mult * sqrt(n))
    dense = [v for v in 1:n if length(adj[v]) > threshold]
    @test hub in dense
    @test sort(p[(n - length(dense) + 1):n]) == dense   # stripped rows come last
    # Default dense_mult must NOT strip the hub here (threshold 100 > degree 99),
    # and still produce a valid permutation.
    pdef = PureSparse.order(AMDOrdering(), n, colptr, rowval)
    @test sort(pdef) == 1:n

    # Disconnected components: two separate cliques, no cross-contamination — each
    # eliminates independently with zero fill.
    n = 8
    adj = clique!(clique!([Set{Int}() for _ in 1:n], 1:4), 5:8)
    colptr, rowval = adj_to_full_csc(n, adj)
    for alg in (AMDOrdering(), AMDOrdering(aggressive = false))
        p = PureSparse.order(alg, n, colptr, rowval)
        @test sort(p) == 1:n
        @test elimination_fill(n, adj, p) == n + nedges(adj)
    end
end

@testitem "AMD fill quality vs CHOLMOD's fill-reducing permutation" setup = [AMDOracle] begin
    # Quality smoke test (design.md §2.2 output validation). AMD.jl could not be added
    # to the test environment (its Pkg resolve is blocked by an unrelated pre-existing
    # TypeContracts compat pin), so the reference ordering is CHOLMOD's own permutation
    # observed black-box through SparseArrays' public API (output only — clean-room
    # safe per CLAUDE.md req 1). Both orderings are scored with the SAME fill
    # simulator, so the comparison is apples-to-apples; the wide 1.5x slack is a
    # badly-broken-implementation gate, not a tuning target.
    using Random, SparseArrays, LinearAlgebra
    rng = MersenneTwister(45)

    function cholmod_perm_of(adj, n)
        I_ = Int[]
        J_ = Int[]
        for j in 1:n, i in adj[j]
            push!(I_, i)
            push!(J_, j)
        end
        append!(I_, 1:n)
        append!(J_, 1:n)
        V = ones(Float64, length(I_))
        A = sparse(I_, J_, V, n, n)
        # diagonally dominant SPD so cholesky succeeds; only the PATTERN matters here
        for j in 1:n
            A[j, j] = n + 1.0
        end
        return LinearAlgebra.cholesky(Symmetric(A, :L)).p
    end

    # 2D grid Laplacian pattern (the classic sparse SPD stress case) + random patterns
    k = 12
    n = k * k
    adj = [Set{Int}() for _ in 1:n]
    idx(r, c) = (c - 1) * k + r
    for c in 1:k, r in 1:k
        for (dr, dc) in ((1, 0), (0, 1))
            rr, cc = r + dr, c + dc
            (1 <= rr <= k && 1 <= cc <= k) || continue
            push!(adj[idx(r, c)], idx(rr, cc))
            push!(adj[idx(rr, cc)], idx(r, c))
        end
    end
    cases = Any[(n, adj)]
    for (nn, dens) in ((80, 0.05), (150, 0.02), (120, 0.1))
        push!(cases, (nn, random_adj(rng, nn, dens)))
    end
    for (nn, aa) in cases
        colptr, rowval = adj_to_full_csc(nn, aa)
        pours = PureSparse.order(AMDOrdering(), nn, colptr, rowval)
        pref = cholmod_perm_of(aa, nn)
        fours = elimination_fill(nn, aa, pours)
        fref = elimination_fill(nn, aa, pref)
        @test fours <= 1.5 * fref
    end
end
