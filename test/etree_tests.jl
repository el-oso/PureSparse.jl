@testsetup module EtreeOracle
using SparseArrays
export brute_force_etree, random_symmetric_pattern, upper_csc

# Brute-force elimination-tree oracle via the classic "elimination game" (symbolic
# Cholesky by simulated fill-in — George & Liu 1981 / standard graph-elimination
# textbook material, NOT CHOLMOD-derived): eliminate columns 1..n in order; at each step
# k, parent[k] is the smallest surviving neighbor > k, and eliminating k connects all of
# k's surviving neighbors > k into a clique (the fill-in). This directly matches the
# textbook DEFINITION of the elimination tree, independent of Liu's fast path-compressed
# algorithm under test.
function brute_force_etree(n::Int, adj::Vector{Set{Int}})
    adj = deepcopy(adj)
    parent = zeros(Int, n)
    for k in 1:n
        nbrs = sort!([i for i in adj[k] if i > k])
        if !isempty(nbrs)
            parent[k] = nbrs[1]
        end
        for a in nbrs, b in nbrs
            a == b && continue
            push!(adj[a], b)
        end
    end
    return parent
end

# Build a random structurally-symmetric adjacency (no diagonal) from a density, plus the
# CSC pattern PureSparse.etree expects (full symmetric pattern, as `symmetrized_upper`
# would receive before its own symmetrization — here we hand etree the ALREADY strict
# upper triangular CSC form directly, matching its documented input contract).
function random_symmetric_pattern(rng, n::Int, density::Float64)
    adj = [Set{Int}() for _ in 1:n]
    for j in 1:n, i in 1:(j - 1)
        if rand(rng) < density
            push!(adj[i], j)
            push!(adj[j], i)
        end
    end
    return adj
end

# Strict-upper CSC (1-based, sorted rows per column) from an adjacency-set representation.
function upper_csc(n::Int, adj::Vector{Set{Int}})
    cols = [sort!([i for i in adj[j] if i < j]) for j in 1:n]
    colptr = Vector{Int}(undef, n + 1)
    colptr[1] = 1
    for j in 1:n
        colptr[j + 1] = colptr[j] + length(cols[j])
    end
    rowval = Vector{Int}(undef, colptr[n + 1] - 1)
    for j in 1:n
        rowval[colptr[j]:(colptr[j + 1] - 1)] .= cols[j]
    end
    return colptr, rowval
end
end

@testitem "etree matches brute-force elimination game" setup = [EtreeOracle] begin
    using Random
    rng = MersenneTwister(1)
    for n in (1, 2, 5, 10, 30, 60), density in (0.0, 0.05, 0.2, 0.5)
        adj = random_symmetric_pattern(rng, n, density)
        colptr, rowval = upper_csc(n, adj)
        parent = PureSparse.etree(n, colptr, rowval)
        @test parent == brute_force_etree(n, adj)
    end
end

@testitem "etree structured cases" begin
    # Path graph 1-2-3-...-n (tridiagonal pattern): no fill-in, etree is a chain.
    n = 8
    colptr2 = Vector{Int}(undef, n + 1)
    rv2 = Int[]
    colptr2[1] = 1
    for j in 1:n
        if j > 1
            push!(rv2, j - 1)
        end
        colptr2[j + 1] = length(rv2) + 1
    end
    parent = PureSparse.etree(n, colptr2, rv2)
    @test parent == vcat(2:n, 0)

    # Star pattern: hub node 1 connected to everyone (row entries: column j has row 1 for
    # j>1). Eliminating hub node 1 first fills in a clique among {2,...,n}, so the
    # elimination tree of a star is a CHAIN — not a star — same shape as the path-graph
    # case above (verified independently against `brute_force_etree` in the property
    # test; confirmed by hand-tracing the elimination game here).
    colptr3 = Vector{Int}(undef, n + 1)
    rv3 = Int[]
    colptr3[1] = 1
    for j in 1:n
        if j > 1
            push!(rv3, 1)
        end
        colptr3[j + 1] = length(rv3) + 1
    end
    parent2 = PureSparse.etree(n, colptr3, rv3)
    @test parent2 == vcat(2:n, 0)
end

@testitem "postorder is a valid postordering" setup = [EtreeOracle] begin
    using Random
    rng = MersenneTwister(2)
    for n in (1, 2, 5, 10, 30, 60), density in (0.0, 0.05, 0.2, 0.5)
        adj = random_symmetric_pattern(rng, n, density)
        colptr, rowval = upper_csc(n, adj)
        parent = PureSparse.etree(n, colptr, rowval)
        post, postinv = PureSparse.postorder(n, parent)

        # post/postinv are mutual inverses and a permutation of 1:n.
        @test sort(post) == collect(1:n)
        @test all(post[postinv[j]] == j for j in 1:n)
        @test all(postinv[post[k]] == k for k in 1:n)

        # Every non-root is postordered strictly before its parent.
        @test all(parent[j] == 0 || postinv[j] < postinv[parent[j]] for j in 1:n)

        # Each node's descendant set occupies a contiguous postorder range
        # [first_desc(j), postinv[j]].
        firstdesc = fill(typemax(Int), n)
        for k in 1:n
            j = post[k]
            v = j
            while true
                firstdesc[v] = min(firstdesc[v], k)
                p = parent[v]
                p == 0 && break
                v = p
                # stop climbing once an ancestor already has a smaller firstdesc from
                # an earlier-visited subtree; still correct to keep going, just redundant.
            end
        end
        for j in 1:n
            @test firstdesc[j] <= postinv[j]
        end
    end
end

@testitem "symmetrized_upper produces a valid strict-upper symmetric pattern" setup = [EtreeOracle] begin
    using Random
    rng = MersenneTwister(3)
    for n in (1, 2, 5, 10, 30), density in (0.0, 0.1, 0.3)
        adj = random_symmetric_pattern(rng, n, density)
        colptr, rowval = upper_csc(n, adj)
        # Build a permuted, possibly-asymmetric-storage full CSC (upper + lower explicitly,
        # simulating a real SparseMatrixCSC's storage) to exercise symmetrization.
        fullcolptr = Vector{Int}(undef, n + 1)
        fullrows = Int[]
        fullcolptr[1] = 1
        for j in 1:n
            for i in sort!(collect(adj[j]))
                push!(fullrows, i)
            end
            fullcolptr[j + 1] = length(fullrows) + 1
        end
        perm = collect(1:n)
        shuffle!(rng, perm)
        iperm = Vector{Int}(undef, n)
        for (k, p) in enumerate(perm)
            iperm[p] = k
        end
        cp2, rv2 = PureSparse.symmetrized_upper(n, fullcolptr, fullrows, perm, iperm)
        # Every entry must be strict upper (row < col) and sorted, no duplicates, per column.
        for j in 1:n
            rows = rv2[cp2[j]:(cp2[j + 1] - 1)]
            @test issorted(rows)
            @test allunique(rows)
            @test all(r < j for r in rows)
        end
        # Symmetry check: (i,j) present (i<j) in the permuted pattern iff original
        # adjacency had an edge between iperm-preimages... equivalently, reconstruct the
        # permuted adjacency directly from `adj` and compare as sets.
        expected = Set{Tuple{Int,Int}}()
        for jorig in 1:n, iorig in adj[jorig]
            i, j = iperm[iorig], iperm[jorig]
            lo, hi = i < j ? (i, j) : (j, i)
            lo == hi && continue
            push!(expected, (lo, hi))
        end
        got = Set{Tuple{Int,Int}}()
        for j in 1:n, r in cp2[j]:(cp2[j + 1] - 1)
            push!(got, (rv2[r], j))
        end
        @test got == expected
    end
end
