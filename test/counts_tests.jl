@testsetup module CountsOracle
using Random
export brute_force_colcounts, random_upper_csc

# Brute-force column-count oracle via the same "elimination game" as EtreeOracle
# (etree_tests.jl) — eliminate columns 1..n in order, track fill-in explicitly; here we
# also record, for each column j, 1 (diagonal) + the number of surviving neighbors > j at
# the moment j is eliminated. This is exactly colcount[j] by definition, independent of
# Gilbert-Ng-Peyton's fast skeleton-leaf algorithm under test.
function brute_force_colcounts(n::Int, adj::Vector{Set{Int}})
    adj = deepcopy(adj)
    colcount = zeros(Int, n)
    for k in 1:n
        nbrs = [i for i in adj[k] if i > k]
        colcount[k] = 1 + length(nbrs)
        for a in nbrs, b in nbrs
            a == b && continue
            push!(adj[a], b)
        end
    end
    return colcount
end

# Random structurally-symmetric adjacency + its strict-upper CSC (mirrors
# EtreeOracle.random_symmetric_pattern/upper_csc in etree_tests.jl, duplicated here to
# keep this testsetup self-contained rather than depending on file-scan order).
function random_upper_csc(rng, n::Int, density::Float64)
    adj = [Set{Int}() for _ in 1:n]
    for j in 1:n, i in 1:(j - 1)
        if rand(rng) < density
            push!(adj[i], j)
            push!(adj[j], i)
        end
    end
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
    return adj, colptr, rowval
end
end

@testitem "column_counts matches brute-force elimination game (postordered patterns)" setup = [CountsOracle] begin
    using Random
    rng = MersenneTwister(11)
    for n in (1, 2, 5, 10, 30, 60), density in (0.0, 0.05, 0.2, 0.5)
        adj, colptr, rowval = random_upper_csc(rng, n, density)
        parent = PureSparse.etree(n, colptr, rowval)
        post, postinv = PureSparse.postorder(n, parent)
        # Relabel BOTH the pattern and a fresh brute-force adjacency by the postorder, so
        # the oracle and the fast algorithm agree on which "column j" means what.
        cp2, rv2 = PureSparse.relabel_pattern(n, colptr, rowval, postinv)
        parent2 = PureSparse.etree(n, cp2, rv2)  # re-derive parent in the new labeling
        adj2 = [Set{Int}() for _ in 1:n]
        for jorig in 1:n, iorig in adj[jorig]
            push!(adj2[postinv[jorig]], postinv[iorig])
        end
        colcount = PureSparse.column_counts(n, cp2, rv2, parent2)
        @test colcount == brute_force_colcounts(n, adj2)
        @test PureSparse.nnz_l(colcount) == sum(colcount)
        @test PureSparse.chol_flops(colcount) == sum(abs2, colcount)
    end
end

@testitem "column_counts structured cases" begin
    # Diagonal-only (empty upper pattern): every column has just its own diagonal.
    n = 6
    colptr0 = ones(Int, n + 1)
    parent0 = zeros(Int, n)
    @test PureSparse.column_counts(n, colptr0, Int[], parent0) == ones(Int, n)

    # Dense/complete graph, natural order (already postordered: parent[j] = j+1 for j<n):
    # colcount[j] = n - j + 1 (every later column is a filled-in neighbor).
    colptr1 = Vector{Int}(undef, n + 1)
    rv1 = Int[]
    colptr1[1] = 1
    for j in 1:n
        for i in 1:(j - 1)
            push!(rv1, i)
        end
        colptr1[j + 1] = length(rv1) + 1
    end
    parent1 = PureSparse.etree(n, colptr1, rv1)
    @test parent1 == vcat(2:n, 0)  # confirms this pattern is already postordered
    colcount1 = PureSparse.column_counts(n, colptr1, rv1, parent1)
    @test colcount1 == collect(n:-1:1)

    # Path graph (tridiagonal, no fill-in): colcount[j] = 2 for interior, 1 for last.
    colptr2 = Vector{Int}(undef, n + 1)
    rv2 = Int[]
    colptr2[1] = 1
    for j in 1:n
        if j > 1
            push!(rv2, j - 1)
        end
        colptr2[j + 1] = length(rv2) + 1
    end
    parent2 = PureSparse.etree(n, colptr2, rv2)
    colcount2 = PureSparse.column_counts(n, colptr2, rv2, parent2)
    @test colcount2 == vcat(fill(2, n - 1), 1)
end
