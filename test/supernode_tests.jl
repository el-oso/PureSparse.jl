@testsetup module SupernodeHelpers
using Random
using PureSparse
export postordered_pipeline, random_upper_csc2

function random_upper_csc2(rng, n::Int, density::Float64)
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
    return colptr, rowval
end

# Full pipeline through postordering, matching how `symbolic()` will eventually drive it:
# etree -> postorder -> relabel -> re-etree -> colcounts.
function postordered_pipeline(n, colptr, rowval)
    parent0 = PureSparse.etree(n, colptr, rowval)
    post, postinv = PureSparse.postorder(n, parent0)
    cp2, rv2 = PureSparse.relabel_pattern(n, colptr, rowval, postinv)
    parent = PureSparse.etree(n, cp2, rv2)
    colcount = PureSparse.column_counts(n, cp2, rv2, parent)
    return parent, colcount
end
end

@testitem "fundamental_supernodes: valid partition, covers 1:n" setup = [SupernodeHelpers] begin
    using Random
    rng = MersenneTwister(21)
    for n in (1, 2, 5, 10, 30, 60), density in (0.0, 0.05, 0.2, 0.5)
        colptr, rowval = random_upper_csc2(rng, n, density)
        parent, colcount = postordered_pipeline(n, colptr, rowval)
        nsuper, super = PureSparse.fundamental_supernodes(n, parent, colcount)
        @test length(super) == nsuper + 1
        @test super[1] == 1
        @test super[nsuper + 1] == n + 1
        @test issorted(super)
        @test all(super[s] < super[s + 1] for s in 1:nsuper)  # every supernode nonempty
        @test 1 <= nsuper <= n
    end
end

@testitem "fundamental_supernodes structured cases" begin
    # Path graph (tridiagonal): interior colcounts are all 2 (no exact +1 nesting between
    # consecutive interior columns, so they stay separate supernodes) EXCEPT the trailing
    # pair (n-1, n), whose L-patterns are {n-1,n} and {n} — genuinely nested with no
    # fill (a path's last edge is trivially a dense 2x2 corner) — so those two columns
    # DO form one legitimate fundamental supernode. Verified by hand: colcount[n-1]=2 ==
    # colcount[n]+1 = 1+1 = 2.
    n = 8
    colptr = Vector{Int}(undef, n + 1)
    rv = Int[]
    colptr[1] = 1
    for j in 1:n
        j > 1 && push!(rv, j - 1)
        colptr[j + 1] = length(rv) + 1
    end
    parent = PureSparse.etree(n, colptr, rv)
    colcount = PureSparse.column_counts(n, colptr, rv, parent)
    nsuper, super = PureSparse.fundamental_supernodes(n, parent, colcount)
    @test nsuper == n - 1
    @test super == vcat(collect(1:(n - 1)), n + 1)

    # Dense graph, natural postordered numbering (chain etree, single-child throughout):
    # colcounts nest perfectly -> merges into exactly one supernode.
    colptr2 = Vector{Int}(undef, n + 1)
    rv2 = Int[]
    colptr2[1] = 1
    for j in 1:n
        for i in 1:(j - 1)
            push!(rv2, i)
        end
        colptr2[j + 1] = length(rv2) + 1
    end
    parent2 = PureSparse.etree(n, colptr2, rv2)
    colcount2 = PureSparse.column_counts(n, colptr2, rv2, parent2)
    nsuper2, super2 = PureSparse.fundamental_supernodes(n, parent2, colcount2)
    @test nsuper2 == 1
    @test super2 == [1, n + 1]
end

@testitem "supernode_tree: snode_of/sparent consistency" setup = [SupernodeHelpers] begin
    using Random
    rng = MersenneTwister(22)
    for n in (2, 5, 10, 30, 60), density in (0.05, 0.2, 0.5)
        colptr, rowval = random_upper_csc2(rng, n, density)
        parent, colcount = postordered_pipeline(n, colptr, rowval)
        nsuper, super = PureSparse.fundamental_supernodes(n, parent, colcount)
        snode_of, sparent = PureSparse.supernode_tree(n, nsuper, super, parent)

        @test all(1 <= snode_of[j] <= nsuper for j in 1:n)
        # snode_of respects the partition boundaries.
        for s in 1:nsuper, j in super[s]:(super[s + 1] - 1)
            @test snode_of[j] == s
        end
        # sparent[s] == 0 iff s's last column is an etree root; else it's a HIGHER
        # supernode index (postorder: parents always come after all their descendants).
        for s in 1:nsuper
            lastcol = super[s + 1] - 1
            if parent[lastcol] == 0
                @test sparent[s] == 0
            else
                @test sparent[s] == snode_of[parent[lastcol]]
                @test sparent[s] > s
            end
        end
    end
end

@testitem "relaxed_amalgamation: coarsens the fundamental partition validly" setup = [SupernodeHelpers] begin
    using Random
    rng = MersenneTwister(23)
    for n in (1, 2, 5, 10, 30, 80), density in (0.0, 0.03, 0.1, 0.3)
        colptr, rowval = random_upper_csc2(rng, n, density)
        parent, colcount = postordered_pipeline(n, colptr, rowval)
        nsuper, super = PureSparse.fundamental_supernodes(n, parent, colcount)
        nsuper2, super2 = PureSparse.relaxed_amalgamation(n, nsuper, super, parent, colcount)

        # Still a valid partition of 1:n.
        @test super2[1] == 1
        @test super2[nsuper2 + 1] == n + 1
        @test issorted(super2)
        @test all(super2[s] < super2[s + 1] for s in 1:nsuper2)
        # Amalgamation only merges: never more (post-merge) supernodes than fundamental ones.
        @test nsuper2 <= nsuper
        # Every fundamental boundary is either a surviving boundary or was absorbed into
        # a wider merged supernode — i.e. super2 is a subsequence of super (coarsening).
        @test issubset(Set(super2), Set(super))
    end
end

@testitem "relaxed_amalgamation respects the zero-fraction tier bound" setup = [SupernodeHelpers] begin
    using Random
    rng = MersenneTwister(24)
    # Build a lookup of original fundamental-supernode boundaries so we can tell which
    # final supernodes actually resulted from a merge (width > 1 fundamental supernode)
    # vs. survived untouched (nothing to check for those).
    for n in (20, 50, 100), density in (0.02, 0.08, 0.2)
        colptr, rowval = random_upper_csc2(rng, n, density)
        parent, colcount = postordered_pipeline(n, colptr, rowval)
        nsuper, super = PureSparse.fundamental_supernodes(n, parent, colcount)
        nsuper2, super2 = PureSparse.relaxed_amalgamation(n, nsuper, super, parent, colcount)
        fundamental_starts = Set(super)
        for s in 1:nsuper2
            j0, j1 = super2[s], super2[s + 1] - 1
            width = j1 - j0 + 1
            # Skip supernodes that are exactly one untouched fundamental supernode — no
            # merge decision was made for them, so there's no z-bound to check.
            j1 + 1 in fundamental_starts && j0 in fundamental_starts &&
                count(b -> j0 <= b < j1 + 1, super) == 1 && continue
            true_nnz = sum(colcount[j0:j1])
            rows_est = colcount[j0]
            cells = rows_est * width
            cells == 0 && continue
            z = 1.0 - true_nnz / cells
            tier = PureSparse._amalg_tier(width, PureSparse.AMALG_COLS)
            # This final range was assembled by at least one accepted merge, whose LAST
            # step's z-check (by additivity of colcount sums over contiguous ranges) is
            # exactly this z computed over the full final range — so it must satisfy the
            # tier bound it was accepted under.
            @test tier != 0
            @test z <= PureSparse.AMALG_ZMAX[tier] + 1.0e-9
        end
    end
end

@testitem "relaxed_amalgamation structured cases" begin
    # Dense graph (already one fundamental supernode) stays one supernode.
    n = 10
    colptr = Vector{Int}(undef, n + 1)
    rv = Int[]
    colptr[1] = 1
    for j in 1:n
        for i in 1:(j - 1)
            push!(rv, i)
        end
        colptr[j + 1] = length(rv) + 1
    end
    parent = PureSparse.etree(n, colptr, rv)
    colcount = PureSparse.column_counts(n, colptr, rv, parent)
    nsuper, super = PureSparse.fundamental_supernodes(n, parent, colcount)
    nsuper2, super2 = PureSparse.relaxed_amalgamation(n, nsuper, super, parent, colcount)
    @test nsuper2 == 1

    # n=1 / n=0 edge cases don't error.
    @test PureSparse.fundamental_supernodes(1, [0], [1]) == (1, [1, 2])
    p0, c0 = Int[], Int[]
    nsuper0, super0 = PureSparse.fundamental_supernodes(0, p0, c0)
    @test nsuper0 == 0
    @test super0 == [1]
end
