@testsetup module SupernodeHelpers
using Random
using PureSparse
export postordered_pipeline, random_upper_csc2, true_column_patterns

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
# etree -> postorder -> relabel -> re-etree -> colcounts. Returns the POSTORDERED pattern
# too (needed by supernode_rowind).
function postordered_pipeline(n, colptr, rowval)
    parent0 = PureSparse.etree(n, colptr, rowval)
    post, postinv = PureSparse.postorder(n, parent0)
    cp2, rv2 = PureSparse.relabel_pattern(n, colptr, rowval, postinv)
    parent = PureSparse.etree(n, cp2, rv2)
    colcount = PureSparse.column_counts(n, cp2, rv2, parent)
    return parent, colcount, cp2, rv2
end

# Brute-force TRUE per-column L-pattern via the same elimination game as
# EtreeOracle/CountsOracle: eliminate columns 1..n in order, track fill-in explicitly;
# true_column_patterns[j] = the exact set of rows i>j with L[i,j] != 0 at the moment
# column j is eliminated (independent of the Gilbert-Ng-Peyton / supernode-merging
# machinery under test — this is the ground truth the superset invariant is checked
# against, design.md §3.4/§9.1 point 3).
function true_column_patterns(n::Int, colptr::Vector{Int}, rowval::Vector{Int})
    adj = [Set{Int}() for _ in 1:n]
    for j in 1:n, p in colptr[j]:(colptr[j + 1] - 1)
        i = rowval[p]
        push!(adj[i], j)
        push!(adj[j], i)
    end
    patterns = Vector{Set{Int}}(undef, n)
    for k in 1:n
        nbrs = Set(i for i in adj[k] if i > k)
        patterns[k] = nbrs
        for a in nbrs, b in nbrs
            a == b && continue
            push!(adj[a], b)
        end
    end
    return patterns
end
end

@testitem "fundamental_supernodes: valid partition, covers 1:n" setup = [SupernodeHelpers] begin
    using Random
    rng = MersenneTwister(21)
    for n in (1, 2, 5, 10, 30, 60), density in (0.0, 0.05, 0.2, 0.5)
        colptr, rowval = random_upper_csc2(rng, n, density)
        parent, colcount, _, _ = postordered_pipeline(n, colptr, rowval)
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
        parent, colcount, _, _ = postordered_pipeline(n, colptr, rowval)
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
        parent, colcount, _, _ = postordered_pipeline(n, colptr, rowval)
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
        parent, colcount, _, _ = postordered_pipeline(n, colptr, rowval)
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
            # Exact merged-block height (single-range-root derivation in the
            # `relaxed_amalgamation` docstring): width + colcount[last column] - 1.
            height = width + colcount[j1] - 1
            cells = height * width
            cells == 0 && continue
            z = 1.0 - true_nnz / cells
            tier = PureSparse._amalg_tier(width, PureSparse.AMALG_COLS)
            # This final range was assembled by at least one accepted merge; the LAST
            # accepted merge extended the block to exactly this final range (merges only
            # grow `start` leftward), and its z-check used exactly this height and this
            # colcount sum — so the final range must satisfy the tier bound it was
            # accepted under.
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

@testitem "supernode_rowind: superset invariant (design §3.4/§9.1 point 3)" setup = [SupernodeHelpers] begin
    using Random
    rng = MersenneTwister(31)
    for n in (1, 2, 5, 10, 30, 60), density in (0.0, 0.03, 0.1, 0.3, 0.6)
        colptr0, rowval0 = random_upper_csc2(rng, n, density)
        parent, colcount, cp, rv = postordered_pipeline(n, colptr0, rowval0)
        truth = true_column_patterns(n, cp, rv)

        for (nsuper, super) in (
            PureSparse.fundamental_supernodes(n, parent, colcount),
            PureSparse.relaxed_amalgamation(n, PureSparse.fundamental_supernodes(n, parent, colcount)..., parent, colcount),
        )
            rowind_ptr, rowind, snode_of, sparent, muz, mer = PureSparse.supernode_rowind(
                n, cp, rv, parent, nsuper, super)

            # rowind_ptr/rowind form a valid CSC-style structure of the right total size.
            @test rowind_ptr[1] == 1
            @test rowind_ptr[nsuper + 1] == length(rowind) + 1
            @test issorted(rowind_ptr)

            for s in 1:nsuper
                j0, j1 = super[s], super[s + 1] - 1
                rows_s = Set(rowind[rowind_ptr[s]:(rowind_ptr[s + 1] - 1)])
                @test issorted(rowind[rowind_ptr[s]:(rowind_ptr[s + 1] - 1)])
                @test allunique(rowind[rowind_ptr[s]:(rowind_ptr[s + 1] - 1)])
                # s's own columns are all present (diagonal block).
                for j in j0:j1
                    @test j in rows_s
                end
                # SUPERSET INVARIANT: for every column j in s, rowind[s] restricted to
                # rows >= j is a superset of column j's TRUE L-pattern.
                for j in j0:j1
                    true_below = truth[j]  # rows i>j with L[i,j] != 0
                    @test true_below ⊆ rows_s
                end
            end
            @test muz >= 0
            @test mer >= 0
        end
    end
end

@testitem "2D grid Laplacian: superset invariant + z-bound under multi-pass amalgamation" setup = [SupernodeHelpers] begin
    # 2D grid Laplacians under AMD have BUSHY etrees (most nodes have 2+ children) — the
    # case where the fixpoint amalgamation absorbs several siblings into one parent
    # across passes. The random-pattern items above barely exercise that; this pins the
    # §3.4 superset invariant and the per-block zero-fraction bound on exactly the
    # partition shape the multi-pass merge produces (ROADMAP task 7b').
    using SparseArrays
    nx = ny = 24
    n = nx * ny
    idx(i, j) = (j - 1) * nx + i
    I = Int[]; J = Int[]; V = Float64[]
    for j in 1:ny, i in 1:nx
        p = idx(i, j)
        i > 1 && (push!(I, p); push!(J, idx(i - 1, j)); push!(V, -1.0))
        j > 1 && (push!(I, p); push!(J, idx(i, j - 1)); push!(V, -1.0))
        push!(I, p); push!(J, p); push!(V, 4.0)
    end
    A = sparse(I, J, V, n, n)   # lower triangle stored, as `symbolic` reads it

    S = PureSparse.symbolic(A)

    # Ground-truth per-column L-patterns on the PERMUTED pattern (S.super/S.rowind live
    # in the final AMD∘postorder labeling S.perm).
    Ifull, Jfull, _ = findnz(A + A')
    up_I = Int[]; up_J = Int[]
    for (i, j) in zip(Ifull, Jfull)
        pi, pj = S.iperm[i], S.iperm[j]
        pi < pj && (push!(up_I, pi); push!(up_J, pj))
    end
    P = sparse(up_I, up_J, trues(length(up_I)), n, n)
    truth = true_column_patterns(n, Vector{Int}(P.colptr), Vector{Int}(P.rowval))

    # Fundamental partition on the same (final) labeling, to tell merged blocks (which
    # faced a z-test) from untouched fundamental ones (which never did — their z under
    # the rectangle-cells convention can legitimately exceed every tier, e.g. the dense
    # trailing root block where the diagonal block's strict-upper triangle counts as
    # zeros).
    _, super_fund = PureSparse.fundamental_supernodes(n, S.parent, S.colcount)
    fundamental_starts = Set(super_fund)
    merged_count = 0

    for s in 1:S.nsuper
        j0, j1 = S.super[s], S.super[s + 1] - 1
        rng = S.rowind_ptr[s]:(S.rowind_ptr[s + 1] - 1)
        rows_s = Set(S.rowind[rng])
        @test issorted(S.rowind[rng])
        # Exact-height derivation check: stored height == width + colcount[last col] - 1.
        @test length(rng) == (j1 - j0 + 1) + S.colcount[j1] - 1
        for j in j0:j1
            @test j in rows_s                 # own columns present (diagonal block)
            @test truth[j] ⊆ rows_s           # SUPERSET INVARIANT (design §3.4)
        end
        # Zero-fraction tier bound on every MERGED block (same skip logic as the random
        # item: a final range that is exactly one untouched fundamental supernode never
        # faced a z-test).
        width = j1 - j0 + 1
        is_untouched = j0 in fundamental_starts && (j1 + 1) in fundamental_starts &&
            count(b -> j0 <= b < j1 + 1, super_fund) == 1
        if !is_untouched
            merged_count += 1
            tier = PureSparse._amalg_tier(width, PureSparse.AMALG_COLS)
            z = 1.0 - sum(S.colcount[j0:j1]) / (length(rng) * width)
            # The LAST accepted merge covered exactly this final range with exactly this
            # height, so the tier bound it was accepted under must hold here.
            @test tier != 0
            @test z <= PureSparse.AMALG_ZMAX[tier] + 1.0e-9
        end
    end
    # The whole point of the multi-pass version: bushy-etree merges actually happen.
    @test merged_count > 0
end

@testitem "supernode_rowind workspace bounds are internally consistent" setup = [SupernodeHelpers] begin
    using Random
    rng = MersenneTwister(32)
    for n in (10, 30, 60), density in (0.03, 0.1, 0.3)
        colptr0, rowval0 = random_upper_csc2(rng, n, density)
        parent, colcount, cp, rv = postordered_pipeline(n, colptr0, rowval0)
        nsuper, super = PureSparse.fundamental_supernodes(n, parent, colcount)
        rowind_ptr, rowind, snode_of, sparent, max_update_size, max_extend_rows =
            PureSparse.supernode_rowind(n, cp, rv, parent, nsuper, super)

        # max_extend_rows independently recomputed from the partition directly.
        expected_extend = 0
        for s in 1:nsuper
            nrow = Int(rowind_ptr[s + 1] - rowind_ptr[s])
            ncol = Int(super[s + 1] - super[s])
            expected_extend = max(expected_extend, nrow - ncol)
        end
        @test max_extend_rows == expected_extend
        @test max_update_size >= 0
    end
end
