@testsetup module ATAOracle
using Random, SparseArrays, LinearAlgebra
export random_rect, dense_ata_pattern, elimination_fill_adj, adj_of_pattern,
    stdlib_spqr_nnzR, greedy_mindeg_fill

# Random m×n sparse pattern (not necessarily structurally full rank), as a SparseMatrixCSC.
random_rect(rng, m::Int, n::Int, density::Float64) = sprand(rng, m, n, density)

# Brute-force pattern(AᵀA) via SparseArrays' own multiply — the oracle ata_pattern is
# checked against (black-box stdlib output only, clean-room safe per CLAUDE.md req 1;
# forming AᵀA explicitly here in the TEST is fine, it's exactly what ata_pattern exists
# to avoid doing in the actual symbolic pipeline).
function dense_ata_pattern(A::SparseMatrixCSC)
    n = size(A, 2)
    B = A' * A
    return (B .!= 0) .& .!Matrix(I, n, n)
end

# Adjacency-list view of a full symmetric CSC pattern (colptr/rowval, both triangles) —
# for reuse with the same elimination-game fill simulator amd_tests.jl uses.
function adj_of_pattern(n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}) where {Ti<:Integer}
    adj = [Set{Int}() for _ in 1:n]
    for j in 1:n, p in colptr[j]:(colptr[j + 1] - 1)
        push!(adj[j], Int(rowval[p]))
    end
    return adj
end

# nnz(L) == nnz(R) (design_qr.md H1: the star-matrix identity) for a GIVEN column
# permutation, via the same elimination-game fill simulator as amd_tests.jl.
function elimination_fill_adj(n::Int, adj, perm)
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

# stdlib SuiteSparseQR's own nnz(R) for a given A — black-box output oracle (CLAUDE.md
# req 1: output observation via SparseArrays is clean-room safe, source is never read).
stdlib_spqr_nnzR(A::SparseMatrixCSC) = nnz(qr(A).R)

# Greedy (not globally optimal) minimum-degree fill — same reference simulator as
# amd_tests.jl's `exact_mindeg_fill` (name kept distinct here since it is not a true
# brute-force-over-all-permutations minimum, just a strong greedy baseline; mirrored for
# consistency with the existing AMD test convention rather than pulling in a
# permutations-enumeration dependency).
function greedy_mindeg_fill(n::Int, adj)
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
end

@testitem "csc_transpose matches SparseArrays' own transpose" setup = [ATAOracle] begin
    using Random, SparseArrays
    rng = MersenneTwister(101)
    for _ in 1:300
        m = rand(rng, 0:12)
        n = rand(rng, 0:12)
        A = random_rect(rng, m, n, rand(rng, (0.05, 0.2, 0.5)))
        colptr2, rowval2 = PureSparse.csc_transpose(m, n, A.colptr, A.rowval)
        got = SparseMatrixCSC(n, m, colptr2, rowval2, ones(Int, length(rowval2)))
        @test got == (sparse(A') .!= 0) .* 1
        # every column's row indices must be sorted ascending (no separate sort needed)
        for i in 1:m
            seg = view(rowval2, colptr2[i]:(colptr2[i + 1] - 1))
            @test issorted(seg)
        end
    end
end

@testitem "ata_pattern matches brute-force pattern(AᵀA)" setup = [ATAOracle] begin
    using Random, SparseArrays
    rng = MersenneTwister(202)
    for _ in 1:500
        m = rand(rng, 0:12)
        n = rand(rng, 0:12)
        A = random_rect(rng, m, n, rand(rng, (0.05, 0.15, 0.3, 0.6)))
        colptr2, rowval2 = PureSparse.ata_pattern(m, n, A.colptr, A.rowval)
        n == 0 && (@test isempty(rowval2); continue)
        got = SparseMatrixCSC(n, n, colptr2, rowval2, ones(Int, length(rowval2))) .!= 0
        ref = dense_ata_pattern(A)
        @test got == ref
        @test got == got'  # genuinely symmetric
        for j in 1:n
            @test issorted(view(rowval2, colptr2[j]:(colptr2[j + 1] - 1)))
        end
    end
end

@testitem "ata_pattern edge cases: empty, single row/column, disconnected" begin
    using SparseArrays
    for (m, n) in ((0, 0), (1, 1), (1, 5), (5, 1), (3, 0), (0, 3))
        A = spzeros(m, n)
        colptr2, rowval2 = PureSparse.ata_pattern(m, n, A.colptr, A.rowval)
        @test isempty(rowval2)
        @test length(colptr2) == n + 1
    end
    # a fully disconnected column (no row touches it) must get an empty pattern row/col
    A = spzeros(4, 4)
    A[1, 1] = 1.0
    A[2, 1] = 1.0
    A[3, 3] = 1.0
    colptr2, rowval2 = PureSparse.ata_pattern(4, 4, A.colptr, A.rowval)
    @test colptr2[2] == colptr2[3]  # column 2 (never touched) has zero entries
    @test colptr2[4] == colptr2[5]  # column 4 (never touched) has zero entries
end

@testitem "order_columns(AMDOrdering) returns valid permutations" setup = [ATAOracle] begin
    using Random
    rng = MersenneTwister(303)
    for _ in 1:300
        m = rand(rng, 0:15)
        n = rand(rng, 0:15)
        A = random_rect(rng, m, n, rand(rng, (0.05, 0.2, 0.4)))
        p = PureSparse.order_columns(PureSparse.AMDOrdering(), m, n, A.colptr, A.rowval)
        @test length(p) == n
        n > 0 && @test sort(p) == collect(1:n)
    end
end

@testitem "order_columns(GivenOrdering)/(NaturalOrdering) pass through / identity" begin
    using SparseArrays
    A = sprand(6, 4, 0.5)
    p = collect(4:-1:1)
    @test PureSparse.order_columns(PureSparse.GivenOrdering(p), 6, 4, A.colptr, A.rowval) == p
    @test PureSparse.order_columns(PureSparse.NaturalOrdering(), 6, 4, A.colptr, A.rowval) == collect(1:4)
    @test_throws DimensionMismatch PureSparse.order_columns(
        PureSparse.GivenOrdering([1, 2]), 6, 4, A.colptr, A.rowval,
    )
end

@testitem "AMD-on-AᵀA fill quality: sane vs natural order, tiny-graph greedy-mindeg gate" setup = [ATAOracle] begin
    # Quality smoke test, same spirit and same 2x-slack discipline as amd_tests.jl's
    # "AMD fill is close to exact minimum degree on tiny graphs" item (that test's
    # "exact" reference is itself a strong greedy heuristic, not a true brute-force
    # global minimum — mirrored here rather than re-derived).
    using Random
    rng = MersenneTwister(404)

    for _ in 1:80
        m = rand(rng, 2:10)
        n = rand(rng, 2:8)
        A = random_rect(rng, m, n, rand(rng, (0.1, 0.3, 0.5)))
        colptr2, rowval2 = PureSparse.ata_pattern(m, n, A.colptr, A.rowval)
        adj = adj_of_pattern(n, colptr2, rowval2)
        p = PureSparse.order_columns(PureSparse.AMDOrdering(), m, n, A.colptr, A.rowval)
        got = elimination_fill_adj(n, adj, p)
        fgreedy = greedy_mindeg_fill(n, adj)
        @test got <= 2 * fgreedy
    end

    # Larger random cases: AMD-on-AᵀA fill must not exceed the natural order's fill.
    nbetter = 0
    ntotal = 60
    for _ in 1:ntotal
        m = rand(rng, 10:40)
        n = rand(rng, 10:30)
        A = random_rect(rng, m, n, rand(rng, (0.05, 0.1, 0.2)))
        colptr2, rowval2 = PureSparse.ata_pattern(m, n, A.colptr, A.rowval)
        adj = adj_of_pattern(n, colptr2, rowval2)
        natural_fill = elimination_fill_adj(n, adj, collect(1:n))
        p = PureSparse.order_columns(PureSparse.AMDOrdering(), m, n, A.colptr, A.rowval)
        amd_fill = elimination_fill_adj(n, adj, p)
        @test amd_fill <= natural_fill
        amd_fill < natural_fill && (nbetter += 1)
    end
    @test nbetter > ntotal ÷ 2  # AMD should win outright on most random unstructured cases
end
