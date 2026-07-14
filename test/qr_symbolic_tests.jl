@testsetup module QRSymbolicOracle
using Random, SparseArrays, PureSparse
export random_rect_qr, ata_etree_colcount, star_etree_colcount

random_rect_qr(rng, m::Int, n::Int, density::Float64) = sprand(rng, m, n, density)

# Reference path: etree/colcount computed from pattern(AᵀA) (brute-force, via
# ata_pattern — already independently verified against SparseArrays' A'*A in
# ata_tests.jl) under a given column permutation.
function ata_etree_colcount(A::SparseMatrixCSC, perm::Vector{Int})
    m, n = size(A)
    iperm = Vector{Int}(undef, n)
    for (k, p) in enumerate(perm)
        iperm[p] = k
    end
    acolptr, arowval = PureSparse.ata_pattern(m, n, A.colptr, A.rowval)
    ucolptr, urowval = PureSparse.symmetrized_upper(n, acolptr, arowval, perm, iperm)
    parent = PureSparse.etree(n, ucolptr, urowval)
    cc = PureSparse.column_counts(n, ucolptr, urowval, parent)
    return parent, cc
end

# Star-matrix path (design_qr.md §3.2, H1): the pattern under test.
function star_etree_colcount(A::SparseMatrixCSC, perm::Vector{Int})
    m, n = size(A)
    iperm = Vector{Int}(undef, n)
    for (k, p) in enumerate(perm)
        iperm[p] = k
    end
    scolptr, srowval = PureSparse.star_pattern(m, n, A.colptr, A.rowval, iperm)
    idperm = collect(1:n)
    ucolptr, urowval = PureSparse.symmetrized_upper(n, scolptr, srowval, idperm, idperm)
    parent = PureSparse.etree(n, ucolptr, urowval)
    cc = PureSparse.column_counts(n, ucolptr, urowval, parent)
    return parent, cc, scolptr, srowval
end
end

@testitem "H1: star matrix etree/colcount match pattern(AᵀA)'s, under random permutations" setup = [QRSymbolicOracle] begin
    using Random
    rng = MersenneTwister(55)
    for _ in 1:500
        m = rand(rng, 1:15)
        n = rand(rng, 1:15)
        A = random_rect_qr(rng, m, n, rand(rng, (0.05, 0.15, 0.3, 0.6)))
        perm = randperm(rng, n)
        parent_a, cc_a = ata_etree_colcount(A, perm)
        parent_s, cc_s, _, _ = star_etree_colcount(A, perm)
        @test parent_s == parent_a
        @test cc_s == cc_a
    end
end

@testitem "H1: identity permutation and structured cases (banded, disconnected, dense row)" setup = [QRSymbolicOracle] begin
    using SparseArrays
    # Banded rectangular
    m, n = 12, 8
    I_ = Int[]; J_ = Int[]; V_ = Float64[]
    for j in 1:n, i in max(1, j - 1):min(m, j + 2)
        push!(I_, i); push!(J_, j); push!(V_, 1.0)
    end
    A = sparse(I_, J_, V_, m, n)
    perm = collect(1:n)
    parent_a, cc_a = ata_etree_colcount(A, perm)
    parent_s, cc_s, scolptr, srowval = star_etree_colcount(A, perm)
    @test parent_s == parent_a
    @test cc_s == cc_a
    @test length(srowval) <= nnz(A)  # |S| <= |A|, design_qr.md §3.2

    # A column touched by no row's leftmost entry (disconnected-ish structure): still
    # must agree with the AᵀA reference.
    A2 = spzeros(5, 4)
    A2[1, 1] = 1.0
    A2[1, 2] = 1.0
    A2[2, 3] = 1.0
    A2[2, 4] = 1.0
    A2[3, 1] = 1.0
    p2 = collect(1:4)
    pa, ca = ata_etree_colcount(A2, p2)
    ps, cs, _, _ = star_etree_colcount(A2, p2)
    @test ps == pa
    @test cs == ca

    # One dense row spanning every column (worst-case star fan-out)
    m3, n3 = 6, 10
    A3 = spzeros(m3, n3)
    A3[1, :] .= 1.0                    # dense row 1
    for j in 1:n3
        A3[mod1(j, m3 - 1) + 1, j] = 1.0
    end
    A3 = sparse(A3)
    p3 = collect(1:n3)
    pa3, ca3 = ata_etree_colcount(A3, p3)
    ps3, cs3, scolptr3, srowval3 = star_etree_colcount(A3, p3)
    @test ps3 == pa3
    @test cs3 == ca3
    @test length(srowval3) <= nnz(A3)
end

@testitem "star_pattern edge cases and structural sanity" begin
    using SparseArrays
    for (m, n) in ((0, 0), (1, 1), (1, 5), (5, 1))
        A = spzeros(m, n)
        p = collect(1:n)
        colptr2, rowval2 = PureSparse.star_pattern(m, n, A.colptr, A.rowval, p)
        @test length(colptr2) == n + 1
        @test isempty(rowval2)
    end
    # every column's stored entries must be sorted, no self-loops (diagonal), no dup
    using Random
    rng = MersenneTwister(909)
    for _ in 1:200
        m = rand(rng, 1:14)
        n = rand(rng, 1:14)
        A = sprand(rng, m, n, rand(rng, (0.1, 0.3, 0.5)))
        p = randperm(rng, n)
        iperm = Vector{Int}(undef, n)
        for (k, v) in enumerate(p)
            iperm[v] = k
        end
        colptr2, rowval2 = PureSparse.star_pattern(m, n, A.colptr, A.rowval, iperm)
        @test length(rowval2) <= nnz(A)
        for k in 1:n
            seg = view(rowval2, colptr2[k]:(colptr2[k + 1] - 1))
            @test issorted(seg)
            @test allunique(seg)
            @test !(k in seg)   # no self-loop: a row's leftmost column never lists itself
        end
    end
end
