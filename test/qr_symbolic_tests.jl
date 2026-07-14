@testsetup module QRSymbolicOracle
using Random, SparseArrays, PureSparse
export random_rect_qr, ata_etree_colcount, star_etree_colcount, full_qr_symbolic

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

# Full symbolic assembly (design_qr.md §3, M5a tasks 4+5): ordering -> star pattern ->
# postorder -> R structure (rcount) -> V structure (rperm/riperm/mb/vptr/vrowind/
# pivotslot/vcount). Mirrors driver.jl's `symbolic()` composition but WITHOUT the
# merge-aware two-pass postorder (design_qr.md §3.2: "No amalgamation priority needed
# in M5a — no supernodes"), a single default postorder pass suffices.
function full_qr_symbolic(A::SparseMatrixCSC, ordering)
    m, n = size(A)
    fcperm = PureSparse.order_columns(ordering, m, n, A.colptr, A.rowval)
    fciperm = Vector{Int}(undef, n)
    for (k, p) in enumerate(fcperm)
        fciperm[p] = k
    end
    scolptr, srowval = PureSparse.star_pattern(m, n, A.colptr, A.rowval, fciperm)
    idp = collect(1:n)
    ucolptr, urowval = PureSparse.symmetrized_upper(n, scolptr, srowval, idp, idp)
    parent0 = PureSparse.etree(n, ucolptr, urowval)
    post, postinv = PureSparse.postorder(n, parent0)
    cp2, rv2 = PureSparse.relabel_pattern(n, ucolptr, urowval, postinv)
    parent = PureSparse.etree(n, cp2, rv2)
    rcount = PureSparse.column_counts(n, cp2, rv2, parent)
    cperm = Vector{Int}(undef, n)
    for orig in 1:n
        cperm[postinv[fciperm[orig]]] = orig
    end
    ciperm = Vector{Int}(undef, n)
    for (k, p) in enumerate(cperm)
        ciperm[p] = k
    end
    _, _, leftcol = PureSparse.row_leftcol(m, n, A.colptr, A.rowval, ciperm)
    rperm, riperm, mb, vptr, vrowind, pivotslot, vcount = PureSparse.qr_row_structure(m, n, parent, leftcol)
    return (; cperm, ciperm, parent, rcount, rperm, riperm, mb, vptr, vrowind, pivotslot, vcount, leftcol)
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

@testitem "qr_row_structure matches the design doc's three hand-worked examples" begin
    # A = [1 1 1] (m=1,n=3): star etree is the chain 1->2->3, leftcol(row1)=1.
    rperm, riperm, mb, vptr, vrowind, pivotslot, vcount =
        PureSparse.qr_row_structure(1, 3, [2, 3, 0], [1])
    @test vcount == [1, 0, 0]
    @test pivotslot == [1, 0, 0]
    @test mb == 1

    # A = [0 1] (m=1,n=2): leftcol(row1)=2, both columns isolated (parent=[0,0]).
    rperm2, riperm2, mb2, vptr2, vrowind2, pivotslot2, vcount2 =
        PureSparse.qr_row_structure(1, 2, [0, 0], [2])
    @test vcount2 == [0, 1]
    @test pivotslot2 == [0, 1]
    @test mb2 == 1

    # A = [1 1; 0 0] (m=2,n=2): leftcol=[1,0] (row 2 null), parent=[2,0].
    rperm3, riperm3, mb3, vptr3, vrowind3, pivotslot3, vcount3 =
        PureSparse.qr_row_structure(2, 2, [2, 0], [1, 0])
    @test vcount3 == [1, 0]
    @test pivotslot3 == [1, 0]
    @test mb3 == 1
end

@testitem "H2: row-path consistency property, full symbolic pipeline" setup = [QRSymbolicOracle] begin
    # design_qr.md §3.4's consistency property: for every physical row p, the set of
    # columns {k : p ∈ S_k} is a contiguous ascending path in the (postordered) column
    # etree starting at p's leftcol, ending EITHER where p retires as pivot (at an
    # internal node or a root) OR at a LIVE root without retiring (legitimate for the
    # m > n case — clarified in design_qr.md during this task, see the edited §3.4
    # bullet). The one thing that must never happen: terminating at a DEAD root.
    using Random, SparseArrays
    rng = MersenneTwister(77)
    for _ in 1:3000
        m = rand(rng, 1:16)
        n = rand(rng, 1:16)
        A = random_rect_qr(rng, m, n, rand(rng, (0.03, 0.08, 0.15, 0.3, 0.6)))
        r = full_qr_symbolic(A, PureSparse.AMDOrdering())

        # self-consistency: segment length == vcount[k]; pivot is first entry & the min
        for k in 1:n
            seg = view(r.vrowind, r.vptr[k]:(r.vptr[k + 1] - 1))
            @test length(seg) == r.vcount[k]
            if !isempty(seg)
                @test seg[1] == r.pivotslot[k]
                @test minimum(seg) == r.pivotslot[k]
                @test allunique(seg)
            else
                @test r.pivotslot[k] == 0
            end
        end

        membership = [Int[] for _ in 1:r.mb]
        for k in 1:n, idx in r.vptr[k]:(r.vptr[k + 1] - 1)
            push!(membership[r.vrowind[idx]], k)
        end
        for p in 1:r.mb
            ks = sort(membership[p])
            @test !isempty(ks)
            orig_r = r.rperm[p]
            lc = r.leftcol[orig_r]
            @test lc != 0
            @test ks[1] == lc
            for t in 1:(length(ks) - 1)
                @test r.parent[ks[t]] == ks[t + 1]
            end
            last_k = ks[end]
            is_pivot_here = r.pivotslot[last_k] == p
            is_root = r.parent[last_k] == 0
            @test is_pivot_here || is_root
            if is_root && !is_pivot_here
                @test r.vcount[last_k] > 0   # never terminates at a DEAD root
            end
        end
    end
end

@testitem "V-structure edge cases: fully null matrix, single dense row, all-singleton columns" setup = [QRSymbolicOracle] begin
    using SparseArrays
    # Fully null m x n matrix: every column dead, mb == 0.
    A = spzeros(5, 4)
    r = full_qr_symbolic(A, PureSparse.AMDOrdering())
    @test r.mb == 0
    @test all(==(0), r.vcount)
    @test all(==(0), r.pivotslot)

    # One dense row touching every column: extreme fan-out for the V-structure.
    A2 = spzeros(3, 6)
    A2[1, :] .= 1.0
    A2[2, 3] = 1.0
    A2[3, 5] = 1.0
    A2 = sparse(A2)
    r2 = full_qr_symbolic(A2, PureSparse.AMDOrdering())
    @test r2.mb == 3  # 3 nonzero rows, none null
    @test sum(r2.vcount) == sum(length(r2.vrowind[r2.vptr[k]:(r2.vptr[k + 1] - 1)]) for k in 1:6)

    # Square identity-like pattern (every column its own singleton row): no V fill at all.
    A3 = sparse(1:5, 1:5, ones(5), 5, 5)
    r3 = full_qr_symbolic(A3, PureSparse.AMDOrdering())
    @test r3.mb == 5
    @test all(==(1), r3.vcount)  # each column's own single assigned row, no inheritance
end
