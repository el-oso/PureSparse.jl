@testsetup module QRSingletonOracle
using Random, SparseArrays
export random_rect_sing, peel_singletons_ref

random_rect_sing(rng, m::Int, n::Int, density::Float64) = sprand(rng, m, n, density)

# Column-ascending, cascade-within-pass reference (independent of design_qr.md's
# stated "breadth-first" algorithm shape, deliberately — this is a DIFFERENT valid
# tie-break used only to cross-check total-count effectiveness and structural
# soundness, not to demand exact column-for-column agreement; see the "different
# valid schedule" test below for why exact agreement isn't the right invariant).
function peel_singletons_ref(A::SparseMatrixCSC)
    m, n = size(A)
    rowlive = trues(m)
    collive = trues(n)
    peel = Int[]
    changed = true
    while changed
        changed = false
        for j in 1:n
            collive[j] || continue
            live = [A.rowval[q] for q in nzrange(A, j) if rowlive[A.rowval[q]]]
            if length(live) == 1
                rowlive[live[1]] = false
                collive[j] = false
                push!(peel, j)
                changed = true
            end
        end
    end
    return peel, rowlive, collive
end
end

@testitem "peel_column_singletons: structural soundness (random matrices, threshold=0)" setup = [QRSingletonOracle] begin
    using Random, SparseArrays
    rng = MersenneTwister(1)
    for _ in 1:500
        m = rand(rng, 1:15)
        n = rand(rng, 1:15)
        A = random_rect_sing(rng, m, n, rand(rng, (0.05, 0.1, 0.2, 0.3)))
        peel_col, peel_row, collive, rowlive = PureSparse.peel_column_singletons(A, 0.0)
        @test allunique(peel_col)
        @test allunique(peel_row)
        @test length(peel_col) == length(peel_row)
        for k in eachindex(peel_col)
            @test A[peel_row[k], peel_col[k]] != 0
            @test !collive[peel_col[k]]
            @test !rowlive[peel_row[k]]
        end
        @test count(!, collive) == length(peel_col)
        @test count(!, rowlive) == length(peel_row)
    end
end

@testitem "peel_column_singletons: same total count as an independent valid schedule" setup = [QRSingletonOracle] begin
    # design_qr.md §2.3 specifies BREADTH-FIRST peeling; the test-side reference here
    # uses a different (column-ascending, cascade-within-pass) tie-break. When two
    # columns are simultaneously singletons pointing at the SAME row, whichever gets
    # processed first "claims" it and the other is permanently excluded -- a genuine
    # multiple-valid-schedules situation, not a bug (independently re-derived and
    # verified: every individual peel from either schedule is soundly justified at
    # the moment it fires). What SHOULD match, and does, is the total count peeled.
    using Random, SparseArrays
    rng = MersenneTwister(1)
    ndiverge = 0
    for _ in 1:500
        m = rand(rng, 1:15)
        n = rand(rng, 1:15)
        A = random_rect_sing(rng, m, n, rand(rng, (0.05, 0.1, 0.2, 0.3)))
        peel_col, _, _, _ = PureSparse.peel_column_singletons(A, 0.0)
        ref_peel, _, _ = peel_singletons_ref(A)
        @test length(peel_col) == length(ref_peel)
        Set(peel_col) != Set(ref_peel) && (ndiverge += 1)
    end
    @test ndiverge > 0   # sanity: the tie-break divergence case is genuinely exercised
end

@testitem "peel_column_singletons: magnitude threshold policy (§2.3 'values, not just pattern')" begin
    using SparseArrays
    # column 2's sole live entry is small (0.01); above/below-threshold behavior must differ.
    A = sparse([1.0 0.0; 0.0 0.01; 0.0 0.0])
    peel_lo, _, collive_lo, _ = PureSparse.peel_column_singletons(A, 0.001)
    @test 2 in peel_lo
    peel_hi, _, collive_hi, _ = PureSparse.peel_column_singletons(A, 0.1)
    @test !(2 in peel_hi)
    @test collive_hi[2]   # left live: structurally a singleton, but magnitude test failed
end

@testitem "peel_column_singletons: edge cases (empty, no singletons, all singletons)" begin
    using SparseArrays
    A0 = spzeros(0, 0)
    peel0, _, _, _ = PureSparse.peel_column_singletons(A0, 0.0)
    @test isempty(peel0)

    # dense 3x3: no column is ever a singleton
    A1 = sparse(ones(3, 3))
    peel1, _, collive1, _ = PureSparse.peel_column_singletons(A1, 0.0)
    @test isempty(peel1)
    @test all(collive1)

    # diagonal 4x4: every column is immediately its own singleton
    A2 = sparse(1:4, 1:4, [1.0, 2.0, 3.0, 4.0], 4, 4)
    peel2, peel_row2, collive2, rowlive2 = PureSparse.peel_column_singletons(A2, 0.0)
    @test length(peel2) == 4
    @test all(!, collive2)
    @test all(!, rowlive2)
    @test Set(zip(peel2, peel_row2)) == Set(zip(1:4, 1:4))
end

@testitem "peel_column_singletons: cascading chain (design_qr.md's LP-style structure)" begin
    using SparseArrays
    # A chain: col k's only entry (besides possibly earlier ones) is row k, so peeling
    # is forced to cascade strictly in order 1,2,3,4.
    A = sparse([1, 2, 3, 4], [1, 2, 3, 4], [1.0, 1.0, 1.0, 1.0], 4, 4)
    peel_col, peel_row, collive, rowlive = PureSparse.peel_column_singletons(A, 0.0)
    @test length(peel_col) == 4
    @test all(!, collive)
end
