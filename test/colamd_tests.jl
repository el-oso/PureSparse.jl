@testsetup module COLAMDOracle
using Random, SparseArrays
export peel_singletons, hidden_band_with_dense_row

# Breadth-first column-singleton peeling (test-side reference implementation of the
# SPQR-paper §2.1 pre-elimination, design_qr.md §2.3 — the pipeline stage stdlib SPQR
# ALWAYS applies before its ordering; PureSparse gets it in M5a task 9, separate from
# COLAMD). Returns (peeled columns in peel order, surviving row mask, surviving column
# mask). Quadratic re-scan is fine at test sizes.
function peel_singletons(A::SparseMatrixCSC)
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

# Column-permuted banded pattern (the band is hidden, so natural order is BAD) plus one
# completely dense final row. The dense row makes the COLAMD upper bound on the first
# pivot row the full column set (paper §4: "A single dense row in A renders all our
# bounds useless"), so an implementation that fails to withhold it degenerates to
# near-arbitrary ordering; a withholding one recovers the band. Returns (A, Aband).
function hidden_band_with_dense_row(rng, m::Int, n::Int, halfbw::Int)
    I_ = Int[]
    J_ = Int[]
    for j in 1:n, _ in 1:4
        push!(I_, clamp(j + rand(rng, (-halfbw):halfbw, ), 1, m - 1))
        push!(J_, j)
    end
    Aband = sparse(I_, J_, ones(length(I_)), m - 1, n)[:, randperm(rng, n)]
    A = vcat(Aband, sparse(fill(1, n), 1:n, ones(n), 1, n))
    return A, Aband
end
end

@testitem "COLAMD returns valid permutations on random patterns and edge cases" setup = [ATAOracle] begin
    using Random, SparseArrays
    rng = MersenneTwister(51)
    for _ in 1:400
        m = rand(rng, 0:15)
        n = rand(rng, 0:15)
        A = random_rect(rng, m, n, rand(rng, (0.05, 0.2, 0.5, 0.9)))
        # default; tiny mults (forces dense withholding, floor permitting); huge mults
        # (disables withholding entirely)
        for alg in (COLAMDOrdering(),
                COLAMDOrdering(dense_row_mult = 0.01, dense_col_mult = 0.01),
                COLAMDOrdering(dense_row_mult = 1e9, dense_col_mult = 1e9))
            p = PureSparse.order_columns(alg, m, n, A.colptr, A.rowval)
            @test length(p) == n
            n > 0 && @test sort(p) == collect(1:n)
        end
    end
    alg = COLAMDOrdering()
    # n = 0, n = 1, m = 0 (every column null)
    @test PureSparse.order_columns(alg, 4, 0, [1], Int[]) == Int[]
    @test PureSparse.order_columns(alg, 3, 1, [1, 3], [1, 2]) == [1]
    @test PureSparse.order_columns(alg, 0, 5, fill(1, 6), Int[]) == collect(1:5)
    # fully dense small matrix (one giant super-column; mass elimination in one pivot)
    A = sparse(ones(7, 5))
    @test sort(PureSparse.order_columns(alg, 7, 5, A.colptr, A.rowval)) == collect(1:5)
    # null column and null row embedded in a sparse pattern
    A = spzeros(5, 5)
    A[1, 1] = 1; A[2, 1] = 1; A[2, 3] = 1; A[4, 4] = 1; A[4, 5] = 1  # col 2, row 3 null
    p = PureSparse.order_columns(alg, 5, 5, A.colptr, A.rowval)
    @test sort(p) == collect(1:5)
    @test p[end] == 2                     # the null column is ordered last
    # l_k = 0 branch (design_qr.md §2.2 pt 2, D9): NOT-strong-Hall inputs where a pivot
    # row represents zero non-pivotal candidate rows and must be discarded, never
    # referenced. Duplicate single-entry columns are the minimal trigger.
    A = sparse([1, 1], [1, 2], ones(2), 1, 2)
    @test sort(PureSparse.order_columns(alg, 1, 2, A.colptr, A.rowval)) == [1, 2]
    A = sparse([1, 1, 1, 2, 2], [1, 2, 3, 2, 3], ones(5), 2, 3)
    @test sort(PureSparse.order_columns(alg, 2, 3, A.colptr, A.rowval)) == [1, 2, 3]
    # duplicated column blocks: exercises super-columns + mass elimination + l_k = 0
    B = random_rect(rng, 20, 5, 0.4)
    A = hcat(B, B, B)
    @test sort(PureSparse.order_columns(alg, 20, 15, A.colptr, A.rowval)) == collect(1:15)
    # Ti = Int32 passes through
    A = random_rect(rng, 30, 25, 0.15)
    p32 = PureSparse.order_columns(alg, 30, 25, Int32.(A.colptr), Int32.(A.rowval))
    @test p32 isa Vector{Int32}
    @test sort(p32) == 1:25
end

@testitem "COLAMD fill is close to greedy minimum degree on tiny matrices" setup = [ATAOracle] begin
    # Same 2x-slack-vs-greedy-mindeg discipline as amd_tests.jl's tiny-graph gate:
    # column fill under the COLAMD permutation is elimination-game fill on
    # pattern(AᵀA) (the star-matrix identity, design_qr.md §3.1/H1), scored with the
    # same simulator as ata_tests.jl. Catches a badly-broken implementation, not
    # tie-breaking differences.
    using Random
    rng = MersenneTwister(52)
    for _ in 1:120
        m = rand(rng, 2:10)
        n = rand(rng, 2:8)
        A = random_rect(rng, m, n, rand(rng, (0.1, 0.3, 0.5)))
        colptr2, rowval2 = PureSparse.ata_pattern(m, n, A.colptr, A.rowval)
        adj = adj_of_pattern(n, colptr2, rowval2)
        p = PureSparse.order_columns(COLAMDOrdering(), m, n, A.colptr, A.rowval)
        @test elimination_fill_adj(n, adj, p) <= 2 * greedy_mindeg_fill(n, adj)
    end
    # medium sizes, where super-columns, aggressive absorption and garbage collection
    # are actually exercised
    for (m, n, d) in ((60, 40, 0.1), (80, 60, 0.08), (50, 50, 0.15))
        A = random_rect(rng, m, n, d)
        colptr2, rowval2 = PureSparse.ata_pattern(m, n, A.colptr, A.rowval)
        adj = adj_of_pattern(n, colptr2, rowval2)
        p = PureSparse.order_columns(COLAMDOrdering(), m, n, A.colptr, A.rowval)
        @test elimination_fill_adj(n, adj, p) <= 2 * greedy_mindeg_fill(n, adj)
    end
end

@testitem "COLAMD ordering quality: nnz(R) within 1.15x of stdlib SPQR" setup = [ATAOracle, COLAMDOracle] begin
    # design_qr.md §2.2 ordering-quality guardrail: nnz(R) under our COLAMD ≤ 1.15×
    # nnz(R) under stdlib SPQR's default ordering, black-box (§9.2: our permutation is
    # imposed via ordering=ORDERING_FIXED on the column-pre-permuted A; SPQR source
    # never read).
    using Random, SparseArrays, LinearAlgebra
    rng = MersenneTwister(53)
    nnzR_fixed(A, p) = nnz(qr(A[:, p]; ordering = SparseArrays.SPQR.ORDERING_FIXED).R)

    # Square and tall (m ≥ n): raw comparison.
    for (m, n, d) in ((60, 40, 0.1), (120, 80, 0.05), (200, 100, 0.03), (150, 150, 0.04),
            (100, 60, 0.15), (300, 200, 0.02), (400, 150, 0.03), (500, 300, 0.01))
        for _ in 1:2
            A = random_rect(rng, m, n, d)
            p = PureSparse.order_columns(COLAMDOrdering(), m, n, A.colptr, A.rowval)
            @test sort(p) == collect(1:n)
            @test nnzR_fixed(A, p) <= 1.15 * stdlib_spqr_nnzR(A)
        end
    end

    # Wide (m < n): stdlib SPQR ALWAYS peels column singletons before ordering (SPQR
    # paper §2.1), and sparse wide random matrices routinely peel away almost entirely
    # (measured during implementation: whole 87×152 instances peeled to nothing,
    # giving stdlib zero-fill nnz(R) = nnz(A) no matter which ordering arm is
    # selected). Singleton pre-elimination is a SEPARATE pipeline stage in this design
    # (design_qr.md §2.3, M5a task 9) — comparing raw COLAMD against
    # singletons+COLAMD conflates the two stages, so the reference peel is composed in
    # front here, mirroring what the real pipeline (and SPQR itself) does. With it,
    # measured quality is at parity (worst 1.002 over 40 wide instances).
    for (m, n, d) in ((80, 120, 0.05), (50, 90, 0.1), (100, 180, 0.06), (60, 140, 0.08),
            (90, 130, 0.07), (110, 160, 0.04))
        for _ in 1:2
            A = random_rect(rng, m, n, d)
            peel, rowlive, collive = peel_singletons(A)
            rest = [j for j in 1:n if collive[j]]
            B = A[findall(rowlive), rest]
            p2 = PureSparse.order_columns(COLAMDOrdering(), size(B, 1), size(B, 2),
                B.colptr, B.rowval)
            p = vcat(peel, rest[p2])
            @test sort(p) == collect(1:n)
            @test nnzR_fixed(A, p) <= 1.15 * stdlib_spqr_nnzR(A)
        end
    end
end

@testitem "COLAMD dense-row withholding and dense/null column placement" setup = [ATAOracle, COLAMDOracle] begin
    using Random, SparseArrays
    rng = MersenneTwister(54)

    # A dense row must not poison the ordering (design_qr.md §2.2 pt 5). Hidden-band
    # construction: the unwithheld dense row collapses every COLAMD bound to the full
    # column set, degenerating pivot selection to near-arbitrary — measured ~3x worse
    # fill on the banded part. The observable is elimination fill of each permutation
    # on pattern(AbandᵀAband) (the dense row itself densifies AᵀA for EVERY ordering,
    # so it is excluded from the metric; withholding fixes the ordering, not R's
    # density — §1.1 non-goals).
    m, n = 301, 400
    A, Aband = hidden_band_with_dense_row(rng, m, n, 6)
    @test n > max(PureSparse.COLAMD_DENSE_FLOOR, 10.0 * sqrt(n))  # the full row trips the default threshold
    colptr2, rowval2 = PureSparse.ata_pattern(m - 1, n, Aband.colptr, Aband.rowval)
    adj = adj_of_pattern(n, colptr2, rowval2)
    p_with = PureSparse.order_columns(COLAMDOrdering(), m, n, A.colptr, A.rowval)
    p_wo = PureSparse.order_columns(COLAMDOrdering(dense_row_mult = 1e9), m, n,
        A.colptr, A.rowval)
    @test sort(p_with) == collect(1:n)
    @test sort(p_wo) == collect(1:n)
    fill_with = elimination_fill_adj(n, adj, p_with)
    fill_wo = elimination_fill_adj(n, adj, p_wo)
    # measured 9038 vs 28731 on this seed; 0.6 is a wide badly-broken gate, not a tune
    @test fill_with <= 0.6 * fill_wo

    # Placement blocks ([T] §4.2.3 / docstring): [active | newly-null | dense | null],
    # ascending within each block. m = 40, n = 20, floor = 16:
    #   row 1 dense (18 > 16 entries, withheld); col 19 dense (20 > 16 entries);
    #   col 18 touched only by the dense row → newly null; col 20 empty → null.
    m, n = 40, 20
    I_ = Int[]
    J_ = Int[]
    for j in 1:18                       # dense row 1 covers columns 1..18
        push!(I_, 1)
        push!(J_, j)
    end
    for j in 1:17                       # live background for the active columns
        push!(I_, 1 + j)
        push!(J_, j)
        push!(I_, 2 + j)
        push!(J_, j)
    end
    for i in 2:21                       # dense column 19
        push!(I_, i)
        push!(J_, 19)
    end
    A = sparse(I_, J_, ones(length(I_)), m, n)
    alg = COLAMDOrdering(dense_row_mult = 0.5, dense_col_mult = 0.5)  # thresholds at the floor, 16
    p = PureSparse.order_columns(alg, m, n, A.colptr, A.rowval)
    @test sort(p) == collect(1:n)
    @test p[20] == 20                   # null column very last
    @test p[19] == 19                   # dense column just before it
    @test p[18] == 18                   # newly-null column just before the dense block
    @test sort(p[1:17]) == collect(1:17)
end
