@testsetup module QRFrontalSymbolicHelpers
using Random, SparseArrays
export random_rect_frontal
random_rect_frontal(rng, m::Int, n::Int, density::Float64) = sprand(rng, m, n, density)
end

@testitem "symbolic_qr_frontal: design_qr_m5b.md §A3.3 worked example (hand-verified numbers)" begin
    using SparseArrays
    # A is 7x5, already in final permuted column order; row patterns:
    # r1:{1,2,4} r2:{1,2,5} r3:{2,4} r4:{3,4} r5:{3,5} r6:{4,5} r7:{4}
    I = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7]
    J = [1, 2, 4, 1, 2, 5, 2, 4, 3, 4, 3, 5, 4, 5, 4]
    A = sparse(I, J, ones(length(I)), 7, 5)

    sym = PureSparse.symbolic_qr(A; ordering = PureSparse.NaturalOrdering())
    @test sym.parent == [2, 4, 4, 5, 0]
    @test sym.rcount == [4, 3, 3, 2, 1]

    # amalg_cols=(0,0,0) disables merging so the raw (non-fundamental) 2-front
    # partition survives intact — the design's worked example is illustrating the
    # supernode/assembly-simulation step in isolation, before any amalgamation.
    fsym = PureSparse.symbolic_qr_frontal(sym, A; fundamental = false, amalg_cols = (0, 0, 0))
    @test fsym.nfront == 2
    @test fsym.fsuper == [1, 3, 6]              # front 1: cols 1-2, front 2: cols 3-5
    @test fsym.fparent == [2, 0]                # front tree 1 -> 2 (root)
    @test fsym.fcolind[fsym.fcolptr[1]:(fsym.fcolptr[2] - 1)] == [1, 2, 4, 5]  # front 1 cols
    @test fsym.fcolind[fsym.fcolptr[2]:(fsym.fcolptr[3] - 1)] == [3, 4, 5]     # front 2 cols
    @test fsym.fmmax == [3, 6]
    @test fsym.fcrmax == [2, 0]
    @test fsym.nnzVF == 30                      # 3*4 + 6*3
    @test fsym.nnzRF == 13                      # front1: 4+3, front2: 3+2+1
    @test fsym.max_front_rows == 6
    @test fsym.max_front_cols == 4

    # the FUNDAMENTAL (3-condition) partition splits into 3 fronts on this same matrix
    # (§A2.2's own worked contrast: column 4 has two etree children)
    fsym3 = PureSparse.symbolic_qr_frontal(sym, A; fundamental = true, amalg_cols = (0, 0, 0))
    @test fsym3.nfront == 3
end

@testitem "symbolic_qr_frontal: rejects sym.n1 > 0 and mismatched A" begin
    using SparseArrays, LinearAlgebra
    A = sparse([1.0 0.5 0.7; 0.0 0.9 0.3; 0.0 0.2 0.5])
    sym1 = PureSparse.symbolic_qr(A; ordering = PureSparse.AMDOrdering())  # n1==0 here
    Awrong = sparse(rand(4, 4) .+ I(4))
    @test_throws DimensionMismatch PureSparse.symbolic_qr_frontal(sym1, Awrong)

    # symbolic_qr_frontal only accepts a block-level (n1==0) QRSymbolic — it never sees
    # the peeled singleton block directly, so simulate an n1>0 sym via qr()'s own
    # composition (design_qr.md §2.3's staircase matrix, fully-singleton by construction).
    Asing = sparse(1:4, 1:4, [2.0, 3.0, 4.0, 5.0], 4, 4)
    Fsing = PureSparse.qr(Asing; ordering = PureSparse.AMDOrdering(), tol = 0)
    @test Fsing.sym.n1 > 0
    @test_throws ArgumentError PureSparse.symbolic_qr_frontal(Fsing.sym, Asing)
end

@testitem "symbolic_qr_frontal: structural invariants on random matrices" setup = [QRFrontalSymbolicHelpers] begin
    using Random, SparseArrays
    rng = MersenneTwister(11)
    for _ in 1:200
        m = rand(rng, 2:30)
        n = rand(rng, 1:min(m, 25))
        A = random_rect_frontal(rng, m, n, rand(rng, (0.1, 0.2, 0.4)))
        sym = PureSparse.symbolic_qr(A; ordering = PureSparse.AMDOrdering())
        nb = length(sym.parent)
        nb == 0 && continue
        fsym = PureSparse.symbolic_qr_frontal(sym, A)

        # every block column belongs to exactly one front, fronts partition 1:nb
        @test fsym.fsuper[1] == 1
        @test fsym.fsuper[fsym.nfront + 1] == nb + 1
        @test issorted(fsym.fsuper)

        # fcolind: front f's own pivotal columns (its own super range) are its first
        # p_f entries, sorted; the whole front column list is sorted ascending.
        for f in 1:fsym.nfront
            p_f = fsym.fsuper[f + 1] - fsym.fsuper[f]
            cols = fsym.fcolind[fsym.fcolptr[f]:(fsym.fcolptr[f + 1] - 1)]
            @test issorted(cols)
            @test cols[1:p_f] == collect(fsym.fsuper[f]:(fsym.fsuper[f + 1] - 1))
        end

        # arowptr sums to mb (every physical row assigned to exactly one leftcol bucket
        # or is a null row, and arowptr only counts the assigned ones)
        @test fsym.arowptr[1] == 1
        @test issorted(fsym.arowptr)
        @test fsym.arowptr[nb + 1] - 1 <= sym.mb

        # capacities: fcrmax_f <= c_f (trapezoid clamp) and <= fmmax_f
        for f in 1:fsym.nfront
            n_f = fsym.fcolptr[f + 1] - fsym.fcolptr[f]
            p_f = fsym.fsuper[f + 1] - fsym.fsuper[f]
            c_f = n_f - p_f
            @test fsym.fcrmax[f] <= c_f
            @test fsym.fcrmax[f] <= fsym.fmmax[f]
            @test fsym.fmmax[f] >= 0
        end

        # a root front (fparent==0) always has crmax==0's irrelevant (nothing to pass
        # up), but its OWN mmax must be >= its A-row count (no children double-counted
        # incorrectly): spot-check the sum identity mmax_f == a_f + sum(children crmax)
        for f in 1:fsym.nfront
            a_f = fsym.arowptr[fsym.fsuper[f + 1]] - fsym.arowptr[fsym.fsuper[f]]
            childsum = 0
            for cp in fsym.fchildptr[f]:(fsym.fchildptr[f + 1] - 1)
                childsum += fsym.fcrmax[fsym.fchildren[cp]]
            end
            @test fsym.fmmax[f] == a_f + childsum
        end

        # row-form: rowptr/rowcol cover exactly nnz(A) entries, each row's columns sorted
        @test fsym.rowptr[m + 1] - 1 == nnz(A)
        for r in 1:m
            cols = fsym.rowcol[fsym.rowptr[r]:(fsym.rowptr[r + 1] - 1)]
            @test issorted(cols)
        end

        # atrans is a valid permutation of 1:nnz(A) into row-form slot space
        @test sort(fsym.atrans) == collect(1:nnz(A))

        # padded R storage is at least as large as the true R storage
        @test fsym.nnzRF >= sym.nnzR
    end
end

@testitem "symbolic_qr_frontal: fundamental keyword produces a valid (possibly different) partition" setup = [QRFrontalSymbolicHelpers] begin
    using Random, SparseArrays
    rng = MersenneTwister(12)
    n_more_fronts = 0
    ntested = 0
    for _ in 1:100
        m = rand(rng, 3:25)
        n = rand(rng, 1:min(m, 20))
        A = random_rect_frontal(rng, m, n, rand(rng, (0.15, 0.3)))
        sym = PureSparse.symbolic_qr(A; ordering = PureSparse.AMDOrdering())
        nb = length(sym.parent)
        nb == 0 && continue
        ntested += 1
        f_default = PureSparse.symbolic_qr_frontal(sym, A; fundamental = false, amalg_cols = (0, 0, 0))
        f_fund = PureSparse.symbolic_qr_frontal(sym, A; fundamental = true, amalg_cols = (0, 0, 0))
        # fundamental=true is a REFINEMENT of fundamental=false's partition (superset of
        # boundaries) — the fundamental partition can only split fronts further, never merge.
        @test f_fund.nfront >= f_default.nfront
        f_fund.nfront > f_default.nfront && (n_more_fronts += 1)
    end
    @test ntested > 30
end
