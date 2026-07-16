@testsetup module QRFrontalNumericHelpers
using Random, SparseArrays, LinearAlgebra
export random_rect_frontal2, dense_R_frontal, normal_eq_resid

random_rect_frontal2(rng, m::Int, n::Int, density::Float64) = sprand(rng, m, n, density)

# Dense n×n R (abstract row-of-R layout, indexed by GLOBAL pivotal column) built from
# a QRFrontFactor's row-wise rval/frptr/fcolind storage — the frontal-path analogue of
# qr_numeric_tests.jl's own `dense_R` for QRFactor.
function dense_R_frontal(F)
    fsym = F.fsym
    nb = length(fsym.base.parent)
    R = zeros(eltype(F.rval), nb, nb)
    for k in 1:nb
        F.fpivotrow[k] == 0 && continue
        f = fsym.fsnode[k]
        colslo = Int(fsym.fcolptr[f])
        n_f = Int(fsym.fcolptr[f + 1] - fsym.fcolptr[f])
        pos = k - Int(fsym.fsuper[f]) + 1
        p = Int(fsym.frptr[k])
        for jc in pos:n_f
            gcol = fsym.fcolind[colslo + jc - 1]
            R[k, gcol] = F.rval[p]
            p += 1
        end
    end
    return R
end

normal_eq_resid(A, x, b) = norm(A' * (b - A * x))
end

@testitem "qr_frontal numeric: design_qr_m5b.md §A3.3 worked example (exact RᵀR/solve)" setup=[QRFrontalNumericHelpers] begin
    using SparseArrays, LinearAlgebra, Random

    I = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7]
    J = [1, 2, 4, 1, 2, 5, 2, 4, 3, 4, 3, 5, 4, 5, 4]
    rng = MersenneTwister(1)
    A = sparse(I, J, randn(rng, length(I)) .+ 3.0, 7, 5)

    F = PureSparse.qr_frontal(A; ordering = PureSparse.NaturalOrdering())
    @test F.stats.rank == 5
    @test F.stats.n_dead == 0

    R = dense_R_frontal(F)
    Ad = Matrix(A)[:, F.fsym.base.cperm]
    @test maximum(abs.(R' * R .- Ad' * Ad)) < 1e-10

    b = Float64.(1:7)
    x = PureSparse.solve!(zeros(5), F, b)
    @test normal_eq_resid(A, x, b) < 1e-10
end

@testitem "qr_frontal numeric: random sweep vs M5a :column oracle (RᵀR, normal eqns, x agreement at full rank)" setup=[QRFrontalNumericHelpers] begin
    using SparseArrays, LinearAlgebra, Random

    for seed in 1:60
        rng = MersenneTwister(seed)
        m = rand(rng, 5:40)
        n = rand(rng, 3:min(m, 25))
        dens = rand(rng, 0.1:0.05:0.5)
        A = random_rect_frontal2(rng, m, n, dens)

        F = PureSparse.qr_frontal(A; ordering = PureSparse.NaturalOrdering())
        R = dense_R_frontal(F)
        nb = length(F.fsym.base.parent)
        Ad = Matrix(A)[:, F.fsym.base.cperm]
        @test maximum(abs.(R' * R .- Ad' * Ad)) < 1e-8 * max(1, norm(Ad)^2)

        b = randn(MersenneTwister(seed + 999), m)
        x = PureSparse.solve!(zeros(n), F, b)
        @test normal_eq_resid(A, x, b) < 1e-6 * max(1, norm(b))

        # x-agreement with M5a's own (already-gated) column path is only a valid
        # cross-check at full rank: rank-deficient basic solutions aren't unique
        # (the two paths can break dead-column ties differently), so that case is
        # covered by the normal-equations check above instead (§A9.2's own oracle
        # requirement is agreement on the WELL-POSED case).
        if F.stats.n_dead == 0
            Fcol = PureSparse.qr(A; ordering = PureSparse.NaturalOrdering(), singletons = false)
            xc = PureSparse.solve!(zeros(n), Fcol, b)
            @test norm(x - xc) < 1e-6 * max(1, norm(xc))
        end
    end
end

@testitem "qr_frontal numeric: rank-deficient (duplicate column) still hits the normal-equations oracle" setup=[QRFrontalNumericHelpers] begin
    using SparseArrays, LinearAlgebra, Random

    rng = MersenneTwister(777)
    A = random_rect_frontal2(rng, 20, 10, 0.3)
    A[:, 5] = A[:, 3]

    F = PureSparse.qr_frontal(A; ordering = PureSparse.NaturalOrdering(), tol = 1e-10)
    @test F.stats.n_dead >= 1
    @test F.stats.rank == 9

    b = randn(MersenneTwister(777 + 999), 20)
    x = PureSparse.solve!(zeros(10), F, b)
    @test normal_eq_resid(A, x, b) < 1e-8
end

@testitem "qr_frontal numeric: qr!/solve!/apply_Q!/apply_Qt! are zero-allocation warm (design_qr_m5b.md §A9.6, task 16e)" setup=[QRFrontalNumericHelpers] begin
    using SparseArrays, LinearAlgebra, Random

    for (label, A, tol) in (
        ("full rank", random_rect_frontal2(MersenneTwister(11), 40, 25, 0.2), nothing),
        ("rank-deficient", let
            Ar = random_rect_frontal2(MersenneTwister(777), 20, 10, 0.3)
            Ar[:, 5] = Ar[:, 3]
            Ar
        end, 1e-10),
    )
        sym = PureSparse.symbolic_qr(A; ordering = PureSparse.NaturalOrdering())
        fsym = PureSparse.symbolic_qr_frontal(sym, A; fundamental = false)
        F = PureSparse.QRFrontFactor{Float64,Int}(fsym)
        PureSparse.qr!(F, A; tol)   # warm up
        @test (@allocated PureSparse.qr!(F, A; tol)) == 0

        m, n = size(A)
        b = randn(MersenneTwister(1), m)
        x = zeros(n)
        PureSparse.solve!(x, F, b)   # warm up
        @test (@allocated PureSparse.solve!(x, F, b)) == 0

        y = randn(MersenneTwister(2), fsym.base.mb)
        PureSparse.apply_Qt!(y, F)
        @test (@allocated PureSparse.apply_Qt!(y, F)) == 0
        PureSparse.apply_Q!(y, F)
        @test (@allocated PureSparse.apply_Q!(y, F)) == 0
    end
end

@testitem "qr_frontal numeric: large-front block-size consistency — ftau slab vs numeric panel cap (regression: NB(0,0)≠NB(front) OOB)" setup=[QRFrontalNumericHelpers] begin
    using SparseArrays, LinearAlgebra, Random
    # qr_block_size comes from PureBLAS but is a TRANSITIVE dep here (not in test/'s own
    # Project.toml) — reach it through PureSparse's own binding, not a bare `using PureBLAS`.
    qbs = PureSparse.qr_block_size

    # Regression for the `qr_block_size(0,0)` vs `qr_block_size(max_front_rows,
    # max_front_cols)` mismatch: the symbolic ftau T-slab was budgeted with the former,
    # the numeric loop packed pcount×pcount T's capped by the latter. `qr_block_size` is
    # dimension-dependent (8 for small/zero dims, 16 once a front is large enough), so on
    # a matrix with a large-enough front the packing overwrote `ftau` by up to one panel's
    # worth — a genuine OOB write, silent-corruption on Zen3, hard SIGSEGV on Zen5. Every
    # PRE-EXISTING frontal test used small fronts (nb==8 both places, no mismatch), so
    # none caught it; this one deliberately builds a front large enough to make nb==16.
    #
    # A fully-dense 400×384 matrix has a single 400×384 front — big enough to step
    # qr_block_size up to 16 (measured: the 8→16 boundary sits between a 256×256 and a
    # 384×384 front). Density 1.0 is fine here: the point is the front SIZE, not sparsity.
    rng = MersenneTwister(31337)
    A = sparse(randn(rng, 400, 384))
    ordering = PureSparse.COLAMDOrdering()
    F = PureSparse.qr_frontal(A; ordering)

    # (1) The test must actually exercise the nb>baseline regime — otherwise it's vacuous
    # (a future shrink of this matrix below the 8→16 boundary would silently defang it).
    @test F.fsym.nb > qbs(0, 0)
    # (2) THE single-source-of-truth invariant, checked directly: the numeric loop's panel
    # width cap (`size(ws.Tm,1)`) must equal the nb the ftau slab was budgeted with
    # (`fsym.nb`). This fails deterministically on ANY hardware if the two NB uses ever
    # diverge again — independent of whether the resulting OOB happens to crash. THIS is
    # the OOB-regression this test primarily guards, and it PASSES with the fix in place.
    @test size(F.ws.Tm, 1) == F.fsym.nb
    @test F.fsym.nb == qbs(F.fsym.max_front_rows, F.fsym.max_front_cols)

    # (3)/(4) End-to-end correctness of the BLOCKED path (front element count 153600 >
    # QR_FRONTAL_UNBLOCKED_THRESHOLD, so this takes the wy_t!/wy_apply! blocked kernel,
    # NOT the scalar fallback every other frontal test hits). Regression for the
    # dropped-group bug: the numeric loop emitted only ONE NB-clamped group per split
    # trigger, so any front whose pending width exceeded NB at a trigger (every
    # dense-ish front — all min-cols coincide and only the sentinel triggers) silently
    # lost every column past NB (rank NB instead of n_f, wrong R, ~1e-3 LSQ residual).
    # Fixed by emitting ⌈width/NB⌉ consecutive ≤NB-wide groups per trigger
    # (frontal_numeric.jl's while-loop over cols_left).
    R = dense_R_frontal(F)
    Ad = Matrix(A)[:, F.fsym.base.cperm]
    @test maximum(abs.(R' * R .- Ad' * Ad)) < 1e-6 * max(1, norm(Ad)^2)

    b = randn(MersenneTwister(31338), 400)
    x = PureSparse.solve!(zeros(384), F, b)
    @test normal_eq_resid(A, x, b) < 1e-6 * max(1, norm(b))

    PureSparse.qr!(F, A)   # warm refactor at nb==16 must at least stay in-bounds
    Rw = dense_R_frontal(F)
    @test maximum(abs.(Rw' * Rw .- Ad' * Ad)) < 1e-6 * max(1, norm(Ad)^2)
end
