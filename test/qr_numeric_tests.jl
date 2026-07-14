@testsetup module QRNumericOracle
using Random, SparseArrays, LinearAlgebra
export random_rect_qr2, dense_R, gram_R, reconstruct_A_from_QR, relerr

random_rect_qr2(rng, m::Int, n::Int, density::Float64) = sprand(rng, m, n, density)

relerr(a, b) = norm(a - b) / max(norm(b), eps())

# Dense R (abstract n x n row-of-R layout, NOT the physical-row embedding —
# gram_R/RᵀR is invariant to internal row labeling, design_qr.md §3.4/§4.1) from a
# QRFactor's row-wise storage.
function dense_R(F)
    n = F.sym.n - F.sym.n1
    R = zeros(eltype(F.rval), n, n)
    for k in 1:n
        for c in F.sym.rptr[k]:(F.ws.rcursor[k] - 1)
            j = F.rcolind[c]
            R[k, j] = F.rval[c]
        end
    end
    return R
end

gram_R(F) = (R = dense_R(F); R' * R)

# Direct Q*R = A reconstruction (design_qr.md §4.1/§4.4): R's row k is stored at the
# PHYSICAL row pivotslot[k] (not the abstract row index k — the whole point of the
# B2 fix, §3.4), so the reflector application below places it there before applying
# H_1...H_n (in REVERSE column order, since H_n...H_1*A = R means A = H_1...H_n*R).
# Only meaningful when mb >= n (every column retires a distinct physical row);
# returns `missing` otherwise (RᵀR=AᵀA is the general-case invariant instead).
function reconstruct_A_from_QR(F, A::SparseMatrixCSC)
    m, n = size(A)
    mb = F.sym.mb
    mb < n && return missing
    T = eltype(F.rval)
    X = zeros(T, mb, n)
    for k in 1:n
        piv = F.sym.pivotslot[k]
        piv == 0 && continue
        for c in F.sym.rptr[k]:(F.ws.rcursor[k] - 1)
            j = F.rcolind[c]
            X[piv, j] = F.rval[c]
        end
    end
    for k in n:-1:1
        beta = F.beta[k]
        beta == zero(T) && continue
        vlo, vhi = F.sym.vptr[k], F.sym.vptr[k + 1] - 1
        idxs = F.sym.vrowind[vlo:vhi]
        v = F.vval[vlo:vhi]
        for col in 1:n
            w = beta * sum(v[t] * X[idxs[t], col] for t in eachindex(idxs))
            for t in eachindex(idxs)
                X[idxs[t], col] -= w * v[t]
            end
        end
    end
    Aperm = zeros(T, m, n)
    for j in 1:n
        origcol = F.sym.cperm[j]
        for p in A.colptr[origcol]:(A.colptr[origcol + 1] - 1)
            physrow = F.sym.riperm[A.rowval[p]]
            physrow > mb && continue
            Aperm[physrow, j] = A.nzval[p]
        end
    end
    return X, Aperm[1:mb, :]
end
end

@testitem "RᵀR = AᵀA (fundamental QR identity), random rectangular matrices" setup = [QRNumericOracle] begin
    using Random, SparseArrays, LinearAlgebra
    rng = MersenneTwister(3)
    for _ in 1:300
        m = rand(rng, 1:14)
        n = rand(rng, 1:min(m, 12))
        A = random_rect_qr2(rng, m, n, rand(rng, (0.1, 0.2, 0.4, 0.6)))
        # tol=0 disables M5a task 8's rank detection (§5.3): this test checks the raw
        # numeric loop's fundamental identity, which only holds exactly when no column
        # is dropped as rank-deficient (dropping loses information by construction,
        # §5.2) — rank detection's own effect on RᵀR gets its own dedicated test.
        # singletons=false: task 9's peeling is ON by default REGARDLESS of tol (tol
        # only relaxes the magnitude test, §2.3) — this test's own helpers (gram_R)
        # assume the raw block-only (n1==0) layout; singleton composition gets its own
        # dedicated tests (qr_singleton_compose_tests.jl).
        F = PureSparse.qr(A; ordering = PureSparse.AMDOrdering(), tol = 0, singletons = false)
        G = gram_R(F)
        # R's columns are in FINAL PERMUTED order (design_qr.md §1.4 cperm) — row
        # permutation doesn't matter for AᵀA (cancels: (PA)ᵀ(PA) = AᵀA for any
        # permutation P), but the COLUMN order must match R's before comparing.
        Ad = Matrix(A)[:, F.sym.cperm]
        @test isapprox(G, Ad' * Ad, atol = 1.0e-8)
    end
end

@testitem "Q*R = A direct reconstruction (well-determined cases)" setup = [QRNumericOracle] begin
    using Random, SparseArrays
    rng = MersenneTwister(11)
    ntested = 0
    for _ in 1:300
        m = rand(rng, 1:14)
        n = rand(rng, 1:min(m, 12))
        A = random_rect_qr2(rng, m, n, rand(rng, (0.15, 0.3, 0.5, 0.7)))
        F = PureSparse.qr(A; ordering = PureSparse.AMDOrdering(), tol = 0, singletons = false)  # raw identity
        res = reconstruct_A_from_QR(F, A)
        ismissing(res) && continue
        X, Aperm = res
        ntested += 1
        @test isapprox(X, Aperm, atol = 1.0e-8)
    end
    @test ntested > 100   # sanity: most random trials should be well-determined
end

@testitem "BigFloat precision oracle: RᵀR agrees to near-full precision" begin
    using Random, SparseArrays
    rng = MersenneTwister(66)
    for _ in 1:20
        m = rand(rng, 2:10)
        n = rand(rng, 1:min(m, 8))
        A = sprand(rng, m, n, 0.4)
        Abig = SparseMatrixCSC{BigFloat,Int}(A)
        # tol=0 in both runs: the default τ scales with eps(T), which differs
        # astronomically between Float64 and BigFloat -- comparing the FULL
        # (undropped) R in both is the fair, apples-to-apples precision check.
        F = PureSparse.qr(A; ordering = PureSparse.AMDOrdering(), tol = 0, singletons = false)
        Fbig = PureSparse.qr(Abig; ordering = PureSparse.AMDOrdering(), tol = 0, singletons = false)
        n2 = F.sym.n - F.sym.n1
        Rd = zeros(n2, n2)
        Rb = zeros(BigFloat, n2, n2)
        for k in 1:n2
            for c in F.sym.rptr[k]:(F.ws.rcursor[k] - 1)
                Rd[k, F.rcolind[c]] = F.rval[c]
            end
            for c in Fbig.sym.rptr[k]:(Fbig.ws.rcursor[k] - 1)
                Rb[k, Fbig.rcolind[c]] = Fbig.rval[c]
            end
        end
        Gd = Rd' * Rd
        Gb = Float64.(Rb' * Rb)
        @test isapprox(Gd, Gb, atol = 1.0e-9)
    end
end

@testitem "B3 fix: numerically-zero live column does not crash, beta set to 0" begin
    using SparseArrays
    # Column 2's only entry gets fully cancelled by column 1's reflector: construct A
    # so that after applying H_1, column 2's remaining pattern is exactly zero.
    # A = [1 1; 1 1] under natural order: column 1 = [1,1], reflector zeros row 2;
    # applied to column 2 = [1,1] (identical), it ALSO fully zeros out (both columns
    # proportional) -- a genuinely reachable zero-norm-after-apply case.
    A = sparse([1, 2, 1, 2], [1, 1, 2, 2], [1.0, 1.0, 1.0, 1.0], 2, 2)
    F = PureSparse.qr(A; ordering = PureSparse.NaturalOrdering())
    @test F.ok
    @test F.beta[2] == 0.0   # column 2 fully cancelled -> zero-norm guard triggers
    # R still satisfies RᵀR = AᵀA even with a dead-value (not dead-pattern) column
    n = F.sym.n
    R = zeros(n, n)
    for k in 1:n, c in F.sym.rptr[k]:(F.ws.rcursor[k] - 1)
        R[k, F.rcolind[c]] = F.rval[c]
    end
    Ad = Matrix(A)
    @test isapprox(R' * R, Ad' * Ad, atol = 1.0e-10)
end

@testitem "Structurally dead column (vcount[k]==0): diagonal slot present and zero" begin
    using SparseArrays
    # A = [1 1 1] (design_qr.md §3.4's own worked example): columns 2,3 structurally
    # dead IN THE BLOCK PIPELINE this test means to exercise -- but every column here
    # is ALSO a structural singleton (m=1, so each has exactly one entry), so task 9's
    # default-on peeling would intercept it entirely; singletons=false forces the
    # matrix through the block pipeline this test is actually about.
    A = sparse([1, 1, 1], [1, 2, 3], [1.0, 1.0, 1.0], 1, 3)
    F = PureSparse.qr(A; ordering = PureSparse.NaturalOrdering(), singletons = false)
    @test F.ok
    @test F.beta[2] == 0.0
    @test F.beta[3] == 0.0
    @test F.sym.pivotslot[2] == 0
    @test F.sym.pivotslot[3] == 0
    # every column structurally owns its diagonal slot (rcount[k] >= 1 always)
    for k in 1:3
        @test F.sym.rptr[k + 1] > F.sym.rptr[k]
    end
    # column 1's diagonal is the only nonzero entry anywhere in R
    @test F.rval[F.sym.rptr[1]] != 0.0
    @test F.rval[F.sym.rptr[2]] == 0.0
    @test F.rval[F.sym.rptr[3]] == 0.0
end

@testitem "qr edge cases: empty matrix, single column, single row" begin
    using SparseArrays
    A0 = spzeros(0, 0)
    F0 = PureSparse.qr(A0; ordering = PureSparse.NaturalOrdering())
    @test F0.ok
    @test F0.sym.nnzR == 0

    A1 = sparse(reshape([2.0, 0.0, 3.0], 3, 1))
    F1 = PureSparse.qr(A1; ordering = PureSparse.NaturalOrdering())
    @test F1.ok
    R1 = F1.rval[F1.sym.rptr[1]]
    @test isapprox(R1^2, 2.0^2 + 3.0^2, atol = 1.0e-10)

    # A2's 3 columns are each their own singleton (m=1, so every column has exactly
    # one entry) -- task 9's singleton peeling (default tol, which does NOT disable
    # peeling itself, only the magnitude-threshold strictness -- design_qr.md §2.3)
    # peels ONE of them (the rest become permanently dead once the sole row dies), so
    # n1 can be > 0 here. Reconstruct the FULL R (R11/R12 in full-column-index space,
    # the block part offset by n1) rather than assuming n1 == 0.
    A2 = sparse(reshape([1.0, 2.0, 3.0], 1, 3))
    F2 = PureSparse.qr(A2; ordering = PureSparse.NaturalOrdering())
    @test F2.ok
    n = F2.sym.n
    n1 = F2.sym.n1
    R2 = zeros(n, n)
    for k in 1:n1, c in F2.r1ptr[k]:(F2.r1ptr[k + 1] - 1)
        R2[k, F2.r1colind[c]] = F2.r1val[c]
    end
    nb = length(F2.sym.parent)
    for k in 1:nb, c in F2.sym.rptr[k]:(F2.ws.rcursor[k] - 1)
        R2[n1 + k, n1 + F2.rcolind[c]] = F2.rval[c]
    end
    Ad2 = Matrix(A2)[:, F2.sym.cperm]
    @test isapprox(R2' * R2, Ad2' * Ad2, atol = 1.0e-10)
end

@testitem "qr!: refactorize with new values on the same pattern" begin
    using Random, SparseArrays
    rng = MersenneTwister(202)
    A = sprand(rng, 8, 5, 0.4)
    # qr! requires sym.n1 == 0 (§2.3): force singletons off on the initial factor so
    # this refactor-focused test can't randomly hit the n1>0 rejection path.
    F = PureSparse.qr(A; ordering = PureSparse.AMDOrdering(), tol = 0, singletons = false)
    A2 = SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, A.nzval .+ 1.0)
    PureSparse.qr!(F, A2; tol = 0)
    n = F.sym.n
    R = zeros(n, n)
    for k in 1:n, c in F.sym.rptr[k]:(F.ws.rcursor[k] - 1)
        R[k, F.rcolind[c]] = F.rval[c]
    end
    A2d = Matrix(A2)[:, F.sym.cperm]
    @test isapprox(R' * R, A2d' * A2d, atol = 1.0e-8)
end
