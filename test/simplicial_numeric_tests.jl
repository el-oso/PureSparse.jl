# Simplicial (column-by-column) LLᵀ/LDLᵀ path and the automatic simplicial-versus-
# supernodal choice (design.md §4.5). The shared oracle items in llt_tests.jl and
# ldlt_tests.jl run both paths; this file covers what is specific to the simplicial one:
# the exact pattern of L, the by-row copy of the values, the switch, the exact failing
# column, the update/downdate conversion, and zero allocation on the warm paths.

@testsetup module SimplicialNumericHelpers
using SparseArrays, LinearAlgebra, Random
export dense_L_cols, dense_L_rows, permuted_dense, relerr, random_spd, random_sqd_kkt, laplacian_2d

# Dense L from the by-column storage (sym.lcolptr/lrowval, F.lx).
function dense_L_cols(F)
    sym = F.sym
    n = sym.n
    L = zeros(eltype(F.lx), n, n)
    for j in 1:n, p in Int(sym.lcolptr[j]):(Int(sym.lcolptr[j + 1]) - 1)
        L[Int(sym.lrowval[p]), j] = F.lx[p]
    end
    return L
end

# Dense L from the by-row storage (sym.lrowptr/lcolval, F.ltx).
function dense_L_rows(F)
    sym = F.sym
    n = sym.n
    L = zeros(eltype(F.ltx), n, n)
    for k in 1:n, t in Int(sym.lrowptr[k]):(Int(sym.lrowptr[k + 1]) - 1)
        L[k, Int(sym.lcolval[t])] = F.ltx[t]
    end
    return L
end

# Dense P·A·Pᵀ from a SparseMatrixCSC read via its lower triangle.
function permuted_dense(F, A::SparseMatrixCSC)
    n = F.sym.n
    Ad = zeros(eltype(A), n, n)
    for j in 1:n, p in A.colptr[j]:(A.colptr[j + 1] - 1)
        i = A.rowval[p]
        i >= j || continue
        Ad[i, j] = A.nzval[p]
        Ad[j, i] = A.nzval[p]
    end
    perm = Int.(F.sym.perm)
    return Ad[perm, perm]
end

relerr(a, b) = norm(a .- b) / max(norm(b), eps(real(float(eltype(b)))))

function random_spd(rng, n::Int, density::Float64; T = Float64)
    I = Int[]; J = Int[]; V = T[]
    rowsum = zeros(T, n)
    for j in 1:n, i in (j + 1):n
        rand(rng) < density || continue
        v = T(randn(rng))
        push!(I, i); push!(J, j); push!(V, v)
        rowsum[i] += abs(v); rowsum[j] += abs(v)
    end
    for j in 1:n
        push!(I, j); push!(J, j); push!(V, rowsum[j] + one(T))
    end
    return sparse(I, J, V, n, n)
end

function random_sqd_kkt(rng, npos::Int, nneg::Int, density::Float64)
    n = npos + nneg
    I = Int[]; J = Int[]; V = Float64[]
    rowsum = zeros(n)
    addsym!(i, j, v) = (
        push!(I, i); push!(J, j); push!(V, v);
        i != j && (push!(I, j); push!(J, i); push!(V, v));
        rowsum[i] += abs(v); i != j && (rowsum[j] += abs(v))
    )
    for j in 1:npos, i in (j + 1):npos
        rand(rng) < density && addsym!(i, j, randn(rng))
    end
    for j in 1:nneg, i in (j + 1):nneg
        rand(rng) < density && addsym!(npos + i, npos + j, randn(rng))
    end
    for j in 1:npos, i in 1:nneg
        rand(rng) < density && addsym!(npos + i, j, randn(rng))
    end
    for j in 1:n
        v = (rowsum[j] + 1.0) * (j <= npos ? 1.0 : -1.0)
        push!(I, j); push!(J, j); push!(V, v)
    end
    return sparse(I, J, V, n, n)
end

# 5-point Laplacian with Dirichlet boundary (strictly diagonally dominant), lower stored.
function laplacian_2d(nx::Int, ny::Int)
    n = nx * ny
    idx(i, j) = (j - 1) * nx + i
    I = Int[]; J = Int[]; V = Float64[]
    for j in 1:ny, i in 1:nx
        p = idx(i, j)
        i > 1 && (push!(I, p); push!(J, idx(i - 1, j)); push!(V, -1.0))
        j > 1 && (push!(I, p); push!(J, idx(i, j - 1)); push!(V, -1.0))
        push!(I, p); push!(J, p); push!(V, 4.0)
    end
    return sparse(I, J, V, n, n)
end
end

@testitem "simplicial symbolic: exact pattern of L by columns and by rows" setup = [SimplicialNumericHelpers] begin
    using Random, SparseArrays, LinearAlgebra
    rng = MersenneTwister(91)
    for n in (1, 2, 5, 10, 30, 80), density in (0.0, 0.05, 0.2, 0.5)
        A = random_spd(rng, n, density)
        sym = PureSparse.symbolic(A; method = :simplicial)
        @test sym.simplicial
        @test iszero(sym.nsuper)
        nnzL = sym.nnzL
        @test length(sym.lrowval) == nnzL == length(sym.lcolval)
        @test sym.lcolptr[n + 1] == nnzL + 1 == sym.lrowptr[n + 1]
        @test [sym.lcolptr[j + 1] - sym.lcolptr[j] for j in 1:n] == sym.colcount
        # by columns: diagonal first, rows ascending
        for j in 1:n
            col = sym.lrowval[sym.lcolptr[j]:(sym.lcolptr[j + 1] - 1)]
            @test col[1] == j
            @test issorted(col) && allunique(col)
        end
        # by rows: the diagonal last, the other columns before their etree ancestors,
        # and the same (row, column) set as the by-column pattern
        pairs_cols = Set((Int(sym.lrowval[p]), j) for j in 1:n for p in sym.lcolptr[j]:(sym.lcolptr[j + 1] - 1))
        pairs_rows = Set((k, Int(sym.lcolval[t])) for k in 1:n for t in sym.lrowptr[k]:(sym.lrowptr[k + 1] - 1))
        @test pairs_cols == pairs_rows
        for k in 1:n
            row = Int.(sym.lcolval[sym.lrowptr[k]:(sym.lrowptr[k + 1] - 1)])
            @test row[end] == k
            pos = Dict(i => t for (t, i) in enumerate(row))
            for i in row[1:(end - 1)]
                p = Int(sym.parent[i])
                !iszero(p) && p < k && haskey(pos, p) && @test pos[i] < pos[p]
            end
        end
        # the pattern of L is the true pattern: every nonzero of the dense Cholesky
        # factor of P·A·Pᵀ lies in it (no fill is missed), and it has nnzL entries
        F = PureSparse.cholesky(sym, A)
        Ld = LinearAlgebra.cholesky(Symmetric(permuted_dense(F, A), :L)).L
        for j in 1:n, i in j:n
            abs(Ld[i, j]) > 1.0e-12 && @test (i, j) in pairs_cols
        end
        # permuted upper pattern: column k holds the rows ≤ k of P·A·Pᵀ, one slot per
        # stored lower entry of A, every slot addressed by amap
        used = filter(!iszero, sym.amap)
        @test length(sym.urowval) == length(used)
        @test sort(used) == eachindex(sym.urowval)
        for k in 1:n, p in sym.ucolptr[k]:(sym.ucolptr[k + 1] - 1)
            @test sym.urowval[p] <= k
        end
    end
end

@testitem "simplicial cholesky!: oracle, by-row copy, non-SPD column, refactorization" setup = [SimplicialNumericHelpers] begin
    using Random, SparseArrays, LinearAlgebra
    rng = MersenneTwister(92)
    for n in (1, 2, 3, 5, 10, 30, 80), density in (0.0, 0.05, 0.2, 0.5)
        A = random_spd(rng, n, density)
        F = PureSparse.cholesky(A; method = :simplicial)
        @test PureSparse.issuccess(F)
        @test iszero(F.stats.fail_col)
        @test F.stats.nnzL == F.sym.nnzL
        Lc = dense_L_cols(F)
        @test Lc == dense_L_rows(F)                  # the two value copies agree exactly
        PAP = permuted_dense(F, A)
        @test relerr(Lc * Lc', PAP) < 1.0e-9
        Lbig = LinearAlgebra.cholesky(Symmetric(BigFloat.(PAP), :L)).L
        @test relerr(Lc, Float64.(Lbig)) < 1.0e-10
        # same L as the supernodal path, up to rounding
        Fs = PureSparse.cholesky(A; method = :supernodal)
        @test Fs.sym.perm == F.sym.perm
        b = randn(rng, n)
        @test relerr(F \ b, Fs \ b) < 1.0e-10
    end

    # exact failing column (factor order): the first non-positive pivot of P·A·Pᵀ
    A = random_spd(rng, 12, 0.3)
    F = PureSparse.cholesky(A; method = :simplicial)
    perm = Int.(F.sym.perm)
    Ad = Matrix(Symmetric(Matrix(A), :L))
    target = perm[7]                       # the 7th pivot in factor order
    Ad[target, target] = -1.0              # SPD up to pivot 6, indefinite at pivot 7
    Abad = sparse(LinearAlgebra.tril(Ad))
    Fb = PureSparse.cholesky(F.sym, Abad)
    @test !PureSparse.issuccess(Fb)
    @test Fb.stats.fail_col == 7
    @test !PureSparse.issuccess(PureSparse.cholesky(sparse(Diagonal([1.0, NaN, 2.0])); method = :simplicial))

    # refactorization with new values on the same pattern, zero allocation
    A1 = random_spd(rng, 60, 0.1)
    sym = PureSparse.symbolic(A1; method = :simplicial)
    F = PureSparse.cholesky(sym, A1)
    L1 = dense_L_cols(F)
    A2 = SparseMatrixCSC(60, 60, copy(A1.colptr), copy(A1.rowval), 2.0 .* A1.nzval)
    PureSparse.cholesky!(F, A2)
    @test PureSparse.issuccess(F)
    @test relerr(dense_L_cols(F), sqrt(2.0) .* L1) < 1.0e-12
    @test iszero(@allocated PureSparse.cholesky!(F, A2))
end

@testitem "simplicial ldlt!: oracle, inertia, signed regularization, zero allocation" setup = [SimplicialNumericHelpers] begin
    using Random, SparseArrays, LinearAlgebra
    rng = MersenneTwister(93)
    for (npos, nneg) in ((1, 1), (3, 2), (10, 6), (30, 20), (45, 30)), density in (0.0, 0.05, 0.2)
        K = random_sqd_kkt(rng, npos, nneg, density)
        F = PureSparse.ldlt(K; n_pos = npos, n_neg = nneg, method = :simplicial)
        @test PureSparse.issuccess(F)
        @test iszero(F.stats.n_perturbed)
        @test (F.stats.n_pos, F.stats.n_neg, F.stats.n_zero) == (npos, nneg, 0)
        L = dense_L_cols(F)
        @test L == dense_L_rows(F)
        @test all(isone, diag(L))
        PKP = permuted_dense(F, K)
        @test relerr(L * Diagonal(F.d) * L', PKP) < 1.0e-9
        # the supernodal path computes the same D
        Fs = PureSparse.ldlt(K; n_pos = npos, n_neg = nneg, method = :supernodal)
        @test relerr(F.d, Fs.d) < 1.0e-10
        b = randn(rng, npos + nneg)
        @test relerr(F \ b, Fs \ b) < 1.0e-9
    end

    # signed regularization on the simplicial path: wrong-sign pivot forced, and the
    # later pivots use the forced value (K = [-2 1; 1 3] under natural order)
    K = sparse([1, 2, 2], [1, 1, 2], [-2.0, 1.0, 3.0], 2, 2)
    F = PureSparse.ldlt(K; signs = [1, 1], ordering = PureSparse.GivenOrdering([1, 2]), method = :simplicial)
    @test PureSparse.issuccess(F)
    @test F.stats.n_perturbed == 1
    @test F.d ≈ [2.0, 2.5]
    @test F.stats.max_perturbation ≈ 4.0
    # zero pivot forced up to the floor, counted as zero in the inertia
    F0 = PureSparse.ldlt(sparse(Diagonal([1.0, 0.0, 2.0])); signs = [1, 1, 1], method = :simplicial)
    @test F0.stats.n_zero == 1 && F0.stats.n_perturbed == 1
    @test all(>(0), F0.d)

    # zero allocation on the warm refactor
    K = random_sqd_kkt(rng, 40, 25, 0.1)
    F = PureSparse.ldlt(K; n_pos = 40, n_neg = 25, method = :simplicial)
    PureSparse.ldlt!(F, K)
    @test iszero(@allocated PureSparse.ldlt!(F, K))
end

@testitem "simplicial solve!: residual gate, split solves, multi-RHS, zero allocation" setup = [SimplicialNumericHelpers] begin
    using Random, SparseArrays, LinearAlgebra
    rng = MersenneTwister(94)
    for n in (1, 2, 5, 30, 120), density in (0.0, 0.05, 0.2)
        A = random_spd(rng, n, density)
        F = PureSparse.cholesky(A; method = :simplicial)
        G = PureSparse.ldlt(A; method = :simplicial)
        Ad = Matrix(Symmetric(Matrix(A), :L))
        b = randn(rng, n)
        for H in (F, G)
            x = H \ b
            @test norm(Ad * x - b) / (norm(Ad) * norm(x) + eps()) < 1.0e-10
            # split solves compose to the full solve in factor order
            y = b[Int.(H.sym.perm)]
            PureSparse.solve_L!(y, H)
            H isa PureSparse.LDLFactor && PureSparse.solve_D!(y, H)
            PureSparse.solve_Lt!(y, H)
            @test relerr(y[Int.(H.sym.iperm)], x) < 1.0e-10
            B = randn(rng, n, 3)
            X = H \ B
            for c in 1:3
                @test relerr(X[:, c], H \ B[:, c]) < 1.0e-10
            end
        end
    end
    A = random_spd(rng, 80, 0.1)
    F = PureSparse.cholesky(A; method = :simplicial)
    G = PureSparse.ldlt(A; method = :simplicial)
    b = randn(rng, 80)
    x = similar(b)
    PureSparse.solve!(x, F, b)
    PureSparse.solve!(x, G, b)
    @test iszero(@allocated PureSparse.solve!(x, F, b))
    @test iszero(@allocated PureSparse.solve!(x, G, b))
    @test_throws DimensionMismatch PureSparse.solve_L!(zeros(79), F)
end

@testitem "simplicial: generic element and index types" setup = [SimplicialNumericHelpers] begin
    using Random, SparseArrays, LinearAlgebra
    rng = MersenneTwister(95)
    A64 = random_spd(rng, 40, 0.15)
    A32 = SparseMatrixCSC{Float32, Int32}(A64)
    F = PureSparse.cholesky(A32; method = :simplicial)
    @test F isa PureSparse.SupernodalFactor{Float32, Int32}
    @test F.sym.simplicial
    b = randn(rng, Float32, 40)
    x = F \ b
    @test norm(Matrix(Symmetric(Matrix(A32), :L)) * x - b) / norm(b) < 1.0e-4
    Abig = SparseMatrixCSC{BigFloat, Int}(A64)
    Fbig = PureSparse.cholesky(Abig; method = :simplicial)
    Lbig = dense_L_cols(Fbig)
    @test relerr(Lbig * Lbig', permuted_dense(Fbig, Abig)) < 1.0e-30
end

@testitem "simplicial: automatic choice, override, and validation" setup = [SimplicialNumericHelpers] begin
    using Random, SparseArrays, LinearAlgebra
    rng = MersenneTwister(96)
    # a narrow banded matrix (predictor about 7: little work per stored entry) takes
    # the simplicial path
    A = let n = 2000, bw = 6
        I = Int[]; J = Int[]; V = Float64[]
        rowsum = zeros(n)
        for j in 1:n, i in (j + 1):min(n, j + bw)
            v = randn(rng)
            push!(I, i); push!(J, j); push!(V, v)
            rowsum[i] += abs(v); rowsum[j] += abs(v)
        end
        for j in 1:n
            push!(I, j); push!(J, j); push!(V, rowsum[j] + 1.0)
        end
        sparse(I, J, V, n, n)
    end
    sym = PureSparse.symbolic(A)
    @test sym.simplicial
    @test sym.flops / sym.nnzL <= PureSparse.SIMPLICIAL_RATIO
    F = PureSparse.cholesky(A)
    @test F isa PureSparse.SupernodalFactor{Float64, Int}
    @test F.sym.simplicial && isempty(F.x) && length(F.lx) == F.sym.nnzL
    G = PureSparse.ldlt(A)
    @test G.sym.simplicial
    b = randn(rng, size(A, 1))
    @test norm(Symmetric(A, :L) * (F \ b) - b) / norm(b) < 1.0e-12
    # the 2D 100x100 Laplacian (predictor about 55) and a dense-ish problem take the
    # supernodal path: their refactorization is far cheaper there (design.md §4.5)
    L2 = laplacian_2d(100, 100)
    syml = PureSparse.symbolic(L2)
    @test !syml.simplicial
    @test syml.flops / syml.nnzL > PureSparse.SIMPLICIAL_RATIO
    D = random_spd(rng, 600, 0.05)
    symd = PureSparse.symbolic(D)
    @test !symd.simplicial
    @test symd.flops / symd.nnzL > PureSparse.SIMPLICIAL_RATIO
    @test !PureSparse.cholesky(D).sym.simplicial
    # explicit overrides, both directions, same type either way
    @test PureSparse.symbolic(A; method = :supernodal).nsuper > 0
    @test PureSparse.symbolic(D; method = :simplicial).simplicial
    @test typeof(PureSparse.cholesky(D; method = :simplicial)) == typeof(PureSparse.cholesky(D; method = :supernodal))
    @test typeof(PureSparse.ldlt(D; method = :simplicial)) == typeof(PureSparse.ldlt(D; method = :supernodal))
    @test_throws ArgumentError PureSparse.symbolic(A; method = :fast)
    @test_throws ArgumentError PureSparse.cholesky(A; method = :fast)
    # the switch is written without a division: an empty matrix is fine
    @test PureSparse.symbolic(spzeros(0, 0)).simplicial
end

@testitem "simplicial: updowndate! from a simplicial-mode LDLFactor" setup = [SimplicialNumericHelpers] begin
    using Random, SparseArrays, LinearAlgebra
    rng = MersenneTwister(97)
    for (npos, nneg) in ((6, 4), (25, 15))
        n = npos + nneg
        K = random_sqd_kkt(rng, npos, nneg, 0.2)
        F = PureSparse.ldlt(K; n_pos = npos, n_neg = nneg, method = :simplicial)
        G = PureSparse.simplicial(F; grow = Float64(n))
        # the conversion carries the exact strictly-lower pattern and the values
        @test G.stats.nnzL == F.sym.nnzL
        Lf = dense_L_cols(F)
        Lg = Matrix{Float64}(I, n, n)
        for j in 1:n, p in Int(G.colptr[j]):(Int(G.colptr[j]) + Int(G.colnnz[j]) - 1)
            Lg[Int(G.rowval[p]), j] = G.nzval[p]
        end
        @test Lg == Lf
        @test G.d == F.d
        # update then downdate round-trips, and the update matches a refactorization
        w = zeros(n); w[1:npos] .= randn(rng, npos)
        @test PureSparse.updowndate!(G, w, 1) === :ok
        Kw = K + sparse(w * w')
        Fw = PureSparse.ldlt(Kw; n_pos = npos, n_neg = nneg, method = :simplicial, ordering = PureSparse.GivenOrdering(Int.(F.sym.perm)))
        b = randn(rng, n)
        @test relerr(G \ b, Fw \ b) < 1.0e-9
        @test PureSparse.updowndate!(G, w, -1) === :ok
        @test relerr(G \ b, F \ b) < 1.0e-9
    end
end
