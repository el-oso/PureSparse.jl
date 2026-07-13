@testsetup module LDLTHelpers
using SparseArrays, LinearAlgebra
export dense_unit_L, permuted_dense, relerr, random_sqd_kkt, oracle_unit_ldl

# Reconstruct the dense n x n unit-lower L from an LDLFactor's panels. Same
# strictly-upper-of-diagonal-block skip as llt_tests.jl's dense_L (those positions hold
# stale/undefined data by the same storage convention — see that helper's comment); the
# diagonal positions hold explicit 1s written by the base case.
function dense_unit_L(F)
    sym = F.sym
    n = sym.n
    L = zeros(eltype(F.x), n, n)
    for s in 1:sym.nsuper
        j0 = Int(sym.super[s]); j1 = Int(sym.super[s + 1]) - 1
        nscol = j1 - j0 + 1
        rp0 = Int(sym.rowind_ptr[s])
        nsrow = Int(sym.rowind_ptr[s + 1]) - rp0
        panel = reshape(view(F.x, Int(sym.px[s]):(Int(sym.px[s + 1]) - 1)), nsrow, nscol)
        for k in 1:nsrow, c in 1:nscol
            k < c && k <= nscol && continue   # strictly-upper of the diagonal block: undefined, skip
            L[Int(sym.rowind[rp0 + k - 1]), j0 + c - 1] = panel[k, c]
        end
    end
    return L
end

# Dense P*A*P' for the factor's permutation, from a SparseMatrixCSC read via its lower
# triangle (symbolic()'s documented input convention). Same as llt_tests.jl.
function permuted_dense(F, A::SparseMatrixCSC)
    sym = F.sym
    n = sym.n
    Ad = zeros(eltype(A), n, n)
    for j in 1:n, p in A.colptr[j]:(A.colptr[j + 1] - 1)
        i = A.rowval[p]
        i >= j || continue
        Ad[i, j] = A.nzval[p]
        Ad[j, i] = A.nzval[p]
    end
    perm = Int.(sym.perm)
    return Ad[perm, perm]
end

relerr(a, b) = norm(a .- b) / max(norm(b), eps(real(float(eltype(b)))))

# Random symmetric quasi-definite KKT matrix [H A'; A -C] with H (npos x npos) and
# C (nneg x nneg) diagonally-dominant SPD and a random sparse coupling block A. Stored
# with BOTH triangles (the symbolic driver reads the lower one). Vanderbei 1995: such a
# matrix is strongly factorizable with inertia exactly (npos, nneg, 0).
function random_sqd_kkt(rng, npos::Int, nneg::Int, density::Float64)
    n = npos + nneg
    I = Int[]; J = Int[]; V = Float64[]
    rowsum = zeros(n)
    addsym!(i, j, v) = (push!(I, i); push!(J, j); push!(V, v);
        i != j && (push!(I, j); push!(J, i); push!(V, v));
        rowsum[i] += abs(v); i != j && (rowsum[j] += abs(v)))
    for j in 1:npos, i in (j + 1):npos              # H strict lower
        rand(rng) < density && addsym!(i, j, randn(rng))
    end
    for j in 1:nneg, i in (j + 1):nneg              # C strict lower
        rand(rng) < density && addsym!(npos + i, npos + j, randn(rng))
    end
    for j in 1:npos, i in 1:nneg                    # coupling block A
        rand(rng) < density && addsym!(npos + i, j, randn(rng))
    end
    for j in 1:n                                    # diagonally dominant: +/- per block
        v = (rowsum[j] + 1.0) * (j <= npos ? 1.0 : -1.0)
        push!(I, j); push!(J, j); push!(V, v)
    end
    return sparse(I, J, V, n, n)
end

# From-scratch dense unit-LDL^T without pivoting (Golub & Van Loan, symmetric indefinite
# factorization with fixed 1x1 pivots), generic over the element type — the independent
# oracle the supernodal factorization is compared against.
function oracle_unit_ldl(Ad::AbstractMatrix{TT}) where {TT}
    n = size(Ad, 1)
    M = Matrix{TT}(copy(Ad))
    L = Matrix{TT}(LinearAlgebra.I, n, n)
    d = zeros(TT, n)
    for j in 1:n
        d[j] = M[j, j]
        for i in (j + 1):n
            L[i, j] = M[i, j] / d[j]
        end
        for k in (j + 1):n, i in k:n
            M[i, k] -= d[j] * L[i, j] * L[k, j]
        end
    end
    return L, d
end
end

@testitem "ldlt: L·D·Lᵀ reconstructs P·K·Pᵀ and matches the dense oracle (SQD, no perturbation)" setup = [LDLTHelpers] begin
    using Random, LinearAlgebra, SparseArrays
    rng = MersenneTwister(71)
    for (npos, nneg) in ((1, 1), (3, 2), (10, 6), (30, 20), (45, 30)), density in (0.0, 0.05, 0.2)
        K = random_sqd_kkt(rng, npos, nneg, density)
        F = PureSparse.ldlt(K; n_pos = npos, n_neg = nneg)
        @test PureSparse.issuccess(F)
        @test F.stats.n_perturbed == 0        # natural pivots already have the right signs
        L = dense_unit_L(F)
        PKP = permuted_dense(F, K)
        @test relerr(L * Diagonal(F.d) * L', PKP) < 1.0e-9
        # elementwise against the independent dense unit-LDLᵀ oracle
        Lref, dref = oracle_unit_ldl(PKP)
        @test relerr(L, Lref) < 1.0e-8
        @test relerr(F.d, dref) < 1.0e-8
    end
end

@testitem "ldlt: BigFloat precision oracle on small SQD matrices" setup = [LDLTHelpers] begin
    using Random, LinearAlgebra
    rng = MersenneTwister(72)
    for (npos, nneg) in ((2, 1), (3, 3), (5, 4))
        K = random_sqd_kkt(rng, npos, nneg, 0.4)
        F = PureSparse.ldlt(K; n_pos = npos, n_neg = nneg)
        PKP = permuted_dense(F, K)
        Lref, dref = oracle_unit_ldl(BigFloat.(PKP))
        @test relerr(dense_unit_L(F), Float64.(Lref)) < 1.0e-12
        @test relerr(F.d, Float64.(dref)) < 1.0e-12
    end
end

@testitem "ldlt: inertia matches SQD construction (n_pos/n_neg/n_zero)" setup = [LDLTHelpers] begin
    using Random, SparseArrays, LinearAlgebra
    rng = MersenneTwister(73)
    for (npos, nneg) in ((4, 3), (12, 8), (25, 25)), density in (0.05, 0.3)
        K = random_sqd_kkt(rng, npos, nneg, density)
        F = PureSparse.ldlt(K; n_pos = npos, n_neg = nneg)
        @test F.stats.n_pos == npos
        @test F.stats.n_neg == nneg
        @test F.stats.n_zero == 0
        @test 0.0 < F.stats.rcond_est <= 1.0
    end
    # explicit-diagonal control case: signs are read off directly
    Kd = sparse(Diagonal([4.0, -2.0, 7.0, -1.5, 3.0]))
    Fd = PureSparse.ldlt(Kd; signs = [1, -1, 1, -1, 1])
    @test (Fd.stats.n_pos, Fd.stats.n_neg, Fd.stats.n_zero) == (3, 2, 0)
    @test Fd.stats.n_perturbed == 0
end

@testitem "ldlt: signed regularization forces wrong-sign and zero pivots, F.ok stays true" setup = [LDLTHelpers] begin
    using SparseArrays, LinearAlgebra
    # wrong-sign pivot: -3 forced to +3 under signs=[+1,+1,+1] (d <- sign * max(delta,|d|))
    K = sparse(Diagonal([2.0, -3.0, 5.0]))
    F = PureSparse.ldlt(K; signs = [1, 1, 1])
    @test PureSparse.issuccess(F)
    @test F.stats.n_perturbed == 1
    @test F.stats.max_perturbation ≈ 6.0
    d_orig = F.d[Int.(F.sym.iperm)]           # back to original column order
    @test d_orig ≈ [2.0, 3.0, 5.0]
    # the factorization reconstructs the REGULARIZED matrix, not A, at the forced entry
    L = dense_unit_L(F)
    R = L * Diagonal(F.d) * L'
    @test relerr(R, permuted_dense(F, sparse(Diagonal([2.0, 3.0, 5.0])))) < 1.0e-12
    @test relerr(R, permuted_dense(F, K)) > 0.1

    # zero pivot: forced up to the magnitude floor delta = ldlt_delta * max|A|
    K0 = sparse(Diagonal([1.0, 0.0, 2.0]))
    F0 = PureSparse.ldlt(K0; signs = [1, 1, 1])
    @test PureSparse.issuccess(F0)
    @test F0.stats.n_perturbed == 1
    @test F0.stats.n_zero == 1
    d0 = F0.d[Int.(F0.sym.iperm)]
    @test d0[2] > 0.0                          # forced positive, tiny but nonzero
    @test d0[[1, 3]] ≈ [1.0, 2.0]

    # free signs (signs omitted): magnitude floor only, never a sign flip
    Ff = PureSparse.ldlt(K)                    # K has a healthy -3 pivot
    @test Ff.stats.n_perturbed == 0
    @test (Ff.stats.n_pos, Ff.stats.n_neg) == (2, 1)
end

@testitem "ldlt!: width-1/2 diagonal-block fast path matches the general ger! path" setup = [LDLTHelpers] begin
    # Mirror of llt.jl's fast-path regression guard (n=64 OB-arm perf fix). Width-1
    # already made zero kernel calls before this fix (documented no-op mirror in
    # ldlt.jl); width-2 replaces its single ger! trailing-update call with an inline
    # muladd. Pin correctness on a matrix with the SAME width-[1,2] supernode
    # partition as llt_tests.jl's Aiso case, plus a dedicated check that signed
    # regularization correctly propagates through the inlined width-2 trailing
    # update (the fast path uses `dj` AFTER regularization, not the original pivot —
    # this is the one place a naive port could silently use the wrong value).
    using Random, LinearAlgebra, SparseArrays

    # (a) same widths=[1,2] partition as llt_tests.jl's Aiso, free signs.
    A1 = sparse([1, 2, 2], [1, 1, 2], [1.0, 0.3, 2.0], 2, 2)
    Aiso = blockdiag(sparse([1.0;;]), A1)
    Fiso = PureSparse.ldlt(Aiso)
    @test PureSparse.issuccess(Fiso)
    L = dense_unit_L(Fiso)
    R = L * Diagonal(Fiso.d) * L'
    @test relerr(R, permuted_dense(Fiso, Aiso)) < 1.0e-10

    # (b) width-2 regularization propagation: K = [[-2,1],[1,3]] under signs=[+1,+1]
    # forces d1 from -2 to +2; the trailing update for column 2 must use the
    # REGULARIZED d1=2 (not the original -2) to compute d2. Hand-derived: L21 =
    # 1/2 = 0.5, d2 = 3 - 2*0.5^2 = 2.5, and L*D*L' must reconstruct the REGULARIZED
    # matrix [[2,1],[1,3]], not the original K. Order pinned via GivenOrdering so
    # AMD can't swap which column processes first (it does, on this tiny 2x2 input,
    # which would otherwise silently invalidate the hand-derived values below).
    K = sparse([1, 2, 2], [1, 1, 2], [-2.0, 1.0, 3.0], 2, 2)
    nat_order = PureSparse.GivenOrdering([1, 2])
    @assert PureSparse.symbolic(K; ordering = nat_order).nsuper == 1   # one width-2 supernode
    F = PureSparse.ldlt(K; signs = [1, 1], ordering = nat_order)
    @test PureSparse.issuccess(F)
    @test F.stats.n_perturbed == 1
    d = F.d[Int.(F.sym.iperm)]
    @test d ≈ [2.0, 2.5]
    Lb = dense_unit_L(F)
    Rb = Lb * Diagonal(F.d) * Lb'
    Kreg = sparse([1, 2, 2], [1, 1, 2], [2.0, 1.0, 3.0], 2, 2)
    @test relerr(Rb, permuted_dense(F, Kreg)) < 1.0e-10
    @test relerr(Rb, permuted_dense(F, K)) > 0.1   # not the unregularized matrix
end

@testitem "ldlt: solve! residual gate on SQD systems + multi-RHS" setup = [LDLTHelpers] begin
    using Random, LinearAlgebra, SparseArrays
    rng = MersenneTwister(74)
    for (npos, nneg) in ((2, 1), (8, 5), (30, 20), (50, 35)), density in (0.0, 0.1)
        K = random_sqd_kkt(rng, npos, nneg, density)
        n = npos + nneg
        F = PureSparse.ldlt(K; n_pos = npos, n_neg = nneg)
        @test F.stats.n_perturbed == 0        # so K itself (not a regularized K) is solved
        b = randn(rng, n)
        x = F \ b
        Kd = zeros(n, n)
        for j in 1:n, p in K.colptr[j]:(K.colptr[j + 1] - 1)
            i = K.rowval[p]
            i >= j || continue
            Kd[i, j] = K.nzval[p]
            Kd[j, i] = K.nzval[p]
        end
        @test norm(Kd * x - b) / (norm(Kd) * norm(x) + eps()) < 1.0e-8
        B = randn(rng, n, 3)
        X = F \ B
        for c in 1:3
            @test relerr(X[:, c], F \ B[:, c]) < 1.0e-10
        end
    end
end

@testitem "ldlt!: refactorize with new values on the same pattern" setup = [LDLTHelpers] begin
    using Random, LinearAlgebra, SparseArrays
    rng = MersenneTwister(75)
    npos, nneg = 20, 15
    K1 = random_sqd_kkt(rng, npos, nneg, 0.1)
    sym = PureSparse.symbolic(K1)
    signs = vcat(fill(1, npos), fill(-1, nneg))
    F = PureSparse.ldlt(sym, K1; signs)
    L1 = dense_unit_L(F)
    d1 = copy(F.d)
    @test relerr(L1 * Diagonal(d1) * L1', permuted_dense(F, K1)) < 1.0e-9

    # same PATTERN, scaled values (scaling preserves SQD and the sign pattern)
    K2 = SparseMatrixCSC(K1.m, K1.n, copy(K1.colptr), copy(K1.rowval), 3.0 .* K1.nzval)
    PureSparse.ldlt!(F, K2)
    @test PureSparse.issuccess(F)
    @test F.stats.n_perturbed == 0
    L2 = dense_unit_L(F)
    @test relerr(L2 * Diagonal(F.d) * L2', permuted_dense(F, K2)) < 1.0e-9
    @test F.d ≈ 3.0 .* d1                      # D scales, unit-L is scale-invariant
    @test relerr(L2, L1) < 1.0e-12
end

@testitem "ldlt!: refactorize is genuinely zero-allocation (M2 gate, CLAUDE.md req 5)" setup = [LDLTHelpers] begin
    using Random, LinearAlgebra, SparseArrays
    rng = MersenneTwister(77)
    # Two shapes: a generic SQD KKT, and a lopsided one (few coupled variables, wide
    # blocks) to also exercise wide-descendant/short-update-block pairs — the small-k1
    # case whose chunk width used to exceed max_extend_rows under the old flat
    # max_update_size-sized `cd` staging buffer.
    for (npos, nneg, dens) in ((20, 15, 0.3), (60, 8, 0.25))
        n = npos + nneg
        K = random_sqd_kkt(rng, npos, nneg, dens)
        signs = vcat(fill(1, npos), fill(-1, nneg))
        F = PureSparse.ldlt(K; signs)
        PureSparse.ldlt!(F, K)   # warm up (any first-touch allocation happens here)
        allocs = @allocated PureSparse.ldlt!(F, K)
        # History: 1120 bytes remained after M1 task 7 fixed the shared `c` update
        # buffer -> 0 after `Workspace.cd` (the L·D scaled-copy staging buffer) changed
        # from a flat `Vector{T}(max_update_size)` needing a fresh `_panelview`
        # unsafe_wrap per chunk to a pre-allocated square
        # `Matrix{T}(max_extend_rows, max_extend_rows)` used via
        # `view(cdbuf, 1:k1, 1:wk)` — in-bounds because k1 ≤ max_extend_rows (same
        # containment proof as `c`) and the chunk width wk is capped at the buffer's
        # own column capacity by construction (derivation in types.jl's Workspace
        # docstring, cap in ldlt.jl's update loop).
        @test allocs == 0
        # zero-alloc must not have cost correctness on this factor
        b = randn(rng, n)
        x = F \ b
        @test norm(Matrix(K) * x - b) / norm(b) < 1.0e-8
        @test (F.stats.n_pos, F.stats.n_neg, F.stats.n_zero) == (npos, nneg, 0)
    end
end

@testitem "ldlt: n_pos/n_neg convenience is exactly equivalent to explicit signs" setup = [LDLTHelpers] begin
    using Random
    rng = MersenneTwister(76)
    npos, nneg = 12, 9
    K = random_sqd_kkt(rng, npos, nneg, 0.15)
    F1 = PureSparse.ldlt(K; signs = vcat(fill(Int8(1), npos), fill(Int8(-1), nneg)))
    F2 = PureSparse.ldlt(K; n_pos = npos, n_neg = nneg)
    @test F1.signs == F2.signs
    @test F1.d == F2.d
    @test F1.x == F2.x
end

@testitem "ldlt: argument validation" setup = [LDLTHelpers] begin
    using Random, SparseArrays
    rng = MersenneTwister(77)
    K = random_sqd_kkt(rng, 3, 2, 0.2)
    @test_throws DimensionMismatch PureSparse.ldlt(K; signs = [1, 1])
    @test_throws ArgumentError PureSparse.ldlt(K; signs = [1, 1, 1, 2, -1])
    @test_throws ArgumentError PureSparse.ldlt(K; signs = fill(1, 5), n_pos = 3, n_neg = 2)
    @test_throws ArgumentError PureSparse.ldlt(K; n_pos = 3, n_neg = 3)
end
