@testsetup module LLTHelpers
using SparseArrays, LinearAlgebra
export dense_L, permuted_dense, relerr, random_spd_matrix

# Reconstruct the dense n x n L (lower triangular) from a SupernodalFactor's panels, for
# comparison against a dense oracle. Test-only.
#
# IMPORTANT: `potrf!(Ldiag; uplo='L')` (like LAPACK) only ever WRITES the lower triangle
# of a supernode's own nscol x nscol diagonal block — the strictly-upper part is never
# touched by LOAD (which only ever writes lower-triangle destinations, design §4.2) or by
# the update loop (design §4.3's scatter is lower-triangle-only) or by potrf!'s own
# internal computation, and is consequently STALE/UNDEFINED garbage from whatever the
# buffer held before this factorization pass. `cholesky!`/`solve!` never read it (`trsm!`
# with uplo='L' only references the lower triangle by BLAS convention). This helper MUST
# skip it too (k < c within the diagonal sub-block, i.e. row-local-index < col-local-
# index) — copying it in produced a real-looking-but-spurious L*L' mismatch that cost
# significant debugging time before being traced to this helper, not to cholesky!/solve!
# itself (verified independently against a full dense LAPACK oracle on the captured
# pre-potrf block and against the true dense Cholesky of the whole permuted matrix).
function dense_L(F)
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

# Dense P*A*P' for the factor's permutation, from a SparseMatrixCSC A read via its lower
# triangle (symbolic()'s documented input convention).
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

relerr(a, b) = norm(a .- b) / max(norm(b), eps(real(eltype(b))))

function random_spd_matrix(rng, n::Int, density::Float64)
    I = Int[]; J = Int[]; V = Float64[]
    rowsum = zeros(n)
    for j in 1:n, i in (j + 1):n
        rand(rng) < density || continue
        v = randn(rng)
        push!(I, i); push!(J, j); push!(V, v)
        rowsum[i] += abs(v); rowsum[j] += abs(v)
    end
    for j in 1:n
        push!(I, j); push!(J, j); push!(V, rowsum[j] + 1.0)  # diagonally dominant -> SPD
    end
    return sparse(I, J, V, n, n)
end
end

@testitem "cholesky!: L*L' reconstructs P*A*P' (dense reconstruction oracle)" setup = [LLTHelpers] begin
    using Random, LinearAlgebra, SparseArrays
    rng = MersenneTwister(61)
    for n in (1, 2, 3, 5, 10, 30, 60), density in (0.0, 0.05, 0.2, 0.5)
        A = random_spd_matrix(rng, n, density)
        F = PureSparse.cholesky(A)
        @test PureSparse.issuccess(F)
        L = dense_L(F)
        PAP = permuted_dense(F, A)
        @test relerr(L * L', PAP) < 1.0e-9
    end
end

@testitem "cholesky!: refactorize with new values on the same pattern" setup = [LLTHelpers] begin
    using Random, LinearAlgebra, SparseArrays
    rng = MersenneTwister(62)
    n = 40
    A1 = random_spd_matrix(rng, n, 0.1)
    sym = PureSparse.symbolic(A1)
    F = PureSparse.cholesky(sym, A1)
    L1 = dense_L(F)
    @test relerr(L1 * L1', permuted_dense(F, A1)) < 1.0e-9

    # Same PATTERN (same colptr/rowval), different values.
    A2 = SparseMatrixCSC(n, n, copy(A1.colptr), copy(A1.rowval), randn(rng, length(A1.nzval)))
    rowsum = zeros(n)
    for j in 1:n, p in A2.colptr[j]:(A2.colptr[j + 1] - 1)
        i = A2.rowval[p]
        i > j && (rowsum[i] += abs(A2.nzval[p]); rowsum[j] += abs(A2.nzval[p]))
    end
    for j in 1:n, p in A2.colptr[j]:(A2.colptr[j + 1] - 1)
        A2.rowval[p] == j && (A2.nzval[p] = rowsum[j] + 1.0)
    end
    PureSparse.cholesky!(F, A2)
    @test PureSparse.issuccess(F)
    L2 = dense_L(F)
    @test relerr(L2 * L2', permuted_dense(F, A2)) < 1.0e-9
    @test relerr(L1, L2) > 1.0e-6   # sanity: actually refactored (different values)
end

@testitem "cholesky!: refactorize allocations are bounded (zero-alloc hardening is M1 task 7)" setup = [LLTHelpers] begin
    using Random
    rng = MersenneTwister(63)
    n = 50
    A = random_spd_matrix(rng, n, 0.08)
    sym = PureSparse.symbolic(A)
    F = PureSparse.cholesky(sym, A)
    PureSparse.cholesky!(F, A)   # warm up (any first-touch allocation happens here)
    allocs = @allocated PureSparse.cholesky!(F, A)
    # NOT zero yet: `unsafe_wrap`'s Array *header* (not the underlying data) allocates
    # per panel view, needed to avoid a ~90s-per-kernel PureBLAS JIT compile blowup on
    # `reshape(view(...))`'s ReshapedArray-of-SubArray type (see src/numeric/llt.jl's
    # `_panelview` docstring). True zero-alloc needs pre-cached panel views reused across
    # calls — M1 task list item 7 ("refactorize/allocation hardening"), a deliberately
    # separate, later step (ROADMAP.md). This test only guards against a REGRESSION
    # (e.g. an accidental O(n) or O(nnz) allocation), not the eventual zero target.
    @test allocs > 0
    @test allocs < 50_000
end

@testitem "cholesky!: reports non-SPD input via issuccess/F.ok" begin
    using SparseArrays, LinearAlgebra
    n = 4
    # Indefinite: diag all -1 (negative pivot immediately).
    A = sparse(Diagonal(fill(-1.0, n)))
    F = PureSparse.cholesky(A)
    @test !PureSparse.issuccess(F)
    @test F.stats.fail_col >= 1
end

@testitem "solve!: A*x ≈ b residual gate" setup = [LLTHelpers] begin
    using Random, LinearAlgebra, SparseArrays
    rng = MersenneTwister(64)
    for n in (1, 2, 5, 10, 30, 60), density in (0.0, 0.05, 0.2)
        A = random_spd_matrix(rng, n, density)
        F = PureSparse.cholesky(A)
        b = randn(rng, n)
        x = F \ b
        Ad = zeros(n, n)
        for j in 1:n, p in A.colptr[j]:(A.colptr[j + 1] - 1)
            i = A.rowval[p]
            i >= j || continue
            Ad[i, j] = A.nzval[p]
            Ad[j, i] = A.nzval[p]
        end
        resid = norm(Ad * x - b) / (norm(Ad) * norm(x) + eps())
        @test resid < 1.0e-8
    end
end

@testitem "solve!: multi-RHS matches single-RHS column by column" setup = [LLTHelpers] begin
    using Random, LinearAlgebra
    rng = MersenneTwister(65)
    n = 25
    A = random_spd_matrix(rng, n, 0.15)
    F = PureSparse.cholesky(A)
    B = randn(rng, n, 3)
    X = F \ B
    for c in 1:3
        xc = F \ B[:, c]
        @test relerr(X[:, c], xc) < 1.0e-10
    end
end

@testitem "cholesky!: BigFloat precision oracle on small matrices" setup = [LLTHelpers] begin
    using Random, LinearAlgebra, SparseArrays
    rng = MersenneTwister(66)
    for n in (3, 5, 8)
        A = random_spd_matrix(rng, n, 0.4)
        F = PureSparse.cholesky(A)
        L = dense_L(F)
        PAP = permuted_dense(F, A)
        Lbig = LinearAlgebra.cholesky(Symmetric(BigFloat.(PAP), :L)).L
        # Compare Gram matrices (factor itself isn't unique in sign per column in
        # general, but for a genuine Cholesky with positive diagonal it IS unique; still
        # compare via L*L' to sidestep any transcription-order ambiguity).
        @test relerr(Float64.(Lbig * Lbig'), L * L') < 1.0e-12
    end
end
