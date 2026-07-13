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

@testitem "cholesky!: refactorize is genuinely zero-allocation (M1 task 7, CLAUDE.md req 5)" setup = [LLTHelpers] begin
    using Random
    rng = MersenneTwister(63)
    n = 50
    A = random_spd_matrix(rng, n, 0.08)
    sym = PureSparse.symbolic(A)
    F = PureSparse.cholesky(sym, A)
    PureSparse.cholesky!(F, A)   # warm up (any first-touch allocation happens here)
    allocs = @allocated PureSparse.cholesky!(F, A)
    # History: 7392 -> 2576 bytes after caching F.panels (built once, reused across
    # calls instead of re-`unsafe_wrap`ping every supernode's panel/panel_d every
    # call) -> 0 after also fixing the variable-shaped update-block buffer: instead of
    # `_panelview(cbuf, 1, ctot, k1)` (a fresh `unsafe_wrap` — small Array-header
    # allocation — every call), `Workspace.c` is now a single pre-allocated
    # `Matrix{T}(max_extend_rows, max_extend_rows)` and the update block is
    # `view(cbuf, 1:ctot, 1:k1)` — zero-alloc because both ctot and k1 are provably
    # ≤ max_extend_rows for every (descendant, ancestor) pair (types.jl's Workspace
    # docstring has the derivation) and a `view` of an already-allocated Matrix costs
    # nothing (measured directly: 0 bytes, vs 48 for `reshape`, 80 for `unsafe_wrap`).
    @test allocs == 0
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

@testitem "cholesky!: width-1/2 diagonal-block fast path matches the general trsm!/potrf! path" setup = [LLTHelpers] begin
    # Regression guard for the inline kernel-call bypass (n=64 OB-arm perf fix): a
    # supernode of width 1 or 2 skips potrf!/trsm! entirely and does the arithmetic
    # directly (see llt.jl's "factor diagonal block" / "panel solve" steps). Pin
    # correctness against the dense oracle on matrices specifically chosen to force
    # width-1/2 supernodes, including the EXACT n=64 sweep matrix that originally
    # exposed the OB-arm dip (benchmark/size_sweep.jl's MersenneTwister(2026)
    # sequence, reproduced here so a future refactor can't silently regress the case
    # this fix targets).
    using Random, LinearAlgebra, SparseArrays

    # (a) exact n=64 sweep matrix: 16 supernodes, 9 of them width 1-3 (6 structurally
    # isolated columns with zero fill, per ROADMAP's root-cause writeup).
    A64 = let rng64 = MersenneTwister(2026), A = nothing
        for n in (2, 4, 8, 16, 32, 64)
            A = random_spd_matrix(rng64, n, 0.05)
        end
        A
    end
    F64 = PureSparse.cholesky(A64)
    @test PureSparse.issuccess(F64)
    L64 = dense_L(F64)
    PAP64 = permuted_dense(F64, A64)
    @test relerr(L64 * L64', PAP64) < 1.0e-10

    # (b) block-diag of an isolated width-1 column (diag entry with no off-diagonal
    # fill at all -- the structurally-unmergeable case from the n=64 matrix) with an
    # unrelated 2-column block -- verified via symbolic() to produce widths [1, 2],
    # exercising BOTH fast-path widths in one factorization.
    A1 = sparse([1, 2, 2], [1, 1, 2], [1.0, 0.3, 2.0], 2, 2)
    Aiso = blockdiag(sparse([1.0;;]), A1)
    @assert [s2 - s1 for (s1, s2) in zip(PureSparse.symbolic(Aiso).super, PureSparse.symbolic(Aiso).super[2:end])] == [1, 2]
    Fiso = PureSparse.cholesky(Aiso)
    @test PureSparse.issuccess(Fiso)
    Liso = dense_L(Fiso)
    @test relerr(Liso * Liso', permuted_dense(Fiso, Aiso)) < 1.0e-10

    # (c) failure signaling still correct through both fast-path widths.
    Aneg1 = sparse(Diagonal([-1.0]))
    @test !PureSparse.issuccess(PureSparse.cholesky(Aneg1))
    # width-2, negative SECOND pivot (Schur complement goes negative: d2 = 0.5 - 2^2/4 < 0)
    Aneg2b = sparse([1, 2, 2], [1, 1, 2], [4.0, 2.0, 0.5], 2, 2)
    Fneg2b = PureSparse.cholesky(Aneg2b)
    @test !PureSparse.issuccess(Fneg2b)
    # width-2, negative FIRST pivot
    Aneg2a = sparse([1, 2, 2], [1, 1, 2], [-4.0, 2.0, 5.0], 2, 2)
    Fneg2a = PureSparse.cholesky(Aneg2a)
    @test !PureSparse.issuccess(Fneg2a)
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
