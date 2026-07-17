# pure_potrf_dev.jl — DEV SCRATCH (M6): pure-KernelAbstractions device Cholesky potrf('L')
# and right-lower-transpose TRSM (B := B·L⁻ᵀ), the two vendor calls still left in
# ext/gpu_numeric.jl (CUSOLVER.potrf! / CUBLAS.trsm!). Self-contained candidate replacement:
# defines its own kernels (gemm/syrk bodies copied from the proven ext/PureSparseCUDAExt.jl
# kernel), validates vs cuSOLVER/cuBLAS on contiguous blocks AND strided panel views
# (lda ≠ n, the real supernode-panel call shape), prints relerr + CUDA.@elapsed-median timing.
#
# Algorithms (standard textbook, Golub & Van Loan "Matrix Computations" §4.2, blocked
# outer-product Cholesky; clean-room, no vendor/CHOLMOD source consulted):
#   potrf:  blocked right-looking, panel width 32 —
#           [diag base kernel (one workgroup, shared-mem outer-product Cholesky)]
#           → [panel solve A21 := A21·L11⁻ᵀ (base trsm kernel)]
#           → [trailing update A22 -= A21·A21ᵀ, LOWER-only syrk kernel].
#   trsm:   blocked left-looking over 64-wide column blocks of B —
#           [one gemm  B_j -= X_prior·L_slabᵀ]  → [base solve vs the 64×64 diagonal block].
#           Row i of B is independent in X·Lᵀ = B: x[i,j] = (b[i,j] − Σ_{k<j} x[i,k]·L[j,k])/L[j,j],
#           so the base kernel is one thread per row, sequential over the ≤64 block columns.
#
# Zero device-pool allocation per call: drivers only take views + launch kernels; the only
# scratch is @localmem. potrf takes a caller-provided 1-element Int32 device vector `d_info`
# (caller zeroes it; first non-positive pivot writes its 1-based column, factorization
# continues with NaNs — host checks after sync, like the (ok, fail_col) contract in
# gpu_numeric.jl).
#
# Run (galen): julia --project=$HOME/Documents/claude/gpu_probe benchmark/gpu/pure_potrf_dev.jl

using CUDA
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @localmem, @private, @synchronize, get_backend
using Base.Cartesian: @nexprs
using LinearAlgebra
using Random
using Printf
using CUDA.CUSOLVER: potrf!
using CUDA.CUBLAS: trsm!

# =======================================================================================
# Kernels
# =======================================================================================

# Pure device GEMM C = α·A·Bᵀ + β·C — verbatim copy of ext/PureSparseCUDAExt.jl's proven
# _gemm_nt_4x4! (beats cuBLAS FP64 1.14× on galen). Used for the trsm left-looking update.
@kernel unsafe_indices = true function _gemm_nt_4x4!(C, @Const(A), @Const(B), alpha, beta, M, N, K)
    T = eltype(C)
    li = @index(Local, NTuple)
    gi = @index(Group, NTuple)
    tx = li[1]; ty = li[2]
    tid = (ty - 1) * 16 + (tx - 1)
    br = (gi[1] - 1) * 64
    bc = (gi[2] - 1) * 64
    As = @localmem T (64, 8)
    Bs = @localmem T (64, 8)
    acc = @private T (4, 4)
    @inbounds for i in 1:4, j in 1:4
        acc[i, j] = zero(T)
    end
    k0 = 0
    nt = div(K + 7, 8)
    for _ in 1:nt
        @inbounds for t in 1:2
            p = tid + (t - 1) * 256
            ml = p & 63; kl = p >> 6
            gr = br + ml; gk = k0 + kl
            As[ml + 1, kl + 1] = (gr < M && gk < K) ? A[gr + 1, gk + 1] : zero(T)
            gc = bc + ml
            Bs[ml + 1, kl + 1] = (gc < N && gk < K) ? B[gc + 1, gk + 1] : zero(T)
        end
        @synchronize
        @inbounds for kk in 1:8
            @nexprs 4 i -> (a_i = As[(tx - 1) * 4 + i, kk])
            @nexprs 4 j -> (b_j = Bs[(ty - 1) * 4 + j, kk])
            @nexprs 4 i -> @nexprs 4 j -> (acc[i, j] = muladd(a_i, b_j, acc[i, j]))
        end
        @synchronize
        k0 += 8
    end
    @inbounds for i in 1:4, j in 1:4
        gr = br + (tx - 1) * 4 + (i - 1)
        gc = bc + (ty - 1) * 4 + (j - 1)
        if gr < M && gc < N
            C[gr + 1, gc + 1] = beta == zero(T) ? alpha * acc[i, j] :
                                muladd(alpha, acc[i, j], beta * C[gr + 1, gc + 1])
        end
    end
end

function gpu_gemm_nt!(C, A, B, alpha, beta)
    M, K = size(A); N = size(B, 1)
    backend = get_backend(C)
    kern = _gemm_nt_4x4!(backend, (16, 16))
    kern(C, A, B, alpha, beta, M, N, K; ndrange = (cld(M, 64) * 16, cld(N, 64) * 16))
    return C
end

# LOWER-only SYRK C = α·A·Aᵀ + β·C (C n×n, A n×K): the gemm kernel with B = A, plus
# (a) a uniform whole-workgroup skip of tiles strictly above the diagonal (halves the flops —
# this is the triangular-syrk variant flagged as the Phase-2 delta in PureSparseCUDAExt.jl),
# (b) a gr ≥ gc write mask so the strict upper triangle is never touched (the supernode
# panel's strict-upper diag-block cells must stay 0, cf. _scatter_add! in gpu_numeric.jl).
@kernel unsafe_indices = true function _syrk_ln_4x4!(C, @Const(A), alpha, beta, N, K)
    T = eltype(C)
    li = @index(Local, NTuple)
    gi = @index(Group, NTuple)
    tx = li[1]; ty = li[2]
    tid = (ty - 1) * 16 + (tx - 1)
    br = (gi[1] - 1) * 64
    bc = (gi[2] - 1) * 64
    As = @localmem T (64, 8)
    Bs = @localmem T (64, 8)
    acc = @private T (4, 4)
    if br + 64 > bc                       # group-uniform: skip tiles strictly above diagonal
        @inbounds for i in 1:4, j in 1:4
            acc[i, j] = zero(T)
        end
        k0 = 0
        nt = div(K + 7, 8)
        for _ in 1:nt
            @inbounds for t in 1:2
                p = tid + (t - 1) * 256
                ml = p & 63; kl = p >> 6
                gr = br + ml; gk = k0 + kl
                As[ml + 1, kl + 1] = (gr < N && gk < K) ? A[gr + 1, gk + 1] : zero(T)
                gc = bc + ml
                Bs[ml + 1, kl + 1] = (gc < N && gk < K) ? A[gc + 1, gk + 1] : zero(T)
            end
            @synchronize
            @inbounds for kk in 1:8
                @nexprs 4 i -> (a_i = As[(tx - 1) * 4 + i, kk])
                @nexprs 4 j -> (b_j = Bs[(ty - 1) * 4 + j, kk])
                @nexprs 4 i -> @nexprs 4 j -> (acc[i, j] = muladd(a_i, b_j, acc[i, j]))
            end
            @synchronize
            k0 += 8
        end
        @inbounds for i in 1:4, j in 1:4
            gr = br + (tx - 1) * 4 + (i - 1)
            gc = bc + (ty - 1) * 4 + (j - 1)
            if gr < N && gc < N && gr >= gc
                C[gr + 1, gc + 1] = beta == zero(T) ? alpha * acc[i, j] :
                                    muladd(alpha, acc[i, j], beta * C[gr + 1, gc + 1])
            end
        end
    end
end

function gpu_syrk_ln!(C, A, alpha, beta)
    N, K = size(A)
    backend = get_backend(C)
    kern = _syrk_ln_4x4!(backend, (16, 16))
    kern(C, A, alpha, beta, N, K; ndrange = (cld(N, 64) * 16, cld(N, 64) * 16))
    return C
end

# Base Cholesky: factor the kb×kb (kb ≤ 32) leading block of A in place, lower triangle only.
# One (32,32) workgroup; classic shared-memory outer-product Cholesky (G&VL alg 4.2.1):
# per column j — sqrt pivot, scale column j, rank-1 update of the trailing lower triangle.
# All @synchronize points are workgroup-uniform (kb is uniform). Non-positive pivot: writes
# the 1-based global column j0+j-1 into info[1] (first failure only) and continues with NaN —
# the host checks info after sync (LAPACK-style early exit is not worth a device round-trip).
@kernel unsafe_indices = true function _potrf_base32!(A, kb, info, j0)
    T = eltype(A)
    li = @index(Local, NTuple)
    tx = li[1]; ty = li[2]
    As = @localmem T (32, 32)
    piv = @localmem T (1,)                # 1/sqrt(pivot) — FP64 division is slow on GPU,
    inb = tx <= kb && ty <= kb            # do it once and multiply
    @inbounds if inb
        As[tx, ty] = tx >= ty ? A[tx, ty] : zero(T)
    end
    @synchronize
    for j in 1:kb
        if tx == j && ty == j
            @inbounds begin
                d = As[j, j]
                if !(d > zero(T)) && info[1] == Int32(0)
                    info[1] = Int32(j0 + j - 1)
                end
                s = sqrt(d)
                As[j, j] = s
                piv[1] = one(T) / s
            end
        end
        @synchronize
        if ty == j && tx > j && tx <= kb
            @inbounds As[tx, j] *= piv[1]
        end
        @synchronize
        if ty > j && tx >= ty && tx <= kb
            @inbounds As[tx, ty] = muladd(-As[tx, j], As[ty, j], As[tx, ty])
        end
        @synchronize
    end
    @inbounds if inb && tx >= ty
        A[tx, ty] = As[tx, ty]
    end
end

# Base right-lower-transpose TRSM: B := B·L⁻ᵀ for kb ≤ 64, B m×kb, L kb×kb lower (non-unit).
# One thread per row of B (rows are independent in X·Lᵀ = B); L staged in shared (lower only —
# the strict upper of the L view is other panel data and is never read). The inner Σ re-reads
# this thread's own already-written B[i,k] from global: the m×64 block fits in AD104's L2,
# and same-thread read-after-write needs no sync.
# ponytail: fixed (64,64) shared tile (32 KB) even for kb=32 callers — costs some occupancy
# on the potrf panel solve; split a 32-wide variant only if that base solve shows up in Nsight.
@kernel unsafe_indices = true function _trsm_rlt_base!(B, @Const(L), m, kb)
    T = eltype(B)
    li = @index(Local, Linear)
    gi = @index(Group, Linear)
    Ls = @localmem T (64, 64)
    Ld = @localmem T (64,)                # 1/L[j,j] — avoid the slow FP64 division per row·col
    p = li
    @inbounds while p <= kb * kb
        r = (p - 1) % kb + 1
        c = (p - 1) ÷ kb + 1
        Ls[r, c] = r >= c ? L[r, c] : zero(T)
        p += 256
    end
    @inbounds if li <= kb
        Ld[li] = one(T) / L[li, li]
    end
    @synchronize
    i = (gi - 1) * 256 + li
    if i <= m
        @inbounds for j in 1:kb
            s = B[i, j]
            for k in 1:(j - 1)
                s = muladd(-B[i, k], Ls[j, k], s)
            end
            B[i, j] = s * Ld[j]
        end
    end
end

# =======================================================================================
# Drivers (host-blocked; work on strided views — only launch kernels, no device-pool allocs)
# =======================================================================================

"""
    gpu_trsm_rlt!(B, L) -> B

Pure-KA `B := B·L⁻ᵀ` (`CUBLAS.trsm!('R','L','T','N',1,L,B)` semantics): `L` n×n lower
triangular non-unit, `B` m×n; both may be strided device views. Blocked left-looking over
64-wide column blocks: one gemm (B_j -= X_prior·L[jrows,prior]ᵀ) + one base solve per block.
"""
function gpu_trsm_rlt!(B, L)
    m, n = size(B)
    @assert size(L, 1) == n && size(L, 2) == n "gpu_trsm_rlt!: L is $(size(L)), expected ($n,$n)"
    (m == 0 || n == 0) && return B
    T = eltype(B)
    backend = get_backend(B)
    tk = _trsm_rlt_base!(backend, 256)
    nb = 64
    for j0 in 1:nb:n
        jb = min(nb, n - j0 + 1)
        Bj = view(B, :, j0:(j0 + jb - 1))
        if j0 > 1
            gpu_gemm_nt!(Bj, view(B, :, 1:(j0 - 1)), view(L, j0:(j0 + jb - 1), 1:(j0 - 1)),
                         -one(T), one(T))
        end
        tk(Bj, view(L, j0:(j0 + jb - 1), j0:(j0 + jb - 1)), m, jb; ndrange = cld(m, 256) * 256)
    end
    return B
end

"""
    gpu_potrf!(A, d_info; nb=32) -> A

Pure-KA in-place lower Cholesky (`CUSOLVER.potrf!('L',A)` semantics): only the lower
triangle of the n×n device matrix (strided view OK) is read/written. Two-level blocked
right-looking: outer panel width `nb` (diag block factored by an inner 32-wide loop of
base-kernel + base-trsm + small syrk; panel solve via the blocked gpu_trsm_rlt!) then ONE
K=`nb` lower-syrk trailing update per outer panel. MEASURED (galen sweep 32/64/128): nb=32
is fastest at every n — the base-trsm panel-solve flops scale with nb and run latency-bound,
which beats the saved trailing-syrk passes; widening nb is a net loss. Keep nb=32.
`d_info` is a caller-provided, caller-ZEROED 1-element Int32 device vector; after
`CUDA.synchronize()` a non-zero value is the 1-based column of the first non-positive pivot
(the factorization keeps going and produces NaNs past it, mirroring the (ok,fail_col)
host-check contract in gpu_numeric.jl).
"""
function gpu_potrf!(A, d_info; nb::Int = 32)
    n = size(A, 1)
    @assert size(A, 2) == n "gpu_potrf!: A is $(size(A)), expected square"
    n == 0 && return A
    T = eltype(A)
    backend = get_backend(A)
    pk = _potrf_base32!(backend, (32, 32))
    tk = _trsm_rlt_base!(backend, 256)
    for k0 in 1:nb:n
        kb = min(nb, n - k0 + 1)
        Akk = view(A, k0:(k0 + kb - 1), k0:(k0 + kb - 1))
        for i0 in 1:32:kb                  # inner 32-blocked factor of the kb×kb diag block
            ib = min(32, kb - i0 + 1)
            D = view(Akk, i0:(i0 + ib - 1), i0:(i0 + ib - 1))
            pk(D, ib, d_info, k0 + i0 - 1; ndrange = (32, 32))
            q0 = i0 + ib
            if q0 <= kb
                Pn = view(Akk, q0:kb, i0:(i0 + ib - 1))
                tk(Pn, D, kb - q0 + 1, ib; ndrange = cld(kb - q0 + 1, 256) * 256)
                gpu_syrk_ln!(view(Akk, q0:kb, q0:kb), Pn, -one(T), one(T))
            end
        end
        r0 = k0 + kb
        if r0 <= n
            A21 = view(A, r0:n, k0:(k0 + kb - 1))
            gpu_trsm_rlt!(A21, Akk)        # wide panel solve, reuses the blocked trsm
            gpu_syrk_ln!(view(A, r0:n, r0:n), A21, -one(T), one(T))
        end
    end
    return A
end

# =======================================================================================
# Validation + timing vs cuSOLVER / cuBLAS
# =======================================================================================

med(v) = sort(v)[cld(length(v), 2)]

# In-place ops mutate their input, so each rep restores from `src` first and the median
# restore-copy time is subtracted (never time an op on already-factored/NaN data: cuSOLVER
# early-exits on info>0 and would mismeasure).
function time_op!(op!, dst, src; reps = 11)
    copyto!(dst, src); op!(dst); CUDA.synchronize()             # warm (compile tax)
    tc = [Float64(CUDA.@elapsed copyto!(dst, src)) for _ in 1:reps]
    tt = [Float64(CUDA.@elapsed begin
                      copyto!(dst, src); op!(dst)
                  end) for _ in 1:reps]
    return max(med(tt) - med(tc), 0.0)
end

function make_spd(n)
    A = randn(n, n)
    return A * A' + n * I                 # well-conditioned SPD
end

function run_potrf()
    T = Float64
    @printf("\n== potrf('L'): pure blocked-KA vs cuSOLVER (Float64) ==\n")
    @printf("%6s  %10s  %5s  %10s  %10s  %10s  %11s  %8s\n",
            "n", "relerr", "", "nb=32(ms)", "nb=64", "nb=128", "cuSOLVER", "vend/best")
    for n in (1, 3, 16, 64, 200, 500, 1000, 1500)
        S = make_spd(n)
        src = CuArray{T}(S); dP = similar(src); dV = similar(src)
        info = CUDA.zeros(Int32, 1)
        copyto!(dP, src); gpu_potrf!(dP, info); CUDA.synchronize()
        copyto!(dV, src); potrf!('L', dV); CUDA.synchronize()
        Lp = tril(Array(dP)); Lv = tril(Array(dV))
        re = norm(Lp - Lv) / norm(Lv)
        @assert Array(info)[1] == 0
        tps = [time_op!(X -> gpu_potrf!(X, info; nb), dP, src) for nb in (32, 64, 128)]
        tv = time_op!(X -> potrf!('L', X), dV, src)
        @printf("%6d  %10.2e  %5s  %10.3f  %10.3f  %10.3f  %11.3f  %7.2fx\n",
                n, re, re <= 1e-13 ? "PASS" : "FAIL",
                tps[1] * 1e3, tps[2] * 1e3, tps[3] * 1e3, tv * 1e3, tv / minimum(tps))
        foreach(CUDA.unsafe_free!, (src, dP, dV, info))
    end
end

function run_trsm()
    T = Float64
    @printf("\n== trsm R/L/T/N (B := B·L⁻ᵀ): pure blocked-KA vs cuBLAS (Float64) ==\n")
    @printf("%6s %6s  %10s  %5s  %11s  %11s  %8s\n",
            "m", "n", "relerr", "", "pure (ms)", "cuBLAS", "vend/pure")
    for (m, n) in ((48, 16), (192, 64), (2000, 200), (8000, 500), (20000, 1000), (30000, 1500))
        dL = CuArray{T}(make_spd(n)); potrf!('L', dL)
        src = CuArray{T}(randn(m, n)); dP = similar(src); dV = similar(src)
        copyto!(dP, src); gpu_trsm_rlt!(dP, dL); CUDA.synchronize()
        copyto!(dV, src); trsm!('R', 'L', 'T', 'N', one(T), dL, dV); CUDA.synchronize()
        hv = Array(dV)
        re = norm(Array(dP) - hv) / norm(hv)
        tp = time_op!(X -> gpu_trsm_rlt!(X, dL), dP, src)
        tv = time_op!(X -> trsm!('R', 'L', 'T', 'N', one(T), dL, X), dV, src)
        @printf("%6d %6d  %10.2e  %5s  %11.3f  %11.3f  %7.2fx\n",
                m, n, re, re <= 1e-13 ? "PASS" : "FAIL", tp * 1e3, tv * 1e3, tv / tp)
        foreach(CUDA.unsafe_free!, (dL, src, dP, dV))
    end
end

# The real call shape (gpu_numeric.jl step 3/4): P is an nsrow×nscol panel, the diag block
# is view(P,1:n,1:n) (STRIDED, lda = nsrow ≠ n) and B is view(P,n+1:nsrow,1:n). Run the full
# potrf+trsm sequence pure vs vendor on identical panels.
function run_panel()
    T = Float64
    @printf("\n== strided panel views (lda=nsrow≠n): potrf(diag view) + trsm(below view) ==\n")
    @printf("%6s %6s  %10s  %5s  %11s  %11s  %8s\n",
            "nsrow", "nscol", "relerr", "", "pure (ms)", "vendor", "vend/pure")
    for (nsrow, n) in ((48, 16), (200, 200), (300, 200), (4000, 500), (20000, 1000), (24000, 1500))
        P = randn(nsrow, n)
        P[1:n, 1:n] = make_spd(n)
        src = CuArray{T}(P); dP = similar(src); dV = similar(src)
        info = CUDA.zeros(Int32, 1)
        pure! = X -> begin
            dg = view(X, 1:n, 1:n)
            gpu_potrf!(dg, info)
            nsrow > n && gpu_trsm_rlt!(view(X, (n + 1):nsrow, 1:n), dg)
        end
        vend! = X -> begin
            dg = view(X, 1:n, 1:n)
            potrf!('L', dg)
            nsrow > n && trsm!('R', 'L', 'T', 'N', one(T), dg, view(X, (n + 1):nsrow, 1:n))
        end
        copyto!(dP, src); pure!(dP); CUDA.synchronize()
        copyto!(dV, src); vend!(dV); CUDA.synchronize()
        @assert Array(info)[1] == 0
        hp = Array(dP); hv = Array(dV)
        @inbounds for j in 2:n, i in 1:(j - 1)
            hp[i, j] = 0.0; hv[i, j] = 0.0    # strict upper of diag block: untouched by both
        end
        re = norm(hp - hv) / norm(hv)
        tp = time_op!(pure!, dP, src)
        tv = time_op!(vend!, dV, src)
        @printf("%6d %6d  %10.2e  %5s  %11.3f  %11.3f  %7.2fx\n",
                nsrow, n, re, re <= 1e-13 ? "PASS" : "FAIL", tp * 1e3, tv * 1e3, tv / tp)
        foreach(CUDA.unsafe_free!, (src, dP, dV, info))
    end
end

function main()
    Random.seed!(20260717)
    println("GPU: ", CUDA.name(CUDA.device()), "   CUDA ", CUDA.runtime_version())
    run_potrf()
    run_trsm()
    run_panel()
    println("\ndone.")
end

main()
