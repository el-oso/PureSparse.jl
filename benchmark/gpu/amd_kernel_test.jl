# Standalone AMD (ROCm) validation of the PURE device kernels — proves the portability claim.
#
# The kernel files ext/gpu_dense.jl + ext/gpu_ldlt_dense.jl are backend-generic (KernelAbstractions
# + Atomix; no CUDA intrinsics after the genericization). This script includes them under
# AMDGPU's ROCBackend and checks that (1) they COMPILE + RUN on AMD, (2) the factor matches a
# CPU reference at machine precision. The pure gemm kernel is copied verbatim from
# ext/PureSparseCUDAExt.jl (it is already backend-generic; copied only to avoid pulling in the
# CUDA-gated extension module).
#
#   julia --project=/home/el_oso/Documents/claude/amd_probe benchmark/gpu/amd_kernel_test.jl

using AMDGPU, KernelAbstractions, LinearAlgebra, Random, Statistics, Printf
using KernelAbstractions: @kernel, @index, @localmem, @private, @synchronize, get_backend, @atomic
using Base.Cartesian: @nexprs

@assert AMDGPU.functional() "AMDGPU not functional"
const BK = ROCBackend()
roc(A) = ROCArray(A)

# ---- pure gemm C = α·A·Bᵀ + β·C (copied from ext/PureSparseCUDAExt.jl; backend-generic) ----
@kernel unsafe_indices = true function _gemm_nt_4x4!(C, @Const(A), @Const(B), alpha, beta, M, N, K)
    T = eltype(C)
    li = @index(Local, NTuple); gi = @index(Group, NTuple)
    tx = li[1]; ty = li[2]; tid = (ty - 1) * 16 + (tx - 1)
    br = (gi[1] - 1) * 64; bc = (gi[2] - 1) * 64
    As = @localmem T (64, 8); Bs = @localmem T (64, 8); acc = @private T (4, 4)
    @inbounds for i in 1:4, j in 1:4; acc[i, j] = zero(T); end
    k0 = 0; nt = div(K + 7, 8)
    for _ in 1:nt
        @inbounds for t in 1:2
            p = tid + (t - 1) * 256; ml = p & 63; kl = p >> 6
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
        gr = br + (tx - 1) * 4 + (i - 1); gc = bc + (ty - 1) * 4 + (j - 1)
        if gr < M && gc < N
            C[gr + 1, gc + 1] = beta == zero(T) ? alpha * acc[i, j] :
                                muladd(alpha, acc[i, j], beta * C[gr + 1, gc + 1])
        end
    end
end
function gpu_gemm_nt!(C, A, B, alpha, beta)
    M, K = size(A); N = size(B, 1)
    kern = _gemm_nt_4x4!(get_backend(C), (16, 16))
    kern(C, A, B, alpha, beta, M, N, K; ndrange = (cld(M, 64) * 16, cld(N, 64) * 16))
    return C
end
gpu_syrk_nt!(C, A, alpha, beta) = gpu_gemm_nt!(C, A, A, alpha, beta)

# ---- the backend-generic fused-front kernels (the whole point of the port) ----
const EXT = joinpath(@__DIR__, "..", "..", "ext")
include(joinpath(EXT, "gpu_dense.jl"))        # gpu_front!, gpu_potrf!, gpu_trsm_rlt!, FrontWS, syrk_ln
include(joinpath(EXT, "gpu_ldlt_dense.jl"))   # gpu_ldlt_front!, LDLFrontWS

rel(a, b) = norm(a - b) / max(norm(b), eps())
zl(A) = (B = copy(A); for j in axes(B,2), i in 1:(j-1); B[i,j] = 0.0; end; B)   # zero strict upper

# ================= test 1: pure gemm =================
function t_gemm(M, N, K)
    A = randn(M, K); B = randn(N, K); C = zeros(M, N)
    dC = roc(C); gpu_gemm_nt!(dC, roc(A), roc(B), 1.0, 0.0); KernelAbstractions.synchronize(BK)
    r = rel(Array(dC), A * B')
    @printf("  gemm %d×%d×%d   relerr=%.2e  %s\n", M, N, K, r, r < 1e-12 ? "OK" : "FAIL")
    return r < 1e-12
end

# ================= test 2: fused Cholesky front =================
function t_front(nscol, below; mode = :auto)
    rng = MersenneTwister(1); nsrow = nscol + below
    Mm = randn(rng, nscol, nscol); P11 = Mm' * Mm + nscol * I         # SPD
    A21 = randn(rng, below, nscol)
    P = zeros(nsrow, nscol); P[1:nscol, 1:nscol] = P11; P[(nscol+1):nsrow, 1:nscol] = A21
    L11 = Matrix(cholesky(Symmetric(P11, :L)).L)                       # CPU reference
    L21 = A21 / L11'
    ws = FrontWS(BK, Float64, cld(nscol, 64))
    dP = roc(P); gpu_front!(dP, nscol, ws; mode); KernelAbstractions.synchronize(BK)
    H = Array(dP)
    rL = rel(zl(H[1:nscol, 1:nscol]), zl(L11)); rP = below > 0 ? rel(H[(nscol+1):nsrow, 1:nscol], L21) : 0.0
    @printf("  front(%s) %d×%d   relL11=%.2e relL21=%.2e  %s\n", mode, nscol, below, rL, rP,
            (rL < 1e-10 && rP < 1e-10) ? "OK" : "FAIL")
    return rL < 1e-10 && rP < 1e-10
end

# ================= test 3: fused signed-LDL front =================
# CPU signed-LDL reference (the cpu_multifrontal_ldlt! loop, dense) + the kernel's exact
# inertia classification (pre-perturbation dj, running post-perturbation dmax)
function cpu_ldl!(P, signs, delta, zeta)
    nsrow, nscol = size(P); dvals = zeros(nscol); dmax = 0.0
    np = 0; nn = 0; nz = 0
    for j in 1:nscol
        dj = P[j, j]; adj = abs(dj); sg = signs[j]
        if adj <= zeta * max(dmax, delta); nz += 1
        elseif dj > 0; np += 1
        else; nn += 1; end
        wrong = (sg == 1 && !(dj > 0)) || (sg == -1 && !(dj < 0))
        if wrong || adj < delta
            target = sg == 0 ? (signbit(dj) ? -1.0 : 1.0) : Float64(sg); dj = target * max(delta, adj)
        end
        dvals[j] = dj; dmax = max(dmax, abs(dj)); P[j, j] = 1.0; invd = 1 / dj
        for i in (j+1):nsrow; P[i, j] *= invd; end
        for c in (j+1):nscol, r in (j+1):nsrow; P[r, c] -= dj * P[r, j] * P[c, j]; end
    end
    return dvals, (np, nn, nz)
end
function t_ldl_front(nscol, below; mode = :auto)
    rng = MersenneTwister(2); nsrow = nscol + below; n1 = cld(nscol, 2)
    Mm = randn(rng, nscol, nscol); K11 = Mm' * Mm + nscol * I
    for i in (n1+1):nscol; K11[i, i] = -K11[i, i]; end                # make some pivots negative (SQD-ish)
    A21 = randn(rng, below, nscol)
    P = zeros(nsrow, nscol); P[1:nscol, 1:nscol] = K11; P[(nscol+1):nsrow, 1:nscol] = A21
    signs = Int8[i ≤ n1 ? 1 : -1 for i in 1:nscol]
    delta = 1e-13 * maximum(abs, K11); zeta = eps()
    Pref = copy(P); dref, iref = cpu_ldl!(Pref, signs, delta, zeta)
    ws = LDLFrontWS(BK, Float64)
    dP = roc(P); dv = KernelAbstractions.zeros(BK, Float64, nscol)
    gpu_ldlt_front!(dP, nscol, roc(signs), dv, delta, zeta, ws; mode); KernelAbstractions.synchronize(BK)
    H = Array(dP)
    st = Array(ws.stats); igpu = (Int(st[1]), Int(st[2]), Int(st[3]))
    # note: gpu front folds D⁻¹ into L21 (unit L21), reference does not — compare only L11 + D
    rL = rel(zl(H[1:nscol, 1:nscol]), zl(Pref[1:nscol, 1:nscol])); rD = rel(Array(dv), dref)
    ok = rL < 1e-9 && rD < 1e-9 && igpu == iref
    @printf("  ldl-front(%s) %d×%d   relL11=%.2e relD=%.2e  inertia gpu=%s ref=%s  %s\n",
            mode, nscol, below, rL, rD, igpu, iref, ok ? "OK" : "FAIL")
    return ok
end

println("AMDGPU: ", AMDGPU.functional(), "  device: ", AMDGPU.device())
println("\n[1] pure gemm");        ok1 = all([t_gemm(256, 256, 128), t_gemm(1000, 500, 300)])
println("[2] fused Cholesky front")
ok2 = all([t_front(55, 754; mode = :fused3), t_front(234, 1436; mode = :fused3),
           t_front(300, 2000; mode = :fused3), t_front(64, 0; mode = :fused3),
           t_front(55, 754; mode = :fused2), t_front(234, 1436; mode = :fused2),
           t_front(300, 2000; mode = :fused2),
           t_front(55, 754; mode = :fused), t_front(234, 1436; mode = :fused),
           t_front(300, 2000; mode = :fused)])
println("[3] fused signed-LDL front")
ok3 = all([t_ldl_front(55, 754; mode = :fused2), t_ldl_front(234, 1436; mode = :fused2),
           t_ldl_front(55, 754; mode = :fused), t_ldl_front(234, 1436; mode = :fused)])
println("\n", (ok1 && ok2 && ok3) ? "ALL PURE KERNELS RUN + MATCH ON AMD ✓" : "SOME FAILED — see above")
