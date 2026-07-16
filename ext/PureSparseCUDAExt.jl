# PureSparse GPU offload (M6) — CUDA/KernelAbstractions weak-dep extension.
# design_gpu.md. Loaded only when a user has BOTH CUDA and KernelAbstractions.
# Core `src/` has zero GPU hooks; this ext defines the GPU factor path and adds methods
# to PureSparse's existing generic entry points (symbolic/cholesky!/…), plus the pure
# device dense kernels (design_gpu.md §3, the pure-primary Option-1 strategy).
#
# Phase 2.1 (this file, first landing): the proven pure-Julia dense kernels only.
# gemm/syrk beat cuBLAS FP64 by 1.14× (benchmark/gpu/ka_gemm.jl, verified on galen) and
# are portable (KernelAbstractions → NVIDIA/AMD/Intel) and generic over the element type.
# The frontier/scheduler/GPUSymbolic/GPUSupernodalFactor land in Phase 2.2+ (after the
# design_gpu.md v2 review → v3).
module PureSparseCUDAExt

using PureSparse: PureSparse
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @localmem, @private, @synchronize, get_backend
using Base.Cartesian: @nexprs

# ---------------------------------------------------------------------------------------
# Pure device GEMM: C = α·A·Bᵀ + β·C   (design_gpu.md §3/§4.1 trailing update)
#   A is M×K, B is N×K, C is M×N, all column-major (device arrays). The supernode
#   trailing update is exactly this shape; C -= A·Bᵀ is (α,β) = (-1, 1).
#   4×4 register micro-tile, 64×64 block, BK=8, 256 threads. The `muladd` is load-bearing:
#   Julia is IEEE-strict and will NOT contract a*b+acc into an FMA without it — omitting it
#   doubles FP64 instruction count and halves throughput (design_gpu.md §0). Generic over T.
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
        @inbounds for t in 1:2                    # 512 tile elems / 256 threads
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
            # β==0 must OVERWRITE (BLAS semantics), not read C — the trailing update scatters
            # into a freshly-allocated, uninitialized cbuf, and 0*NaN=NaN would corrupt it
            # (design.md §4.3 relies on overwrite-at-β=0). Both v2 reviewers flagged this.
            C[gr + 1, gc + 1] = beta == zero(T) ? alpha * acc[i, j] :
                                muladd(alpha, acc[i, j], beta * C[gr + 1, gc + 1])
        end
    end
end

"""
    gpu_gemm_nt!(C, A, B, α, β) -> C

`C = α·A·Bᵀ + β·C` on device (A: M×K, B: N×K, C: M×N; column-major KA-backed arrays).
The supernode trailing update (`C -= A·Bᵀ` is `α,β = -1,1`). Generic over the element type;
portable across KA backends. Beats cuBLAS FP64 1.14× on galen (design_gpu.md §0/§3).
"""
function gpu_gemm_nt!(C, A, B, alpha, beta)
    M, K = size(A); N = size(B, 1)
    @assert size(C) == (M, N) "gpu_gemm_nt!: C is $(size(C)), expected ($M,$N)"
    @assert size(B, 2) == K "gpu_gemm_nt!: A K=$K, B K=$(size(B,2)) mismatch"
    backend = get_backend(C)
    kern = _gemm_nt_4x4!(backend, (16, 16))
    kern(C, A, B, alpha, beta, M, N, K; ndrange = (cld(M, 64) * 16, cld(N, 64) * 16))
    return C
end

"""
    gpu_syrk_nt!(C, A, α, β) -> C

Symmetric rank-k self-update `C = α·A·Aᵀ + β·C` (supernode diagonal-block trailing update,
design_gpu.md §4.1). Phase 2.1: routed through the gemm kernel with `B = A` (computes the
full square; correct but does ~2× the necessary work on the symmetric block). The
triangular-only syrk-shaped variant is the §3 productization delta, tracked for Phase 2.
"""
gpu_syrk_nt!(C, A, alpha, beta) = gpu_gemm_nt!(C, A, A, alpha, beta)

end # module
