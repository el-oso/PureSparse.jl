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
using CUDA
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @localmem, @private, @synchronize, get_backend
using Base.Cartesian: @nexprs

# Host-side frontier partition (design_gpu.md §5.2) — pure, no CUDA dep, CPU-unit-testable.
include("frontier.jl")

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

# =======================================================================================
# Pure device Cholesky potrf('L') + right-lower-transpose trsm (amendment C — portable, no
# cuSOLVER/cuBLAS). Fable-authored, galen-validated ≤3.9e-16 vs the vendor pair at n∈{1..1500}
# incl strided views; competitive-to-winning at crown-supernode sizes (benchmark/gpu/
# pure_potrf_dev.jl). Reuses gpu_gemm_nt! above; adds a triangular-skip lower syrk.

# LOWER-only SYRK C = α·A·Aᵀ + β·C (C n×n, A n×K): the gemm kernel with B = A, plus a uniform
# whole-workgroup skip of tiles strictly above the diagonal (halves flops) + a gr ≥ gc write
# mask so the strict upper triangle is never touched (panel strict-upper diag cells stay 0).
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

# Base (32,32)-workgroup shared-memory outer-product Cholesky for a kb≤32 diag block; sqrt pivot
# → column scale by 1/sqrt(pivot) (FP64 division is slow — do it once) → rank-1 trailing update.
# Non-positive pivot writes the 1-based global column into caller-zeroed 1-elt Int32 `info`
# (first failure only) and continues with NaN (amendment D deferred-devinfo contract).
@kernel unsafe_indices = true function _potrf_base32!(A, kb, info, j0)
    T = eltype(A)
    li = @index(Local, NTuple)
    tx = li[1]; ty = li[2]
    As = @localmem T (32, 32)
    piv = @localmem T (1,)                # 1/sqrt(pivot)
    inb = tx <= kb && ty <= kb
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
# One thread per row of B (rows independent in X·Lᵀ = B); L (lower only) + reciprocal diag in
# shared. Same-thread read-after-write of B[i,k] needs no sync.
@kernel unsafe_indices = true function _trsm_rlt_base!(B, @Const(L), m, kb)
    T = eltype(B)
    li = @index(Local, Linear)
    gi = @index(Group, Linear)
    Ls = @localmem T (64, 64)
    Ld = @localmem T (64,)                # 1/L[j,j]
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

# Pure-KA `B := B·L⁻ᵀ` (CUBLAS.trsm!('R','L','T','N',1,L,B) semantics); L n×n lower non-unit,
# B m×n; both may be strided device views. Blocked left-looking over 64-wide column blocks.
function gpu_trsm_rlt!(B, L)
    m, n = size(B)
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

# Pure-KA in-place lower Cholesky (CUSOLVER.potrf!('L',A) semantics), lower triangle only,
# strided view OK. Two-level blocked right-looking; nb=32 measured fastest at every n on galen.
# `d_info` = caller-zeroed 1-elt Int32 device vec; after sync, nonzero = 1-based first non-pos
# pivot column (factor keeps going, NaNs past it — amendment D).
function gpu_potrf!(A, d_info; nb::Int = 32, col0::Int = 0)
    n = size(A, 1)
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
            pk(D, ib, d_info, col0 + k0 + i0 - 1; ndrange = (32, 32))   # d_info ← global pivot col
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

# ---------------------------------------------------------------------------------------
# GPUSymbolic (design_gpu.md §2.3): the CPU Symbolic + the upward-closed frontier partition
# (§5.2) + pattern arrays uploaded ONCE to device (§4.2 — host keeps the pattern, so refactors
# do 0 pattern H2D). Immutable, shared by reference. Phase 2.2: pattern residency + frontier +
# sizing; the per-cross-edge ir/rs scatter structure and device factor land in 2.3.
struct GPUSymbolic{Ti,VI}
    cpu::PureSparse.Symbolic{Ti}
    on_gpu::Vector{Bool}          # frontier membership per supernode (§5.2)
    gpu_order::Vector{Ti}         # GPU supernodes ascending = finalize order
    boundary::Vector{Ti}          # CPU supernodes with a GPU ancestor (§5.3 persist set)
    frontier_cutoff::Float64
    d_rowind::VI                  # device pattern arrays (uploaded once)
    d_rowind_ptr::VI
    d_super::VI
    d_snode_of::VI
    d_amap::VI                    # assembly map A.nzval-position -> x offset (0 = skip), §4/§9.A
    xlen::Int                     # length of the packed factor storage x (= px[end]-1)
    bytes::NamedTuple             # §5.3 device-memory budget
end

"""
    gpu_symbolic(A; ordering, backend=CUDABackend(), frontier_cutoff) -> GPUSymbolic

Build a device-resident symbolic analysis: CPU `symbolic` + the upward-closed etree frontier
(design_gpu.md §5.2) + a one-time pattern upload. `frontier_cutoff` is the per-supernode
factor+update flop threshold (its calibrated default is a remaining Phase-0 item, §8.3).
Asserts the §10.2 upward-closure invariant. Only `T ∈ {Float32,Float64}` reach here (§1);
callers of other `T` use the CPU path.
"""
function gpu_symbolic(A::PureSparse.SparseArrays.SparseMatrixCSC{T,Ti};
                      ordering, backend = CUDABackend(),
                      frontier_cutoff::Real) where {T,Ti}
    cpu = PureSparse.symbolic(A; ordering)
    ns = cpu.nsuper
    on_gpu = Vector{Bool}(undef, ns)
    frontier_partition!(on_gpu, ns, cpu.super, cpu.sparent, cpu.colcount, Float64(frontier_cutoff))
    frontier_invariant_holds(on_gpu, ns, cpu.rowind, cpu.rowind_ptr, cpu.snode_of) ||
        error("gpu_symbolic: upward-closure invariant violated (design_gpu.md §10.2)")
    gpu_order = Ti[s for s in 1:ns if on_gpu[s]]
    boundary = boundary_supernodes(on_gpu, ns, cpu.rowind, cpu.rowind_ptr, cpu.snode_of)
    bytes = gpu_device_bytes(cpu.super, cpu.rowind_ptr, boundary, cpu.nnzL,
                             cpu.max_extend_rows, sizeof(T))
    # one-time pattern upload (§4.2 — never re-uploaded on refactor)
    d_rowind = CuArray(cpu.rowind)
    d_rowind_ptr = CuArray(cpu.rowind_ptr)
    d_super = CuArray(cpu.super)
    d_snode_of = CuArray(cpu.snode_of)
    d_amap = CuArray(cpu.amap)
    xlen = Int(cpu.px[cpu.nsuper + 1]) - 1        # packed factor storage length
    return GPUSymbolic{Ti,typeof(d_rowind)}(cpu, on_gpu, gpu_order, boundary,
                                            Float64(frontier_cutoff),
                                            d_rowind, d_rowind_ptr, d_super, d_snode_of,
                                            d_amap, xlen, bytes)
end

# ---------------------------------------------------------------------------------------
# Device assembly (design_gpu.md §4): scatter A.nzval into the packed factor storage `dx` via
# `amap` (amap[p] = destination offset, 0 = skip). One thread per nonzero — the parallel form
# of llt.jl's assembly loop (`fill!(x,0); x[amap[p]] = A.nzval[p]`). `dx` must be pre-zeroed
# (structural-fill positions not touched by amap stay 0). The A-value H2D recurs per refactor
# (§7); the amap upload is one-time (§9.A "0 pattern H2D").
@kernel unsafe_indices = true function _assemble!(dx, @Const(nzval), @Const(amap), nnz)
    p = @index(Global)
    if p ≤ nnz
        @inbounds begin
            m = amap[p]
            (m != 0) && (dx[m] = nzval[p])
        end
    end
end

"""
    gpu_assemble!(dx, nzval, amap) -> dx

`dx .= 0; dx[amap[p]] = nzval[p]` for `amap[p] ≠ 0`, on device (design_gpu.md §4).
"""
function gpu_assemble!(dx, nzval, amap)
    fill!(dx, zero(eltype(dx)))
    backend = get_backend(dx)
    n = length(nzval)
    _assemble!(backend, 256)(dx, nzval, amap, n; ndrange = cld(n, 256) * 256)
    return dx
end

# Multifrontal symbolic + CPU numeric (design_gpu.md §M, amendment F) — pure, before the GPU loop.
include("multifrontal.jl")
# Synchronous all-GPU Cholesky numeric loops (left-looking §4 + multifrontal §M).
include("gpu_numeric.jl")

end # module
