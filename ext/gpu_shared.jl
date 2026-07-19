# ==============================  SHARED GPU ENGINE (backend-generic)  ============================
# design_gpu.md (M6) + design_gpu_multibackend.md §B1 (M8). Included by every per-backend extension
# (PureSparseCUDAExt, PureSparseAMDGPUExt, …). The enclosing ext module supplies `using CUDA` /
# `using AMDGPU` (for the device-array type + `_default_backend()`); everything here is written
# against KernelAbstractions + the shim below, so it compiles + runs on any KA backend.
#
# On the shipped `:auto` path the ONLY backend-specific device ops are the two shim helpers
# (`_dev_zeros`, `_dev_upload`) and `_default_backend()` (per-ext); the `:vendor` reference arm
# (cuSOLVER/cuBLAS, guarded by `_vendor_available()`) is the sole CUDA-only exception. Everything
# else on :auto — allocation via `_dev_zeros`, upload via
# `_dev_upload`, `fill!`, `KernelAbstractions.synchronize`, `Array(x)` for D2H — is backend-neutral.
using KernelAbstractions: @kernel, @index, @localmem, @private, @synchronize, get_backend, @atomic
using Base.Cartesian: @nexprs
const KA = KernelAbstractions

# --- backend shim (design_gpu_multibackend.md §B1) — the whole device-alloc/upload surface -------
# CUDA.zeros(T, dims...)  →  _dev_zeros(backend, T, dims...)
# CuArray(host)           →  _dev_upload(backend, host)   (KA has no one-call upload: allocate+copy)
_dev_zeros(backend, ::Type{T}, dims...) where {T} = KA.zeros(backend, T, dims...)
function _dev_upload(backend, host::AbstractArray)
    d = KA.allocate(backend, eltype(host), size(host)...)
    copyto!(d, host)
    return d
end

# The `:vendor`/`frontmode=:vendor` reference arm (§8 gate arm 4) calls cuSOLVER/cuBLAS directly and
# is the ONE exception to "backend-generic": those calls live in gpu_dense.jl + gpu_numeric.jl's vendor
# branches, guarded by `_vendor_available()`. Each per-backend ext MUST define `_vendor_available()`
# exactly once (CUDA → true, others → false) — it is NOT defined here, so a shared-def-plus-override
# (which errors as "method overwriting during precompilation" on Julia 1.12) is impossible by design.

# Host-side frontier partition (design_gpu.md §5.2) — pure, no backend dep, CPU-unit-testable.
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

# --- Pure device dense kernels (potrf/trsm/front) — amendment C portability ---
include("gpu_dense.jl")
include("gpu_ldlt_dense.jl")   # fused signed-LDL front (the LDLᵀ analogue of gpu_front!)
include("gpu_solve.jl")        # level-scheduled batched triangular solve (backend-generic)

# ---------------------------------------------------------------------------------------
# GPUSymbolic (design_gpu.md §2.3): the CPU Symbolic + the upward-closed frontier partition
# (§5.2) + pattern arrays uploaded ONCE to device (§4.2 — host keeps the pattern, so refactors
# do 0 pattern H2D). Immutable, shared by reference.
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
    gpu_symbolic(A; ordering, backend=_default_backend(), frontier_cutoff) -> GPUSymbolic

Build a device-resident symbolic analysis: CPU `symbolic` + the upward-closed etree frontier
(design_gpu.md §5.2) + a one-time pattern upload. `backend` defaults to the enclosing extension's
device (CUDABackend / ROCBackend / …). `frontier_cutoff` is the per-supernode factor+update flop
threshold. Asserts the §10.2 upward-closure invariant. Only `T ∈ {Float32,Float64}` reach here
(§1); callers of other `T` use the CPU path.
"""
function gpu_symbolic(A::PureSparse.SparseArrays.SparseMatrixCSC{T,Ti};
                      ordering, backend = _default_backend(),
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
    d_rowind = _dev_upload(backend, cpu.rowind)
    d_rowind_ptr = _dev_upload(backend, cpu.rowind_ptr)
    d_super = _dev_upload(backend, cpu.super)
    d_snode_of = _dev_upload(backend, cpu.snode_of)
    d_amap = _dev_upload(backend, cpu.amap)
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
# GPU numeric engines (design_gpu.md §4/§M) — backend-generic (shim). See the orientation table at
# the top of gpu_numeric.jl.
include("gpu_numeric.jl")
