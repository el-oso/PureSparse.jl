# M6 Phase-0 GPU probe (design: Fable M6 review, 2026-07-16). Measurement ONLY — no
# product code, commits to no architecture. Answers the milestone's biggest open question
# (decision D1): can a pure-Julia FP64 gemm/syrk kernel reach ~0.8x cuBLAS on the RTX 4070
# (Ada, FP64 1:64, no FP64 tensor cores)? If yes -> ship pure kernels (keeps generic-T,
# the "Pure Julia Ecosystem" ethos). If no -> escalate the philosophy-vs-perf fork to the
# user with numbers. Also records CUBLAS FP64 rates, kernel-launch latency, and PCIe
# bandwidth (the crossover inputs that replace the underived gpu_flop_threshold=2e9).
#
# Run on galen:  julia --project=~/Documents/claude/gpu_probe phase0_probe.jl
# Results -> benchmark/results/gpu_phase0_<host>.json  (regenerate plots from JSON, never re-run).

using CUDA
using CUDA.CUBLAS: gemm!, syrk!
using LinearAlgebra
using Chairmarks
using Printf

const HAVE_JSON = try; @eval import JSON; true; catch; false; end

const T = Float64

# ---- pure-Julia tiled FP64 gemm: C := A*Bᵀ-style panel update C -= A*Bᵀ is the supernode
# shape (syrk/gemm trailing update). Plain shared-memory tiled kernel — deliberately simple;
# the D1 question is whether *simple pure Julia* saturates the FP64 pipe, not whether we can
# out-engineer cuBLAS. TILE chosen for 4070 (48KB smem, 128 threads/block starting point).
const TILE = 16

function pure_gemm_nt_kernel!(C, A, B, alpha, beta, M, N, K)
    # C[MxN] = alpha*A[MxK]*B[NxK]ᵀ + beta*C   (B stored NxK, we read Bᵀ)
    row = (blockIdx().x - 1) * TILE + threadIdx().x
    col = (blockIdx().y - 1) * TILE + threadIdx().y
    As = CuStaticSharedArray(T, (TILE, TILE))
    Bs = CuStaticSharedArray(T, (TILE, TILE))
    acc = zero(T)
    tx = threadIdx().x; ty = threadIdx().y
    ntiles = cld(K, TILE)
    for t in 0:ntiles-1
        ka = t * TILE + ty
        @inbounds As[tx, ty] = (row <= M && ka <= K) ? A[row, ka] : zero(T)
        kb = t * TILE + tx
        @inbounds Bs[tx, ty] = (col <= N && kb <= K) ? B[col, kb] : zero(T)
        sync_threads()
        for kk in 1:TILE
            @inbounds acc += As[tx, kk] * Bs[kk, ty]
        end
        sync_threads()
    end
    if row <= M && col <= N
        @inbounds C[row, col] = alpha * acc + beta * C[row, col]
    end
    return nothing
end

function pure_gemm_nt!(C, A, B, alpha, beta)
    M, K = size(A); N = size(B, 1)
    threads = (TILE, TILE)
    blocks = (cld(M, TILE), cld(N, TILE))
    @cuda threads=threads blocks=blocks pure_gemm_nt_kernel!(C, A, B, alpha, beta, M, N, K)
    return C
end

# ---- register-blocked variant: each thread computes a TM×TN micro-tile (the standard
# high-performance GEMM structure — reuses each shared-memory load across TM·TN outputs,
# which is what the naive kernel above leaves on the table). This is the honest D1 test:
# "can *well-structured* pure Julia approach cuBLAS FP64", not "can a trivial kernel".
using Base.Cartesian: @nexprs
const BM = 64; const BN = 64; const BK = 8; const TM = 4; const TN = 4  # 16×16 threads

function pure_gemm_nt_reg_kernel!(C, A, B, alpha, beta, M, N, K)
    tx = threadIdx().x; ty = threadIdx().y            # 1..16
    tid = (ty - 1) * 16 + (tx - 1)                    # 0..255
    br = (blockIdx().x - 1) * BM                       # block row origin (0-based)
    bc = (blockIdx().y - 1) * BN                       # block col origin
    As = CuStaticSharedArray(T, (BM, BK))
    Bs = CuStaticSharedArray(T, (BN, BK))
    @nexprs 4 i -> @nexprs 4 j -> (acc_i_j = zero(T))
    k0 = 0
    while k0 < K
        @nexprs 2 t -> begin                          # 512 elems / 256 threads = 2 each
            p = tid + (t - 1) * 256
            ml = p & 63; kl = p >> 6
            gr = br + ml; gk = k0 + kl
            @inbounds As[ml + 1, kl + 1] = (gr < M && gk < K) ? A[gr + 1, gk + 1] : zero(T)
        end
        @nexprs 2 t -> begin
            p = tid + (t - 1) * 256
            nl = p & 63; kl = p >> 6
            gc = bc + nl; gk = k0 + kl
            @inbounds Bs[nl + 1, kl + 1] = (gc < N && gk < K) ? B[gc + 1, gk + 1] : zero(T)
        end
        sync_threads()
        @inbounds for kk in 1:BK
            @nexprs 4 i -> (a_i = As[(tx - 1) * TM + i, kk])
            @nexprs 4 j -> (b_j = Bs[(ty - 1) * TN + j, kk])
            @nexprs 4 i -> @nexprs 4 j -> (acc_i_j += a_i * b_j)
        end
        sync_threads()
        k0 += BK
    end
    @nexprs 4 i -> @nexprs 4 j -> begin
        gr = br + (tx - 1) * TM + (i - 1)
        gc = bc + (ty - 1) * TN + (j - 1)
        if gr < M && gc < N
            @inbounds C[gr + 1, gc + 1] = alpha * acc_i_j + beta * C[gr + 1, gc + 1]
        end
    end
    return nothing
end

function pure_gemm_nt_reg!(C, A, B, alpha, beta)
    M, K = size(A); N = size(B, 1)
    @cuda threads=(16, 16) blocks=(cld(M, BM), cld(N, BN)) pure_gemm_nt_reg_kernel!(C, A, B, alpha, beta, M, N, K)
    return C
end

gflops(m, n, k, secs) = 2.0 * m * n * k / secs / 1e9

function probe_gemm(results)
    # Supernode-relevant panel shapes: tall-skinny (trailing update), K = panel width.
    shapes = [(m, k) for m in (2000, 8000, 20000) for k in (32, 64, 128, 256, 512)]
    rows = []
    for (m, k) in shapes
        N = m                      # square-ish trailing block C[m x m]; use n = min(m, 4096) to bound smem/time
        n = min(m, 4096)
        A = CUDA.rand(T, m, k)
        B = CUDA.rand(T, n, k)
        C = CUDA.zeros(T, m, n)
        # cuBLAS C = A*Bᵀ  (gemm!, 'N','T')
        bref = @be CUDA.@sync(gemm!('N', 'T', one(T), A, B, zero(T), C)) seconds=1
        tref = minimum(x.time for x in bref.samples)
        # pure naive kernel
        bpure = @be CUDA.@sync(pure_gemm_nt!(C, A, B, one(T), zero(T))) seconds=1
        tpure = minimum(x.time for x in bpure.samples)
        # pure register-blocked kernel
        breg = @be CUDA.@sync(pure_gemm_nt_reg!(C, A, B, one(T), zero(T))) seconds=1
        treg = minimum(x.time for x in breg.samples)
        # correctness check (once) — cuBLAS reference vs both pure kernels
        gemm!('N', 'T', one(T), A, B, zero(T), C); CUDA.synchronize(); Cref = Array(C)
        pure_gemm_nt!(C, A, B, one(T), zero(T)); CUDA.synchronize(); Cpure = Array(C)
        pure_gemm_nt_reg!(C, A, B, one(T), zero(T)); CUDA.synchronize(); Creg = Array(C)
        relerr = norm(Cpure - Cref) / max(norm(Cref), eps())
        relerr_reg = norm(Creg - Cref) / max(norm(Cref), eps())
        r = (m=m, n=n, k=k,
             cublas_gflops=round(gflops(m, n, k, tref), digits=1),
             pure_gflops=round(gflops(m, n, k, tpure), digits=1),
             reg_gflops=round(gflops(m, n, k, treg), digits=1),
             pure_over_cublas=round(gflops(m,n,k,tpure)/gflops(m,n,k,tref), digits=3),
             reg_over_cublas=round(gflops(m,n,k,treg)/gflops(m,n,k,tref), digits=3),
             relerr=relerr, relerr_reg=relerr_reg)
        push!(rows, r)
        @printf("gemm m=%6d n=%5d k=%4d  cuBLAS=%7.1f  naive=%7.1f(%.2f)  reg=%7.1f(%.2f) GF  relerr reg=%.1e\n",
                m, n, k, r.cublas_gflops, r.pure_gflops, r.pure_over_cublas, r.reg_gflops, r.reg_over_cublas, relerr_reg)
        CUDA.unsafe_free!(A); CUDA.unsafe_free!(B); CUDA.unsafe_free!(C)
    end
    results["gemm"] = rows
    return rows
end

noop_kernel() = nothing

function probe_launch_latency(results)
    launch() = (@cuda threads=1 blocks=1 noop_kernel(); nothing)
    b = @be CUDA.@sync(launch()) seconds=1
    lat = minimum(x.time for x in b.samples)
    results["launch_latency_us"] = round(lat * 1e6, digits=2)
    @printf("kernel launch latency: %.2f us\n", lat * 1e6)
end

function probe_pcie(results)
    n = 64 * 1024 * 1024 ÷ sizeof(T)   # 64 MB
    bytes = n * sizeof(T)
    d = CUDA.zeros(T, n)
    srcs = Any[("pageable", Array{T}(undef, n))]
    try
        push!(srcs, ("pinned", CUDA.pin(Array{T}(undef, n))))
    catch e
        @warn "pinned-buffer probe skipped" exception=e
    end
    for (nm, src) in srcs
        bh2d = @be CUDA.@sync(copyto!(d, src)) seconds=1
        bd2h = @be CUDA.@sync(copyto!(src, d)) seconds=1
        h2d = bytes / minimum(x.time for x in bh2d.samples) / 1e9
        d2h = bytes / minimum(x.time for x in bd2h.samples) / 1e9
        results["pcie_$(nm)_h2d_GBs"] = round(h2d, digits=1)
        results["pcie_$(nm)_d2h_GBs"] = round(d2h, digits=1)
        @printf("PCIe %-8s  H2D=%.1f GB/s  D2H=%.1f GB/s\n", nm, h2d, d2h)
    end
end

function main()
    CUDA.functional() || error("CUDA not functional on this host")
    results = Dict{String,Any}()
    dev = CUDA.device()
    results["gpu"] = name(dev)
    results["capability"] = string(CUDA.capability(dev))
    results["total_mem_GB"] = round(CUDA.totalmem(dev) / 1e9, digits=1)
    @printf("GPU: %s  cc=%s  mem=%.1f GB\n", results["gpu"], results["capability"], results["total_mem_GB"])
    probe_launch_latency(results)
    probe_pcie(results)
    probe_gemm(results)
    # D1 verdict — on the register-blocked kernel (the proven-fast structure), realistic widths
    ratios = [r.reg_over_cublas for r in results["gemm"] if r.k >= 128]
    med = sort(ratios)[cld(length(ratios), 2)]
    naive_med = let v = [r.pure_over_cublas for r in results["gemm"] if r.k >= 128]; sort(v)[cld(length(v),2)]; end
    results["D1_median_reg_over_cublas_k>=128"] = round(med, digits=3)
    results["D1_median_naive_over_cublas_k>=128"] = round(naive_med, digits=3)
    verdict = med >= 0.8 ? "SHIP PURE" : (med >= 0.65 ? "PROMISING (tune further before deciding)" : "ESCALATE philosophy-vs-perf fork to user")
    results["D1_verdict"] = verdict
    @printf("\n=== D1: median reg-blocked/cuBLAS (k>=128) = %.2f (naive was %.2f) -> %s ===\n", med, naive_med, verdict)
    if HAVE_JSON
        try
            host = gethostname()
            dir = joinpath(@__DIR__, "..", "results"); mkpath(dir)
            open(joinpath(dir, "gpu_phase0_$(host).json"), "w") do io
                JSON.print(io, results, 2)
            end
            println("wrote results JSON")
        catch e
            @warn "JSON dump skipped" exception=e
        end
    else
        @info "JSON not in env; results printed above only"
    end
    return results
end

main()
