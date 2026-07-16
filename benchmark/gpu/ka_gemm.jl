# Final round: generic-T 4x4 kernel (the winner from kernel_v2.jl) as (a) raw CUDA.jl
# and (b) KernelAbstractions.jl, benchmarked vs cuBLAS on the full shape grid in one
# session. Measures the KA-vs-@cuda delta for the portability decision.
using CUDA
using KernelAbstractions
using CUDA.CUBLAS: gemm!
using LinearAlgebra
using Chairmarks
using Printf
import JSON
using Base.Cartesian: @nexprs

# ---------------- raw CUDA.jl, generic over element type ----------------
function gemm_nt_4x4!(C, A, B, alpha, beta, M, N, K)
    T = eltype(C)
    tx = threadIdx().x; ty = threadIdx().y
    tid = (ty - 1) * 16 + (tx - 1)
    br = (blockIdx().x - 1) * 64
    bc = (blockIdx().y - 1) * 64
    As = CuStaticSharedArray(T, (64, 8))
    Bs = CuStaticSharedArray(T, (64, 8))
    @nexprs 4 i -> @nexprs 4 j -> (acc_i_j = zero(T))
    k0 = 0
    while k0 < K
        @nexprs 2 t -> begin
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
        @inbounds for kk in 1:8
            @nexprs 4 i -> (a_i = As[(tx - 1) * 4 + i, kk])
            @nexprs 4 j -> (b_j = Bs[(ty - 1) * 4 + j, kk])
            @nexprs 4 i -> @nexprs 4 j -> (acc_i_j = muladd(a_i, b_j, acc_i_j))
        end
        sync_threads()
        k0 += 8
    end
    @nexprs 4 i -> @nexprs 4 j -> begin
        gr = br + (tx - 1) * 4 + (i - 1)
        gc = bc + (ty - 1) * 4 + (j - 1)
        if gr < M && gc < N
            @inbounds C[gr + 1, gc + 1] = muladd(alpha, acc_i_j, beta * C[gr + 1, gc + 1])
        end
    end
    return nothing
end
function launch_cuda!(C, A, B, alpha, beta)
    M, K = size(A); N = size(B, 1)
    @cuda threads=(16,16) blocks=(cld(M,64), cld(N,64)) gemm_nt_4x4!(C, A, B, alpha, beta, M, N, K)
    return C
end

# ---------------- KernelAbstractions port, same structure ----------------
@kernel unsafe_indices=true function gemm_nt_4x4_ka!(C, @Const(A), @Const(B), alpha, beta, M, N, K)
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
            nl = ml
            gc = bc + nl
            Bs[nl + 1, kl + 1] = (gc < N && gk < K) ? B[gc + 1, gk + 1] : zero(T)
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
            C[gr + 1, gc + 1] = muladd(alpha, acc[i, j], beta * C[gr + 1, gc + 1])
        end
    end
end
function launch_ka!(C, A, B, alpha, beta)
    M, K = size(A); N = size(B, 1)
    backend = KernelAbstractions.get_backend(C)
    kern = gemm_nt_4x4_ka!(backend, (16, 16))
    kern(C, A, B, alpha, beta, M, N, K; ndrange=(cld(M, 64) * 16, cld(N, 64) * 16))
    return C
end

gflops(m, n, k, secs) = 2.0 * m * n * k / secs / 1e9

function bench_grid()
    shapes = [(m, k) for m in (2000, 8000, 20000) for k in (32, 64, 128, 256, 512)]
    rows = []
    for (m, k) in shapes
        n = min(m, 4096)
        T = Float64
        A = CUDA.rand(T, m, k); B = CUDA.rand(T, n, k); C = CUDA.zeros(T, m, n)
        bref = @be CUDA.@sync(gemm!('N', 'T', one(T), A, B, zero(T), C)) seconds=1
        tref = minimum(x.time for x in bref.samples)
        bc_ = @be CUDA.@sync(launch_cuda!(C, A, B, one(T), zero(T))) seconds=1
        tc = minimum(x.time for x in bc_.samples)
        bk_ = @be CUDA.@sync(launch_ka!(C, A, B, one(T), zero(T))) seconds=1
        tk = minimum(x.time for x in bk_.samples)
        gemm!('N', 'T', one(T), A, B, zero(T), C); CUDA.synchronize(); Cref = Array(C)
        launch_cuda!(C, A, B, one(T), zero(T)); CUDA.synchronize(); Cc = Array(C)
        launch_ka!(C, A, B, one(T), zero(T)); CUDA.synchronize(); Ck = Array(C)
        rec = norm(Cc - Cref) / max(norm(Cref), eps())
        rek = norm(Ck - Cref) / max(norm(Cref), eps())
        r = (m=m, n=n, k=k,
             cublas=round(gflops(m,n,k,tref), digits=1),
             cuda4x4=round(gflops(m,n,k,tc), digits=1),
             ka4x4=round(gflops(m,n,k,tk), digits=1),
             cuda_over_cublas=round(tref/tc, digits=3),
             ka_over_cublas=round(tref/tk, digits=3),
             ka_over_cuda=round(tc/tk, digits=3),
             relerr_cuda=rec, relerr_ka=rek)
        push!(rows, r)
        @printf("m=%6d n=%5d k=%4d  cuBLAS=%6.1f  cuda=%6.1f(%.2fx)  ka=%6.1f(%.2fx, %.2fx-vs-cuda)  relerr=%.1e/%.1e\n",
                m, n, k, r.cublas, r.cuda4x4, r.cuda_over_cublas, r.ka4x4, r.ka_over_cublas, r.ka_over_cuda, rec, rek)
        CUDA.unsafe_free!(A); CUDA.unsafe_free!(B); CUDA.unsafe_free!(C)
    end
    return rows
end

function f32_spotcheck()
    T = Float32
    m, n, k = 8000, 4096, 256
    A = CUDA.rand(T, m, k); B = CUDA.rand(T, n, k); C = CUDA.zeros(T, m, n)
    gemm!('N', 'T', one(T), A, B, zero(T), C); CUDA.synchronize(); Cref = Array(C)
    launch_cuda!(C, A, B, one(T), zero(T)); CUDA.synchronize(); Cc = Array(C)
    launch_ka!(C, A, B, one(T), zero(T)); CUDA.synchronize(); Ck = Array(C)
    rec = norm(Cc - Cref) / norm(Cref); rek = norm(Ck - Cref) / norm(Cref)
    b1 = @be CUDA.@sync(gemm!('N', 'T', one(T), A, B, zero(T), C)) seconds=1
    b2 = @be CUDA.@sync(launch_cuda!(C, A, B, one(T), zero(T))) seconds=1
    g1 = gflops(m,n,k, minimum(x.time for x in b1.samples))
    g2 = gflops(m,n,k, minimum(x.time for x in b2.samples))
    @printf("Float32 spot: cuBLAS=%.0f GF  cuda4x4=%.0f GF  relerr cuda=%.1e ka=%.1e\n", g1, g2, rec, rek)
    return (cublas_f32=round(g1, digits=1), cuda4x4_f32=round(g2, digits=1), relerr_cuda=rec, relerr_ka=rek)
end

function main()
    rows = bench_grid()
    f32 = f32_spotcheck()
    out = Dict("grid" => rows, "f32_spotcheck" => Dict(pairs(f32)),
               "gpu" => CUDA.name(CUDA.device()), "clock_locked_mhz" => 1920)
    open(joinpath(homedir(), "Documents/claude/gpu_probe/kernel_tuning_ka_final.json"), "w") do io
        JSON.print(io, out, 2)
    end
    println("wrote kernel_tuning_ka_final.json")
end
main()
