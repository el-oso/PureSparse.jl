# Diagnose why the register-blocked kernel gave zero speedup over naive: register count,
# LOCAL memory (>0 => accumulator spills, the prime suspect), shared mem, occupancy.
using CUDA

const T = Float64
const TILE = 16
const BM = 64; const BN = 64; const BK = 8; const TM = 4; const TN = 4
using Base.Cartesian: @nexprs

function pure_gemm_nt_kernel!(C, A, B, alpha, beta, M, N, K)
    row = (blockIdx().x - 1) * TILE + threadIdx().x
    col = (blockIdx().y - 1) * TILE + threadIdx().y
    As = CuStaticSharedArray(T, (TILE, TILE)); Bs = CuStaticSharedArray(T, (TILE, TILE))
    acc = zero(T); tx = threadIdx().x; ty = threadIdx().y; ntiles = cld(K, TILE)
    for t in 0:ntiles-1
        ka = t*TILE + ty; @inbounds As[tx,ty] = (row<=M && ka<=K) ? A[row,ka] : zero(T)
        kb = t*TILE + tx; @inbounds Bs[tx,ty] = (col<=N && kb<=K) ? B[col,kb] : zero(T)
        sync_threads()
        for kk in 1:TILE; @inbounds acc += As[tx,kk]*Bs[kk,ty]; end
        sync_threads()
    end
    if row<=M && col<=N; @inbounds C[row,col] = alpha*acc + beta*C[row,col]; end
    return nothing
end

function pure_gemm_nt_reg_kernel!(C, A, B, alpha, beta, M, N, K)
    tx = threadIdx().x; ty = threadIdx().y; tid = (ty-1)*16 + (tx-1)
    br = (blockIdx().x-1)*BM; bc = (blockIdx().y-1)*BN
    As = CuStaticSharedArray(T, (BM, BK)); Bs = CuStaticSharedArray(T, (BN, BK))
    @nexprs 4 i -> @nexprs 4 j -> (acc_i_j = zero(T))
    k0 = 0
    while k0 < K
        @nexprs 2 t -> begin
            p = tid + (t-1)*256; ml = p & 63; kl = p >> 6; gr = br+ml; gk = k0+kl
            @inbounds As[ml+1, kl+1] = (gr<M && gk<K) ? A[gr+1, gk+1] : zero(T)
        end
        @nexprs 2 t -> begin
            p = tid + (t-1)*256; nl = p & 63; kl = p >> 6; gc = bc+nl; gk = k0+kl
            @inbounds Bs[nl+1, kl+1] = (gc<N && gk<K) ? B[gc+1, gk+1] : zero(T)
        end
        sync_threads()
        @inbounds for kk in 1:BK
            @nexprs 4 i -> (a_i = As[(tx-1)*TM + i, kk])
            @nexprs 4 j -> (b_j = Bs[(ty-1)*TN + j, kk])
            @nexprs 4 i -> @nexprs 4 j -> (acc_i_j += a_i * b_j)
        end
        sync_threads(); k0 += BK
    end
    @nexprs 4 i -> @nexprs 4 j -> begin
        gr = br + (tx-1)*TM + (i-1); gc = bc + (ty-1)*TN + (j-1)
        if gr<M && gc<N; @inbounds C[gr+1, gc+1] = alpha*acc_i_j + beta*C[gr+1, gc+1]; end
    end
    return nothing
end

M,N,K = 8000, 4096, 256
A = CUDA.rand(T,M,K); B = CUDA.rand(T,N,K); C = CUDA.zeros(T,M,N)
kn = @cuda launch=false pure_gemm_nt_kernel!(C,A,B,one(T),zero(T),M,N,K)
kr = @cuda launch=false pure_gemm_nt_reg_kernel!(C,A,B,one(T),zero(T),M,N,K)
for (nm,k) in (("naive",kn),("reg",kr))
    reg = CUDA.registers(k); mem = CUDA.memory(k)
    println("$nm: registers/thread=$reg  local(spill)=$(mem.local) B  shared=$(mem.shared) B")
    println("   max threads/block for this kernel: ", CUDA.maxthreads(k))
end
