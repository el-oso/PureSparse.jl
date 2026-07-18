# M7 Phase-0c probe (design_qr_gpu v3 §R3.2 / Opus review BLOCKER-1): measure γ, the batched
# WY-apply gemm ratio = cublasDgemmStridedBatched / pure-KA-batched, on the ACTUAL trailing shapes.
#
# Opus's verdict: the whole GPU-QR win (pure TSQR *and* the vendor-hybrid floor) hinges on γ, NOT
# the panel. M6 measured 1.14x only on SINGLE nt gemms; the WY-apply needs BATCHED tn (W=VᵀC, K=rb,
# tiny nb×n_trail output — cuBLAS's strong split-K regime) + BATCHED nn (C-=V·M, K=nb). If γ≈1.0 the
# front budget inverts and neither pure nor hybrid beats geqrf. This settles it before any TSQR code.
#
# The pure kernels below are the M6 _gemm_nt_4x4! structure (4×4 register tiles + muladd, IEEE-strict)
# adapted to (a) 3D batch index and (b) the tn/nn transpose-load access patterns. A NAIVE pure kernel
# would give a falsely low γ — this must be the optimized structure for the ratio to be honest.
#
# Run on galen:  ~/.juliaup/bin/julia --project=. qr_gamma_phase0.jl
using CUDA, KernelAbstractions, LinearAlgebra, Statistics, JSON
using Base.Cartesian: @nexprs
const KA = KernelAbstractions

# ---- batched tn:  W[:,:,p] = Vᵀ[:,:,p] · C[:,:,p]   (V: K×M, C: K×N, W: M×N; contract first index K)
@kernel unsafe_indices = true function _gemm_tn_batched!(W, @Const(V), @Const(C), M, N, K)
    T = eltype(W)
    li = @index(Local, NTuple); gi = @index(Group, NTuple)
    tx = li[1]; ty = li[2]; p = gi[3]
    tid = (ty - 1) * 16 + (tx - 1)
    br = (gi[1] - 1) * 64; bc = (gi[2] - 1) * 64
    As = @localmem T (64, 8); Bs = @localmem T (64, 8)
    acc = @private T (4, 4)
    @inbounds for i in 1:4, j in 1:4
        acc[i, j] = zero(T)
    end
    k0 = 0; ntile = div(K + 7, 8)
    for _ in 1:ntile
        @inbounds for t in 1:2
            q = tid + (t - 1) * 256
            ml = q & 63; kl = q >> 6
            gk = k0 + kl                       # contracted index (rows of V and C)
            gr = br + ml                       # output row  = col of V
            As[ml + 1, kl + 1] = (gr < M && gk < K) ? V[gk + 1, gr + 1, p] : zero(T)   # V[k, i]
            gc = bc + ml                       # output col  = col of C
            Bs[ml + 1, kl + 1] = (gc < N && gk < K) ? C[gk + 1, gc + 1, p] : zero(T) # C[k, j]
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
        (gr < M && gc < N) && (W[gr + 1, gc + 1, p] = acc[i, j])
    end
end

# ---- batched nn:  Y[:,:,p] = V[:,:,p] · M2[:,:,p]   (V: R×K, M2: K×N, Y: R×N; contract K=nb)
@kernel unsafe_indices = true function _gemm_nn_batched!(Y, @Const(V), @Const(M2), R, N, K)
    T = eltype(Y)
    li = @index(Local, NTuple); gi = @index(Group, NTuple)
    tx = li[1]; ty = li[2]; p = gi[3]
    tid = (ty - 1) * 16 + (tx - 1)
    br = (gi[1] - 1) * 64; bc = (gi[2] - 1) * 64
    As = @localmem T (64, 8); Bs = @localmem T (64, 8)
    acc = @private T (4, 4)
    @inbounds for i in 1:4, j in 1:4
        acc[i, j] = zero(T)
    end
    k0 = 0; ntile = div(K + 7, 8)
    for _ in 1:ntile
        @inbounds for t in 1:2
            q = tid + (t - 1) * 256
            ml = q & 63; kl = q >> 6
            gk = k0 + kl
            gr = br + ml
            As[ml + 1, kl + 1] = (gr < R && gk < K) ? V[gr + 1, gk + 1, p] : zero(T)  # V[r, k]
            gc = bc + ml
            Bs[ml + 1, kl + 1] = (gc < N && gk < K) ? M2[gk + 1, gc + 1, p] : zero(T) # M2[k, j]
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
        (gr < R && gc < N) && (Y[gr + 1, gc + 1, p] = acc[i, j])
    end
end

pure_tn!(W, V, C, M, N, K, P) = (_gemm_tn_batched!(get_backend(W), (16, 16, 1))(W, V, C, M, N, K;
    ndrange = (cld(M, 64) * 16, cld(N, 64) * 16, P)); KA.synchronize(get_backend(W)))
pure_nn!(Y, V, M2, R, N, K, P) = (_gemm_nn_batched!(get_backend(Y), (16, 16, 1))(Y, V, M2, R, N, K;
    ndrange = (cld(R, 64) * 16, cld(N, 64) * 16, P)); KA.synchronize(get_backend(Y)))

function gpu_ms(f, reps)
    f(); CUDA.synchronize()
    ts = Float64[]
    for _ in 1:reps
        push!(ts, CUDA.@elapsed f())
    end
    return median(ts) * 1e3
end

const RB = 192; const NB = 32; const REPS = 200
const SHAPES = [(nt, P) for nt in (256, 512, 1024, 2048) for P in (16, 32, 43)]

println("M7 Phase-0c: batched WY-apply gemm  γ = cublasStridedBatched / pure-KA   (rb=$RB, nb=$NB)")
println("GPU: ", CUDA.name(CUDA.device()))
println(rpad("n_trail×P", 12), rpad("shape", 8),
        rpad("errchk", 11), rpad("pure ms", 10), rpad("cublas ms", 11), "γ (cublas/pure)")
results = Dict{String,Any}[]
for (ntr, P) in SHAPES
    # tn:  W = Vᵀ C.  V: rb×nb×P, C: rb×ntr×P, W: nb×ntr×P
    V = CUDA.randn(Float64, RB, NB, P); Ct = CUDA.randn(Float64, RB, ntr, P)
    W = CUDA.zeros(Float64, NB, ntr, P)
    pure_tn!(W, V, Ct, NB, ntr, RB, P)
    Wc = CUDA.zeros(Float64, NB, ntr, P)
    CUDA.CUBLAS.gemm_strided_batched!('T', 'N', 1.0, V, Ct, 0.0, Wc)
    etn = norm(Array(W) - Array(Wc)) / norm(Array(Wc))
    @assert etn < 1e-8 "tn kernel WRONG (rel err $etn) at n_trail=$ntr P=$P — refusing to time garbage"
    tn_pure = gpu_ms(() -> pure_tn!(W, V, Ct, NB, ntr, RB, P), REPS)
    tn_cub  = gpu_ms(() -> CUDA.CUBLAS.gemm_strided_batched!('T', 'N', 1.0, V, Ct, 0.0, Wc), REPS)
    # nn:  Y = V M2.   V: rb×nb×P, M2: nb×ntr×P, Y: rb×ntr×P
    M2 = CUDA.randn(Float64, NB, ntr, P); Y = CUDA.zeros(Float64, RB, ntr, P)
    pure_nn!(Y, V, M2, RB, ntr, NB, P)
    Yc = CUDA.zeros(Float64, RB, ntr, P)
    CUDA.CUBLAS.gemm_strided_batched!('N', 'N', 1.0, V, M2, 0.0, Yc)
    enn = norm(Array(Y) - Array(Yc)) / norm(Array(Yc))
    @assert enn < 1e-8 "nn kernel WRONG (rel err $enn) at n_trail=$ntr P=$P"
    nn_pure = gpu_ms(() -> pure_nn!(Y, V, M2, RB, ntr, NB, P), REPS)
    nn_cub  = gpu_ms(() -> CUDA.CUBLAS.gemm_strided_batched!('N', 'N', 1.0, V, M2, 0.0, Yc), REPS)
    g_tn = tn_cub / tn_pure; g_nn = nn_cub / nn_pure
    g_wy = (tn_cub + nn_cub) / (tn_pure + nn_pure)     # combined WY-apply γ (what the front model uses)
    push!(results, Dict("n_trail" => ntr, "P" => P, "err_tn" => etn, "err_nn" => enn,
        "tn_pure" => tn_pure, "tn_cub" => tn_cub, "nn_pure" => nn_pure, "nn_cub" => nn_cub,
        "gamma_tn" => g_tn, "gamma_nn" => g_nn, "gamma_wy" => g_wy))
    println(rpad("$(ntr)×$(P)", 12), rpad("e=$(round(etn,sigdigits=1))/$(round(enn,sigdigits=1))", 16),
            "γtn=", round(g_tn, digits = 3), "  γnn=", round(g_nn, digits = 3),
            "  γWY=", round(g_wy, digits = 3))
end
gwy = [x["gamma_wy"] for x in results]; gtn = [x["gamma_tn"] for x in results]; gnn = [x["gamma_nn"] for x in results]
println("\nγ_WY (combined, front model uses this):  min ", round(minimum(gwy), digits = 3),
        "  median ", round(median(gwy), digits = 3), "  max ", round(maximum(gwy), digits = 3))
println("γ_tn (tall-K, the hard one):             min ", round(minimum(gtn), digits = 3),
        "  median ", round(median(gtn), digits = 3), "  max ", round(maximum(gtn), digits = 3))
println("γ_nn (small-K):                          min ", round(minimum(gnn), digits = 3),
        "  median ", round(median(gnn), digits = 3), "  max ", round(maximum(gnn), digits = 3))
println("\nInterpretation (Opus BLOCKER-1): γ_WY ≥ ~1.1 → TSQR pure win plausible; γ_WY ≈ 1.0 → budget")
println("inverts, neither pure nor vendor-hybrid beats geqrf per-front; M7 → beat-SPQR-end-to-end only.")
open("qr_gamma_phase0.json", "w") do io
    JSON.print(io, Dict("gpu" => CUDA.name(CUDA.device()), "rb" => RB, "nb" => NB,
        "reps" => REPS, "results" => results), 2)
end
println("wrote qr_gamma_phase0.json")
