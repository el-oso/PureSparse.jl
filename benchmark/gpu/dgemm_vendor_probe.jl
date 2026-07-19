# M8 Phase-0 probe (design_gpu_multibackend.md §B3 / both reviews' BLOCKER 1): does the pure-KA
# FP64 gemm win port off NVIDIA, or is "pure ≥ vendor" foreclosed by the vendor's FP64 MATRIX CORES?
#
# THE ONE NUMBER THAT SETTLES M8'S PREMISE. Compares the shipped pure kernel `_gemm_nt_4x4!`
# (4×4 register tiles + muladd, IEEE-strict — the M6 gemm) against the VENDOR DGEMM (cuBLAS on
# NVIDIA, rocBLAS on AMD, oneMKL on Intel — reached via LinearAlgebra.mul!) at crown-front shapes.
#
# Interpretation (γ = vendor_ms / pure_ms  =  pure's speedup over the vendor):
#   NVIDIA 4070 (Ada, NO FP64 matrix path): expect γ ≈ 1.1–1.14  → reproduces M6, sanity-checks the probe.
#   AMD CDNA (MI210/300, MFMA-f64):         expect γ ≈ 0.5        → rocBLAS uses matrix cores pure can't;
#                                                                   clause-2 (pure ≥ vendor) FORECLOSED.
#   AMD CDNA, γ ≈ 1.0:                       rocBLAS ISN'T using MFMA here → clause-2 alive, worth pursuing.
#   Intel Max (XMX-f64):                     same logic vs oneMKL.
#
# RUN (auto-detects the backend from whichever GPU package is installed):
#   NVIDIA:  julia --project -e 'using CUDA'    then  julia --project benchmark/gpu/dgemm_vendor_probe.jl
#   AMD:     julia --project -e 'using AMDGPU'   then  julia --project benchmark/gpu/dgemm_vendor_probe.jl
#   Intel:   julia --project -e 'using oneAPI'   then  julia --project benchmark/gpu/dgemm_vendor_probe.jl
# (i.e. just have exactly one of CUDA / AMDGPU / oneAPI in the active project; the script picks it up.)

using KernelAbstractions, LinearAlgebra, Statistics, Printf
using Base.Cartesian: @nexprs
const KA = KernelAbstractions

# ---- backend detection: use whichever GPU package is installed in the active project ----
const BE = if Base.find_package("AMDGPU") !== nothing && (@isdefined(AMDGPU) || (try; @eval(using AMDGPU); true; catch; false; end))
    (name = "AMD / rocBLAS", arr = AMDGPU.ROCArray, sync = AMDGPU.synchronize, dev = "gfx")
elseif Base.find_package("oneAPI") !== nothing && (try; @eval(using oneAPI); true; catch; false; end)
    (name = "Intel / oneMKL", arr = oneAPI.oneArray, sync = oneAPI.synchronize, dev = "xe")
elseif Base.find_package("CUDA") !== nothing && (try; @eval(using CUDA); true; catch; false; end)
    (name = "NVIDIA / cuBLAS", arr = CUDA.CuArray, sync = CUDA.synchronize, dev = "sm")
else
    error("install exactly one of AMDGPU / oneAPI / CUDA in the active project")
end

# ---- the shipped pure kernel: C = α·A·Bᵀ  (A: M×K, B: N×K, C: M×N); 4×4 reg tiles, muladd ----
@kernel unsafe_indices = true function _gemm_nt_4x4!(C, @Const(A), @Const(B), alpha, beta, M, N, K)
    T = eltype(C)
    li = @index(Local, NTuple); gi = @index(Group, NTuple)
    tx = li[1]; ty = li[2]
    tid = (ty - 1) * 16 + (tx - 1)
    br = (gi[1] - 1) * 64; bc = (gi[2] - 1) * 64
    As = @localmem T (64, 8); Bs = @localmem T (64, 8)
    acc = @private T (4, 4)
    @inbounds for i in 1:4, j in 1:4
        acc[i, j] = zero(T)
    end
    k0 = 0; nt = div(K + 7, 8)
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
        gr = br + (tx - 1) * 4 + (i - 1); gc = bc + (ty - 1) * 4 + (j - 1)
        if gr < M && gc < N
            C[gr + 1, gc + 1] = beta == zero(T) ? alpha * acc[i, j] :
                                muladd(alpha, acc[i, j], beta * C[gr + 1, gc + 1])
        end
    end
end

function pure_gemm_nt!(C, A, B)   # C = A·Bᵀ
    M, K = size(A); N = size(B, 1)
    _gemm_nt_4x4!(get_backend(C), (16, 16))(C, A, B, 1.0, 0.0, M, N, K;
        ndrange = (cld(M, 64) * 16, cld(N, 64) * 16))
    return C
end

# ---- warm median wall time (ms) over reps, device-synced ----
function gpu_ms(f, reps)
    f(); BE.sync()
    ts = Float64[]
    for _ in 1:reps
        t0 = time_ns(); f(); BE.sync(); push!(ts, (time_ns() - t0) / 1e6)
    end
    return median(ts)
end

const SHAPES = (512, 1024, 2048, 4096)   # square crown-front gemm  C(m×m) = A(m×m)·Bᵀ(m×m)
const REPS = 50

println("M8 Phase-0: pure-KA FP64 gemm  vs  VENDOR DGEMM  (γ = vendor/pure = pure's speedup)")
println("backend: ", BE.name)
println(rpad("m=n=k", 10), rpad("relerr", 12), rpad("pure ms", 11), rpad("vendor ms", 12), "γ (>1 pure wins)")
results = Dict{String,Any}[]
for m in SHAPES
    A = BE.arr(randn(m, m)); B = BE.arr(randn(m, m))
    Cp = BE.arr(zeros(m, m)); Cv = BE.arr(zeros(m, m))
    pure_gemm_nt!(Cp, A, B)
    mul!(Cv, A, transpose(B))                         # vendor DGEMM (cuBLAS/rocBLAS/oneMKL)
    relerr = norm(Array(Cp) - Array(Cv)) / norm(Array(Cv))
    t_pure = gpu_ms(() -> pure_gemm_nt!(Cp, A, B), REPS)
    t_vend = gpu_ms(() -> mul!(Cv, A, transpose(B)), REPS)
    g = t_vend / t_pure
    push!(results, Dict("m" => m, "relerr" => relerr, "pure_ms" => t_pure, "vendor_ms" => t_vend, "gamma" => g))
    @printf("%-10s %-12.2g %-11.4g %-12.4g %.3f\n", "$m", relerr, t_pure, t_vend, g)
end
gs = [r["gamma"] for r in results]
@printf("\nγ over shapes:  min %.3f  median %.3f  max %.3f\n", minimum(gs), median(gs), maximum(gs))
println("VERDICT: γ≈1.1 → pure wins (M6, no FP64 matrix cores). γ≈0.5 → vendor uses FP64 matrix",
        " cores (MFMA/XMX)\n         the pure KA kernel can't emit → clause-2 (pure ≥ vendor) FORECLOSED on this backend.")

# save datapoints (project rule: results→JSON first)
try
    using JSON
    open(joinpath(@__DIR__, "..", "results", "dgemm_vendor_probe_$(BE.dev).json"), "w") do io
        JSON.print(io, Dict("backend" => BE.name, "reps" => REPS, "results" => results), 2)
    end
    println("wrote results/dgemm_vendor_probe_$(BE.dev).json")
catch
end
