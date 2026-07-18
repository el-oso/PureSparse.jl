# M7 Phase-0 probe (design_qr_gpu.md §Q2.2 / §Q5 Phase 0): measure r = pure-panel-QR / geqrf.
#
# The whole M7 GPU-QR bet hinges on ONE number: can a pure single-workgroup Householder panel
# factorization stay within ~1.49x of cuSOLVER geqrf on tall skinny crown-front panels? (At a
# trailing-fraction t=0.8, overall >=1.0x needs r <= 1.49 — see §Q0.) This script measures it.
#
# Pure kernel: ONE workgroup owns the m x nb panel in global memory (too tall for shared mem),
# streaming column-by-column: reduction for ||x|| -> reflector -> in-panel WY-apply. This is the
# direct analogue of geqrf's own (serial-over-columns) panel kernel — the "fair fight" of §Q2.2.
#
# Run on galen:  ~/.juliaup/bin/julia --project=. qr_panel_phase0.jl
using CUDA, KernelAbstractions, LinearAlgebra, Statistics, JSON
const KA = KernelAbstractions

# ---- pure single-workgroup Householder panel QR (LAPACK dlarfg convention, v[1]=1) ----
@kernel unsafe_indices = true function _panel_qr!(A, tau, m, nb)
    T = eltype(A)
    li = @index(Local, Linear)
    WG = @uniform prod(@groupsize())
    red = @localmem T (256,)
    sh  = @localmem T (2,)
    @inbounds for j in 1:nb
        # 1. sum of squares of x = A[j:m, j]
        s = zero(T)
        p = li
        while p <= m - j + 1
            v = A[j + p - 1, j]
            s = muladd(v, v, s)
            p += WG
        end
        red[li] = s
        @synchronize
        d = WG >> 1
        while d >= 1
            (li <= d) && (red[li] += red[li + d])
            @synchronize
            d >>= 1
        end
        # 2. thread 1 forms the reflector: beta (R diag), tau, tail scale
        if li == 1
            ajj = A[j, j]
            nrm = sqrt(red[1])
            beta = ajj >= zero(T) ? -nrm : nrm
            tau[j] = (beta - ajj) / beta
            sh[1] = one(T) / (ajj - beta)     # scale for v tail so v[1]=1
            sh[2] = beta
            A[j, j] = beta
        end
        @synchronize
        scale = sh[1]
        # 3. scale the tail rows j+1..m -> Householder vector v (v[1]=1 implicit)
        p = li
        while p <= m - j
            A[j + p, j] *= scale
            p += WG
        end
        @synchronize
        # 4. apply H = I - tau v vᵀ to trailing columns k=j+1..nb
        for k in j+1:nb
            s2 = zero(T)
            p = li
            while p <= m - j
                s2 = muladd(A[j + p, j], A[j + p, k], s2)
                p += WG
            end
            red[li] = s2
            @synchronize
            d = WG >> 1
            while d >= 1
                (li <= d) && (red[li] += red[li + d])
                @synchronize
                d >>= 1
            end
            if li == 1
                sh[1] = tau[j] * (A[j, k] + red[1])   # w = tau*(vᵀ A[:,k]),  v[1]=1
            end
            @synchronize
            w = sh[1]
            (li == 1) && (A[j, k] -= w)
            p = li
            while p <= m - j
                A[j + p, k] = muladd(-w, A[j + p, j], A[j + p, k])
                p += WG
            end
            @synchronize
        end
    end
end

function panel_qr!(A::CuMatrix{T}, tau::CuVector{T}; wg = 256) where {T}
    m, nb = size(A)
    _panel_qr!(get_backend(A), (wg,))(A, tau, m, nb; ndrange = (wg,))
    KA.synchronize(get_backend(A))
    return A, tau
end

# ---- correctness: rebuild Q from (v,tau) on CPU, check ||A0 - Q*R|| ----
function reconstruct_resid(A0, Afac, tau, m, nb)
    R = triu(Afac[1:nb, 1:nb])
    Q = Matrix{Float64}(I, m, nb)          # thin Q columns
    Qfull = Matrix{Float64}(I, m, m)
    for j in nb:-1:1
        v = zeros(m); v[j] = 1.0
        v[j+1:m] = Afac[j+1:m, j]
        Qfull = (I - tau[j] * v * v') * Qfull
    end
    QR = Qfull[:, 1:nb] * R
    return norm(QR - A0) / norm(A0)
end

# ---- median GPU time (ms) over reps, warm ----
function gpu_ms(f, reps)
    f(); CUDA.synchronize()
    ts = Float64[]
    for _ in 1:reps
        push!(ts, CUDA.@elapsed f())
    end
    return median(ts) * 1e3
end

if abspath(PROGRAM_FILE) == @__FILE__  # main block only when run directly (not on include)
const SHAPES = [(m, nb) for m in (512, 1024, 2048, 4096, 8192) for nb in (32, 48, 64)]
const REPS = 200

println("M7 Phase-0: pure single-workgroup panel-QR  vs  cuSOLVER geqrf  (r = pure/geqrf)")
println("GPU: ", CUDA.name(CUDA.device()))
println(rpad("m x nb", 14), rpad("resid", 12), rpad("pure ms", 12), rpad("geqrf ms", 12), "r = pure/geqrf")
results = Dict{String,Any}[]
for (m, nb) in SHAPES
    A0 = randn(m, nb)
    # pure
    Ap = CuMatrix(copy(A0)); taup = CUDA.zeros(Float64, nb)
    panel_qr!(Ap, taup)
    resid = reconstruct_resid(A0, Array(Ap), Array(taup), m, nb)
    t_pure = gpu_ms(() -> (copyto!(Ap, A0); panel_qr!(Ap, taup)), REPS)
    # geqrf (vendor) — copy back each rep so it factors the same input
    Ag = CuMatrix(copy(A0))
    t_geqrf = gpu_ms(() -> (copyto!(Ag, A0); CUDA.CUSOLVER.geqrf!(Ag)), REPS)
    r = t_pure / t_geqrf
    push!(results, Dict("m" => m, "nb" => nb, "resid" => resid,
                        "pure_ms" => t_pure, "geqrf_ms" => t_geqrf, "r" => r))
    println(rpad("$(m)x$(nb)", 14), rpad(round(resid, sigdigits = 2), 12),
            rpad(round(t_pure, sigdigits = 4), 12), rpad(round(t_geqrf, sigdigits = 4), 12),
            round(r, digits = 3))
end
rs = [x["r"] for x in results]
println("\nr: min ", round(minimum(rs), digits = 3), "  median ", round(median(rs), digits = 3),
        "  max ", round(maximum(rs), digits = 3),
        "   (need r <= 1.49 for overall >=1.0x at t=0.8)")
open("qr_panel_phase0.json", "w") do io
    JSON.print(io, Dict("gpu" => CUDA.name(CUDA.device()), "reps" => REPS, "results" => results), 2)
end
println("wrote qr_panel_phase0.json")
end  # main block guard
