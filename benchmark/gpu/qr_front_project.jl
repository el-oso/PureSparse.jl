# M7 Phase-0 part 2 (design_qr_gpu.md §Q2 / §Q0): the FRONT-level ratio, not the panel-in-isolation.
#
# Probe 1 (qr_panel_phase0.jl) showed the pure single-workgroup panel is 3-10x slower than geqrf
# STANDALONE. But the panel is only ~2*NB/n_f of a wide front's flops; the trailing WY-apply (which
# we own at ~1.14x cuBLAS, the M6 result) dominates. The design question is the FRONT ratio:
#
#   pure_front  ≈  Σ_panels pure_panel(m_p, NB)  +  (geqrf_front − Σ_panels geqrf_panel(m_p, NB)) / 1.14
#                  \_________ our slow panel _________/    \_ geqrf's trailing, we do it 1.14x faster _/
#   ratio = pure_front / geqrf_front           (>=... we want <= 1.0 for clause-1 of §Q4)
#
# Every term is measured fresh here (geqrf full front, geqrf per-panel down the staircase, pure
# per-panel); only the 1.14x trailing advantage is assumed (established for gpu_gemm_nt! in M6).
# This is CONSERVATIVE to the pure path in two ways: (a) it charges pure the SAME trailing as geqrf
# then only divides by 1.14 (real owned gemm may do better/worse — measured later), (b) it ignores
# that a smaller NB shrinks the panel penalty (NB is a free tunable, swept here).
#
# Run on galen:  ~/.juliaup/bin/julia --project=. qr_front_project.jl
using CUDA, KernelAbstractions, LinearAlgebra, Statistics, JSON
include("qr_panel_phase0.jl")   # brings _panel_qr!, panel_qr!, gpu_ms  (its main block also runs;
                                #   harmless — reuses the same GPU. If you want to skip it, guard it.)

# geqrf time (ms), warm median, factoring a fresh m x n each rep
function geqrf_ms(m, n, reps)
    A0 = randn(m, n); A = CuMatrix(A0)
    gpu_ms(() -> (copyto!(A, A0); CUDA.CUSOLVER.geqrf!(A)), reps)
end
function purepanel_ms(m, nb, reps)
    A0 = randn(m, nb); A = CuMatrix(A0); tau = CUDA.zeros(Float64, nb)
    gpu_ms(() -> (copyto!(A, A0); panel_qr!(A, tau)), reps)
end

const CROWN = [(2048, 512), (2048, 1024), (4096, 1024), (4096, 2048), (8192, 2048), (8192, 4096)]
const NBS = (32, 48, 64)
const TRAIL_ADV = 1.14      # our owned trailing gemm vs cuBLAS (M6)
const REPS2 = 100

println("\n\n=== M7 Phase-0 part 2: FRONT-level projected ratio (pure / geqrf) ===")
println("GPU: ", CUDA.name(CUDA.device()), "   (trailing advantage assumed ", TRAIL_ADV, "x)")
println(rpad("front", 14), rpad("NB", 5), rpad("geqrf ms", 11), rpad("ΣpurePan", 11),
        rpad("ΣgeqrfPan", 11), rpad("proj pure", 11), "front ratio")
out = Dict{String,Any}[]
for (m, n) in CROWN
    tgf = geqrf_ms(m, n, REPS2)
    for nb in NBS
        spure = 0.0; sgeqrf = 0.0
        c = 1
        while c <= n
            mp = m - c + 1
            w = min(nb, n - c + 1)
            spure  += purepanel_ms(mp, w, REPS2)
            sgeqrf += geqrf_ms(mp, w, REPS2)
            c += nb
        end
        proj = spure + max(tgf - sgeqrf, 0.0) / TRAIL_ADV
        ratio = proj / tgf
        push!(out, Dict("m" => m, "n" => n, "nb" => nb, "geqrf_front_ms" => tgf,
                        "sum_pure_panel_ms" => spure, "sum_geqrf_panel_ms" => sgeqrf,
                        "proj_pure_ms" => proj, "ratio" => ratio))
        println(rpad("$(m)x$(n)", 14), rpad(nb, 5), rpad(round(tgf, sigdigits = 4), 11),
                rpad(round(spure, sigdigits = 4), 11), rpad(round(sgeqrf, sigdigits = 4), 11),
                rpad(round(proj, sigdigits = 4), 11), round(ratio, digits = 3))
    end
end
best = Dict{Tuple{Int,Int},Float64}()
for r in out
    k = (r["m"], r["n"]); v = r["ratio"]
    best[k] = haskey(best, k) ? min(best[k], v) : v
end
println("\nbest-NB front ratio per shape (pure/geqrf, <1.0 = pure wins):")
for (k, v) in sort(collect(best))
    println("  ", rpad("$(k[1])x$(k[2])", 12), round(v, digits = 3))
end
vals = collect(values(best))
println("\nacross crown shapes @ best NB:  min ", round(minimum(vals), digits = 3),
        "  median ", round(median(vals), digits = 3), "  max ", round(maximum(vals), digits = 3))
open("qr_front_project.json", "w") do io
    JSON.print(io, Dict("gpu" => CUDA.name(CUDA.device()), "trail_adv" => TRAIL_ADV,
                        "results" => out, "best_per_shape" => [Dict("m"=>k[1],"n"=>k[2],"ratio"=>v) for (k,v) in best]), 2)
end
println("wrote qr_front_project.json")
