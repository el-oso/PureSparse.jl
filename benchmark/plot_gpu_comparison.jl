# Regenerates the M6 GPU multifrontal figures for docs/src/benchmarking.md from the
# ALREADY-SAVED benchmark JSON (CLAUDE.md "Benchmarking": results→JSON first, plots
# regenerate from saved JSON only — this script never re-runs a benchmark).
#
# The GPU warm refactor is device-resident and zero-alloc → its per-sample time is
# near-deterministic, so these are MEDIAN line/bar figures (a violin would be a flat,
# uninformative line — the same reasoning the QR gate figure records). The formal pinned
# SPD+SQD stratum gate (design_gpu.md §8) is a separate, later artifact.
#
# Input:  benchmark/results/gpu_multifrontal_galen.json
# Outputs: docs/src/assets/gpu_ldlt_speedup.png
#          docs/src/assets/gpu_front_kernel.png
#          docs/src/assets/gpu_arena.png
#   julia --project=benchmark benchmark/plot_gpu_comparison.jl
using JSON, Statistics
using StatsPlots
gr()

const RESULTS = joinpath(@__DIR__, "results")
const OUT = joinpath(@__DIR__, "..", "docs", "src", "assets")
isdir(OUT) || mkpath(OUT)
D = JSON.parsefile(joinpath(RESULTS, "gpu_multifrontal_galen.json"))

const C_PURE = "#2a78d6"    # pure Julia (the shipped path)
const C_VEND = "#eb6834"    # cuSOLVER/cuBLAS (reference)
const C_MID1 = "#9db8d6"    # pure intermediate arcs (light)
const C_MID2 = "#6f96c9"
const C_ARENA = "#3fa66a"
const C_TGT = "#8a8a8a"     # target line

# ---- Figure 1: LDLᵀ KKT speedup arc (the headline: pure reaches + beats vendor) ----
let l = D["ldlt_kkt"], s = l["series"]
    x = Float64.(l["H_dim"])
    p = plot(size = (760, 470), legend = :topleft, framestyle = :box,
             xlabel = "KKT size  (H = d³ grid Laplacian)", ylabel = "speedup vs single-thread CPU ldlt!",
             title = "GPU multifrontal LDLᵀ — hybrid factor vs CPU (galen RTX 4070)",
             titlefontsize = 10, xticks = (x, string.(Int.(x), "³")), ylim = (0, 5.6))
    hline!(p, [5.0]; color = C_TGT, ls = :dash, lw = 1.2, label = "5× target")
    plot!(p, x, Float64.(s["vendor (cuSOLVER+cuBLAS)"]); color = C_VEND, lw = 2, marker = :diamond,
          ms = 5, label = "vendor (cuSOLVER+cuBLAS)")
    plot!(p, x, Float64.(s["pure v1 (standalone trsm)"]); color = C_MID1, lw = 1.5, marker = :circle,
          ms = 3, ls = :dot, label = "pure — standalone trsm")
    plot!(p, x, Float64.(s["pure v3 (fused signed-LDL front)"]); color = C_PURE, lw = 2.6, marker = :circle,
          ms = 6, label = "pure — fused signed-LDL front (shipped)")
    # annotate the 44³ crossover
    annotate!(p, x[end], 5.08 + 0.22, text("5.08×", 8, C_PURE, :center))
    annotate!(p, x[end], 5.04 - 0.28, text("5.04×", 7, C_VEND, :center))
    savefig(p, joinpath(OUT, "gpu_ldlt_speedup.png"))
    println("wrote gpu_ldlt_speedup.png")
end

# ---- Figure 2: fused signed-LDL front vs vendor front, per real crown-front shape ----
let f = D["ldlt_front_kernel"]
    labs = ["$(n)×$(b)" for (n, b) in zip(f["shape_nscol"], f["shape_below"])]
    sp = Float64.(f["speedup"])
    cols = [v >= 1 ? C_PURE : C_VEND for v in sp]
    p = bar(labs, sp; color = cols, legend = false, framestyle = :box, size = (760, 440),
            xrotation = 30, ylabel = "pure fused front  /  vendor front  (×)",
            title = "Fused signed-LDL front vs CPU-diag+cuBLAS front, by crown-front shape",
            titlefontsize = 10, bar_width = 0.62)
    hline!(p, [1.0]; color = :black, ls = :dash, lw = 1, label = "")
    for (i, v) in enumerate(sp)
        annotate!(p, i, v + 0.18, text(string(v, "×"), 7, v >= 1 ? C_PURE : C_VEND, :center))
    end
    annotate!(p, 1.6, 6.6, text("flop-weighted aggregate: 4.42×\n(the nscol³ CPU-diag round-trip is removed)",
                                8, :left, "#333333"))
    savefig(p, joinpath(OUT, "gpu_front_kernel.png"))
    println("wrote gpu_front_kernel.png")
end

# ---- Figure 3: bounded arena vs monotonic (memory that unlocked the large KKTs) ----
let a = D["arena"]
    x = Float64.(a["H_dim"])
    mono_gb = Float64.(a["monotonic_words"]) .* 8 ./ 1e9
    bnd_gb = Float64.(a["bounded_words"]) .* 8 ./ 1e9
    p = plot(size = (760, 440), legend = :topleft, framestyle = :box,
             xlabel = "problem size  (H = d³ grid)", ylabel = "update-matrix arena  (GB)",
             title = "Bounded stack-with-compaction arena vs monotonic (3D grid)",
             titlefontsize = 10, xticks = (x, string.(Int.(x), "³")))
    plot!(p, x, mono_gb; color = C_VEND, lw = 2, marker = :diamond, ms = 5, label = "monotonic (Σ all U's)")
    plot!(p, x, bnd_gb; color = C_ARENA, lw = 2.6, marker = :circle, ms = 6, label = "bounded (work slot + stack)")
    for (i, r) in enumerate(Float64.(a["ratio"]))
        annotate!(p, x[i], mono_gb[i] + maximum(mono_gb) * 0.03, text(string(r, "× smaller"), 7, C_ARENA, :center))
    end
    savefig(p, joinpath(OUT, "gpu_arena.png"))
    println("wrote gpu_arena.png")
end

println("GPU figures regenerated into docs/src/assets/")
