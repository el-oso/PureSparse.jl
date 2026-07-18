# Regenerates the M6 GPU multifrontal figures for docs/src/benchmarking.md from ALREADY-SAVED
# JSON (CLAUDE.md "Benchmarking": results→JSON first, plots regenerate from saved JSON only —
# never re-runs a benchmark).
#
# The GPU warm refactor+solve is device-resident and near-deterministic, so these are MEDIAN
# line/bar figures (a violin would be a flat line — the same reasoning the QR gate figure records).
#
# Inputs:  benchmark/results/gpu_gate_galen.json        (§8 gate: pure/vendor/CHOLMOD factor+solve)
#          benchmark/results/gpu_chol_sweep_galen.json  (Cholesky front pure-vs-cuSOLVER by shape)
#          benchmark/results/gpu_multifrontal_galen.json (bounded-arena sizes)
# Outputs: docs/src/assets/gpu_gate_ratios.png, gpu_chol_allsizes.png, gpu_arena.png
#   julia --project=benchmark benchmark/plot_gpu_comparison.jl
using JSON, Statistics
using StatsPlots
gr()

const RES = joinpath(@__DIR__, "results")
const OUT = joinpath(@__DIR__, "..", "docs", "src", "assets")
isdir(OUT) || mkpath(OUT)

const C_VEND = "#eb6834"    # cuSOLVER/cuBLAS
const C_CHM  = "#8a56c2"    # CHOLMOD
const C_PURE = "#2a78d6"    # pure Julia (the shipped path)
const C_TGT  = "#8a8a8a"

# ---- Figure 1: §8 gate — pure-GPU speedup over the vendor-GPU and CHOLMOD, per matrix ----
let G = JSON.parsefile(joinpath(RES, "gpu_gate_galen.json"))["results"]
    R = [r for r in G if haskey(r, "gpu_ps")]
    labs = [replace(r["matrix"], "spd_grid_" => "SPD ", "sqd_kkt_" => "SQD ") * "³" for r in R]
    x = 1:length(R)
    vend = [Float64(r["gpu_vendor"]) / Float64(r["gpu_ps"]) for r in R]         # pure vs vendor
    chm  = [min(Float64(r["cholmod_own"]), Float64(r["cholmod_sameperm"])) / Float64(r["gpu_ps"]) for r in R]
    p = plot(size = (820, 470), framestyle = :box, yscale = :log10, legend = :topleft,
             ylabel = "pure-GPU speedup  (×, log)", xrotation = 35,
             xticks = (x, labs), ylim = (0.8, 80),
             title = "GPU multifrontal factor+solve: pure Julia vs the CUDA vendor + CHOLMOD (galen)",
             titlefontsize = 10)
    hline!(p, [1.0]; color = C_TGT, ls = :dash, lw = 1.2, label = "parity (1.0×)")
    plot!(p, x, vend; color = C_PURE, lw = 2.4, marker = :circle, ms = 6,
          label = "pure vs cuSOLVER/cuBLAS  (≥1.0× at every size)")
    plot!(p, x, chm; color = C_CHM, lw = 2, marker = :diamond, ms = 5, ls = :dot,
          label = "pure vs CHOLMOD+OpenBLAS")
    annotate!(p, x[end], chm[end] * 1.15, text(string(round(chm[end], digits = 0), "×"), 7, C_CHM, :center))
    annotate!(p, x[1], vend[1] * 1.12, text(string(round(vend[1], digits = 2), "×"), 7, C_PURE, :center))
    savefig(p, joinpath(OUT, "gpu_gate_ratios.png"))
    println("wrote gpu_gate_ratios.png")
end

# ---- Figure 2: Cholesky front ≥ 1.0× cuSOLVER at EVERY size (the user's hard requirement) ----
let S = JSON.parsefile(joinpath(RES, "gpu_chol_sweep_galen.json"))
    p = plot(size = (760, 450), framestyle = :box, legend = :topright,
             xlabel = "front width  nscol  (diagonal block dimension)",
             ylabel = "pure gpu_front!  /  cuSOLVER+cuBLAS  (×)",
             title = "Pure Cholesky front beats cuSOLVER+cuBLAS at every size",
             titlefontsize = 10, ylim = (0.8, 2.15))
    hline!(p, [1.0]; color = C_TGT, ls = :dash, lw = 1.2, label = "parity (1.0×)")
    plot!(p, Float64.(S["below_1000_nscol"]), Float64.(S["below_1000"]);
          color = C_PURE, lw = 2.4, marker = :circle, ms = 6, label = "below = 1000 (typical crown panel)")
    plot!(p, Float64.(S["nscol"]), Float64.(S["below_186"]);
          color = "#3fa66a", lw = 2.4, marker = :square, ms = 5, label = "below = 186 (potrf-dominated, hardest)")
    annotate!(p, 1536, 1.14 - 0.07, text("1.14× (worst)", 7, "#3fa66a", :center))
    savefig(p, joinpath(OUT, "gpu_chol_allsizes.png"))
    println("wrote gpu_chol_allsizes.png")
end

# ---- Figure 3: bounded arena vs monotonic (kept) ----
let a = JSON.parsefile(joinpath(RES, "gpu_multifrontal_galen.json"))["arena"]
    x = Float64.(a["H_dim"])
    mono = Float64.(a["monotonic_words"]) .* 8 ./ 1e9; bnd = Float64.(a["bounded_words"]) .* 8 ./ 1e9
    p = plot(size = (760, 440), legend = :topleft, framestyle = :box,
             xlabel = "problem size  (H = d³ grid)", ylabel = "update-matrix arena  (GB)",
             title = "Bounded stack-with-compaction arena vs monotonic (3D grid)", titlefontsize = 10,
             xticks = (x, string.(Int.(x), "³")))
    plot!(p, x, mono; color = C_VEND, lw = 2, marker = :diamond, ms = 5, label = "monotonic (Σ all U's)")
    plot!(p, x, bnd; color = "#3fa66a", lw = 2.6, marker = :circle, ms = 6, label = "bounded (work slot + stack)")
    for (i, r) in enumerate(Float64.(a["ratio"]))
        annotate!(p, x[i], mono[i] + maximum(mono) * 0.03, text(string(r, "× smaller"), 7, "#3fa66a", :center))
    end
    savefig(p, joinpath(OUT, "gpu_arena.png"))
    println("wrote gpu_arena.png")
end
println("GPU figures regenerated into docs/src/assets/")
