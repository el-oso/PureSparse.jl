# Regenerates the M5 sparse-QR comparison plots for docs/src/benchmarking.md from the
# ALREADY-SAVED benchmark JSON (CLAUDE.md "Benchmarking": results→JSON first, plots
# regenerate from saved JSON only — this script never re-runs a benchmark).
#
# The gate figure is the D13 gate: PureSparse's WARM `qr!` refactor (zero-allocation,
# the same path M1/M2/M4 gate on) vs SuiteSparseQR's COLD factorization (SPQR exposes no
# analyze-once/refactor split, so its cold is its best case — the comparison is the
# corrected §9.3/D13 methodology). It is a grouped MEDIAN bar chart, not the old
# cold-vs-cold violins: the saved gate JSON stores per-sample vectors only for the COLD
# PureSparse arms (`ps_cold_samples`/`ps_frontal_cold_samples`), NOT for the warm arms
# the D13 gate actually compares — it stores the warm MEDIANS (`ps_warm`,
# `ps_frontal_warm`). A median viz is honest here anyway: the warm path is
# zero-allocation and deterministic, so its per-sample distribution is nearly a point —
# a violin of it would be a flat, uninformative line. SPQR cold is drawn from the median
# of its saved cold samples. Clearly labeled "warm qr! vs SPQR cold (D13), medians".
#
# The flagship 7000×4000 figure is a bar chart (median) — too few affordable samples at
# that scale for a meaningful distribution (faer/SPQR cost ~6-9 s/sample).
#
# Inputs (saved measurement snapshots, neuromancer, clock-locked):
#   benchmark/results/qr_gate_neuromancer.json                      — the 16-matrix-arm D13 gate (16/16 PASS)
#   benchmark/results/faer_vs_puresparse_7000x4000_neuromancer.json — the flagship dense-panel case
# The D13 gate 16/16 verdict is confirmed on BOTH clock-locked hosts (neuromancer AND
# galen — benchmark/results/qr_gate_galen.json); the neuromancer JSON is plotted here.
# Outputs:
#   docs/src/assets/qr_gate_strata.png
#   docs/src/assets/qr_faer_comparison.png
#
# Run as a script (plain generator, no run-guard):
#   julia --project=benchmark benchmark/plot_qr_comparison.jl
using JSON, Statistics
using StatsPlots  # re-exports Plots
gr()

const RESULTS = joinpath(@__DIR__, "results")
const OUT = joinpath(@__DIR__, "..", "docs", "src", "assets")
isdir(OUT) || mkpath(OUT)

# Fixed entity colors, shared across both figures (identity follows the entity):
const C_PS = "#2a78d6"     # PureSparse
const C_FAER = "#eb6834"   # faer
const C_SPQR = "#4a3aa7"   # SuiteSparseQR

# Bars under a log y-axis: draw an explicit rectangle anchored at the axis floor
# (Plots/GR's default zero-baseline bars misrender under yscale=:log10).
function logbar!(p, xs, vals, floor_y; width, color, label)
    shapes = [Shape([x - width / 2, x + width / 2, x + width / 2, x - width / 2],
                    [floor_y, floor_y, v, v]) for (x, v) in zip(xs, vals)]
    plot!(p, shapes; color, linecolor = :white, linewidth = 1, label = [label fill("", 1, length(shapes) - 1)])
    return p
end

logticks(lo, hi) = [10.0^e for e in floor(Int, log10(lo)):ceil(Int, log10(hi)) if lo <= 10.0^e <= hi]
ticklabel(v) = v >= 1 ? string(round(Int, v)) : string(round(v; sigdigits = 1))

# The D13 gate quantity for a row: PureSparse best-of(:column, :frontal) WARM refactor
# median (the same choice qr(A; method=:auto) makes) vs SuiteSparseQR cold median.
ps_best_warm_ms(r) = 1e3 * min(Float64(r["ps_warm"]), Float64(r["ps_frontal_warm"]))
ps_best_warm_arm(r) = Float64(r["ps_warm"]) <= Float64(r["ps_frontal_warm"]) ? ":column" : ":frontal"
spqr_cold_ms(r) = 1e3 * median(Float64.(r["spqr_cold_samples"]))

# ── 1) D13 gate: PureSparse best-of warm qr! vs SuiteSparseQR cold, faceted by stratum ──
gate = JSON.parsefile(joinpath(RESULTS, "qr_gate_neuromancer.json"))
rows = gate["results"]

strata = ["i_singleton", "ii_sparse_R", "iii_flop_rich"]
total_pass = 0
total_k = 0
panels = map(strata) do st
    sr = sort(filter(r -> r["stratum"] == st, rows), by = r -> (r["matrix"], r["arm"]))
    labels = [r["matrix"] * "\n(" * r["arm"] * ")" for r in sr]
    k = length(sr)

    ps = [ps_best_warm_ms(r) for r in sr]
    arms = [ps_best_warm_arm(r) for r in sr]
    sp = [spqr_cold_ms(r) for r in sr]
    npass = count(i -> ps[i] < sp[i], 1:k)
    global total_pass += npass
    global total_k += k

    allvals = vcat(ps, sp)
    lo = 10.0^floor(log10(minimum(allvals)) - 0.05)
    hi = 10.0^(log10(maximum(allvals)) + 0.30)
    yt = logticks(lo, hi)
    p = plot(; yscale = :log10, ylims = (lo, hi), yticks = (yt, ticklabel.(yt)),
        xticks = ((1:k) .* 2 .- 0.5, labels), xlims = (0.3, 2k + 0.7), xrotation = 12,
        legend = (st == strata[1] ? :topleft : false),
        title = "$st  —  $npass/$k PASS", titlefontsize = 10,
        tickfontsize = 6, guidefontsize = 8,
        ylabel = "median time [ms]", grid = :y, gridalpha = 0.15,
        framestyle = :box)
    for i in 1:k
        xp = 2i - 1
        xs = 2i
        logbar!(p, [xp], [ps[i]], lo; width = 0.85, color = C_PS,
            label = (i == 1 ? "PureSparse warm qr! (best of :column/:frontal)" : ""))
        logbar!(p, [xs], [sp[i]], lo; width = 0.85, color = C_SPQR,
            label = (i == 1 ? "SuiteSparseQR cold (stdlib)" : ""))
        # arm that won under the PS bar, ms value above each bar
        annotate!(p, [(xp, ps[i] * 1.20, Plots.text(arms[i], 6, :center, "#2a78d6"))])
        annotate!(p, [(xp, lo * 1.9, Plots.text(string(round(ps[i]; sigdigits = 2)), 6, :center, :white))])
        annotate!(p, [(xs, lo * 1.9, Plots.text(string(round(sp[i]; sigdigits = 2)), 6, :center, :white))])
    end
    p
end
plot(panels...; layout = (3, 1), size = (950, 1150),
    plot_title = "M5 QR gate — warm qr! vs SPQR cold (D13), medians — $total_pass/$total_k PASS " *
                 "(neuromancer + galen, clock-locked)",
    plot_titlefontsize = 10, left_margin = 8Plots.mm, bottom_margin = 4Plots.mm)
savefig(joinpath(OUT, "qr_gate_strata.png"))
println("wrote ", joinpath(OUT, "qr_gate_strata.png"), "  ($total_pass/$total_k PASS)")

# ── 2) Flagship 7000×4000: PureSparse :frontal vs faer vs SuiteSparseQR, per density ──
flag = JSON.parsefile(joinpath(RESULTS, "faer_vs_puresparse_7000x4000_neuromancer.json"))
frows = sort(flag["results"], by = r -> r["density"])
k = length(frows)
labels = [string(round(Int, 100 * r["density"])) * "% density\nnnz = " * string(r["nnz"])
          for r in frows]
# Bar chart (median), not violin, here: at this problem size faer/SPQR cost ~6-9 s/sample —
# even at 10 samples each that's too few for a meaningful density estimate. The gate panel
# above is a median bar chart for a different reason (warm path is deterministic).
series = [("PureSparse :frontal", "ps_frontal", C_PS),
          ("faer", "faer", C_FAER),
          ("SuiteSparseQR", "spqr", C_SPQR)]

allvals = [r[key] for r in frows for (_, key, _) in series]
lo = 10.0^floor(log10(minimum(allvals)) - 0.05)
hi = 10.0^(log10(maximum(allvals)) + 0.30)   # headroom for value labels and legend
yt = logticks(lo, hi)
nsamp = length(frows[1]["ps_frontal_samples"])
p2 = plot(; yscale = :log10, ylims = (lo, hi), yticks = (yt, ticklabel.(yt)),
    xticks = ((1:k) .* 3 .- 1, labels), xlims = (0.3, 3k + 0.7), legend = :topright,
    ylabel = "cold factorize, median of $nsamp samples [s]", grid = :y, gridalpha = 0.15,
    framestyle = :box, tickfontsize = 9,
    title = "sparse QR, 7000×4000, neuromancer (clock-locked) — log scale",
    titlefontsize = 11)
for (i, r) in enumerate(frows)
    for (j, (name, key, col)) in enumerate(series)
        x = 3 * (i - 1) + j
        v = r[key]
        logbar!(p2, [x], [v], lo; width = 0.85, color = col, label = (i == 1 ? name : ""))
        annotate!(p2, [(x, v * 1.10, Plots.text(string(round(v; digits = 2)) * "s", 8, :center))])
    end
end
plot!(p2; size = (820, 520))
savefig(joinpath(OUT, "qr_faer_comparison.png"))
println("wrote ", joinpath(OUT, "qr_faer_comparison.png"))
