# Regenerates the M5 sparse-QR comparison plots for docs/src/benchmarking.md from the
# ALREADY-SAVED benchmark JSON (CLAUDE.md "Benchmarking": results→JSON first, plots
# regenerate from saved JSON only — this script never re-runs a benchmark).
#
# The gate-strata figure uses violin + boxplot overlays (project convention — see
# BlazingPorts.jl's bench/harness.jl), not bars of the median alone: the saved JSON
# stores each config's full raw Chairmarks sample vector (`*_samples` keys), not just
# `_median_time`, specifically so sample-to-sample variance (e.g. grid_ls_70x50's
# documented high variance at this scale, ROADMAP.md) is visible rather than hidden
# behind a single point estimate. The flagship 7000×4000 figure stays a bar chart — see
# the comment at its call site for why (too few affordable samples at that scale for a
# meaningful distribution).
#
# Inputs (saved measurement snapshots from galen, clock-locked, 2026-07-15):
#   benchmark/results/qr_gate_galen.json                      — the 16-matrix-arm stratified gate
#   benchmark/results/faer_vs_puresparse_7000x4000_galen.json — the flagship dense-panel case
# Outputs:
#   docs/src/assets/qr_gate_strata.png
#   docs/src/assets/qr_faer_comparison.png
#
# Run as a script (plain generator, no run-guard):
#   julia --project=benchmark benchmark/plot_qr_comparison.jl
using JSON, Statistics
using StatsPlots  # re-exports Plots; provides violin / boxplot
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

# Draws one violin+boxplot column at x position `x` from raw samples `ys`.
function violincol!(p, x, ys; color, label, width = 0.8)
    violin!(p, fill(x, length(ys)), ys; color, alpha = 0.6, linewidth = 0,
        label, bar_width = width)
    boxplot!(p, fill(x, length(ys)), ys; fillalpha = 0.0, linecolor = :black,
        linewidth = 1.1, outliers = false, bar_width = 0.28 * width, label = "")
    return p
end

logticks(lo, hi) = [10.0^e for e in floor(Int, log10(lo)):ceil(Int, log10(hi)) if lo <= 10.0^e <= hi]
ticklabel(v) = v >= 1 ? string(round(Int, v)) : string(round(v; sigdigits = 1))

# Picks whichever PureSparse arm (:column vs :frontal) has the lower median for this
# row — same "best-of" rule the gate itself uses — and returns that arm's RAW sample
# vector (ms) rather than synthesizing a fake per-sample "best-of" series.
function ps_winner_samples_ms(r)
    col = 1e3 .* Float64.(r["ps_cold_samples"])
    front_ok = haskey(r, "ps_frontal_cold_samples") && r["ps_frontal_cold_samples"] !== nothing
    front_ok || return col, :column
    front = 1e3 .* Float64.(r["ps_frontal_cold_samples"])
    return median(front) < median(col) ? (front, :frontal) : (col, :column)
end

# ── 1) Gate set: PureSparse best-of(:column/:frontal) vs SuiteSparseQR, faceted by stratum ──
gate = JSON.parsefile(joinpath(RESULTS, "qr_gate_galen.json"))
rows = gate["results"]

strata = ["i_singleton", "ii_sparse_R", "iii_flop_rich"]
panels = map(strata) do st
    sr = sort(filter(r -> r["stratum"] == st, rows), by = r -> (r["matrix"], r["arm"]))
    labels = [r["matrix"] * "\n(" * r["arm"] * ")" for r in sr]
    k = length(sr)

    ps_all = [ps_winner_samples_ms(r)[1] for r in sr]
    sp_all = [1e3 .* Float64.(r["spqr_cold_samples"]) for r in sr]
    npass = count(i -> median(ps_all[i]) < median(sp_all[i]), 1:k)

    allvals = reduce(vcat, vcat(ps_all, sp_all))
    lo = 10.0^floor(log10(minimum(allvals)) - 0.05)
    hi = 10.0^(log10(maximum(allvals)) + 0.25)
    yt = logticks(lo, hi)
    p = plot(; yscale = :log10, ylims = (lo, hi), yticks = (yt, ticklabel.(yt)),
        xticks = ((1:k) .* 2 .- 0.5, labels), xlims = (0.3, 2k + 0.7), xrotation = 12,
        legend = (st == strata[1] ? :topleft : false),
        title = "$st  —  $npass/$k pass (by median)", titlefontsize = 10,
        tickfontsize = 6, guidefontsize = 8,
        ylabel = "cold qr(A), 20 samples [ms]", grid = :y, gridalpha = 0.15,
        framestyle = :box)
    for i in 1:k
        violincol!(p, 2i - 1, ps_all[i]; color = C_PS,
            label = (i == 1 ? "PureSparse (best of :column/:frontal)" : ""))
        violincol!(p, 2i, sp_all[i]; color = C_SPQR,
            label = (i == 1 ? "SuiteSparseQR (stdlib)" : ""))
    end
    p
end
plot(panels...; layout = (3, 1), size = (950, 1150),
    plot_title = "M5 QR gate, galen (clock-locked, 2026-07-15) — 20-sample distributions, log scale",
    plot_titlefontsize = 11, left_margin = 8Plots.mm, bottom_margin = 4Plots.mm)
savefig(joinpath(OUT, "qr_gate_strata.png"))
println("wrote ", joinpath(OUT, "qr_gate_strata.png"))

# ── 2) Flagship 7000×4000: PureSparse :frontal vs faer vs SuiteSparseQR, per density ──
flag = JSON.parsefile(joinpath(RESULTS, "faer_vs_puresparse_7000x4000_galen.json"))
frows = sort(flag["results"], by = r -> r["density"])
k = length(frows)
labels = [string(round(Int, 100 * r["density"])) * "% density\nnnz = " * string(r["nnz"])
          for r in frows]
# Bar chart (median), not violin, here: at this problem size faer/SPQR cost 4-6s/sample —
# even at 10 samples each (kept modest deliberately, ~10 min total instead of ~30), that's
# too few for a meaningful density estimate. The gate panel above has real 20-sample
# violins because those matrices are cheap enough to sample deeply.
series = [("PureSparse :frontal", "ps_frontal", C_PS),
          ("faer", "faer", C_FAER),
          ("SuiteSparseQR", "spqr", C_SPQR)]

allvals = [r[key] for r in frows for (_, key, _) in series]
lo = 10.0^floor(log10(minimum(allvals)) - 0.05)
hi = 10.0^(log10(maximum(allvals)) + 0.55)   # headroom for both the value labels and the legend
yt = logticks(lo, hi)
nsamp = length(frows[1]["ps_frontal_samples"])
p2 = plot(; yscale = :log10, ylims = (lo, hi), yticks = (yt, ticklabel.(yt)),
    xticks = ((1:k) .* 3 .- 1, labels), xlims = (0.3, 3k + 0.7), legend = :topright,
    ylabel = "cold factorize, median of $nsamp samples [s]", grid = :y, gridalpha = 0.15,
    framestyle = :box, tickfontsize = 9,
    title = "sparse QR, 7000×4000, galen (clock-locked) — log scale",
    titlefontsize = 11)
for (i, r) in enumerate(frows)
    for (j, (name, key, col)) in enumerate(series)
        x = 3 * (i - 1) + j
        v = r[key]
        logbar!(p2, [x], [v], lo; width = 0.85, color = col, label = (i == 1 ? name : ""))
        annotate!(p2, [(x, v * 1.16, Plots.text(string(round(v; digits = 2)) * "s", 8, :center))])
    end
end
plot!(p2; size = (820, 520))
savefig(joinpath(OUT, "qr_faer_comparison.png"))
println("wrote ", joinpath(OUT, "qr_faer_comparison.png"))
