# M4 gate, second clause (design.md §10): "M1 perf gate still holds through the dropin
# entry point." Re-measures the M1 wall-time gate (gate.jl's matrices/methodology:
# Chairmarks medians, 30 samples/1.5s cap, evals=1, single BLAS thread, both own-
# ordering and same-permutation arms) with the PureSparse arm entered EXCLUSIVELY via
# `LinearAlgebra.cholesky` under an active drop-in — the interposition layer (kwarg
# translation, `Symmetric` unwrapping/materialization, getproperty overrides) is
# inside the measured cold call, not bypassed.
#
# Why two subprocesses (not one process like gate.jl): with the drop-in active,
# CHOLMOD's `cholesky` methods for real sparse input are dispatch-shadowed by ours
# (dropin.jl's header — ours are strictly more specific), so the CHOLMOD baseline
# cannot be measured in the same process at all. Stage 1 runs the baseline in a
# dropin-INACTIVE env; stage 2 runs the dropin arm in a dropin-ACTIVE env; both get
# identical matrices (gate_matrices() is seeded) on the same locked-clock machine.
# `DROPIN_ACTIVE` is a compile-time Preference (tuning.jl), hence the per-stage temp
# projects — same pattern as test/dropin_tests.jl.
#
# Usage:  julia --project=benchmark benchmark/dropin_gate.jl           # measure + verdict
#         julia --project=benchmark benchmark/dropin_gate.jl report    # verdict from saved JSON
# Writes: benchmark/results/dropin_gate_$(hostname).json, prints the gate verdict.
# (Any project with JSON works for `report`; the measure stages build their own envs.)

const PKGROOT = normpath(joinpath(@__DIR__, ".."))
const INNER = joinpath(@__DIR__, "dropin_gate_inner.jl")
const REPORT_ONLY = length(ARGS) >= 1 && ARGS[1] == "report"

function make_env(active::Bool)
    envdir = mktempdir()
    write(joinpath(envdir, "LocalPreferences.toml"), """
    [PureSparse]
    dropin_active = $active
    """)
    bootstrap = joinpath(envdir, "bootstrap.jl")
    write(bootstrap, """
    using Pkg
    Pkg.develop(path=raw"$PKGROOT")
    Pkg.add(["Chairmarks", "JSON", "SparseArrays", "LinearAlgebra", "Random", "Statistics"])
    Pkg.instantiate()
    """)
    run(`$(Base.julia_cmd()) --project=$envdir $bootstrap`)
    return envdir
end

json_out = joinpath(mkpath(joinpath(@__DIR__, "results")), "dropin_gate_$(gethostname()).json")

if !REPORT_ONLY
    env_off = make_env(false)
    env_on = make_env(true)
    json_base = joinpath(mktempdir(), "baseline.json")
    run(`$(Base.julia_cmd()) --project=$env_off $INNER baseline $json_base`)
    run(`$(Base.julia_cmd()) --project=$env_on $INNER dropin $json_base $json_out`)
end

# ---- verdict ----
using JSON
payload = JSON.parsefile(json_out)
results = payload["results"]
println("\n=== M4 dropin-entry-point wall-time gate — warm numeric refactor, median seconds ===")
println(rpad("matrix", 22), rpad("arm", 10), rpad("dropin cold", 13), rpad("CHOLMOD cold", 14),
    rpad("PS warm", 12), rpad("CHOLMOD warm", 14), rpad("overhead", 10), "gate")
npass = 0; ntotal = 0
for r in results
    for arm in ("own", "sameperm")
        global npass, ntotal
        ntotal += 1
        psw = r["ps_warm_$arm"]; chw = r["cholmod_warm_$arm"]
        psc = r["dropin_cold_$arm"]; chc = r["cholmod_cold_$arm"]
        pass = psw < chw
        pass && (npass += 1)
        ov = arm == "own" ? string(round(psc / r["ps_direct_cold_own"]; digits = 3)) * "x" : "-"
        println(rpad(r["matrix"], 22), rpad(arm, 10),
            rpad(string(round(psc * 1000; digits = 4)) * "ms", 13),
            rpad(string(round(chc * 1000; digits = 4)) * "ms", 14),
            rpad(string(round(psw * 1000; digits = 4)) * "ms", 12),
            rpad(string(round(chw * 1000; digits = 4)) * "ms", 14),
            rpad(ov, 10), pass ? "PASS" : "fail")
    end
end
coldpass = count(r["dropin_cold_$arm"] < r["cholmod_cold_$arm"] for r in results, arm in ("own", "sameperm"))
println("\nWarm gate (gate.jl's contractual number): $npass / $ntotal PASS.")
println("Cold, through-dropin entry point: $coldpass / $ntotal faster than CHOLMOD cold.")
println(npass >= cld(ntotal, 2) ? "OVERALL: PASS" : "OVERALL: NOT YET PASSING")
