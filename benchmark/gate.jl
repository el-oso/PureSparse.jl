# M1 wall-time gate (design.md §9.3, CLAUDE.md requirement 2 — non-negotiable). Chairmarks
# medians, single-thread pinned. 3 of the design's 4 configs (config 4, CHOLMOD+PureBLAS, is
# N/A per design §9.3 D1 — see openblas_backend.jl's header and CLAUDE.md's benchmarking
# section for why):
#
#   1. PureSparse + PureBLAS   (primary)
#   2. PureSparse + OpenBLAS   (kernel-attribution arm, via PureSparseOB — see openblas_backend.jl)
#   3. CHOLMOD (SparseArrays) + OpenBLAS   (baseline)
#
# Both own-ordering (each stack's own AMD) and same-permutation (CHOLMOD fed PureSparse's
# AMD perm and vice versa, via `perm=`/`GivenOrdering`) arms are measured — both are part of
# the gate (design §9.3 D2), not just the flagship own-ordering number.
#
# Usage:
#   julia --project=benchmark benchmark/gate.jl            # measure + save + print gate verdict
#   julia --project=benchmark benchmark/gate.jl report      # print verdict from the last saved JSON only
#
# For a methodologically-valid (locked-clock) run: `sudo ../PureBLAS.jl/bench/fleet_freqlock.sh lock`
# first (PureBLAS's own frequency-locking script — see PureBLAS.jl/CLAUDE.md's benchmarking
# section; this repo does not duplicate that script). An unlocked run still produces real
# numbers, just noisier — the JSON records whether the run claims to be locked.

using PureSparse, SparseArrays, LinearAlgebra, Random, Statistics, JSON, Dates
using Chairmarks: @be

include(joinpath(@__DIR__, "matrices.jl"))
include(joinpath(@__DIR__, "openblas_backend.jl"))

LinearAlgebra.BLAS.set_num_threads(1)

const SAMPLES = 30
const SECONDS = 1.5

_times(b) = Float64[smp.time for smp in b.samples]
_median_time(b) = median(_times(b))

@noinline _ps_cold(A, ordering) = PureSparse.cholesky(A; ordering)
@noinline _ps_warm!(F, A) = PureSparse.cholesky!(F, A)
@noinline _psob_cold(A, sym) = PureSparseOB.cholesky(sym, A)
@noinline _psob_warm!(F, A) = PureSparseOB.cholesky!(F, A)
@noinline _cholmod_cold(A; perm = nothing) =
    isnothing(perm) ? LinearAlgebra.cholesky(Symmetric(A, :L)) :
    LinearAlgebra.cholesky(Symmetric(A, :L); perm)
@noinline _cholmod_warm!(F, A) = LinearAlgebra.cholesky!(F, Symmetric(A, :L))

# One (label, A) gate matrix, one ordering arm ("own" or "same-perm"): measure cold
# (symbolic+numeric) and warm (numeric refactor, the IPM-relevant number) for all 3 configs.
function bench_one(label::String, A::SparseMatrixCSC, arm::String)
    n = size(A, 1)
    result = Dict{String,Any}("matrix" => label, "n" => n, "nnz" => nnz(A), "arm" => arm)

    if arm == "own"
        sym_ps = PureSparse.symbolic(A)
        F_cholmod_cold = LinearAlgebra.cholesky(Symmetric(A, :L))
        ps_ordering = PureSparse.AMDOrdering()
        cholmod_perm = nothing
    else # "same-perm": feed each stack the OTHER's own-ordering permutation
        sym_ps_own = PureSparse.symbolic(A)
        F_cholmod_own = LinearAlgebra.cholesky(Symmetric(A, :L))
        ps_ordering = PureSparse.GivenOrdering(Vector{Int}(F_cholmod_own.p))
        cholmod_perm = Vector{Int}(sym_ps_own.perm)
        sym_ps = PureSparse.symbolic(A; ordering = ps_ordering)
        F_cholmod_cold = LinearAlgebra.cholesky(Symmetric(A, :L); perm = cholmod_perm)
    end

    # --- config 1: PureSparse + PureBLAS ---
    b_cold = @be _ps_cold($A, $ps_ordering) seconds = SECONDS samples = SAMPLES evals = 1
    F1 = PureSparse.cholesky(sym_ps, A)
    b_warm = @be _ps_warm!($F1, $A) seconds = SECONDS samples = SAMPLES evals = 1
    result["ps_pureblas_cold"] = _median_time(b_cold)
    result["ps_pureblas_warm"] = _median_time(b_warm)
    result["nnzL"] = sym_ps.nnzL

    # --- config 2: PureSparse + OpenBLAS (kernel-attribution) ---
    b_cold2 = @be _psob_cold($A, $sym_ps) seconds = SECONDS samples = SAMPLES evals = 1
    F2 = PureSparseOB.cholesky(sym_ps, A)
    b_warm2 = @be _psob_warm!($F2, $A) seconds = SECONDS samples = SAMPLES evals = 1
    result["ps_openblas_cold"] = _median_time(b_cold2)
    result["ps_openblas_warm"] = _median_time(b_warm2)

    # --- config 3: CHOLMOD + OpenBLAS (baseline) ---
    b_cold3 = arm == "own" ?
        (@be _cholmod_cold($A) seconds = SECONDS samples = SAMPLES evals = 1) :
        (@be _cholmod_cold($A; perm = $cholmod_perm) seconds = SECONDS samples = SAMPLES evals = 1)
    Fc = F_cholmod_cold
    b_warm3 = @be _cholmod_warm!($Fc, $A) seconds = SECONDS samples = SAMPLES evals = 1
    result["cholmod_openblas_cold"] = _median_time(b_cold3)
    result["cholmod_openblas_warm"] = _median_time(b_warm3)
    result["nnzL_cholmod"] = nnz(sparse(Fc.L))

    return result
end

function run_gate()
    results = Any[]
    for (label, A) in gate_matrices()
        for arm in ("own", "same-perm")
            @info "benchmarking" label arm
            push!(results, bench_one(label, A, arm))
        end
    end

    host = gethostname()
    payload = Dict(
        "host" => host,
        "julia_version" => string(VERSION),
        "results" => results,
    )
    mkpath(joinpath(@__DIR__, "results"))
    outpath = joinpath(@__DIR__, "results", "gate_$(host).json")
    open(outpath, "w") do io
        JSON.print(io, payload, 2)
    end
    @info "wrote $outpath"
    return payload
end

function print_verdict(payload)
    results = payload["results"]
    println("\n=== M1 wall-time gate (design.md §9.3) — warm numeric refactor, median seconds ===")
    println(rpad("matrix", 22), rpad("arm", 10), rpad("PS+PureBLAS", 14), rpad("PS+OpenBLAS", 14),
        rpad("CHOLMOD+OB", 14), "gate(1<3)")
    npass = 0; ntotal = 0
    for r in results
        ntotal += 1
        pass = r["ps_pureblas_warm"] < r["cholmod_openblas_warm"]
        pass && (npass += 1)
        println(
            rpad(r["matrix"], 22), rpad(r["arm"], 10),
            rpad(string(round(r["ps_pureblas_warm"] * 1000; digits = 4)) * "ms", 14),
            rpad(string(round(r["ps_openblas_warm"] * 1000; digits = 4)) * "ms", 14),
            rpad(string(round(r["cholmod_openblas_warm"] * 1000; digits = 4)) * "ms", 14),
            pass ? "PASS" : "fail",
        )
    end
    println("\n$npass / $ntotal matrix-arm combinations strictly faster than CHOLMOD+OpenBLAS (warm refactor).")
    println("Design §10 M1 gate requires: strictly faster on at least half the set, both arms included.")
    println(npass >= cld(ntotal, 2) ? "OVERALL: PASS" : "OVERALL: NOT YET PASSING")
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 1 && ARGS[1] == "report"
        host = gethostname()
        payload = JSON.parsefile(joinpath(@__DIR__, "results", "gate_$(host).json"))
        print_verdict(payload)
    else
        payload = run_gate()
        print_verdict(payload)
    end
end
