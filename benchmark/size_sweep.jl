# Size sweep (n = 2..2048, powers of 2): CHOLMOD+OpenBLAS vs PureSparse+OpenBLAS vs
# PureSparse+PureBLAS, warm numeric refactor median wall-time. Same 3-arm methodology
# as gate.jl (Chairmarks medians, single-thread pinned) but a continuous size sweep on
# one matrix CLASS (random sparse SPD) rather than the gate's fixed named-matrix set —
# for seeing the crossover behavior (small-n per-call overhead vs large-n compute-bound
# regime), not a pass/fail gate. Own-ordering only (no same-permutation arm — this is a
# size-scaling comparison, not the M1 gate).
#
# Usage:
#   julia --project=benchmark benchmark/size_sweep.jl
#   julia --project=benchmark benchmark/size_sweep.jl report

using PureSparse, SparseArrays, LinearAlgebra, Random, Statistics, JSON
using Chairmarks: @be

include(joinpath(@__DIR__, "matrices.jl"))
include(joinpath(@__DIR__, "openblas_backend.jl"))

LinearAlgebra.BLAS.set_num_threads(1)

const SIZES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
const DENSITY = 0.05
const SAMPLES = 30
const SECONDS = 1.0

_times(b) = Float64[smp.time for smp in b.samples]
_median_time(b) = median(_times(b))

@noinline _ps_warm!(F, A) = PureSparse.cholesky!(F, A)
@noinline _psob_warm!(F, A) = PureSparseOB.cholesky!(F, A)
@noinline _cholmod_warm!(F, A) = LinearAlgebra.cholesky!(F, Symmetric(A, :L))

function bench_size(n::Int; rng::AbstractRNG)
    A = random_spd(n, DENSITY; rng)
    result = Dict{String,Any}("n" => n, "nnz" => nnz(A))

    sym = PureSparse.symbolic(A)
    F1 = PureSparse.cholesky(sym, A)
    b1 = @be _ps_warm!($F1, $A) seconds = SECONDS samples = SAMPLES evals = 1
    result["ps_pureblas_warm"] = _median_time(b1)
    result["nnzL"] = sym.nnzL

    F2 = PureSparseOB.cholesky(sym, A)
    b2 = @be _psob_warm!($F2, $A) seconds = SECONDS samples = SAMPLES evals = 1
    result["ps_openblas_warm"] = _median_time(b2)

    Fc = LinearAlgebra.cholesky(Symmetric(A, :L))
    b3 = @be _cholmod_warm!($Fc, $A) seconds = SECONDS samples = SAMPLES evals = 1
    result["cholmod_openblas_warm"] = _median_time(b3)
    result["nnzL_cholmod"] = nnz(sparse(Fc.L))

    return result
end

function run_sweep()
    rng = MersenneTwister(2026)
    results = [(@info "benchmarking" n; bench_size(n; rng)) for n in SIZES]
    host = gethostname()
    payload = Dict("host" => host, "julia_version" => string(VERSION),
        "density" => DENSITY, "results" => results)
    mkpath(joinpath(@__DIR__, "results"))
    outpath = joinpath(@__DIR__, "results", "size_sweep_$(host).json")
    open(io -> JSON.print(io, payload, 2), outpath, "w")
    @info "wrote $outpath"
    return payload
end

function print_table(payload)
    println("\n=== Size sweep (density=$(payload["density"]), host=$(payload["host"])) — warm numeric refactor, median ===")
    println(rpad("n", 8), rpad("nnz(A)", 10), rpad("PS+PureBLAS", 14), rpad("PS+OpenBLAS", 14),
        rpad("CHOLMOD+OB", 14), rpad("PB/CHOLMOD", 12), "OB/CHOLMOD")
    for r in payload["results"]
        pb, ob, ch = r["ps_pureblas_warm"], r["ps_openblas_warm"], r["cholmod_openblas_warm"]
        println(
            rpad(string(r["n"]), 8), rpad(string(r["nnz"]), 10),
            rpad(string(round(pb * 1e6; digits = 2)) * "us", 14),
            rpad(string(round(ob * 1e6; digits = 2)) * "us", 14),
            rpad(string(round(ch * 1e6; digits = 2)) * "us", 14),
            rpad(string(round(ch / pb; digits = 2)) * "x", 12),
            string(round(ch / ob; digits = 2)) * "x",
        )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 1 && ARGS[1] == "report"
        host = gethostname()
        payload = JSON.parsefile(joinpath(@__DIR__, "results", "size_sweep_$(host).json"))
        print_table(payload)
    else
        payload = run_sweep()
        print_table(payload)
    end
end
