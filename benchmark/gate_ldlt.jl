# M2 SQD/LDLT wall-time gate (design.md §9.3/§9.4, M2 task 8 — mirrors gate.jl's M1
# LLᵀ gate exactly, same 3-arm structure, own-ordering + same-permutation arms, same
# CLAUDE.md requirement 2 methodology). CHOLMOD's sparse `ldlt` (via `SparseArrays`)
# is the baseline — verified directly (not assumed) to handle SQD/indefinite input
# without complaint and to share the same `perm=`/`.p` interface as its `cholesky`.
#
# Usage:
#   julia --project=benchmark benchmark/gate_ldlt.jl
#   julia --project=benchmark benchmark/gate_ldlt.jl report

using PureSparse, SparseArrays, LinearAlgebra, Random, Statistics, JSON
using Chairmarks: @be

include(joinpath(@__DIR__, "matrices.jl"))
include(joinpath(@__DIR__, "openblas_backend.jl"))

LinearAlgebra.BLAS.set_num_threads(1)

const SAMPLES = 30
const SECONDS = 1.5

_times(b) = Float64[smp.time for smp in b.samples]
_median_time(b) = median(_times(b))

@noinline _ps_cold(A, ordering, signs) = PureSparse.ldlt(A; signs, ordering)
@noinline _ps_warm!(F, A) = PureSparse.ldlt!(F, A)
@noinline _psob_cold(A, sym, signs) = PureSparseOB.ldlt(sym, A; signs)
@noinline _psob_warm!(F, A) = PureSparseOB.ldlt!(F, A)
@noinline _cholmod_cold(A; perm = nothing) =
    isnothing(perm) ? LinearAlgebra.ldlt(Symmetric(A, :L)) :
    LinearAlgebra.ldlt(Symmetric(A, :L); perm)
@noinline _cholmod_warm!(F, A) = LinearAlgebra.ldlt!(F, Symmetric(A, :L))

function bench_one(label::String, A::SparseMatrixCSC, npos::Int, nneg::Int, arm::String)
    n = size(A, 1)
    signs = vcat(fill(Int8(1), npos), fill(Int8(-1), nneg))
    result = Dict{String,Any}("matrix" => label, "n" => n, "nnz" => nnz(A), "arm" => arm)

    if arm == "own"
        sym_ps = PureSparse.symbolic(A)
        F_cholmod_cold = LinearAlgebra.ldlt(Symmetric(A, :L))
        ps_ordering = PureSparse.AMDOrdering()
        cholmod_perm = nothing
    else # "same-perm": feed each stack the OTHER's own-ordering permutation
        sym_ps_own = PureSparse.symbolic(A)
        F_cholmod_own = LinearAlgebra.ldlt(Symmetric(A, :L))
        ps_ordering = PureSparse.GivenOrdering(Vector{Int}(F_cholmod_own.p))
        cholmod_perm = Vector{Int}(sym_ps_own.perm)
        sym_ps = PureSparse.symbolic(A; ordering = ps_ordering)
        F_cholmod_cold = LinearAlgebra.ldlt(Symmetric(A, :L); perm = cholmod_perm)
    end

    # --- config 1: PureSparse + PureBLAS ---
    b_cold = @be _ps_cold($A, $ps_ordering, $signs) seconds = SECONDS samples = SAMPLES evals = 1
    F1 = PureSparse.ldlt(sym_ps, A; signs)
    b_warm = @be _ps_warm!($F1, $A) seconds = SECONDS samples = SAMPLES evals = 1
    result["ps_pureblas_cold"] = _median_time(b_cold)
    result["ps_pureblas_warm"] = _median_time(b_warm)
    result["nnzL"] = sym_ps.nnzL
    result["n_perturbed"] = F1.stats.n_perturbed

    # --- config 2: PureSparse + OpenBLAS (kernel-attribution) ---
    b_cold2 = @be _psob_cold($A, $sym_ps, $signs) seconds = SECONDS samples = SAMPLES evals = 1
    F2 = PureSparseOB.ldlt(sym_ps, A; signs)
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
    result["nnzL_cholmod"] = nnz(sparse(Fc.LD))   # CHOLMOD only supports :LD extraction for LDLt factors

    return result
end

function run_gate()
    results = Any[]
    for (label, A, npos, nneg) in sqd_gate_matrices()
        for arm in ("own", "same-perm")
            @info "benchmarking" label arm
            push!(results, bench_one(label, A, npos, nneg, arm))
        end
    end

    host = gethostname()
    payload = Dict(
        "host" => host,
        "julia_version" => string(VERSION),
        "results" => results,
    )
    mkpath(joinpath(@__DIR__, "results"))
    outpath = joinpath(@__DIR__, "results", "gate_ldlt_$(host).json")
    open(outpath, "w") do io
        JSON.print(io, payload, 2)
    end
    @info "wrote $outpath"
    return payload
end

function print_verdict(payload)
    results = payload["results"]
    println("\n=== M2 SQD/LDLT wall-time gate (design.md §9.3/§9.4) — warm numeric refactor, median seconds ===")
    println(rpad("matrix", 20), rpad("arm", 10), rpad("PS+PureBLAS", 14), rpad("PS+OpenBLAS", 14),
        rpad("CHOLMOD+OB", 14), rpad("n_pert", 8), "gate(1<3)")
    npass = 0; ntotal = 0
    for r in results
        ntotal += 1
        pass = r["ps_pureblas_warm"] < r["cholmod_openblas_warm"]
        pass && (npass += 1)
        println(
            rpad(r["matrix"], 20), rpad(r["arm"], 10),
            rpad(string(round(r["ps_pureblas_warm"] * 1000; digits = 4)) * "ms", 14),
            rpad(string(round(r["ps_openblas_warm"] * 1000; digits = 4)) * "ms", 14),
            rpad(string(round(r["cholmod_openblas_warm"] * 1000; digits = 4)) * "ms", 14),
            rpad(string(r["n_perturbed"]), 8),
            pass ? "PASS" : "fail",
        )
    end
    println("\n$npass / $ntotal matrix-arm combinations strictly faster than CHOLMOD+OpenBLAS (warm refactor).")
    println(npass >= cld(ntotal, 2) ? "OVERALL: PASS" : "OVERALL: NOT YET PASSING")
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 1 && ARGS[1] == "report"
        host = gethostname()
        payload = JSON.parsefile(joinpath(@__DIR__, "results", "gate_ldlt_$(host).json"))
        print_verdict(payload)
    else
        payload = run_gate()
        print_verdict(payload)
    end
end
