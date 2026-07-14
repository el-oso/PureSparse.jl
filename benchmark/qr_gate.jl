# M5 sparse QR wall-time gate (design_qr.md §9.3, CLAUDE.md requirement 2 —
# non-negotiable). Chairmarks medians, single-thread pinned. Configs (design_qr.md §9.3
# table; QR needs no OpenBLAS kernel-attribution arm the way Cholesky's gate does — M5a's
# left-looking Householder loop calls no BLAS-3 kernel at all, only PureBLAS's `nrm2`
# level-1 op, design_qr.md's own "M5a needs no new PureBLAS kernels" scoping note):
#
#   1. PureSparse QR (COLAMDOrdering, the §2.1 stated default)      [primary — the gate]
#   2. SuiteSparseQR via SparseArrays.qr (stock)                    [baseline — the gate]
#   3. PureSparse cholesky(AᵀA) normal equations                    [context arm, §1.2]
#
# Both own-ordering and same-permutation arms are measured (design.md D2 discipline,
# design_qr.md §9.2/§9.3) — SPQR has no direct "give me this exact permutation" kwarg from
# Julia (`ordering=ORDERING_FIXED` means "use A's column order as-is", so a given
# permutation is fed by pre-permuting A's columns and using ORDERING_FIXED; PureSparse's
# `GivenOrdering` is the existing, designed-for-this mechanism on its side).
#
# The GATE itself is COLD-vs-COLD only (design_qr.md §9.3: "stdlib exposes no analyze-
# once/refactorize path at all — our warm qr! numbers are reported... but not gated").
# Warm `qr!` is additionally measured wherever `sym.n1==0` naturally holds (strata ii/iii)
# using a `singletons=false`-forced initial factor everywhere, for a uniform reported
# number even on stratum (i) matrices (informative only, matches design's own "the
# IPM/NLLS-relevant numbers" framing of the warm path).
#
# Usage:
#   julia --project=benchmark benchmark/qr_gate.jl            # measure + save + print verdict
#   julia --project=benchmark benchmark/qr_gate.jl report      # print verdict from saved JSON only
#
# For a methodologically-valid (locked-clock) run: `sudo ../PureBLAS.jl/bench/fleet_freqlock.sh
# lock` first (see benchmark/gate.jl's own header for the same note) — this repo does not
# duplicate that script.

using PureSparse, SparseArrays, LinearAlgebra, Random, Statistics, JSON, Dates
using Chairmarks: @be

include(joinpath(@__DIR__, "qr_matrices.jl"))

LinearAlgebra.BLAS.set_num_threads(1)

const SAMPLES = 20
const SECONDS = 1.5

_times(b) = Float64[smp.time for smp in b.samples]
_median_time(b) = median(_times(b))

@noinline _ps_cold(A, ordering) = PureSparse.qr(A; ordering)
@noinline _ps_cold_nosing(A, ordering) = PureSparse.qr(A; ordering, singletons = false)
@noinline _ps_warm!(F, A) = PureSparse.qr!(F, A)
@noinline _ps_solve(F, b) = F \ b
@noinline _spqr_cold(A; ordering = SparseArrays.SPQR.ORDERING_DEFAULT) = SparseArrays.qr(A; ordering)
@noinline _spqr_solve(F, b) = F \ b
@noinline _ata_cholesky_cold(A) = PureSparse.cholesky(sparse(A' * A); ordering = PureSparse.AMDOrdering())

# One (label, A, stratum) gate matrix, one arm ("own" or "same-perm"): cold PureSparse
# QR, cold SuiteSparseQR, warm PureSparse qr! (n1==0 forced), the AᵀA context arm.
function bench_one(label::String, A::SparseMatrixCSC, stratum::String, arm::String)
    m, n = size(A)
    result = Dict{String,Any}("matrix" => label, "m" => m, "n" => n, "nnz" => nnz(A), "stratum" => stratum, "arm" => arm)
    b = randn(m)

    # GivenOrdering's stored permutation is FULL-space length n (design_qr.md §2.1's
    # order_columns contract): singleton peeling reduces the block that order_columns
    # is actually invoked on to n-n1 columns, which would DimensionMismatch against a
    # full-length GivenOrdering perm (confirmed by testing this arm before committing
    # it) — so the "same-perm" arm forces singletons off on the PureSparse side, same
    # as the warm qr! arm already needs to for its own reason (§2.3: qr! rejects
    # sym.n1>0). "own" keeps the realistic product default (singletons on).
    ps_singletons = arm == "own"
    if arm == "own"
        ps_ordering = PureSparse.COLAMDOrdering()
        A_spqr = A
        spqr_kw = (ordering = SparseArrays.SPQR.ORDERING_DEFAULT,)
    else # same-perm: feed each stack the OTHER's own-ordering column permutation
        sym_ps_own = PureSparse.symbolic_qr(A; ordering = PureSparse.COLAMDOrdering())
        ps_own_perm = Vector{Int}(sym_ps_own.cperm)
        F_spqr_own = SparseArrays.qr(A; ordering = SparseArrays.SPQR.ORDERING_DEFAULT)
        spqr_own_perm = Vector{Int}(F_spqr_own.pcol)
        ps_ordering = PureSparse.GivenOrdering(spqr_own_perm)
        A_spqr = A[:, ps_own_perm]
        spqr_kw = (ordering = SparseArrays.SPQR.ORDERING_FIXED,)
    end

    # --- config 1: PureSparse QR (cold, the gate number) ---
    b_cold = ps_singletons ?
        (@be _ps_cold($A, $ps_ordering) seconds = SECONDS samples = SAMPLES evals = 1) :
        (@be _ps_cold_nosing($A, $ps_ordering) seconds = SECONDS samples = SAMPLES evals = 1)
    F1 = PureSparse.qr(A; ordering = ps_ordering, singletons = ps_singletons)
    result["ps_cold"] = _median_time(b_cold)
    result["ps_n1"] = F1.sym.n1
    result["ps_rank"] = F1.stats.rank
    result["ps_nnzR"] = F1.stats.nnzR

    b_solve = @be _ps_solve($F1, $b) seconds = SECONDS samples = SAMPLES evals = 1
    result["ps_solve"] = _median_time(b_solve)

    # --- warm qr! (n1==0 forced, reported not gated) ---
    b_cold_ns = @be _ps_cold_nosing($A, $ps_ordering) seconds = SECONDS samples = SAMPLES evals = 1
    F1ns = PureSparse.qr(A; ordering = ps_ordering, singletons = false)
    b_warm = @be _ps_warm!($F1ns, $A) seconds = SECONDS samples = SAMPLES evals = 1
    result["ps_cold_nosingletons"] = _median_time(b_cold_ns)
    result["ps_warm"] = _median_time(b_warm)

    # --- config 2: SuiteSparseQR (cold, the baseline/gate number) ---
    b_cold2 = @be _spqr_cold($A_spqr; $spqr_kw...) seconds = SECONDS samples = SAMPLES evals = 1
    F2 = SparseArrays.qr(A_spqr; spqr_kw...)
    result["spqr_cold"] = _median_time(b_cold2)

    # SuiteSparseQR's `\` on an underdetermined (m<n) system under ORDERING_FIXED can
    # throw SingularException — its rank-revealing minimum-norm path apparently needs
    # its own reordering freedom (confirmed by testing: lp_slack's same-perm arm on
    # galen hit this on the very first run). PureSparse's own `\` never does (§6.2's
    # basic-solution path handles this by construction) — only SPQR's solve needs the
    # guard here.
    try
        b_solve2 = @be _spqr_solve($F2, $b) seconds = SECONDS samples = SAMPLES evals = 1
        result["spqr_solve"] = _median_time(b_solve2)
    catch e
        result["spqr_solve"] = nothing
        result["spqr_solve_error"] = sprint(showerror, e)
    end

    # --- config 3: PureSparse cholesky(AᵀA) normal-equations context arm (§1.2) ---
    try
        b_ata = @be _ata_cholesky_cold($A) seconds = SECONDS samples = SAMPLES evals = 1
        result["ps_ata_cholesky_cold"] = _median_time(b_ata)
    catch e
        result["ps_ata_cholesky_cold"] = nothing
        result["ps_ata_cholesky_error"] = sprint(showerror, e)
    end

    return result
end

function run_gate()
    results = Any[]
    for (label, A, stratum) in qr_gate_matrices()
        for arm in ("own", "same-perm")
            @info "benchmarking" label stratum arm
            push!(results, bench_one(label, A, stratum, arm))
        end
    end

    host = gethostname()
    payload = Dict(
        "host" => host,
        "julia_version" => string(VERSION),
        "timestamp" => string(Dates.now()),
        "results" => results,
    )
    mkpath(joinpath(@__DIR__, "results"))
    outpath = joinpath(@__DIR__, "results", "qr_gate_$(host).json")
    open(outpath, "w") do io
        JSON.print(io, payload, 2)
    end
    @info "wrote $outpath"
    return payload
end

function print_verdict(payload)
    results = payload["results"]
    println("\n=== M5 sparse QR wall-time gate (design_qr.md §9.3) — cold, median seconds ===")
    println(rpad("matrix", 26), rpad("stratum", 14), rpad("arm", 10), rpad("PS qr()", 12),
        rpad("SPQR", 12), "gate(1<2)")
    npass = 0; ntotal = 0
    by_stratum = Dict{String,Tuple{Int,Int}}()
    for r in results
        ntotal += 1
        pass = r["ps_cold"] < r["spqr_cold"]
        pass && (npass += 1)
        s = r["stratum"]
        (np, nt) = get(by_stratum, s, (0, 0))
        by_stratum[s] = (np + (pass ? 1 : 0), nt + 1)
        println(
            rpad(r["matrix"], 26), rpad(r["stratum"], 14), rpad(r["arm"], 10),
            rpad(string(round(r["ps_cold"] * 1000; digits = 3)) * "ms", 12),
            rpad(string(round(r["spqr_cold"] * 1000; digits = 3)) * "ms", 12),
            pass ? "PASS" : "fail",
        )
    end
    println("\nPer-stratum (H4 stated expectation: (i) should win, (ii) competitive, (iii) may lose):")
    for s in sort(collect(keys(by_stratum)))
        (np, nt) = by_stratum[s]
        println("  ", rpad(s, 16), "$np / $nt")
    end
    println("\n$npass / $ntotal matrix-arm combinations strictly faster than SuiteSparseQR (cold).")
    println("Design_qr.md §9.3 M5 closeout gate requires: EVERY stratum passes, both arms —")
    println("no fudge factor. A stratum loss triggers M5b (multifrontal, §7).")
    println(npass == ntotal ? "OVERALL: PASS (M5 gate met, M5b not required)" : "OVERALL: NOT YET PASSING")
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 1 && ARGS[1] == "report"
        host = gethostname()
        payload = JSON.parsefile(joinpath(@__DIR__, "results", "qr_gate_$(host).json"))
        print_verdict(payload)
    else
        payload = run_gate()
        print_verdict(payload)
    end
end
