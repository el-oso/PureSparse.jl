# M5 sparse QR wall-time gate (design_qr.md §9.3, CLAUDE.md requirement 2 —
# non-negotiable). Chairmarks medians, single-thread pinned. Configs (design_qr.md §9.3
# table; QR needs no OpenBLAS kernel-attribution arm the way Cholesky's gate does — M5a's
# left-looking Householder loop calls no BLAS-3 kernel at all, only PureBLAS's `nrm2`
# level-1 op, design_qr.md's own "M5a needs no new PureBLAS kernels" scoping note):
#
#   1. PureSparse QR (COLAMDOrdering, the §2.1 stated default)      [primary — the gate]
#   2. SuiteSparseQR via SparseArrays.qr (stock)                    [baseline — the gate]
#   3. PureSparse cholesky(AᵀA) normal equations                    [context arm, §1.2]
#   5. faer's sparse QR (Rust, MIT-licensed, via a ccall shim)       [context arm, §9.3 config 5]
#
# faer arm: BlazingPorts.jl's existing bench/rust_compare cdylib shim (dense faer_qr etc.)
# extended with a genuinely new `faer_sparse_qr` entry point (faer::sparse::linalg::
# solvers::{SymbolicQr,Qr} — a different API from the dense one) rather than new FFI
# plumbing from scratch, per design_qr.md's own direction. Reported alongside, NOT part
# of the pass/fail gate (design's own D2 reasoning: faer's ordering/threshold choices
# differ, so a head-to-head gate would conflate ordering quality with kernel throughput).
# Skipped gracefully (not a hard error) if the shared library isn't built on this host.
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
@noinline _ps_frontal_cold(A, ordering) = PureSparse.qr(A; ordering, method = :frontal)
@noinline _spqr_cold(A; ordering = SparseArrays.SPQR.ORDERING_DEFAULT) = SparseArrays.qr(A; ordering)
@noinline _spqr_solve(F, b) = F \ b
@noinline _ata_cholesky_cold(A) = PureSparse.cholesky(sparse(A' * A); ordering = PureSparse.AMDOrdering())

const FAER_LIB = joinpath(homedir(), "Documents", "claude", "BlazingPorts.jl", "bench",
    "rust_compare", "rust", "target", "release", "libblazing_compare.so")
const FAER_AVAILABLE = isfile(FAER_LIB)
FAER_AVAILABLE || @warn "faer shared library not found at $FAER_LIB — faer context arm skipped on this host" *
    " (build via: cd \$(BlazingPorts.jl)/bench/rust_compare/rust && cargo build --release --lib)"

# faer::sparse's Qr::sp_qr() (design_qr.md §9.3 config 5, via BlazingPorts.jl's rust_compare
# shim). Julia's SparseMatrixCSC is 1-indexed; faer's raw-CSC constructor wants 0-indexed.
@noinline function _faer_sparse_qr_cold(A::SparseMatrixCSC)
    m, n = size(A)
    colptr0 = Csize_t.(A.colptr .- 1)
    rowval0 = Csize_t.(A.rowval .- 1)
    return ccall((:faer_sparse_qr, FAER_LIB), Float64,
        (Csize_t, Csize_t, Csize_t, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Float64}),
        m, n, nnz(A), colptr0, rowval0, A.nzval)
end

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

    # --- config 1b: PureSparse QR :frontal (cold, M5b — design_qr_m5b.md §A5.6) ---
    # Same ordering as the :column arm above (COLAMDOrdering for "own", the SPQR-
    # matched GivenOrdering for "same-perm"); the frontal path never carries
    # singletons (§A1.2), so there is no ps_singletons split here. Guarded like the
    # AᵀA/faer context arms below: this is a NEW code path under active development,
    # a genuine error on some gate matrix shouldn't crash the whole run.
    try
        b_frontal = @be _ps_frontal_cold($A, $ps_ordering) seconds = SECONDS samples = SAMPLES evals = 1
        Ffrontal = PureSparse.qr(A; ordering = ps_ordering, method = :frontal)
        result["ps_frontal_cold"] = _median_time(b_frontal)
        result["ps_frontal_rank"] = Ffrontal.stats.rank
        b_frontal_solve = @be _ps_solve($Ffrontal, $b) seconds = SECONDS samples = SAMPLES evals = 1
        result["ps_frontal_solve"] = _median_time(b_frontal_solve)
    catch e
        result["ps_frontal_cold"] = nothing
        result["ps_frontal_error"] = sprint(showerror, e)
    end

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

    # --- config 5: faer sparse QR context arm (§9.3), skipped gracefully if unbuilt ---
    # faer's factorize_symbolic_qr hard-asserts nrows >= ncols (confirmed by reading
    # faer 0.24.1's own source, src/sparse/linalg/qr.rs) — it does not support
    # underdetermined systems at all, unlike PureSparse/SuiteSparseQR. Skip rather than
    # call: a Rust panic crossing the extern "C" boundary aborts the WHOLE Julia process,
    # not just this call (confirmed the hard way — an unguarded call on lp_slack_n300x60,
    # m=300 < n=360, SIGABRTed the first real gate run before this guard was added).
    if FAER_AVAILABLE && m >= n
        try
            b_faer = @be _faer_sparse_qr_cold($A) seconds = SECONDS samples = SAMPLES evals = 1
            result["faer_cold"] = _median_time(b_faer)
        catch e
            result["faer_cold"] = nothing
            result["faer_error"] = sprint(showerror, e)
        end
    else
        result["faer_cold"] = nothing
        FAER_AVAILABLE && (result["faer_note"] = "skipped: faer sparse QR requires nrows >= ncols")
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
    println("\n=== M5 sparse QR wall-time gate (design_qr.md §9.3 / design_qr_m5b.md §A5.6) — cold, median seconds ===")
    println(rpad("matrix", 26), rpad("stratum", 14), rpad("arm", 10), rpad("PS column", 12),
        rpad("PS frontal", 12), rpad("SPQR", 12), rpad("faer", 12), "gate(best PS<SPQR)")
    npass = 0; ntotal = 0
    by_stratum = Dict{String,Tuple{Int,Int}}()
    for r in results
        ntotal += 1
        # the gate is on the BEST of PureSparse's own methods (§A5.6's whole premise:
        # a real caller picks whichever wins, exactly what :auto will automate once
        # its threshold is calibrated) — :column alone is the pre-M5b gate, kept as a
        # column in the table for comparison, not the pass/fail criterion by itself.
        ps_frontal = get(r, "ps_frontal_cold", nothing)
        best_ps = isnothing(ps_frontal) ? r["ps_cold"] : min(r["ps_cold"], ps_frontal)
        pass = best_ps < r["spqr_cold"]
        pass && (npass += 1)
        s = r["stratum"]
        (np, nt) = get(by_stratum, s, (0, 0))
        by_stratum[s] = (np + (pass ? 1 : 0), nt + 1)
        faer_str = isnothing(get(r, "faer_cold", nothing)) ? "n/a" :
            string(round(r["faer_cold"] * 1000; digits = 3)) * "ms"
        frontal_str = isnothing(ps_frontal) ? "n/a" : string(round(ps_frontal * 1000; digits = 3)) * "ms"
        println(
            rpad(r["matrix"], 26), rpad(r["stratum"], 14), rpad(r["arm"], 10),
            rpad(string(round(r["ps_cold"] * 1000; digits = 3)) * "ms", 12),
            rpad(frontal_str, 12),
            rpad(string(round(r["spqr_cold"] * 1000; digits = 3)) * "ms", 12),
            rpad(faer_str, 12),
            pass ? "PASS" : "fail",
        )
    end
    println("\nPer-stratum (H4 stated expectation: (i) should win, (ii) competitive, (iii) may lose):")
    for s in sort(collect(keys(by_stratum)))
        (np, nt) = by_stratum[s]
        println("  ", rpad(s, 16), "$np / $nt")
    end
    println("\n$npass / $ntotal matrix-arm combinations: best-of(PS column, PS frontal) strictly faster than SuiteSparseQR (cold).")
    println("Design_qr.md §9.3 M5 closeout gate requires: EVERY stratum passes, both arms —")
    println("no fudge factor.")
    println(npass == ntotal ? "OVERALL: PASS (M5 gate met)" : "OVERALL: NOT YET PASSING")
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
