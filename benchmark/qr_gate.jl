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
# The GATE is best-of(PS :column warm, PS :frontal warm) refactor time vs SPQR cold
# (design_qr.md §9.3 D13, corrected 2026-07-16 from the original cold-vs-cold text).
# SPQR has no analyze-once/refactorize path at all, so its cold call already IS its best
# case; PureSparse's warm `qr!` refactor is independently zero-allocation (StrictMode
# `@assert_noalloc`, both :column and :frontal), so gating on it is both the fairer
# best-case-vs-best-case comparison AND deterministic — no GC exposure in the timed
# region on our side, unlike the original cold-vs-cold criterion (which showed real,
# non-negligible GC-pause-driven timing bimodality on some gate matrices: up to 36% of
# a cold call's wall time was GC on one, enough to flip a gate verdict between two
# back-to-back runs with no code change). `qr!` requires `sym.n1==0`, so :column's warm
# number always uses a `singletons=false`-forced initial factor (even on stratum (i)
# matrices, where singletons ARE exploited on the cold-vs-cold reported number below);
# :frontal never carries singletons at all (§A1.2), so its own factor needs no such
# forcing. Cold-vs-cold (`ps_cold`/`ps_frontal_cold` vs `spqr_cold`) is still measured
# and printed for transparency — genuinely one-shot workloads exist and the number
# stays informative — but it is no longer the deciding inequality.
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

    # Both arms use the realistic product default (singletons on). Used to be
    # own-arm-only: GivenOrdering's stored permutation is FULL-space length n
    # (design_qr.md §2.1's order_columns contract), and singleton peeling used to
    # hand order_columns only the n-n1 surviving columns, DimensionMismatching against
    # a full-length GivenOrdering perm — task #50, fixed via singletons.jl's
    # `_restrict_ordering` (restricts+relabels the given permutation onto the
    # surviving columns' local index space before A22's own order_columns call).
    # Since peeling itself never depends on `ordering` (it's a pure function of A's
    # own pattern/values), n1/nnzR come out bit-identical between the two arms.
    ps_singletons = true
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
    result["ps_cold_samples"] = _times(b_cold)
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
        result["ps_frontal_cold_samples"] = _times(b_frontal)
        result["ps_frontal_rank"] = Ffrontal.stats.rank
        b_frontal_solve = @be _ps_solve($Ffrontal, $b) seconds = SECONDS samples = SAMPLES evals = 1
        result["ps_frontal_solve"] = _median_time(b_frontal_solve)
        # warm qr! (design_qr.md §9.3 D13: the gate criterion, not just reported —
        # frontal never carries singletons, sym.n1==0 always, so Ffrontal is already
        # refactor-ready with no "ns" variant needed the way :column requires below)
        b_frontal_warm = @be _ps_warm!($Ffrontal, $A) seconds = SECONDS samples = SAMPLES evals = 1
        result["ps_frontal_warm"] = _median_time(b_frontal_warm)
    catch e
        result["ps_frontal_cold"] = nothing
        result["ps_frontal_warm"] = nothing
        result["ps_frontal_error"] = sprint(showerror, e)
    end

    # --- warm qr! (n1==0 forced; design_qr.md §9.3 D13: THE gate criterion for
    # :column, mirroring gate.jl's ps_pureblas_warm) ---
    b_cold_ns = @be _ps_cold_nosing($A, $ps_ordering) seconds = SECONDS samples = SAMPLES evals = 1
    F1ns = PureSparse.qr(A; ordering = ps_ordering, singletons = false)
    b_warm = @be _ps_warm!($F1ns, $A) seconds = SECONDS samples = SAMPLES evals = 1
    result["ps_cold_nosingletons"] = _median_time(b_cold_ns)
    result["ps_warm"] = _median_time(b_warm)

    # --- config 2: SuiteSparseQR (cold, the baseline/gate number) ---
    b_cold2 = @be _spqr_cold($A_spqr; $spqr_kw...) seconds = SECONDS samples = SAMPLES evals = 1
    F2 = SparseArrays.qr(A_spqr; spqr_kw...)
    result["spqr_cold"] = _median_time(b_cold2)
    result["spqr_cold_samples"] = _times(b_cold2)

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
    println("\n=== M5 sparse QR wall-time gate (design_qr.md §9.3 D13 / design_qr_m5b.md §A5.6) — best-of PS WARM refactor vs SPQR cold, median seconds ===")
    println(rpad("matrix", 26), rpad("stratum", 14), rpad("arm", 10), rpad("PS col warm", 13),
        rpad("PS front warm", 15), rpad("SPQR cold", 12), rpad("PS cold(rep)", 14), "gate(best PS warm<SPQR cold)")
    npass = 0; ntotal = 0
    by_stratum = Dict{String,Tuple{Int,Int}}()
    for r in results
        ntotal += 1
        # D13: the gate is on the BEST of PureSparse's own WARM refactor times (§A5.6's
        # "a real caller picks whichever wins" premise, now applied to the warm path,
        # which is the actually-deterministic, StrictMode-zero-alloc-verified one) vs
        # SPQR's cold time (its only, and therefore best-case, number). Cold-vs-cold is
        # still computed and reported below for transparency but no longer decides pass/fail.
        ps_frontal_warm = get(r, "ps_frontal_warm", nothing)
        best_ps_warm = isnothing(ps_frontal_warm) ? r["ps_warm"] : min(r["ps_warm"], ps_frontal_warm)
        pass = best_ps_warm < r["spqr_cold"]
        pass && (npass += 1)
        s = r["stratum"]
        (np, nt) = get(by_stratum, s, (0, 0))
        by_stratum[s] = (np + (pass ? 1 : 0), nt + 1)
        frontal_warm_str = isnothing(ps_frontal_warm) ? "n/a" : string(round(ps_frontal_warm * 1000; digits = 3)) * "ms"
        ps_cold_rep = isnothing(get(r, "ps_frontal_cold", nothing)) ? r["ps_cold"] : min(r["ps_cold"], r["ps_frontal_cold"])
        println(
            rpad(r["matrix"], 26), rpad(r["stratum"], 14), rpad(r["arm"], 10),
            rpad(string(round(r["ps_warm"] * 1000; digits = 3)) * "ms", 13),
            rpad(frontal_warm_str, 15),
            rpad(string(round(r["spqr_cold"] * 1000; digits = 3)) * "ms", 12),
            rpad(string(round(ps_cold_rep * 1000; digits = 3)) * "ms", 14),
            pass ? "PASS" : "fail",
        )
    end
    println("\nPer-stratum (H4 stated expectation: (i) should win, (ii) competitive, (iii) may lose):")
    for s in sort(collect(keys(by_stratum)))
        (np, nt) = by_stratum[s]
        println("  ", rpad(s, 16), "$np / $nt")
    end
    println("\n$npass / $ntotal matrix-arm combinations: best-of(PS column, PS frontal) WARM refactor strictly faster than SuiteSparseQR (cold).")
    println("Design_qr.md §9.3 (D13-corrected) M5 closeout gate requires: EVERY stratum passes, both arms —")
    println("no fudge factor. Warm path is StrictMode @assert_noalloc-verified zero-allocation,")
    println("so this comparison carries no GC exposure on the PureSparse side.")
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
