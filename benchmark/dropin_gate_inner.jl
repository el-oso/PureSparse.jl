# Inner measured stage of dropin_gate.jl (see its header for the two-subprocess
# design). ARGS: "baseline" <out.json>  |  "dropin" <baseline.json> <out.json>.
# Methodology mirrors gate.jl: Chairmarks medians, 30 samples/1.5s cap, evals=1,
# @noinline concrete wrappers, single BLAS thread.

using PureSparse, SparseArrays, LinearAlgebra, Random, Statistics, JSON
using Chairmarks: @be

include(joinpath(@__DIR__, "matrices.jl"))

LinearAlgebra.BLAS.set_num_threads(1)

const SAMPLES = 30
const SECONDS = 1.5
_med(b) = median(Float64[s.time for s in b.samples])

@noinline _stdlib_cold(A) = LinearAlgebra.cholesky(Symmetric(A, :L))
@noinline _stdlib_cold_perm(A, p) = LinearAlgebra.cholesky(Symmetric(A, :L); perm = p)
@noinline _ps_direct_cold(A, ord) = PureSparse.cholesky(A; ordering = ord)
@noinline _ps_warm!(F, A) = PureSparse.cholesky!(F, A)
@noinline _cholmod_warm!(F, A) = LinearAlgebra.cholesky!(F, Symmetric(A, :L))

mode = ARGS[1]

if mode == "baseline"
    # dropin must be OFF: the stdlib name is CHOLMOD here.
    @assert !PureSparse.DROPIN_ACTIVE
    @assert !(_stdlib_cold(sparse(1.0I, 2, 2)) isa PureSparse.SupernodalFactor)
    results = Any[]
    for (label, A) in gate_matrices()
        @info "baseline (CHOLMOD)" label
        r = Dict{String,Any}("matrix" => label, "n" => size(A, 1))
        # own-ordering arm
        F = _stdlib_cold(A)
        r["cholmod_perm"] = Vector{Int}(F.p)   # handed to stage 2's same-perm arm
        r["cholmod_cold_own"] = _med(@be _stdlib_cold($A) seconds = SECONDS samples = SAMPLES evals = 1)
        r["cholmod_warm_own"] = _med(@be _cholmod_warm!($F, $A) seconds = SECONDS samples = SAMPLES evals = 1)
        # same-perm arm: CHOLMOD under PureSparse's own AMD permutation (gate.jl §9.3 D2)
        psperm = Vector{Int}(PureSparse.symbolic(A).perm)
        Fp = _stdlib_cold_perm(A, psperm)
        r["cholmod_cold_sameperm"] = _med(@be _stdlib_cold_perm($A, $psperm) seconds = SECONDS samples = SAMPLES evals = 1)
        r["cholmod_warm_sameperm"] = _med(@be _cholmod_warm!($Fp, $A) seconds = SECONDS samples = SAMPLES evals = 1)
        push!(results, r)
    end
    open(ARGS[2], "w") do io
        JSON.print(io, Dict("results" => results), 2)
    end
else
    @assert mode == "dropin"
    # dropin must be ON: the stdlib name is PureSparse here — this is the entry point
    # under test, interposition included.
    @assert PureSparse.DROPIN_ACTIVE
    @assert _stdlib_cold(sparse(1.0I, 2, 2)) isa PureSparse.SupernodalFactor
    base = Dict(r["matrix"] => r for r in JSON.parsefile(ARGS[2])["results"])
    ord = PureSparse.AMDOrdering()
    results = Any[]
    for (label, A) in gate_matrices()
        @info "dropin (PureSparse via LinearAlgebra.cholesky)" label
        r = base[label]
        cp = Vector{Int}(r["cholmod_perm"])
        # The gate matrices store the LOWER TRIANGLE only (matrices.jl). The dropin
        # factors `sparse(Symmetric(A, :L))` — the FULL symmetric pattern — so warm
        # refactorization of a dropin-produced factor must be fed that same full
        # matrix: `cholesky!`'s contract (design §1.2) is "same pattern as the factor's
        # symbolic". Feeding lower-only A here was measured to fail at column 1 and
        # early-exit, producing impossibly-fast fake warm numbers (caught by the
        # residual guards below, which now make that class of mistake impossible).
        Afull = SparseMatrixCSC(sparse(Symmetric(A, :L)))
        b = randn(MersenneTwister(7), size(A, 1))
        # own-ordering arm, THROUGH the drop-in
        F = _stdlib_cold(A)
        @assert F isa PureSparse.SupernodalFactor
        r["dropin_cold_own"] = _med(@be _stdlib_cold($A) seconds = SECONDS samples = SAMPLES evals = 1)
        # warm refactor of the dropin-produced factor (the drop-in exposes no stdlib
        # warm-refactor name — CHOLMOD's cholesky! takes its own Factor type — so warm
        # goes through PureSparse.cholesky! on the factor the dropin returned; honest
        # scope, documented in the ROADMAP)
        _ps_warm!(F, Afull)
        @assert PureSparse.issuccess(F) "warm refactor failed ($label)"
        x = F \ b
        @assert norm(Symmetric(A, :L) * x - b) / norm(b) < 1e-8 "warm refactor wrong ($label)"
        r["ps_warm_own"] = _med(@be _ps_warm!($F, $Afull) seconds = SECONDS samples = SAMPLES evals = 1)
        # interposition overhead: same cold work via the direct PureSparse API
        r["ps_direct_cold_own"] = _med(@be _ps_direct_cold($A, $ord) seconds = SECONDS samples = SAMPLES evals = 1)
        # same-perm arm: dropin fed CHOLMOD's own perm via the perm= kwarg (this also
        # exercises the kwarg-translation interposition path)
        Fp = _stdlib_cold_perm(A, cp)
        _ps_warm!(Fp, Afull)
        @assert PureSparse.issuccess(Fp) "same-perm warm refactor failed ($label)"
        @assert norm(Symmetric(A, :L) * (Fp \ b) - b) / norm(b) < 1e-8 "same-perm warm wrong ($label)"
        r["dropin_cold_sameperm"] = _med(@be _stdlib_cold_perm($A, $cp) seconds = SECONDS samples = SAMPLES evals = 1)
        r["ps_warm_sameperm"] = _med(@be _ps_warm!($Fp, $Afull) seconds = SECONDS samples = SAMPLES evals = 1)
        push!(results, r)
    end

    # zero-alloc re-check WITH the dropin (getproperty overrides etc.) compiled in —
    # CLAUDE.md requirement 5 must not regress from the parity additions. Same
    # full-pattern rule as the warm arms above, and issuccess asserts so a failing
    # early-exit can't masquerade as a 0-byte success.
    A0 = gate_matrices()[1][2]
    A0full = SparseMatrixCSC(sparse(Symmetric(A0, :L)))
    F0 = _stdlib_cold(A0)
    PureSparse.cholesky!(F0, A0full)
    a_llt = @allocated PureSparse.cholesky!(F0, A0full)
    @assert PureSparse.issuccess(F0)
    G0 = LinearAlgebra.ldlt(Symmetric(A0, :L))
    PureSparse.ldlt!(G0, A0full)
    a_ldlt = @allocated PureSparse.ldlt!(G0, A0full)
    @assert PureSparse.issuccess(G0)
    @assert a_llt == 0 "cholesky! allocated $a_llt bytes with dropin active"
    @assert a_ldlt == 0 "ldlt! allocated $a_ldlt bytes with dropin active"
    @info "zero-alloc re-check with dropin active" a_llt a_ldlt

    payload = Dict(
        "host" => gethostname(),
        "julia_version" => string(VERSION),
        "zero_alloc" => Dict("cholesky!" => a_llt, "ldlt!" => a_ldlt),
        "results" => results,
    )
    open(ARGS[3], "w") do io
        JSON.print(io, payload, 2)
    end
end
