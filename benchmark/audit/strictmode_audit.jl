# Fast StrictMode audit of PureSparse's refactor-and-solve entry points. It needs no proof
# backend; the authoritative all-paths proofs (AllocCheck, JET) run in the test suite through
# StrictModeTest (test/strictmode_guarantees_tests.jl). Three checks, any failure exits non-zero:
#
#   1. Loading PureSparse runs its `@strict_policy :default` load check.
#   2. Every in-place entry point allocates zero bytes in warmed steady state
#      (`@allocated` behind a function barrier).
#   3. A sweep of the methods the workload compiled has no type-stability or boxing failure.
#      StrictMode's allocation verdicts are heuristics, so they are reported, not gated.
#
# Run:
#   julia --project=benchmark/audit benchmark/audit/strictmode_audit.jl
using PureSparse, SparseArrays, Random, LinearAlgebra
using StrictMode: StrictMode, audit, module_policy

StrictMode.checks_enabled() || error("StrictMode checks are disabled in this environment")

function check_noalloc(f, args...)
    f(args...)
    n = @allocated f(args...)
    println(rpad(string(nameof(f), "(", join(nameof.(typeof.(args)), ", "), ")"), 64), n, " bytes")
    iszero(n) || error("$(nameof(f)) allocated $n bytes in warmed steady state")
    return nothing
end

rng = MersenneTwister(11)

rng = MersenneTwister(11)

# --- cholesky! / solve! (SupernodalFactor, M1) ---
n = 50
let
    I = Int[]; J = Int[]; V = Float64[]
    rowsum = zeros(n)
    for j in 1:n, i in (j + 1):n
        rand(rng) < 0.15 || continue
        v = randn(rng)
        push!(I, i); push!(J, j); push!(V, v)
        rowsum[i] += abs(v); rowsum[j] += abs(v)
    end
    for j in 1:n
        push!(I, j); push!(J, j); push!(V, rowsum[j] + 1.0)
    end
    global Achol = sparse(Symmetric(sparse(I, J, V, n, n), :L))
end
sym_chol = PureSparse.symbolic(Achol)
Fchol = PureSparse.cholesky(sym_chol, Achol)
bchol = randn(rng, n)
xchol = zeros(n)
PureSparse.cholesky!(Fchol, Achol)   # warm up
PureSparse.solve!(xchol, Fchol, bchol)

check_noalloc(PureSparse.cholesky!, Fchol, Achol)

check_noalloc(PureSparse.solve!, xchol, Fchol, bchol)

# --- ldlt! / solve! (LDLFactor, M2) ---
npos, nneg = 15, 10
nl = npos + nneg
let
    I = Int[]; J = Int[]; V = Float64[]
    rowsum = zeros(nl)
    addsym!(i, j, v) = (
        push!(I, i); push!(J, j); push!(V, v);
        i != j && (push!(I, j); push!(J, i); push!(V, v));
        rowsum[i] += abs(v); i != j && (rowsum[j] += abs(v))
    )
    for j in 1:npos, i in (j + 1):npos
        rand(rng) < 0.2 && addsym!(i, j, randn(rng))
    end
    for j in 1:nneg, i in (j + 1):nneg
        rand(rng) < 0.2 && addsym!(npos + i, npos + j, randn(rng))
    end
    for j in 1:npos, i in 1:nneg
        rand(rng) < 0.2 && addsym!(npos + i, j, randn(rng))
    end
    for j in 1:nl
        v = (rowsum[j] + 1.0) * (j <= npos ? 1.0 : -1.0)
        push!(I, j); push!(J, j); push!(V, v)
    end
    global Kldlt = sparse(I, J, V, nl, nl)
end
signs = vcat(fill(1, npos), fill(-1, nneg))
Fldlt = PureSparse.ldlt(Kldlt; signs)
bldlt = randn(rng, nl)
xldlt = zeros(nl)
PureSparse.ldlt!(Fldlt, Kldlt)   # warm up
PureSparse.solve!(xldlt, Fldlt, bldlt)

check_noalloc(PureSparse.ldlt!, Fldlt, Kldlt)

check_noalloc(PureSparse.solve!, xldlt, Fldlt, bldlt)

# --- qr! / solve! (QRFactor, M5a :column) ---
Aqr = sprand(rng, 60, 25, 0.15) + sparse(1:25, 1:25, 1.0, 60, 25)
ordering = PureSparse.COLAMDOrdering()
# singletons=false: qr! (refactor) rejects a factor built WITH singleton pre-elimination
# (design_qr.md §2.3 — a singleton set chosen for A's values is invalid for a refactor's
# new values), same requirement as the existing @allocated==0 test for this path.
Fcol = PureSparse.qr(Aqr; ordering, singletons = false)
Aqr2 = SparseMatrixCSC(Aqr.m, Aqr.n, Aqr.colptr, Aqr.rowval, Aqr.nzval .* (1 .+ 0.01 .* randn(rng, nnz(Aqr))))
bqr = randn(rng, Aqr.m)
xqr = zeros(Aqr.n)
PureSparse.qr!(Fcol, Aqr2)   # warm up
PureSparse.solve!(xqr, Fcol, bqr)

check_noalloc(PureSparse.qr!, Fcol, Aqr2)

check_noalloc(PureSparse.solve!, xqr, Fcol, bqr)

# --- qr! / solve! (QRFrontFactor, M5b multifrontal) ---
A = sprand(rng, 60, 25, 0.15) + sparse(1:25, 1:25, 1.0, 60, 25)
F = PureSparse.qr_frontal(A; ordering)
A2 = SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, A.nzval .* (1 .+ 0.01 .* randn(rng, nnz(A))))
b = randn(rng, A.m)
x = zeros(A.n)
PureSparse.qr!(F, A2)   # warm up
PureSparse.solve!(x, F, b)

check_noalloc(PureSparse.qr!, F, A2)

check_noalloc(PureSparse.solve!, x, F, b)

fs = audit(PureSparse; sweep = true, guarantees = module_policy(PureSparse))
gating = filter(f -> f.status === :fail && f.guarantee in (:typestable, :noboxing), fs)
heuristic = count(f -> f.status === :fail && f.guarantee === :noalloc, fs)
println(
    "sweep: ", length(fs), " findings; ", length(gating), " type-stability/boxing failures; ",
    heuristic, " allocation heuristics (reported only)"
)
for f in gating
    println("  ", f.guarantee, "  ", f.func, f.signature, " — ", f.reason)
end
isempty(gating) || error("type-stability or boxing failures in PureSparse")
println("PASS")
