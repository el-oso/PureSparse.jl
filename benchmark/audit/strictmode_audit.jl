# StrictMode zero-alloc guarantee for every warm-refactor entry point CLAUDE.md req 5
# already gates at runtime: cholesky!/ldlt! (M1/M2, both via the shared
# solve!(::_PanelFactor,...)), and qr!/solve! on BOTH sparse-QR factor types — QRFactor
# (M5a :column) and QRFrontFactor (M5b :frontal). `@assert_noalloc` (StrictMode) is
# used here in EMPIRICAL mode (`static = false`), not the default static AllocCheck all-paths
# proof — tried `static = true` (the default under analysis="full") first: it correctly proves
# cholesky!'s call into PureBLAS's `trsm!` -> `_l3_apad` (workspace.jl), a lazily-grown-then-
# cached scratch buffer (`if size(cache) < needed; grow!; end`), STILL CONTAINS the growth
# branch in its compiled code — so static proof reports it as an allocation site regardless of
# whether a prior warm-up call already grew the cache to steady state. This is the exact
# false-positive class PureBLAS's own test/strictmode_tests.jl already documents and works
# around the same way (`static = false`, see its header comment) — not a bug to fix, a
# mismatch between "prove zero allocations under every possible history" (impossible by
# design for any lazily-grown cache) and the actual guarantee wanted here ("zero allocations
# in warmed steady state", which is exactly what the existing `@allocated == 0` tests already
# verify, and what `static = false` checks here too, just via StrictMode's own assertion
# mechanism instead of a bare `@test`).
#
# Isolated in this env (not test/) because all of these functions call StrictMode's OWN
# runtime checks internally (check_refactor_shape/check_finite, src/strict.jl) — running with
# checks_enabled=true globally in test/ would contaminate every existing @allocated==0 gate
# there (see this dir's Project.toml header).
#
# Run:
#   julia --project=benchmark/audit benchmark/audit/strictmode_audit.jl
using PureSparse, SparseArrays, Random, LinearAlgebra
using StrictMode
using AllocCheck, JET   # StrictMode's analysis backend is a weak-dep extension — load it

StrictMode.checks_enabled() || error("StrictMode checks disabled — set [preferences.StrictMode] checks_enabled=true")
StrictMode.backend_available() || error("StrictMode analysis backend not loaded — need `using AllocCheck, JET`")

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

println("Auditing PureSparse.cholesky!(::SupernodalFactor, ::SparseMatrixCSC)...")
@assert_noalloc static = false PureSparse.cholesky!(Fchol, Achol)

println("Auditing PureSparse.solve!(::Vector, ::SupernodalFactor, ::Vector)...")
@assert_noalloc static = false PureSparse.solve!(xchol, Fchol, bchol)

# --- ldlt! / solve! (LDLFactor, M2) ---
npos, nneg = 15, 10
nl = npos + nneg
let
    I = Int[]; J = Int[]; V = Float64[]
    rowsum = zeros(nl)
    addsym!(i, j, v) = (push!(I, i); push!(J, j); push!(V, v);
        i != j && (push!(I, j); push!(J, i); push!(V, v));
        rowsum[i] += abs(v); i != j && (rowsum[j] += abs(v)))
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

println("Auditing PureSparse.ldlt!(::LDLFactor, ::SparseMatrixCSC)...")
@assert_noalloc static = false PureSparse.ldlt!(Fldlt, Kldlt)

println("Auditing PureSparse.solve!(::Vector, ::LDLFactor, ::Vector)...")
@assert_noalloc static = false PureSparse.solve!(xldlt, Fldlt, bldlt)

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

println("Auditing PureSparse.qr!(::QRFactor, ::SparseMatrixCSC)...")
@assert_noalloc static = false PureSparse.qr!(Fcol, Aqr2)

println("Auditing PureSparse.solve!(::Vector, ::QRFactor, ::Vector)...")
@assert_noalloc static = false PureSparse.solve!(xqr, Fcol, bqr)

# --- qr! / solve! (QRFrontFactor, M5b multifrontal) ---
A = sprand(rng, 60, 25, 0.15) + sparse(1:25, 1:25, 1.0, 60, 25)
F = PureSparse.qr_frontal(A; ordering)
A2 = SparseMatrixCSC(A.m, A.n, A.colptr, A.rowval, A.nzval .* (1 .+ 0.01 .* randn(rng, nnz(A))))
b = randn(rng, A.m)
x = zeros(A.n)
PureSparse.qr!(F, A2)   # warm up
PureSparse.solve!(x, F, b)

println("Auditing PureSparse.qr!(::QRFrontFactor, ::SparseMatrixCSC)...")
@assert_noalloc static = false PureSparse.qr!(F, A2)

println("Auditing PureSparse.solve!(::Vector, ::QRFrontFactor, ::Vector)...")
@assert_noalloc static = false PureSparse.solve!(x, F, b)

println("PASS: cholesky!/ldlt!/qr!(::QRFactor)/qr!(::QRFrontFactor) and their solve!s are all alloc-free in warmed steady state (StrictMode @assert_noalloc, empirical mode).")
