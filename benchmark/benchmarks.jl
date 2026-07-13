# PkgBenchmark suite — commit-to-commit self-regression guardrail (mirrors PureBLAS's
# benchmark/benchmarks.jl convention). Measures ABSOLUTE times of PureSparse routines so
# `PkgBenchmark.judge` can flag a slowdown between two commits. Complements `gate.jl` (the
# ABSOLUTE gate vs CHOLMOD+OpenBLAS, design.md §9.3); `judge` here answers "did MY change
# make MY code slower?", which the gate does not.
#
# Run/compare (from repo root):
#   using PkgBenchmark
#   base = benchmarkpkg(PureSparse); PkgBenchmark.writeresults("benchmark/base.json", base)
#   # …make changes…
#   j = judge(PureSparse, "benchmark/base.json")            # or judge(PureSparse, "HEAD", "HEAD~1")
#   PkgBenchmark.export_markdown("benchmark/judge.md", j)

using BenchmarkTools, PureSparse, SparseArrays, Random
include(joinpath(@__DIR__, "matrices.jl"))
Random.seed!(1)
const SUITE = BenchmarkGroup()

sym_group = SUITE["symbolic"] = BenchmarkGroup()
for (n, density) in ((200, 0.02), (1000, 0.005))
    A = random_spd(n, density)
    sym_group["symbolic_amd_n$n"] = @benchmarkable PureSparse.symbolic($A) evals = 1
end

fact = SUITE["cholesky"] = BenchmarkGroup()
for (n, density) in ((200, 0.02), (1000, 0.005))
    A = random_spd(n, density)
    sym = PureSparse.symbolic(A)
    fact["cold_n$n"] = @benchmarkable PureSparse.cholesky($sym, $A) evals = 1
    F = PureSparse.cholesky(sym, A)
    fact["warm_refactor_n$n"] = @benchmarkable PureSparse.cholesky!(F, $A) setup = (F = $F)
end

slv = SUITE["solve"] = BenchmarkGroup()
for (n, density) in ((200, 0.02), (1000, 0.005))
    A = random_spd(n, density)
    F = PureSparse.cholesky(A)
    b = randn(n)
    slv["solve_n$n"] = @benchmarkable PureSparse.solve!(x, $F, $b) setup = (x = similar($b))
end
