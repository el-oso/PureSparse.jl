# Crossover measurement for the simplicial-versus-supernodal switch (design.md §4.5,
# `simplicial_ratio` in src/tuning.jl). For a set of SPD matrices spanning a wide range
# of the switch predictor `flops / nnzL`, both paths are forced through `method=` and
# their analysis, warm refactorization and warm single-vector solve are timed with
# Chairmarks medians, single thread. The printout lists, per matrix, the ratio and the
# simplicial-to-supernodal time ratio of each phase and of refactor+solve, so the
# crossover can be read off; results are saved to JSON first.
#
# Usage (pin to one core, e.g. `taskset -c 3`):
#   julia --project=benchmark benchmark/simplicial_threshold.jl           # measure + save + print
#   julia --project=benchmark benchmark/simplicial_threshold.jl report    # print from the saved JSON

using PureSparse, SparseArrays, LinearAlgebra, Random, Statistics, JSON
using Chairmarks: @be
import PureBLAS

include(joinpath(@__DIR__, "matrices.jl"))

LinearAlgebra.BLAS.set_num_threads(1)
PureBLAS.set_num_threads(1)

const SECONDS = 1.0

_median_us(b) = median(Float64[smp.time for smp in b.samples]) * 1.0e6

function laplacian3d(nx::Int, ny::Int, nz::Int)
    n = nx * ny * nz
    idx(i, j, k) = ((k - 1) * ny + (j - 1)) * nx + i
    I = Int[]; J = Int[]; V = Float64[]
    for k in 1:nz, j in 1:ny, i in 1:nx
        p = idx(i, j, k)
        i > 1 && (push!(I, p); push!(J, idx(i - 1, j, k)); push!(V, -1.0))
        j > 1 && (push!(I, p); push!(J, idx(i, j - 1, k)); push!(V, -1.0))
        k > 1 && (push!(I, p); push!(J, idx(i, j, k - 1)); push!(V, -1.0))
        push!(I, p); push!(J, p); push!(V, 6.0)
    end
    return sparse(I, J, V, n, n)
end

function threshold_matrices()
    rng = MersenneTwister(2026)
    ms = Tuple{String, SparseMatrixCSC{Float64, Int}}[]
    for bw in (2, 4, 8, 12, 16, 24, 32, 64)
        push!(ms, ("banded_n4000_bw$bw", banded_spd(4000, bw; rng)))
    end
    for k in (20, 30, 40, 50, 70, 100, 140, 200)
        push!(ms, ("laplacian2d_$(k)x$(k)", laplacian2d(k, k)))
    end
    for k in (8, 12, 16, 22)
        push!(ms, ("laplacian3d_$(k)^3", laplacian3d(k, k, k)))
    end
    for (n, d) in ((2000, 0.002), (2000, 0.005), (1000, 0.02))
        push!(ms, ("random_n$(n)_d$(d)", random_spd(n, d; rng)))
    end
    return ms
end

@noinline _sym(A, method) = PureSparse.symbolic(A; method)
@noinline _refactor!(F, A) = PureSparse.cholesky!(F, A)
@noinline _solve!(x, F, b) = PureSparse.solve!(x, F, b)

function bench_one(label::String, A::SparseMatrixCSC)
    n = size(A, 1)
    b = randn(MersenneTwister(7), n)
    x = similar(b)
    r = Dict{String, Any}("matrix" => label, "n" => n, "nnz" => nnz(A))
    for method in (:supernodal, :simplicial)
        sym = PureSparse.symbolic(A; method)
        F = PureSparse.cholesky(sym, A)
        _solve!(x, F, b)
        r["ratio"] = sym.flops / sym.nnzL
        r["nnzL"] = sym.nnzL
        pre = String(method)
        r["$(pre)_symbolic_us"] = _median_us(@be _sym($A, $method) seconds = SECONDS)
        r["$(pre)_refactor_us"] = _median_us(@be _refactor!($F, $A) seconds = SECONDS)
        r["$(pre)_solve_us"] = _median_us(@be _solve!($x, $F, $b) seconds = SECONDS)
        r["$(pre)_residual"] = norm(A * x - b) / norm(b)
    end
    return r
end

function run_all()
    results = Any[]
    for (label, A) in threshold_matrices()
        @info "benchmarking" label
        push!(results, bench_one(label, A))
    end
    payload = Dict("host" => gethostname(), "julia_version" => string(VERSION), "results" => results)
    mkpath(joinpath(@__DIR__, "results"))
    outpath = joinpath(@__DIR__, "results", "simplicial_threshold_$(gethostname()).json")
    open(io -> JSON.print(io, payload, 2), outpath, "w")
    @info "wrote $outpath"
    return payload
end

function print_table(payload)
    results = sort(payload["results"]; by = r -> r["ratio"])
    println("\nsimplicial / supernodal time ratios (< 1: simplicial faster); medians, µs")
    println(
        rpad("matrix", 24), rpad("n", 7), rpad("flops/nnzL", 12), rpad("sym S/N", 9), rpad("refac S/N", 11),
        rpad("solve S/N", 11), rpad("refac+solve", 12), rpad("sup refac", 11), rpad("sim refac", 11),
        rpad("sup solve", 11), "sim solve"
    )
    for r in results
        rs = r["simplicial_refactor_us"] / r["supernodal_refactor_us"]
        ss = r["simplicial_solve_us"] / r["supernodal_solve_us"]
        es = (r["simplicial_refactor_us"] + r["simplicial_solve_us"]) /
            (r["supernodal_refactor_us"] + r["supernodal_solve_us"])
        ys = r["simplicial_symbolic_us"] / r["supernodal_symbolic_us"]
        f(v) = rpad(string(round(v; digits = 2)), 11)
        println(
            rpad(r["matrix"], 24), rpad(r["n"], 7), rpad(round(r["ratio"]; digits = 1), 12),
            rpad(string(round(ys; digits = 2)), 9), f(rs), f(ss), rpad(string(round(es; digits = 2)), 12),
            f(round(r["supernodal_refactor_us"]; digits = 0)), f(round(r["simplicial_refactor_us"]; digits = 0)),
            f(round(r["supernodal_solve_us"]; digits = 0)), round(r["simplicial_solve_us"]; digits = 0),
        )
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 1 && ARGS[1] == "report"
        print_table(JSON.parsefile(joinpath(@__DIR__, "results", "simplicial_threshold_$(gethostname()).json")))
    else
        print_table(run_all())
    end
end
