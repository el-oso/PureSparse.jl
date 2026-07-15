# User-requested comparison: 7000x4000 tall matrix, faer vs PureSparse (simplified to
# just the 1% and 10% density endpoints).
using PureSparse, SparseArrays, LinearAlgebra, Random, Statistics, JSON, Dates
using Chairmarks: @be

LinearAlgebra.BLAS.set_num_threads(1)

# SECONDS must comfortably exceed SAMPLES × the slowest PLOTTED series' median (faer/SPQR
# run ~4-6s/call at this size): at the old SECONDS=2.0, faer/SPQR each got exactly 1
# sample (Chairmarks completes an in-progress rep before checking budget, so a 2.0s
# budget against a 4-6s call always yields n=1) — a degenerate, non-distributional
# violin/boxplot. `ps_column` (~60-65s/call) still bottoms out near 1 sample regardless
# of any reasonable budget here; it isn't plotted, so that's fine.
const SAMPLES = 8
const SECONDS = 50.0
_times(b) = Float64[s.time for s in b.samples]
_median_time(b) = median(_times(b))

const FAER_LIB = joinpath(homedir(), "Documents", "claude", "BlazingPorts.jl", "bench",
    "rust_compare", "rust", "target", "release", "libblazing_compare.so")
@noinline function _faer_sparse_qr_cold(A::SparseMatrixCSC)
    m, n = size(A)
    colptr0 = Csize_t.(A.colptr .- 1)
    rowval0 = Csize_t.(A.rowval .- 1)
    return ccall((:faer_sparse_qr, FAER_LIB), Float64,
        (Csize_t, Csize_t, Csize_t, Ptr{Csize_t}, Ptr{Csize_t}, Ptr{Float64}),
        m, n, nnz(A), colptr0, rowval0, A.nzval)
end

@noinline _ps_column_cold(A, ordering) = PureSparse.qr(A; ordering, method = :column)
@noinline _ps_frontal_cold(A, ordering) = PureSparse.qr(A; ordering, method = :frontal)
@noinline _spqr_cold(A) = SparseArrays.qr(A; ordering = SparseArrays.SPQR.ORDERING_DEFAULT)

m, n = 7000, 4000
results = Any[]
for dens in (0.01, 0.10)
    rng = MersenneTwister(hash((m, n, dens)))
    A = sprand(rng, m, n, dens)
    @info "density $dens: nnz=$(nnz(A)) ($(round(100*nnz(A)/(m*n); digits=2))%)"

    ordering = PureSparse.COLAMDOrdering()
    b_col = @be _ps_column_cold($A, $ordering) seconds = SECONDS samples = SAMPLES evals = 1
    b_front = @be _ps_frontal_cold($A, $ordering) seconds = SECONDS samples = SAMPLES evals = 1
    b_spqr = @be _spqr_cold($A) seconds = SECONDS samples = SAMPLES evals = 1
    b_faer = @be _faer_sparse_qr_cold($A) seconds = SECONDS samples = SAMPLES evals = 1

    r = Dict(
        "density" => dens, "nnz" => nnz(A),
        "ps_column" => _median_time(b_col),
        "ps_column_samples" => _times(b_col),
        "ps_frontal" => _median_time(b_front),
        "ps_frontal_samples" => _times(b_front),
        "spqr" => _median_time(b_spqr),
        "spqr_samples" => _times(b_spqr),
        "faer" => _median_time(b_faer),
        "faer_samples" => _times(b_faer),
    )
    push!(results, r)
    println(rpad("density=$dens", 16), rpad("PS column: $(round(r["ps_column"]*1000;digits=2))ms", 24),
        rpad("PS frontal: $(round(r["ps_frontal"]*1000;digits=2))ms", 26),
        rpad("SPQR: $(round(r["spqr"]*1000;digits=2))ms", 20),
        "faer: $(round(r["faer"]*1000;digits=2))ms")
end

payload = Dict("host" => gethostname(), "julia_version" => string(VERSION),
    "timestamp" => string(Dates.now()), "m" => m, "n" => n, "results" => results)
mkpath(joinpath(@__DIR__, "results"))
open(joinpath(@__DIR__, "results", "faer_vs_puresparse_7000x4000_$(gethostname()).json"), "w") do io
    JSON.print(io, payload, 2)
end
println("done")
