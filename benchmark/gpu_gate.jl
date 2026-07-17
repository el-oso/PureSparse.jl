# M6 GPU wall-time gate (design_gpu.md §8, amendment B — the 5× target).
#
# Three arms per matrix on a LARGE stratum (where the GPU crown fronts exist), SPD + SQD:
#   1. CPU  PureSparse+PureBLAS       — cholesky!/ldlt! + solve!, warm, single-thread
#   2. GPU  PureSparse (pure kernels) — device-resident multifrontal factor + device solve
#   3. CHOLMOD+OpenBLAS               — the baseline (both own-ordering and same-perm)
#
# Timed region (amendment B): warm NUMERICAL factor + solve, both on device for the GPU arm
# (no full-factor D2H — only b/x vectors cross). CPU arms: warm factor + solve.
#
# Verdict clauses:
#   (1) GPU factor+solve ≤ CPU-PureSparse factor+solve / 5           [amendment B, 5×]
#   (3) GPU-enabled PureSparse factor+solve < CHOLMOD+OpenBLAS       [req-2 carried forward]
# (Clause 2 — no regression on the M1/M2 gate set — is benchmark/gate.jl + gate_ldlt.jl run
#  separately; the auto frontier keeps those small matrices on CPU, so they are unchanged.)
#
# Frontier cutoff: a small deterministic sweep over top-flop quantiles, best kept (the
# auto-frontier policy targets this near-optimal cutoff; recorded per matrix as `q`).
#
# Methodology (CLAUDE.md req 2): Chairmarks medians for the CPU arms, median of CUDA.@elapsed
# for the GPU arm; single-thread pinned; results→JSON; run on a clock-locked host.
#
# Usage:
#   julia --project=gpu_probe benchmark/gpu_gate.jl          # measure + save + verdict  (needs CUDA/galen)
#   julia --project=gpu_probe benchmark/gpu_gate.jl report   # verdict from the saved JSON only (no GPU)

using PureSparse, SparseArrays, LinearAlgebra, Random, Statistics, JSON, Printf
using Chairmarks: @be

const REPORT_ONLY = length(ARGS) >= 1 && ARGS[1] == "report"

LinearAlgebra.BLAS.set_num_threads(1)
const SAMPLES = 20
const SECONDS = 2.0
const GPU_REPS = 9                      # median of CUDA.@elapsed for the device arm
const TARGET = 5.0                      # amendment B clause-1 margin

_median_time(b) = median(Float64[s.time for s in b.samples])

# CUDA (and the `@elapsed` macro, which must expand at load time) live only in the measure
# path; the report path parses the saved JSON with no GPU dependency.
if !REPORT_ONLY
    using CUDA, KernelAbstractions
    # @eval defers lowering (and the CUDA.@elapsed macro expansion) to runtime, after `using CUDA`
    @eval _gpu_elapsed(f) = median(Float64[CUDA.@elapsed(f()) for _ in 1:GPU_REPS])
end

# ---------------- stratum: large SPD grids + SQD KKTs (≥6, fit ~9 GB) ----------------
grid3d(d) = (n = d^3; A = spzeros(n, n); lin(i, j, k) = ((k - 1) * d + (j - 1)) * d + i;
    for k in 1:d, j in 1:d, i in 1:d
        p = lin(i, j, k); A[p, p] = 6.0
        i < d && (A[p, lin(i + 1, j, k)] = A[lin(i + 1, j, k), p] = -1.0)
        j < d && (A[p, lin(i, j + 1, k)] = A[lin(i, j + 1, k), p] = -1.0)
        k < d && (A[p, lin(i, j, k + 1)] = A[lin(i, j, k + 1), p] = -1.0)
    end; A + 0.1I)

function stratum()
    rng = MersenneTwister(5)
    spd = [("spd_grid_$(d)", d, sparse(grid3d(d))) for d in (28, 32, 36, 40, 44)]
    sqd = NamedTuple[]
    for d in (28, 32, 36, 40, 44)
        H = grid3d(d); n1 = size(H, 1); n2 = n1 ÷ 50
        Ac = sprand(rng, n2, n1, 1.0 / n1); Dm = sparse(2.0I, n2, n2)
        K = sparse([H Ac'; Ac -Dm])
        push!(sqd, (label = "sqd_kkt_$(d)", d = d, K = K, n1 = n1, n2 = n2))
    end
    return spd, sqd
end

# frontier flop quantile → cutoff value (top-flop supernodes go on GPU)
flop_cutoff(cpu, q) = begin
    snf = sort([sum(Float64(cpu.colcount[j])^2 for j in cpu.super[s]:(cpu.super[s + 1] - 1)) for s in 1:cpu.nsuper])
    snf[clamp(round(Int, q * cpu.nsuper), 1, cpu.nsuper)]
end
const QSWEEP = (0.999, 0.995, 0.99, 0.98)

# ============================ SPD =============================
function bench_spd(label, A, ext)
    n = size(A, 1); b = randn(MersenneTwister(1), n)
    r = Dict{String,Any}("matrix" => label, "n" => n, "nnz" => nnz(A), "kind" => "spd")

    # --- arm 1: CPU PureSparse (factor+solve) ---
    sym = PureSparse.symbolic(A); F = PureSparse.cholesky(sym, A)
    fs_cpu() = (PureSparse.cholesky!(F, A); PureSparse.solve!(similar(b), F, b))
    r["cpu_ps"] = _median_time(@be fs_cpu() seconds = SECONDS samples = SAMPLES evals = 1)
    r["nnzL"] = Int(sym.nnzL)

    # --- arm 3: CHOLMOD+OpenBLAS (own + same-perm) ---
    Fc = LinearAlgebra.cholesky(Symmetric(A)); fc_own() = (LinearAlgebra.cholesky!(Fc, Symmetric(A)); Fc \ b)
    r["cholmod_own"] = _median_time(@be fc_own() seconds = SECONDS samples = SAMPLES evals = 1)
    Fcp = LinearAlgebra.cholesky(Symmetric(A); perm = Vector{Int}(sym.perm))
    fc_sp() = (LinearAlgebra.cholesky!(Fcp, Symmetric(A)); Fcp \ b)
    r["cholmod_sameperm"] = _median_time(@be fc_sp() seconds = SECONDS samples = SAMPLES evals = 1)

    # --- arm 2: GPU pure (device factor + device solve), best over the frontier sweep ---
    if !REPORT_ONLY
        best = (Inf, 0.0, 0)
        for q in QSWEEP
            G = ext.gpu_symbolic(A; ordering = PureSparse.AMDOrdering(), frontier_cutoff = flop_cutoff(sym, q))
            ng = count(G.on_gpu); ng == 0 && continue
            M = ext.mf_symbolic(G.cpu)
            (M.arena_peak + G.xlen) * 8 / 1e9 > 9.5 && continue
            xh = Vector{Float64}(undef, G.xlen); ha = Vector{Float64}(undef, max(M.arena_peak, 1))
            da = CUDA.zeros(Float64, max(M.arena_peak, 1)); dz = CUDA.zeros(Float64, G.xlen)
            perm = G.cpu.perm; mer = max(G.cpu.max_extend_rows, 1)
            dup = CUDA.zeros(Float64, mer); dga = CUDA.zeros(Float64, mer)
            db = CuArray(b[perm])
            sched = ext.solve_schedule(G)                 # analysis-once level schedule
            factsolve() = begin
                ext.gpu_multifrontal_hybrid!(xh, dz, ha, da, M, G, A; d2h = false)
                ext.gpu_upload_cpu_panels!(dz, xh, G)
                ext.gpu_solve!(db, dz, G, dup, dga; sched = sched)
            end
            factsolve()                                   # warm
            t = _gpu_elapsed(factsolve)
            t < best[1] && (best = (t, q, ng))
            CUDA.unsafe_free!(da); CUDA.unsafe_free!(dz)
        end
        r["gpu_ps"] = best[1]; r["gpu_q"] = best[2]; r["gpu_fronts"] = best[3]
    end
    return r
end

# ============================ SQD/LDLᵀ =============================
function bench_sqd(nt, ext)
    K = nt.K; n = size(K, 1); n1 = nt.n1
    signs0 = Int8[i ≤ n1 ? 1 : -1 for i in 1:n]; b = randn(MersenneTwister(2), n)
    r = Dict{String,Any}("matrix" => nt.label, "n" => n, "nnz" => nnz(K), "kind" => "sqd")

    # --- arm 1: CPU PureSparse ---
    sym = PureSparse.symbolic(K); F = PureSparse.ldlt(sym, K; signs = signs0)
    fs_cpu() = (PureSparse.ldlt!(F, K); PureSparse.solve!(similar(b), F, b))
    r["cpu_ps"] = _median_time(@be fs_cpu() seconds = SECONDS samples = SAMPLES evals = 1)
    r["nnzL"] = Int(sym.nnzL)

    # --- arm 3: CHOLMOD+OpenBLAS ldlt (own + same-perm) ---
    Fc = LinearAlgebra.ldlt(Symmetric(K, :L)); fc_own() = (LinearAlgebra.ldlt!(Fc, Symmetric(K, :L)); Fc \ b)
    r["cholmod_own"] = _median_time(@be fc_own() seconds = SECONDS samples = SAMPLES evals = 1)
    Fcp = LinearAlgebra.ldlt(Symmetric(K, :L); perm = Vector{Int}(sym.perm))
    fc_sp() = (LinearAlgebra.ldlt!(Fcp, Symmetric(K, :L)); Fcp \ b)
    r["cholmod_sameperm"] = _median_time(@be fc_sp() seconds = SECONDS samples = SAMPLES evals = 1)

    # --- arm 2: GPU pure LDLᵀ (device factor + device solve) ---
    if !REPORT_ONLY
        best = (Inf, 0.0, 0)
        for q in QSWEEP
            G = ext.gpu_symbolic(K; ordering = PureSparse.AMDOrdering(), frontier_cutoff = flop_cutoff(sym, q))
            ng = count(G.on_gpu); ng == 0 && continue
            M = ext.mf_symbolic(G.cpu)
            (M.arena_peak + G.xlen) * 8 / 1e9 > 9.5 && continue
            xh = Vector{Float64}(undef, G.xlen); ha = Vector{Float64}(undef, max(M.arena_peak, 1))
            da = CUDA.zeros(Float64, max(M.arena_peak, 1)); dz = CUDA.zeros(Float64, G.xlen)
            dv = Vector{Float64}(undef, n); dd = CUDA.zeros(Float64, n)
            signs = Vector{Int8}(F.signs)                 # factor-ordering signs
            perm = G.cpu.perm; mer = max(G.cpu.max_extend_rows, 1)
            dup = CUDA.zeros(Float64, mer); dga = CUDA.zeros(Float64, mer); d_dd = CUDA.zeros(Float64, n)
            sched = ext.solve_schedule(G)                 # analysis-once level schedule
            factsolve() = begin
                ext.gpu_multifrontal_ldlt_hybrid!(xh, dz, ha, da, dv, dd, M, G, K, signs; d2h = false)
                ext.gpu_upload_cpu_panels!(dz, xh, G); copyto!(d_dd, 1, dv, 1, n)
                dy = CuArray(b[perm]); ext.gpu_solve_ldlt!(dy, dz, d_dd, G, dup, dga; sched = sched)
            end
            factsolve()
            t = _gpu_elapsed(factsolve)
            t < best[1] && (best = (t, q, ng))
            CUDA.unsafe_free!(da); CUDA.unsafe_free!(dz)
        end
        r["gpu_ps"] = best[1]; r["gpu_q"] = best[2]; r["gpu_fronts"] = best[3]
    end
    return r
end

function run_gate()
    ext = Base.get_extension(PureSparse, :PureSparseCUDAExt)
    @assert !isnothing(ext) "CUDA extension not loaded"
    spd, sqd = stratum()
    results = Any[]
    for (label, _, A) in spd
        @info "gate SPD" label
        push!(results, bench_spd(label, A, ext))
    end
    for nt in sqd
        @info "gate SQD" label = nt.label
        push!(results, bench_sqd(nt, ext))
    end
    host = gethostname()
    payload = Dict("host" => host, "julia_version" => string(VERSION),
                   "gpu" => (CUDA.functional() ? string(CUDA.name(CUDA.device())) : "n/a"),
                   "target" => TARGET, "results" => results)
    mkpath(joinpath(@__DIR__, "results"))
    outpath = joinpath(@__DIR__, "results", "gpu_gate_$(host).json")
    open(outpath, "w") do io; JSON.print(io, payload, 2); end
    @info "wrote $outpath"
    return payload
end

function print_verdict(payload)
    R = payload["results"]; tgt = get(payload, "target", TARGET)
    println("\n=== M6 GPU wall-time gate (design_gpu.md §8, amendment B) — factor+solve, median seconds ===")
    println(@sprintf("host=%s  gpu=%s  target=%.0f×\n", payload["host"], get(payload, "gpu", "?"), tgt))
    println(rpad("matrix", 15), rpad("n", 9), rpad("CPU-PS", 12), rpad("GPU-PS", 12), rpad("CHOLMOD", 12),
            rpad("q", 7), rpad("×vsCPU", 9), rpad("c1(5×)", 8), "c3(<CHM)")
    n1 = 0; n3 = 0; nt = 0
    for r in R
        haskey(r, "gpu_ps") || continue
        nt += 1
        cpu = r["cpu_ps"]; gpu = r["gpu_ps"]; chm = min(r["cholmod_own"], r["cholmod_sameperm"])
        spd = cpu / gpu
        c1 = spd >= tgt; c3 = gpu < chm
        c1 && (n1 += 1); c3 && (n3 += 1)
        println(rpad(r["matrix"], 15), rpad(r["n"], 9),
                rpad(@sprintf("%.2fms", cpu * 1e3), 12), rpad(@sprintf("%.2fms", gpu * 1e3), 12),
                rpad(@sprintf("%.2fms", chm * 1e3), 12), rpad(@sprintf("%.3f", get(r, "gpu_q", 0.0)), 7),
                rpad(@sprintf("%.2f×", spd), 9), rpad(c1 ? "PASS" : "fail", 8), c3 ? "PASS" : "fail")
    end
    println(@sprintf("\nclause 1 (GPU ≥ %.0f× CPU-PureSparse): %d/%d", tgt, n1, nt))
    println(@sprintf("clause 3 (GPU beats CHOLMOD+OpenBLAS):    %d/%d", n3, nt))
    ok = n1 == nt && n3 == nt
    println("\nOVERALL: ", ok ? "PASS" : "NOT YET PASSING (see per-row)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    if REPORT_ONLY
        host = gethostname()
        print_verdict(JSON.parsefile(joinpath(@__DIR__, "results", "gpu_gate_$(host).json")))
    else
        print_verdict(run_gate())
    end
end
