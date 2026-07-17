# Standalone AMD (ROCm) validation of the LEVEL-SCHEDULED batched device solve
# (ext/gpu_solve.jl) — proves the portability claim: the kernels (incl. the bare-atomic
# scatter) COMPILE + RUN on gfx1151 and match a CPU factor solve at machine precision.
# Perf on this FP64-weak iGPU is moot. Pattern follows amd_kernel_test.jl: include the
# backend-generic file under ROCBackend; the factor itself comes from the CPU path
# (PureSparse.cholesky/ldlt) and is uploaded — only the SOLVE runs on device.
#
#   julia --project=/home/el_oso/Documents/claude/amd_probe benchmark/gpu/amd_solve_test.jl

using AMDGPU, KernelAbstractions, LinearAlgebra, SparseArrays, Random, Printf
using KernelAbstractions: @kernel, @index, @synchronize, get_backend, @atomic
using PureSparse

@assert AMDGPU.functional() "AMDGPU not functional"
const BK = ROCBackend()
include(joinpath(@__DIR__, "..", "..", "ext", "gpu_solve.jl"))

rng = MersenneTwister(11)

function dev_solve(sym, x_packed, dvec, b)          # upload factor + pattern, batched solve
    d_x = ROCArray(x_packed)
    d_ri = ROCArray(sym.rowind); d_rp = ROCArray(sym.rowind_ptr); d_su = ROCArray(sym.super)
    sched = solve_schedule(sym.sparent, sym.px, d_ri)
    d_y = ROCArray(b[sym.perm])
    d_d = isnothing(dvec) ? nothing : ROCArray(dvec)
    batched_solve!(d_y, d_x, d_ri, d_rp, d_su, sched, !isnothing(dvec), d_d)
    KernelAbstractions.synchronize(BK)
    x = zeros(length(b)); x[sym.perm] = Array(d_y)
    return x
end

function t_chol(A, label)
    sym = PureSparse.symbolic(A)
    F = PureSparse.cholesky(sym, A)
    b = randn(rng, size(A, 1))
    x = dev_solve(sym, F.x, nothing, b)
    res = norm(A * x - b) / norm(b)
    @printf("  %-12s Cholesky  res=%.2e  %s\n", label, res, res < 1e-8 ? "OK" : "FAIL")
    return res < 1e-8
end

function t_ldlt(K, signs, label)
    sym = PureSparse.symbolic(K)
    F = PureSparse.ldlt(sym, K; signs = signs)
    b = randn(rng, size(K, 1))
    x = dev_solve(sym, F.x, F.d, b)
    res = norm(K * x - b) / norm(b)
    @printf("  %-12s LDLt      res=%.2e  %s\n", label, res, res < 1e-8 ? "OK" : "FAIL")
    return res < 1e-8
end

grid2d(m) = (n = m^2; A = spzeros(n, n);
    for j in 1:m, i in 1:m
        k = (j - 1) * m + i; A[k, k] = 4.0
        i < m && (A[k, k + 1] = A[k + 1, k] = -1.0)
        j < m && (A[k, k + m] = A[k + m, k] = -1.0)
    end; A + 0.05I)

println("AMDGPU: ", AMDGPU.functional(), "  device: ", AMDGPU.device())
A1 = let n = 400; S = sprand(rng, n, n, 0.02); S + S' + n * I end
A2 = grid2d(25)
K = let H = grid2d(20), n1 = 400, n2 = 20
    Ac = sprand(rng, n2, n1, 5.0 / n1)
    sparse([H Ac'; Ac -sparse(2.0I, n2, n2)])
end
signs = Int8[i ≤ 400 ? 1 : -1 for i in 1:420]

ok = all([t_chol(A1, "rand_n400"), t_chol(A2, "grid_25x25"), t_ldlt(K, signs, "kkt_420")])
println(ok ? "\nBATCHED SOLVE RUNS + MATCHES ON AMD" : "\nFAILED — see above")
