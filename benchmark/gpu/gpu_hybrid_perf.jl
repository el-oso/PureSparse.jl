# Sweep the frontier cutoff for the hybrid Cholesky and find the speed crossover vs CPU.
# The naive all-GPU floor was 0.08-0.18x; the hybrid keeps the many small supernodes on CPU and
# offloads only the few large near-root fronts to the GPU. NOTE: this hybrid still D2H's every
# GPU panel back to x_host and re-allocs per call (not the device-resident/device-solve final
# design) — so absolute times are PESSIMISTIC; the sweep SHAPE + the best cutoff are the signal.
using PureSparse, CUDA, KernelAbstractions, SparseArrays, LinearAlgebra, Printf
ext = Base.get_extension(PureSparse, :PureSparseCUDAExt)

function grid3d(nx, ny, nz)
    n = nx*ny*nz; A = spzeros(n, n)
    lin(i,j,k) = ((k-1)*ny + (j-1))*nx + i
    for k in 1:nz, j in 1:ny, i in 1:nx
        p = lin(i,j,k); A[p,p] = 6.0
        i<nx && (A[p,lin(i+1,j,k)]=A[lin(i+1,j,k),p]=-1.0)
        j<ny && (A[p,lin(i,j+1,k)]=A[lin(i,j+1,k),p]=-1.0)
        k<nz && (A[p,lin(i,j,k+1)]=A[lin(i,j,k+1),p]=-1.0)
    end
    A + 0.1I
end

mint(f, n=5) = minimum(f() for _ in 1:n)

for (nx,ny,nz) in [(28,28,28), (32,32,32)]
    A = grid3d(nx,ny,nz)
    S = PureSparse.symbolic(A; ordering=PureSparse.AMDOrdering())
    F = PureSparse.cholesky(S, A)
    snflop = sort([sum(Float64(S.colcount[j])^2 for j in S.super[s]:(S.super[s+1]-1)) for s in 1:S.nsuper])
    gf = 2*S.flops/1e9
    PureSparse.cholesky!(F, A)
    cpu_t = mint(() -> @elapsed(PureSparse.cholesky!(F, A)))
    @printf("\ngrid3d %d³  n=%d  nnzL=%d  %.1f GF   CPU cholesky! = %.1f ms\n",
            nx, nx*ny*nz, S.nnzL, gf, cpu_t*1e3)
    best = (Inf, 0.0, 0)
    for qtl in [0.999, 0.995, 0.99, 0.98, 0.95, 0.90, 0.80, 0.50]
        cut = snflop[clamp(round(Int, qtl*S.nsuper), 1, S.nsuper)]
        G = ext.gpu_symbolic(A; ordering=PureSparse.AMDOrdering(), frontier_cutoff=cut)
        ngpu = count(G.on_gpu); ngpu == 0 && continue
        x_host = Vector{Float64}(undef, G.xlen); d = CUDA.zeros(Float64, G.xlen)
        ext.gpu_cholesky_hybrid!(x_host, d, G, A)   # warm
        t = mint(() -> CUDA.@elapsed(ext.gpu_cholesky_hybrid!(x_host, d, G, A)))
        @printf("  cutoff q=%.3f  GPU=%4d/%d supernodes  hybrid=%.1f ms  vs CPU %.2fx\n",
                qtl, ngpu, S.nsuper, t*1e3, cpu_t/t)
        t < best[1] && (best = (t, qtl, ngpu))
    end
    @printf("  BEST: %.1f ms at q=%.3f (%d GPU supernodes) -> %.2fx vs CPU\n",
            best[1]*1e3, best[2], best[3], cpu_t/best[1])
end
println("\n(hybrid is pessimistic: D2H-per-GPU-panel + per-call allocs not yet removed)")
