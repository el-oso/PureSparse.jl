# Rough perf signal: is the (unoptimized, synchronous, all-GPU) gpu_cholesky_sync! already
# competitive with CPU cholesky! on a FLOP-RICH matrix (large supernodes, where GPU's win isn't
# swamped by small-supernode launch latency)? NOT the gate number (no warm pre-alloc, no hybrid
# frontier, no streams) — just a direction check on whether the GPU approach pays off.
using PureSparse, CUDA, KernelAbstractions, SparseArrays, LinearAlgebra, Printf
ext = Base.get_extension(PureSparse, :PureSparseCUDAExt)

# 3-D grid Laplacian (larger fronts than 2-D → bigger supernodes)
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

for (nx,ny,nz) in [(24,24,24), (30,30,30)]
    A = grid3d(nx,ny,nz)
    G = ext.gpu_symbolic(A; ordering=PureSparse.AMDOrdering(), frontier_cutoff=0.0)
    F = PureSparse.cholesky(G.cpu, A)
    dx = CUDA.zeros(Float64, G.xlen)

    # warm both
    PureSparse.cholesky!(F, A); ext.gpu_cholesky_sync!(dx, G, A)
    # time (median of a few)
    cpu_t = minimum(@elapsed(PureSparse.cholesky!(F, A)) for _ in 1:5)
    gpu_t = minimum(CUDA.@elapsed(ext.gpu_cholesky_sync!(dx, G, A)) for _ in 1:5)
    gf = 2 * G.cpu.flops / 1e9
    @printf("grid3d %d³ (n=%d, nnzL=%d, %.1f GF): CPU %.1f ms  GPU-sync %.1f ms  speedup %.2fx\n",
            nx, nx*ny*nz, G.cpu.nnzL, gf, cpu_t*1e3, gpu_t*1e3, cpu_t/gpu_t)
end
println("(GPU-sync is UNOPTIMIZED: no warm pre-alloc, no hybrid frontier, no streams — a floor, not the gate)")
