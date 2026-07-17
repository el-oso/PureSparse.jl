# Path B payoff: all-GPU multifrontal vs CPU cholesky! on the SAME 3D grids where left-looking
# was 0.72-0.95x. Did front assembly fix the launch-bound problem?
using PureSparse, CUDA, KernelAbstractions, SparseArrays, LinearAlgebra, Printf
ext = Base.get_extension(PureSparse, :PureSparseCUDAExt)

function grid3d(nx,ny,nz)
    n=nx*ny*nz; A=spzeros(n,n); lin(i,j,k)=((k-1)*ny+(j-1))*nx+i
    for k in 1:nz,j in 1:ny,i in 1:nx; p=lin(i,j,k); A[p,p]=6.0
        i<nx&&(A[p,lin(i+1,j,k)]=A[lin(i+1,j,k),p]=-1.0); j<ny&&(A[p,lin(i,j+1,k)]=A[lin(i,j+1,k),p]=-1.0)
        k<nz&&(A[p,lin(i,j,k+1)]=A[lin(i,j,k+1),p]=-1.0) end; A+0.1I
end
mint(f,n=5)=minimum(f() for _ in 1:n)

for (nx,ny,nz) in [(28,28,28),(32,32,32),(40,40,40)]
    A=grid3d(nx,ny,nz)
    G=ext.gpu_symbolic(A; ordering=PureSparse.AMDOrdering(), frontier_cutoff=0.0)
    F=PureSparse.cholesky(G.cpu,A); M=ext.mf_symbolic(G.cpu)
    d_nzval=CUDA.zeros(Float64,G.xlen); d_arena=CUDA.zeros(Float64,max(M.arena_peak,1))
    gf=2*G.cpu.flops/1e9
    PureSparse.cholesky!(F,A); ext.gpu_multifrontal_cholesky!(d_nzval,d_arena,M,G,A)  # warm
    cpu_t=mint(()->@elapsed(PureSparse.cholesky!(F,A)))
    mf_t =mint(()->CUDA.@elapsed(ext.gpu_multifrontal_cholesky!(d_nzval,d_arena,M,G,A)))
    @printf("grid3d %d³  n=%d nnzL=%d %.1f GF  arena %.0f MB   CPU %.1f ms   MF-GPU %.1f ms   %.2fx\n",
            nx, nx*ny*nz, G.cpu.nnzL, gf, M.arena_peak*8/1e6, cpu_t*1e3, mf_t*1e3, cpu_t/mf_t)
end
println("(all-GPU multifrontal, unoptimized: monotonic arena, per-call d_emap upload, no streams)")
