using PureSparse, CUDA, KernelAbstractions, SparseArrays, LinearAlgebra, Printf
ext = Base.get_extension(PureSparse, :PureSparseCUDAExt)
function grid3d(nx,ny,nz)
    n=nx*ny*nz; A=spzeros(n,n); lin(i,j,k)=((k-1)*ny+(j-1))*nx+i
    for k in 1:nz,j in 1:ny,i in 1:nx; p=lin(i,j,k); A[p,p]=6.0
        i<nx&&(A[p,lin(i+1,j,k)]=A[lin(i+1,j,k),p]=-1.0); j<ny&&(A[p,lin(i,j+1,k)]=A[lin(i,j+1,k),p]=-1.0)
        k<nz&&(A[p,lin(i,j,k+1)]=A[lin(i,j,k+1),p]=-1.0) end; A+0.1I
end
mint(f,n=4)=minimum(f() for _ in 1:n)
for d in (52,56)
    A=grid3d(d,d,d); S=PureSparse.symbolic(A); F=PureSparse.cholesky(S,A)
    snf=sort([sum(Float64(S.colcount[j])^2 for j in S.super[s]:(S.super[s+1]-1)) for s in 1:S.nsuper])
    PureSparse.cholesky!(F,A); cpu_t=mint(()->@elapsed(PureSparse.cholesky!(F,A)))
    best=(Inf,0.0,0)
    for qtl in (0.9995,0.999,0.998,0.995,0.99)
        cut=snf[clamp(round(Int,qtl*S.nsuper),1,S.nsuper)]
        G=ext.gpu_symbolic(A;ordering=PureSparse.AMDOrdering(),frontier_cutoff=cut); ng=count(G.on_gpu); ng==0&&continue
        M=ext.mf_symbolic(G.cpu)
        arena_gb=M.arena_peak*8/1e9; factor_gb=G.xlen*8/1e9
        if arena_gb+factor_gb > 10.5; @printf("  d=%d q=%.4f SKIP (arena %.1f + factor %.1f GB > 10.5)\n",d,qtl,arena_gb,factor_gb); continue; end
        xh=Vector{Float64}(undef,G.xlen); ha=Vector{Float64}(undef,max(M.arena_peak,1))
        da=CUDA.zeros(Float64,max(M.arena_peak,1)); dz=CUDA.zeros(Float64,G.xlen)
        ext.gpu_multifrontal_hybrid!(xh,dz,ha,da,M,G,A;d2h=false)
        t=mint(()->CUDA.@elapsed(ext.gpu_multifrontal_hybrid!(xh,dz,ha,da,M,G,A;d2h=false)))
        @printf("  d=%d q=%.4f GPU=%d/%d arena=%.1fGB  %.0f ms  %.2fx\n",d,qtl,ng,S.nsuper,arena_gb,t*1e3,cpu_t/t)
        t<best[1]&&(best=(t,qtl,ng)); CUDA.unsafe_free!(da); CUDA.unsafe_free!(dz)
    end
    @printf("d=%d³ n=%d nnzL=%d %.1fGF  CPU %.0f ms  BEST %.2fx\n",d,d^3,S.nnzL,2*S.flops/1e9,cpu_t*1e3,cpu_t/best[1])
end
