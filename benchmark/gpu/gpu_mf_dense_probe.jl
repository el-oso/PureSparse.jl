# Quick, non-systematic: hybrid-multifrontal speedup for small dense-ish SPD matrices
# (n=200..1000, fill 5%..50%). A different regime from sparse 3D grids: few big fronts.
using PureSparse, CUDA, KernelAbstractions, SparseArrays, LinearAlgebra, Printf, Random
ext = Base.get_extension(PureSparse, :PureSparseCUDAExt)
mint(f,n=5)=minimum(f() for _ in 1:n)
rng=MersenneTwister(7)
@printf("%-6s %-6s %-9s %-9s %-9s %-6s\n","n","fill","nnzL","CPU ms","GPU ms","x")
for n in (200,500,1000), fill in (0.05,0.2,0.5)
    A=sprand(rng,n,n,fill); A=A+A'; A=A+(2*(n*fill)+1)*I    # SPD (diag-dominant)
    S=PureSparse.symbolic(A); F=PureSparse.cholesky(S,A)
    PureSparse.issuccess(F) || (println("  n=$n fill=$fill NOT SPD, skip"); continue)
    snf=sort([sum(Float64(S.colcount[j])^2 for j in S.super[s]:(S.super[s+1]-1)) for s in 1:S.nsuper])
    PureSparse.cholesky!(F,A); cpu_t=mint(()->@elapsed(PureSparse.cholesky!(F,A)))
    best=Inf
    for qtl in (0.0,0.5,0.9,0.98)          # 0.0 = all-GPU; dense => few fronts
        cut = qtl==0.0 ? 0.0 : snf[clamp(round(Int,qtl*S.nsuper),1,S.nsuper)]
        G=ext.gpu_symbolic(A;ordering=PureSparse.AMDOrdering(),frontier_cutoff=cut); count(G.on_gpu)==0 && continue
        M=ext.mf_symbolic(G.cpu)
        xh=Vector{Float64}(undef,G.xlen); ha=Vector{Float64}(undef,max(M.arena_peak,1))
        da=CUDA.zeros(Float64,max(M.arena_peak,1)); dz=CUDA.zeros(Float64,G.xlen)
        ext.gpu_multifrontal_hybrid!(xh,dz,ha,da,M,G,A;d2h=false)
        t=mint(()->CUDA.@elapsed(ext.gpu_multifrontal_hybrid!(xh,dz,ha,da,M,G,A;d2h=false)))
        best=min(best,t)
    end
    @printf("%-6d %-6.2f %-9d %-9.2f %-9.2f %-6.2f\n",n,fill,S.nnzL,cpu_t*1e3,best*1e3,cpu_t/best)
end
