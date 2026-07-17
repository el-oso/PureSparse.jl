# LDLᵀ hybrid perf spot-check: KKT [H Aᵀ; A −D] with H = 3D-grid Laplacian (big fronts),
# hybrid multifrontal LDLᵀ vs ldlt!. Confirms LDLᵀ perf mirrors Cholesky (same engine).
using PureSparse, CUDA, KernelAbstractions, SparseArrays, LinearAlgebra, Printf, Random
ext = Base.get_extension(PureSparse, :PureSparseCUDAExt)
rng = MersenneTwister(5)
function grid3d(d)
    n=d^3; A=spzeros(n,n); lin(i,j,k)=((k-1)*d+(j-1))*d+i
    for k in 1:d,j in 1:d,i in 1:d; p=lin(i,j,k); A[p,p]=6.0
        i<d&&(A[p,lin(i+1,j,k)]=A[lin(i+1,j,k),p]=-1.0); j<d&&(A[p,lin(i,j+1,k)]=A[lin(i,j+1,k),p]=-1.0)
        k<d&&(A[p,lin(i,j,k+1)]=A[lin(i,j,k+1),p]=-1.0) end; A+0.1I
end
mint(f,n=4)=minimum(f() for _ in 1:n)
for d in (24,28,36,40,44)
    H=grid3d(d); n1=size(H,1); n2=n1÷50
    Ac=sprand(rng,n2,n1,1.0/n1); D=sparse(2.0I,n2,n2); K=[H Ac'; Ac -D]
    n=n1+n2; signs=Int8[i≤n1 ? 1 : -1 for i in 1:n]
    G0=ext.gpu_symbolic(K;ordering=PureSparse.AMDOrdering(),frontier_cutoff=0.0)
    F=PureSparse.ldlt(G0.cpu,K;signs=signs); @assert PureSparse.issuccess(F)
    snf=sort([sum(Float64(G0.cpu.colcount[j])^2 for j in G0.cpu.super[s]:(G0.cpu.super[s+1]-1)) for s in 1:G0.cpu.nsuper])
    PureSparse.ldlt!(F,K); cpu_t=mint(()->@elapsed(PureSparse.ldlt!(F,K)))
    @printf("\nKKT H=%d³ n=%d nnzL=%d  CPU ldlt! %.0f ms\n",d,n,G0.cpu.nnzL,cpu_t*1e3)
    best=(Inf,0.0,0)
    for qtl in (0.999,0.998,0.995,0.99)
        cut=snf[clamp(round(Int,qtl*G0.cpu.nsuper),1,G0.cpu.nsuper)]
        G=ext.gpu_symbolic(K;ordering=PureSparse.AMDOrdering(),frontier_cutoff=cut); ng=count(G.on_gpu); ng==0&&continue
        M=ext.mf_symbolic(G.cpu)
        gb=(M.arena_peak+G.xlen)*8/1e9
        if gb>9.5; @printf("  q=%.3f arena %.2f GB > 9.5 — skip\n",qtl,gb); continue; end
        xh=Vector{Float64}(undef,G.xlen); ha=Vector{Float64}(undef,max(M.arena_peak,1))
        da=CUDA.zeros(Float64,max(M.arena_peak,1)); dz=CUDA.zeros(Float64,G.xlen)
        dv=Vector{Float64}(undef,n); dd=CUDA.zeros(Float64,n)
        ext.gpu_multifrontal_ldlt_hybrid!(xh,dz,ha,da,dv,dd,M,G,K,G.cpu isa Any ? Int8[i≤n1 ? 1 : -1 for i in 1:n] : signs;d2h=false)
        t=mint(()->CUDA.@elapsed(ext.gpu_multifrontal_ldlt_hybrid!(xh,dz,ha,da,dv,dd,M,G,K,Int8[i≤n1 ? 1 : -1 for i in 1:n];d2h=false)))
        @printf("  q=%.3f GPU=%d/%d  %.0f ms  %.2fx\n",qtl,ng,G0.cpu.nsuper,t*1e3,cpu_t/t)
        t<best[1]&&(best=(t,qtl,ng)); CUDA.unsafe_free!(da);CUDA.unsafe_free!(dz)
    end
    @printf("  BEST %.2fx\n",cpu_t/best[1])
end
