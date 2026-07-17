# Characterize the GPU-front shapes (nscol, below_s) the multifrontal actually factors at the
# best cutoff — the real optimization target for the pure potrf/trsm (vs Fable's isolated panels).
using PureSparse, CUDA, KernelAbstractions, SparseArrays, LinearAlgebra, Printf, Random
ext = Base.get_extension(PureSparse, :PureSparseCUDAExt)
rng=MersenneTwister(5)
grid3d(d)=(n=d^3; A=spzeros(n,n); lin(i,j,k)=((k-1)*d+(j-1))*d+i;
  for k in 1:d,j in 1:d,i in 1:d; p=lin(i,j,k); A[p,p]=6.0;
    i<d&&(A[p,lin(i+1,j,k)]=A[lin(i+1,j,k),p]=-1.0); j<d&&(A[p,lin(i,j+1,k)]=A[lin(i,j+1,k),p]=-1.0);
    k<d&&(A[p,lin(i,j,k+1)]=A[lin(i,j,k+1),p]=-1.0) end; A+0.1I)
for d in (36,44)
  H=grid3d(d); n1=size(H,1); n2=n1÷50
  Ac=sprand(rng,n2,n1,1.0/n1); D=sparse(2.0I,n2,n2); K=[H Ac'; Ac -D]
  G0=ext.gpu_symbolic(K;ordering=PureSparse.AMDOrdering(),frontier_cutoff=0.0)
  snf=sort([sum(Float64(G0.cpu.colcount[j])^2 for j in G0.cpu.super[s]:(G0.cpu.super[s+1]-1)) for s in 1:G0.cpu.nsuper])
  cut=snf[clamp(round(Int,0.99*G0.cpu.nsuper),1,G0.cpu.nsuper)]
  G=ext.gpu_symbolic(K;ordering=PureSparse.AMDOrdering(),frontier_cutoff=cut)
  sym=G.cpu
  shapes=[(Int(sym.super[s+1])-Int(sym.super[s]),
           Int(sym.rowind_ptr[s+1])-Int(sym.rowind_ptr[s])-(Int(sym.super[s+1])-Int(sym.super[s])))
          for s in 1:sym.nsuper if G.on_gpu[s]]
  ng=length(shapes); nscols=sort([x[1] for x in shapes]); belows=sort([x[2] for x in shapes])
  @printf("\nH=%d³  GPU fronts=%d/%d\n",d,ng,sym.nsuper)
  q(v,p)=v[clamp(round(Int,p*length(v)),1,length(v))]
  @printf("  nscol  (block width): min=%d  p50=%d  p90=%d  max=%d\n",nscols[1],q(nscols,.5),q(nscols,.9),nscols[end])
  @printf("  below_s (panel rows):  min=%d  p50=%d  p90=%d  max=%d\n",belows[1],q(belows,.5),q(belows,.9),belows[end])
  # bucket by nscol to see where the flops concentrate
  for (lo,hi) in ((1,64),(65,200),(201,500),(501,1000),(1001,10^9))
    sel=[x for x in shapes if lo<=x[1]<=hi]
    isempty(sel) && continue
    fl=sum(x[1]^2*x[2] + x[1]^3/3 for x in sel)  # ~trsm+potrf flops
    @printf("  nscol∈[%d,%d]: %d fronts, ~%.1e panel-flops, e.g. shapes %s\n",
            lo,hi,length(sel),fl, string([(x[1],x[2]) for x in sel[1:min(3,end)]]))
  end
end
