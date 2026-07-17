# Diagnose the §8 gate's factor+solve gap: decompose the GPU path (SQD 40³ KKT) into
# factor (persistent buffers vs rebuilt-each-call), make-solve-ready, and solve.
using PureSparse, CUDA, KernelAbstractions, SparseArrays, LinearAlgebra, Statistics, Printf, Random
ext = Base.get_extension(PureSparse, :PureSparseCUDAExt)
med(f,n=9)=median(Float64[CUDA.@elapsed(f()) for _ in 1:n])
grid3d(d)=(n=d^3; A=spzeros(n,n); lin(i,j,k)=((k-1)*d+(j-1))*d+i;
  for k in 1:d,j in 1:d,i in 1:d; p=lin(i,j,k); A[p,p]=6.0;
    i<d&&(A[p,lin(i+1,j,k)]=A[lin(i+1,j,k),p]=-1.0); j<d&&(A[p,lin(i,j+1,k)]=A[lin(i,j+1,k),p]=-1.0);
    k<d&&(A[p,lin(i,j,k+1)]=A[lin(i,j,k+1),p]=-1.0) end; A+0.1I)
rng=MersenneTwister(5); d=40
H=grid3d(d); n1=size(H,1); n2=n1÷50
Ac=sprand(rng,n2,n1,1.0/n1); Dm=sparse(2.0I,n2,n2); K=sparse([H Ac'; Ac -Dm]); n=n1+n2
signs=Int8[i≤n1 ? 1 : -1 for i in 1:n]; b=randn(rng,n)
sym=PureSparse.symbolic(K)
snf=sort([sum(Float64(sym.colcount[j])^2 for j in sym.super[s]:(sym.super[s+1]-1)) for s in 1:sym.nsuper])
cut=snf[clamp(round(Int,0.98*sym.nsuper),1,sym.nsuper)]
G=ext.gpu_symbolic(K;ordering=PureSparse.AMDOrdering(),frontier_cutoff=cut); M=ext.mf_symbolic(G.cpu)
ng=count(G.on_gpu); @printf("SQD 40³: n=%d  GPU fronts=%d/%d\n",n,ng,sym.nsuper)
xh=Vector{Float64}(undef,G.xlen); ha=Vector{Float64}(undef,max(M.arena_peak,1))
da=CUDA.zeros(Float64,max(M.arena_peak,1)); dz=CUDA.zeros(Float64,G.xlen)
dv=Vector{Float64}(undef,n); dd=CUDA.zeros(Float64,n)
perm=G.cpu.perm; mer=max(G.cpu.max_extend_rows,1)
dup=CUDA.zeros(Float64,mer); dga=CUDA.zeros(Float64,mer); d_dd=CUDA.zeros(Float64,n)
# persistent buffers (amendment A)
d_emap=CuArray(M.emap); d_W=CUDA.zeros(Float64,mer,mer); d_dummy=CUDA.zeros(Float64,1,1)
d_Anz=CUDA.zeros(Float64,length(K.nzval)); d_signs=CuArray(signs)
ldlws=Base.get_extension(PureSparse,:PureSparseCUDAExt).LDLFrontWS(get_backend(dz),Float64)
fac_rebuild()=ext.gpu_multifrontal_ldlt_hybrid!(xh,dz,ha,da,dv,dd,M,G,K,signs;d2h=false)
fac_persist()=ext.gpu_multifrontal_ldlt_hybrid!(xh,dz,ha,da,dv,dd,M,G,K,signs;d2h=false,
                d_emap=d_emap,d_W=d_W,d_dummy=d_dummy,d_Anz=d_Anz,d_signs=d_signs,ldlws=ldlws)
fac_persist()  # warm
msr()=(ext.gpu_upload_cpu_panels!(dz,xh,G); copyto!(d_dd,1,dv,1,n))
rhs()=CuArray(b[perm])
sched=ext.solve_schedule(G)   # analysis-once level schedule (built at setup, reused per solve)
dy=CuArray(b[perm]); slv()=ext.gpu_solve_ldlt!(dy,dz,d_dd,G,dup,dga;sched=sched)
msr(); slv()
@printf("  factor (rebuild d_emap/signs/ws each call) : %7.1f ms\n", med(fac_rebuild)*1e3)
@printf("  factor (persistent buffers, amendment A)   : %7.1f ms\n", med(fac_persist)*1e3)
@printf("  make-solve-ready (upload CPU panels + D)    : %7.1f ms\n", med(msr)*1e3)
@printf("  RHS upload (b[perm] H2D)                     : %7.1f ms\n", med(rhs)*1e3)
@printf("  device solve                                 : %7.1f ms\n", med(slv)*1e3)
F=PureSparse.ldlt(sym,K;signs=signs)
cpu=med(()->0.0); using Chairmarks
cput=minimum(x.time for x in (@be (PureSparse.ldlt!(F,K); PureSparse.solve!(similar(b),F,b)) seconds=2 samples=15 evals=1).samples)
@printf("  --- CPU-PureSparse factor+solve             : %7.1f ms\n", cput*1e3)
