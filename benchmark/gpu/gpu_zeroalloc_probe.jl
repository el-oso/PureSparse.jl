# Amendment A probe: warm refactor with persistent device buffers → 0 device-pool alloc + 0 pattern-H2D.
using PureSparse, CUDA, KernelAbstractions, SparseArrays, LinearAlgebra, Random
ext = Base.get_extension(PureSparse, :PureSparseCUDAExt)
rng=MersenneTwister(7)
grid3d(d)=(n=d^3; A=spzeros(n,n); lin(i,j,k)=((k-1)*d+(j-1))*d+i;
  for k in 1:d,j in 1:d,i in 1:d; p=lin(i,j,k); A[p,p]=6.0;
    i<d&&(A[p,lin(i+1,j,k)]=A[lin(i+1,j,k),p]=-1.0); j<d&&(A[p,lin(i,j+1,k)]=A[lin(i,j+1,k),p]=-1.0);
    k<d&&(A[p,lin(i,j,k+1)]=A[lin(i,j,k+1),p]=-1.0) end; A+0.1I)
H=grid3d(24); n1=size(H,1); n2=n1÷50
Ac=sprand(rng,n2,n1,1.0/n1); D=sparse(2.0I,n2,n2); K=[H Ac'; Ac -D]; n=n1+n2
signs=Int8[i≤n1 ? 1 : -1 for i in 1:n]
G0=ext.gpu_symbolic(K;ordering=PureSparse.AMDOrdering(),frontier_cutoff=0.0)
snf=sort([sum(Float64(G0.cpu.colcount[j])^2 for j in G0.cpu.super[s]:(G0.cpu.super[s+1]-1)) for s in 1:G0.cpu.nsuper])
cut=snf[clamp(round(Int,0.99*G0.cpu.nsuper),1,G0.cpu.nsuper)]
G=ext.gpu_symbolic(K;ordering=PureSparse.AMDOrdering(),frontier_cutoff=cut)
M=ext.mf_symbolic(G.cpu)
F=PureSparse.ldlt(G.cpu,K;signs=signs); PureSparse.ldlt!(F,K)   # reference (F.signs = permuted signs)
signs=F.signs                                                   # gpu path expects factor-ordering signs
# persistent buffers (built ONCE, outside the refactor loop)
xh=Vector{Float64}(undef,G.xlen); ha=Vector{Float64}(undef,max(M.arena_peak,1))
da=CUDA.zeros(Float64,max(M.arena_peak,1)); dz=CUDA.zeros(Float64,G.xlen)
dv=Vector{Float64}(undef,n); dd=CUDA.zeros(Float64,n)
d_emap=CuArray(M.emap); mer=max(G.cpu.max_extend_rows,1)
d_W=CUDA.zeros(Float64,mer,mer); d_dummy=CUDA.zeros(Float64,1,1); d_Anz=CUDA.zeros(Float64,length(K.nzval))
call(d2h)=ext.gpu_multifrontal_ldlt_hybrid!(xh,dz,ha,da,dv,dd,M,G,K,signs;d2h=d2h,
                                            d_emap=d_emap,d_W=d_W,d_dummy=d_dummy,d_Anz=d_Anz)
call(false)  # warm
dev = CUDA.@allocated call(false)   # gate path: factor device-resident, no per-refactor D2H
println("GPU=",count(G.on_gpu),"/",G.cpu.nsuper,"  device-pool alloc on warm refactor = ",dev," bytes")
# correctness: d2h=true completes the host factor for comparison to the CPU oracle
call(true)
zsud!(x,S)=(for s in 1:S.nsuper; nsc=Int(S.super[s+1])-Int(S.super[s]); nsr=Int(S.rowind_ptr[s+1])-Int(S.rowind_ptr[s]); b=Int(S.px[s]); for j in 1:nsc,i in 1:(j-1); x[b+(j-1)*nsr+(i-1)]=0.0 end end; x)
relL=norm(zsud!(copy(xh),G.cpu)-zsud!(copy(F.x),G.cpu))/norm(zsud!(copy(F.x),G.cpu))
println("relL=",round(relL,sigdigits=3)," relD=",round(norm(dv-F.d)/norm(F.d),sigdigits=3))
@assert relL<1e-9 "correctness regressed"
# residual device alloc = per-GPU-front `Array(view(panel,1:nscol,1:nscol))` D2H staging of the
# strided diagonal block (CUDA.jl stages a strided view through a contiguous device temp). Killed
# by optimization 3 (pure device LDL of the diagonal — the block never leaves the device).
println(dev==0 ? "ZERO device-pool alloc ✓ (amendment A)" :
        "residual $dev B = strided diag-block D2H staging (→ 0 after pure device potrf/LDL, opt 3)")
