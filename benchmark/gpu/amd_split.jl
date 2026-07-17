using AMDGPU, KernelAbstractions, LinearAlgebra, Random, Printf
using KernelAbstractions: @kernel, @index, @localmem, @private, @synchronize, get_backend, @atomic
using Base.Cartesian: @nexprs
const BK = ROCBackend(); roc(A) = ROCArray(A)
# pure gemm (needed by the kernel files)
@kernel unsafe_indices=true function _gemm_nt_4x4!(C,@Const(A),@Const(B),alpha,beta,M,N,K)
  T=eltype(C); li=@index(Local,NTuple); gi=@index(Group,NTuple); tx=li[1];ty=li[2]; tid=(ty-1)*16+(tx-1)
  br=(gi[1]-1)*64; bc=(gi[2]-1)*64; As=@localmem T (64,8); Bs=@localmem T (64,8); acc=@private T (4,4)
  @inbounds for i in 1:4,j in 1:4; acc[i,j]=zero(T); end
  k0=0; nt=div(K+7,8)
  for _ in 1:nt
    @inbounds for t in 1:2; p=tid+(t-1)*256; ml=p&63; kl=p>>6; gr=br+ml; gk=k0+kl
      As[ml+1,kl+1]=(gr<M&&gk<K) ? A[gr+1,gk+1] : zero(T); gc=bc+ml
      Bs[ml+1,kl+1]=(gc<N&&gk<K) ? B[gc+1,gk+1] : zero(T); end
    @synchronize
    @inbounds for kk in 1:8; @nexprs 4 i->(a_i=As[(tx-1)*4+i,kk]); @nexprs 4 j->(b_j=Bs[(ty-1)*4+j,kk])
      @nexprs 4 i-> @nexprs 4 j->(acc[i,j]=muladd(a_i,b_j,acc[i,j])); end
    @synchronize; k0+=8; end
  @inbounds for i in 1:4,j in 1:4; gr=br+(tx-1)*4+(i-1); gc=bc+(ty-1)*4+(j-1)
    if gr<M&&gc<N; C[gr+1,gc+1]= beta==zero(T) ? alpha*acc[i,j] : muladd(alpha,acc[i,j],beta*C[gr+1,gc+1]); end; end
end
function gpu_gemm_nt!(C,A,B,alpha,beta); M,K=size(A); N=size(B,1)
  _gemm_nt_4x4!(get_backend(C),(16,16))(C,A,B,alpha,beta,M,N,K; ndrange=(cld(M,64)*16,cld(N,64)*16)); C; end
gpu_syrk_nt!(C,A,a,b)=gpu_gemm_nt!(C,A,A,a,b)
include(joinpath(@__DIR__,"..","..","ext","gpu_dense.jl"))
rel(a,b)=norm(a-b)/max(norm(b),eps()); zl(A)=(B=copy(A); for j in axes(B,2),i in 1:(j-1); B[i,j]=0.0; end; B)
function t(nscol,below,mode)
  rng=MersenneTwister(1); nsrow=nscol+below; Mm=randn(rng,nscol,nscol); P11=Mm'*Mm+nscol*I; A21=randn(rng,below,nscol)
  P=zeros(nsrow,nscol); P[1:nscol,1:nscol]=P11; P[(nscol+1):nsrow,1:nscol]=A21
  L11=Matrix(cholesky(Symmetric(P11,:L)).L); L21=A21/L11'
  ws=FrontWS(BK,Float64,cld(nscol,64)); dP=roc(P)
  try
    gpu_front!(dP,nscol,ws;mode=mode); KernelAbstractions.synchronize(BK); H=Array(dP)
    rL=rel(zl(H[1:nscol,1:nscol]),zl(L11)); rP=rel(H[(nscol+1):nsrow,1:nscol],L21)
    @printf("  front(%s) %dx%d  relL=%.2e relP=%.2e  %s\n",mode,nscol,below,rL,rP,(rL<1e-9&&rP<1e-9) ? "OK" : "FAIL")
  catch e; @printf("  front(%s) %dx%d  ERROR %s\n",mode,nscol,below,sprint(showerror,e)[1:min(end,70)]); end
end
println("dev ",AMDGPU.device())
t(55,754,:split_gemm); t(234,1436,:split_gemm); t(300,2000,:split_gemm)
