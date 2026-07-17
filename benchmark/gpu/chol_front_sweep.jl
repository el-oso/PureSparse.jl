# Map where the pure Cholesky front drops below 1.0× vs cuSOLVER potrf + cuBLAS trsm, across nscol.
using PureSparse, CUDA, KernelAbstractions, LinearAlgebra, Random, Statistics, Printf
using CUDA.CUSOLVER: potrf!
using CUDA.CUBLAS: trsm!
ext = Base.get_extension(PureSparse, :PureSparseCUDAExt)
med(f,n=15)=median(Float64[CUDA.@elapsed(f()) for _ in 1:n])
function one(nscol, below)
    rng=MersenneTwister(1); nsrow=nscol+below
    Mm=randn(rng,nscol,nscol); P11=Mm'*Mm+nscol*I; A21=randn(rng,below,nscol)
    P=zeros(nsrow,nscol); P[1:nscol,1:nscol]=P11; P[(nscol+1):nsrow,1:nscol]=A21
    dP0=CuArray(P)
    ws=ext.FrontWS(get_backend(dP0),Float64,cld(nscol,64))
    dP=copy(dP0); ext.gpu_front!(dP,nscol,ws); CUDA.synchronize()  # warm
    pure=med(()->(copyto!(dP,dP0); ext.gpu_front!(dP,nscol,ws)))
    # vendor: cuSOLVER potrf(diag) + cuBLAS trsm(below)
    dV=copy(dP0)
    ven=med(()->begin copyto!(dV,dP0); diag=view(dV,1:nscol,1:nscol); potrf!('L',diag)
        below>0 && trsm!('R','L','T','N',1.0,diag,view(dV,(nscol+1):nsrow,1:nscol)) end)
    @printf("  nscol=%5d below=%5d | pure %8.1fµs  vendor %8.1fµs  | ratio %.2fx %s\n",
            nscol,below,pure*1e6,ven*1e6, ven/pure, ven/pure>=1.0 ? "OK" : "<1.0")
    return ven/pure
end
println("== Cholesky front pure gpu_front!(:auto) vs cuSOLVER+cuBLAS, galen ==")
println("-- below=186 (potrf-dominated, the hard case) --")
for nc in (64,128,256,384,512,768,1024,1280,1536); one(nc,186); end
println("-- below=1000 (typical crown) --")
for nc in (64,128,256,512,1024,1536); one(nc,1000); end
