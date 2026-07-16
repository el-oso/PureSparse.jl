# Does gpu_gemm_nt! give correct results on NESTED VIEWS (reshape-of-view, then sub-view) —
# the exact operand structure gpu_cholesky_sync! feeds it? And at tiny K (small supernode
# widths)? The ext_loadtest only covered plain CuArrays at K≥32.
using PureSparse, CUDA, KernelAbstractions, LinearAlgebra, Random, Printf
ext = Base.get_extension(PureSparse, :PureSparseCUDAExt)

function main()
rng = MersenneTwister(3)
worst = 0.0
for (nsrow, ncol, q, ctot, k1) in [(50,3,5,20,4), (40,1,3,15,1), (60,2,10,25,2),
                                   (100,8,20,50,8), (33,5,7,19,5), (17,1,2,10,1)]
    dx = CuArray(randn(rng, 20000))
    off = 137
    pd = reshape(view(dx, off:(off + nsrow*ncol - 1)), nsrow, ncol)   # reshape-of-view
    Ab = view(pd, q:(q+ctot-1), 1:ncol)     # ctot×ncol nested sub-view
    L1 = view(pd, q:(q+k1-1), 1:ncol)       # k1×ncol nested sub-view
    C = CUDA.zeros(Float64, ctot, k1)     # NB: CUDA.zeros defaults to Float32!
    ext.gpu_gemm_nt!(C, Ab, L1, -1.0, 0.0)
    CUDA.synchronize()
    Cref = -Array(Ab) * Array(L1)'
    re = norm(Array(C) - Cref) / max(norm(Cref), eps())
    worst = max(worst, re)
    @printf("nsrow=%3d ncol=%d ctot=%2d k1=%d  relerr=%.2e\n", nsrow, ncol, ctot, k1, re)
end
println("worst relerr = ", worst)
@assert worst < 1e-12 "gemm-on-nested-views is WRONG (this is the cholesky bug)"
println("gpu_gemm_nt! on nested views + tiny K is CORRECT")
end
main()
