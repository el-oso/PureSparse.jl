# Path B step 4: all-GPU multifrontal Cholesky must match the CPU factor (design_gpu.md §M).
using PureSparse, CUDA, KernelAbstractions, SparseArrays, LinearAlgebra, Random
ext = Base.get_extension(PureSparse, :PureSparseCUDAExt)
ext === nothing && error("PureSparseCUDAExt did not load")

zsud!(x,S)= (for s in 1:S.nsuper; nsc=Int(S.super[s+1])-Int(S.super[s]); nsr=Int(S.rowind_ptr[s+1])-Int(S.rowind_ptr[s]); b=Int(S.px[s]); for j in 1:nsc,i in 1:(j-1); x[b+(j-1)*nsr+(i-1)]=0.0 end end; x)

function test_mf(A, label)
    G = ext.gpu_symbolic(A; ordering=PureSparse.AMDOrdering(), frontier_cutoff=0.0)
    F = PureSparse.cholesky(G.cpu, A); @assert PureSparse.issuccess(F)
    M = ext.mf_symbolic(G.cpu)
    d_nzval = CUDA.zeros(Float64, G.xlen)                 # NB Float64 (CUDA.zeros defaults F32)
    d_arena = CUDA.zeros(Float64, max(M.arena_peak, 1))
    ok, fc = ext.gpu_multifrontal_cholesky!(d_nzval, d_arena, M, G, A)
    @assert ok "multifrontal reported non-SPD (fail_col=$fc) $label"
    relerr = norm(zsud!(Array(d_nzval), G.cpu) - zsud!(copy(F.x), G.cpu)) / norm(zsud!(copy(F.x), G.cpu))
    println(rpad(label,12), " nsuper=", lpad(G.cpu.nsuper,5), " arena=", lpad(M.arena_peak,9),
            " (", round(M.arena_peak*8/1e6,digits=1), " MB)  relerr=", relerr)
    @assert relerr < 1e-10 "$label multifrontal factor mismatch relerr=$relerr"
end

rng = MersenneTwister(31)
test_mf((let n=300; A=sprand(rng,n,n,0.02); A+A'+n*I end), "rand_n300")
test_mf((let nx=25,ny=25; n=nx*ny; A=spzeros(n,n)
    for j in 1:ny,i in 1:nx; k=(j-1)*nx+i; A[k,k]=4.0
        i<nx&&(A[k,k+1]=A[k+1,k]=-1.0); j<ny&&(A[k,k+nx]=A[k+nx,k]=-1.0) end; A+0.05I end), "grid_25x25")
test_mf((let d=12; n=d^3; A=spzeros(n,n); lin(i,j,k)=((k-1)*d+(j-1))*d+i
    for k in 1:d,j in 1:d,i in 1:d; p=lin(i,j,k); A[p,p]=6.0
        i<d&&(A[p,lin(i+1,j,k)]=A[lin(i+1,j,k),p]=-1.0); j<d&&(A[p,lin(i,j+1,k)]=A[lin(i,j+1,k),p]=-1.0)
        k<d&&(A[p,lin(i,j,k+1)]=A[lin(i,j,k+1),p]=-1.0) end; A+0.1I end), "grid3d_12")
test_mf((let n=1000; A=sprand(rng,n,n,0.008); A+A'+2n*I end), "rand_n1000")

println("\nALL-GPU MULTIFRONTAL ORACLE PASSES — device factor matches CPU factor")
