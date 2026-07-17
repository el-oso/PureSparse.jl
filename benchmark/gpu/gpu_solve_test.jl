# Device supernodal solve (design_gpu.md §7, amendment B): A·x=b on device vs CPU F\b.
using PureSparse, CUDA, KernelAbstractions, SparseArrays, LinearAlgebra, Random
ext = Base.get_extension(PureSparse, :PureSparseCUDAExt)
rng = MersenneTwister(9)

function dsolve(G, d_nzval)                         # device solve wrapper (permute/unpermute host-side)
    n = G.cpu.n; perm = G.cpu.perm; mer = max(G.cpu.max_extend_rows, 1)
    b = randn(rng, n)
    d_y = CuArray(b[perm])
    d_upd = CUDA.zeros(Float64, mer); d_gath = CUDA.zeros(Float64, mer)
    ext.gpu_solve!(d_y, d_nzval, G, d_upd, d_gath)
    x = Vector{Float64}(undef, n); x[perm] = Array(d_y)
    return b, x
end

function test(A, label)
    # all-GPU factor (full device factor) + device solve
    G = ext.gpu_symbolic(A; ordering=PureSparse.AMDOrdering(), frontier_cutoff=0.0)
    F = PureSparse.cholesky(G.cpu, A)
    M = ext.mf_symbolic(G.cpu)
    d_nzval = CUDA.zeros(Float64, G.xlen); d_arena = CUDA.zeros(Float64, max(M.arena_peak,1))
    ext.gpu_multifrontal_cholesky!(d_nzval, d_arena, M, G, A)
    b, x = dsolve(G, d_nzval)
    r1 = norm(x - (F\b))/norm(F\b); res1 = norm(A*x - b)/norm(b)

    # hybrid factor -> make-solve-ready (upload CPU panels) -> device solve
    snf = sort([sum(Float64(G.cpu.colcount[j])^2 for j in G.cpu.super[s]:(G.cpu.super[s+1]-1)) for s in 1:G.cpu.nsuper])
    Gh = ext.gpu_symbolic(A; ordering=PureSparse.AMDOrdering(), frontier_cutoff=snf[cld(G.cpu.nsuper,2)])
    Mh = ext.mf_symbolic(Gh.cpu)
    xh = Vector{Float64}(undef, Gh.xlen); ha = Vector{Float64}(undef, max(Mh.arena_peak,1))
    da = CUDA.zeros(Float64, max(Mh.arena_peak,1)); dz = CUDA.zeros(Float64, Gh.xlen)
    ext.gpu_multifrontal_hybrid!(xh, dz, ha, da, Mh, Gh, A; d2h=false)
    ext.gpu_upload_cpu_panels!(dz, xh, Gh)          # make-solve-ready
    b2, x2 = dsolve(Gh, dz)
    r2 = norm(x2 - (F\b2))/norm(F\b2); res2 = norm(A*x2 - b2)/norm(b2)

    println(rpad(label,12)," all-GPU: relerr=",round(r1,sigdigits=3)," res=",round(res1,sigdigits=3),
            "   hybrid: relerr=",round(r2,sigdigits=3)," res=",round(res2,sigdigits=3))
    @assert r1<1e-8 && res1<1e-8 && r2<1e-8 && res2<1e-8 "$label solve mismatch"
end

test((let n=400; A=sprand(rng,n,n,0.02); A+A'+n*I end), "rand_n400")
test((let nx=25,ny=25; n=nx*ny; A=spzeros(n,n)
    for j in 1:ny,i in 1:nx; k=(j-1)*nx+i; A[k,k]=4.0
        i<nx&&(A[k,k+1]=A[k+1,k]=-1.0); j<ny&&(A[k,k+nx]=A[k+nx,k]=-1.0) end; A+0.05I end), "grid_25x25")
test((let d=12; n=d^3; A=spzeros(n,n); lin(i,j,k)=((k-1)*d+(j-1))*d+i
    for k in 1:d,j in 1:d,i in 1:d; p=lin(i,j,k); A[p,p]=6.0
        i<d&&(A[p,lin(i+1,j,k)]=A[lin(i+1,j,k),p]=-1.0); j<d&&(A[p,lin(i,j+1,k)]=A[lin(i,j+1,k),p]=-1.0)
        k<d&&(A[p,lin(i,j,k+1)]=A[lin(i,j,k+1),p]=-1.0) end; A+0.1I end), "grid3d_12")
println("\nDEVICE SOLVE PASSES — A·x=b on device matches CPU (all-GPU + hybrid make-solve-ready)")
