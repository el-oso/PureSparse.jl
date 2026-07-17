# Phase 2.3 hybrid oracle: gpu_cholesky_hybrid! (CPU supernodes on CPU, GPU frontier on device)
# must produce the CPU factor at machine precision, at EVERY frontier cutoff — all-GPU (cut 0),
# genuine hybrid (mid), all-CPU (cut Inf). Validates the CPU/GPU coordination + boundary uploads.
using PureSparse, CUDA, KernelAbstractions, SparseArrays, LinearAlgebra, Random
ext = Base.get_extension(PureSparse, :PureSparseCUDAExt)

function zero_strict_upper_diag!(x, S)  # unused strict-upper diagonal cells (see cholesky_test)
    for s in 1:S.nsuper
        nsc = Int(S.super[s+1]) - Int(S.super[s]); nsr = Int(S.rowind_ptr[s+1]) - Int(S.rowind_ptr[s])
        base = Int(S.px[s])
        for j in 1:nsc, i in 1:(j-1); x[base + (j-1)*nsr + (i-1)] = 0.0; end
    end
    return x
end

function test_hybrid(A, label)
    S0 = PureSparse.symbolic(A; ordering = PureSparse.AMDOrdering())
    F = PureSparse.cholesky(S0, A); @assert PureSparse.issuccess(F)
    xc = zero_strict_upper_diag!(copy(F.x), S0)
    snflop = [sum(Float64(S0.colcount[j])^2 for j in S0.super[s]:(S0.super[s+1]-1)) for s in 1:S0.nsuper]
    sf = sort(snflop)
    for (tag, cut) in [("all-GPU", 0.0), ("hybrid-25%", sf[cld(3*S0.nsuper,4)]),
                       ("hybrid-75%", sf[cld(S0.nsuper,4)]), ("all-CPU", Inf)]
        G = ext.gpu_symbolic(A; ordering = PureSparse.AMDOrdering(), frontier_cutoff = cut)
        x_host = Vector{Float64}(undef, G.xlen)
        d_nzval = CUDA.zeros(Float64, G.xlen)
        ok, fc = ext.gpu_cholesky_hybrid!(x_host, d_nzval, G, A)
        @assert ok "hybrid reported non-SPD (fail_col=$fc) $label/$tag"
        relerr = norm(zero_strict_upper_diag!(x_host, S0) - xc) / norm(xc)
        ngpu = count(G.on_gpu)
        println(rpad("$label/$tag", 26), " GPU=", lpad(ngpu, 4), "/", S0.nsuper,
                " boundary=", lpad(length(G.boundary), 4), "  relerr=", relerr)
        @assert relerr < 1e-10 "$label/$tag: hybrid factor mismatch relerr=$relerr"
    end
end

rng = MersenneTwister(21)
test_hybrid((let n=300; A=sprand(rng,n,n,0.015); A+A'+n*I end), "rand_n300")
test_hybrid((let nx=25,ny=25
    n=nx*ny; A=spzeros(n,n)
    for j in 1:ny, i in 1:nx
        k=(j-1)*nx+i; A[k,k]=4.0
        i<nx && (A[k,k+1]=A[k+1,k]=-1.0); j<ny && (A[k,k+nx]=A[k+nx,k]=-1.0)
    end; A+0.05I
end), "grid_25x25")
test_hybrid((let n=900; A=sprand(rng,n,n,0.008); A+A'+2n*I end), "rand_n900")

println("\nALL HYBRID ORACLE TESTS PASS — CPU/GPU split factor matches CPU factor at every cutoff")
