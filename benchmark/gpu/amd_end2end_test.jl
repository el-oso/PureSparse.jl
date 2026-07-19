# End-to-end Cholesky on ROCm via PureSparseAMDGPUExt (design_gpu_multibackend.md, M8) — the AMD
# mirror of benchmark/gpu/gpu_mf_hybrid_test.jl. Proves the WIRED solver (gpu_symbolic → multifrontal
# hybrid FACTOR → device SOLVE) runs end-to-end on AMD and matches the CPU factor + a residual gate at
# machine precision, at every frontier cutoff. Covers Float64 AND Float32 (the path is generic over T).
# Correctness only; FP64 perf is moot on the iGPU. Run under an env with PureSparse + AMDGPU + KA.
using PureSparse, AMDGPU, KernelAbstractions, SparseArrays, LinearAlgebra, Random
ext = Base.get_extension(PureSparse, :PureSparseAMDGPUExt)
@assert ext !== nothing "PureSparseAMDGPUExt did not load (need AMDGPU + KernelAbstractions present)"
println("PureSparseAMDGPUExt loaded; device = ", AMDGPU.device())

roczeros(::Type{T}, n) where {T} = AMDGPU.ROCArray(zeros(T, n)); rocarr(a) = AMDGPU.ROCArray(a)
zsud!(x, S) = (for s in 1:S.nsuper
    nsc = Int(S.super[s+1]) - Int(S.super[s]); nsr = Int(S.rowind_ptr[s+1]) - Int(S.rowind_ptr[s]); b = Int(S.px[s])
    for j in 1:nsc, i in 1:(j-1); x[b+(j-1)*nsr+(i-1)] = zero(eltype(x)) end
end; x)

function test_chol(A::SparseMatrixCSC{T}, label) where {T}
    S0 = PureSparse.symbolic(A); F = PureSparse.cholesky(S0, A); @assert PureSparse.issuccess(F)
    xc = zsud!(copy(F.x), S0)
    tolf = T == Float32 ? 1e-5 : 1e-10; tols = T == Float32 ? 1e-3 : 1e-8
    snf = sort([sum(Float64(S0.colcount[j])^2 for j in S0.super[s]:(S0.super[s+1]-1)) for s in 1:S0.nsuper])
    for (tag, cut) in [("all-GPU", 0.0), ("hyb-25%", snf[cld(3*S0.nsuper,4)]), ("hyb-75%", snf[cld(S0.nsuper,4)]), ("all-CPU", Inf)]
        G = ext.gpu_symbolic(A; ordering=PureSparse.AMDOrdering(), frontier_cutoff=cut)  # backend defaults to ROCBackend
        M = ext.mf_symbolic(G.cpu)
        x_host = Vector{T}(undef, G.xlen); ha = Vector{T}(undef, max(M.arena_peak, 1))
        da = roczeros(T, max(M.arena_peak, 1)); dz = roczeros(T, G.xlen)
        ok, fc = ext.gpu_multifrontal_hybrid!(x_host, dz, ha, da, M, G, A)
        @assert ok "AMD hybrid-mf non-SPD $label/$tag (failcol=$fc)"
        relerr = norm(zsud!(x_host, S0) - xc) / norm(xc)
        msg = rpad("$label/$tag", 24) * " GPU=$(lpad(count(G.on_gpu),4))/$(S0.nsuper)  relerr=$relerr"
        # device Cholesky SOLVE on the all-GPU cutoff (dz holds the full device factor there)
        if tag == "all-GPU"
            n = S0.n; b = randn(MersenneTwister(7), T, n); perm = S0.perm
            d_y = rocarr(b[perm]); ext.gpu_solve!(d_y, dz, G)
            x = Vector{T}(undef, n); x[perm] = Array(d_y)
            res = norm(A * x - b) / norm(b); msg *= "  solve-res=$res"
            @assert res < tols "$label solve res=$res"
        end
        println(msg)
        @assert relerr < tolf "$label/$tag mismatch $relerr"
    end
end

rng = MersenneTwister(41)
grid3d(d) = (let n=d^3; A=spzeros(n,n); lin(i,j,k)=((k-1)*d+(j-1))*d+i
    for k in 1:d,j in 1:d,i in 1:d; p=lin(i,j,k); A[p,p]=6.0
        i<d&&(A[p,lin(i+1,j,k)]=A[lin(i+1,j,k),p]=-1.0); j<d&&(A[p,lin(i,j+1,k)]=A[lin(i,j+1,k),p]=-1.0)
        k<d&&(A[p,lin(i,j,k+1)]=A[lin(i,j,k+1),p]=-1.0) end; A+0.1I end)
test_chol((let n=300; A=sprand(rng,n,n,0.02); A+A'+n*I end), "rand_n300")
test_chol((let nx=25,ny=25; n=nx*ny; A=spzeros(n,n)
    for j in 1:ny,i in 1:nx; k=(j-1)*nx+i; A[k,k]=4.0
        i<nx&&(A[k,k+1]=A[k+1,k]=-1.0); j<ny&&(A[k,k+nx]=A[k+nx,k]=-1.0) end; A+0.05I end), "grid_25x25")
test_chol(grid3d(12), "grid3d_12")
test_chol(SparseMatrixCSC{Float32,Int}(grid3d(10)), "grid3d_10_f32")   # Float32 end-to-end
println("\n✅ END-TO-END CHOLESKY ON AMD (ROCm) — factor+solve match CPU at every cutoff, F64 & F32")
