# End-to-end Cholesky on ROCm via PureSparseAMDGPUExt (design_gpu_multibackend.md, M8) — the AMD
# mirror of benchmark/gpu/gpu_mf_hybrid_test.jl. Proves the WIRED solver (gpu_symbolic → multifrontal
# hybrid factor → device solve) runs end-to-end on AMD and matches the CPU factor at machine
# precision, at every frontier cutoff (all-CPU → all-GPU). Correctness only; FP64 perf is moot on
# the iGPU. Run: julia --project=<env-with-PureSparse+AMDGPU> benchmark/gpu/amd_end2end_test.jl
using PureSparse, AMDGPU, KernelAbstractions, SparseArrays, LinearAlgebra, Random
ext = Base.get_extension(PureSparse, :PureSparseAMDGPUExt)
@assert ext !== nothing "PureSparseAMDGPUExt did not load (need AMDGPU + KernelAbstractions present)"
println("PureSparseAMDGPUExt loaded; device = ", AMDGPU.device())

roczeros(n) = AMDGPU.ROCArray(zeros(Float64, n))
# zero the strict-upper of each supernode diagonal block (the never-written cells) for comparison
zsud!(x, S) = (for s in 1:S.nsuper
    nsc = Int(S.super[s+1]) - Int(S.super[s]); nsr = Int(S.rowind_ptr[s+1]) - Int(S.rowind_ptr[s]); b = Int(S.px[s])
    for j in 1:nsc, i in 1:(j-1); x[b+(j-1)*nsr+(i-1)] = 0.0 end
end; x)

function test_chol(A, label)
    S0 = PureSparse.symbolic(A); F = PureSparse.cholesky(S0, A); @assert PureSparse.issuccess(F)
    xc = zsud!(copy(F.x), S0)
    snf = sort([sum(Float64(S0.colcount[j])^2 for j in S0.super[s]:(S0.super[s+1]-1)) for s in 1:S0.nsuper])
    for (tag, cut) in [("all-GPU", 0.0), ("hyb-25%", snf[cld(3*S0.nsuper,4)]), ("hyb-75%", snf[cld(S0.nsuper,4)]), ("all-CPU", Inf)]
        G = ext.gpu_symbolic(A; ordering=PureSparse.AMDOrdering(), frontier_cutoff=cut)  # backend defaults to ROCBackend
        M = ext.mf_symbolic(G.cpu)
        x_host = Vector{Float64}(undef, G.xlen); ha = Vector{Float64}(undef, max(M.arena_peak, 1))
        da = roczeros(max(M.arena_peak, 1)); dz = roczeros(G.xlen)
        ok, fc = ext.gpu_multifrontal_hybrid!(x_host, dz, ha, da, M, G, A)
        @assert ok "AMD hybrid-mf non-SPD $label/$tag (failcol=$fc)"
        relerr = norm(zsud!(x_host, S0) - xc) / norm(xc)
        println(rpad("$label/$tag", 22), " GPU=", lpad(count(G.on_gpu), 4), "/", S0.nsuper, "  relerr=", relerr)
        @assert relerr < 1e-10 "$label/$tag mismatch $relerr"
    end
end

rng = MersenneTwister(41)
test_chol((let n=300; A=sprand(rng,n,n,0.02); A+A'+n*I end), "rand_n300")
test_chol((let nx=25,ny=25; n=nx*ny; A=spzeros(n,n)
    for j in 1:ny,i in 1:nx; k=(j-1)*nx+i; A[k,k]=4.0
        i<nx&&(A[k,k+1]=A[k+1,k]=-1.0); j<ny&&(A[k,k+nx]=A[k+nx,k]=-1.0) end; A+0.05I end), "grid_25x25")
test_chol((let d=12; n=d^3; A=spzeros(n,n); lin(i,j,k)=((k-1)*d+(j-1))*d+i
    for k in 1:d,j in 1:d,i in 1:d; p=lin(i,j,k); A[p,p]=6.0
        i<d&&(A[p,lin(i+1,j,k)]=A[lin(i+1,j,k),p]=-1.0); j<d&&(A[p,lin(i,j+1,k)]=A[lin(i,j+1,k),p]=-1.0)
        k<d&&(A[p,lin(i,j,k+1)]=A[lin(i,j,k+1),p]=-1.0) end; A+0.1I end), "grid3d_12")
println("\n✅ END-TO-END CHOLESKY ON AMD (ROCm) — factor matches CPU at every cutoff, machine precision")
