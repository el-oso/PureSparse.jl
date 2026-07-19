# End-to-end LDLᵀ on ROCm via PureSparseAMDGPUExt (M8) — the AMD mirror of gpu_ldlt_e2e_test.jl.
# HYBRID multifrontal signed-LDLᵀ factor + device LDLᵀ solve on a KKT (quasi-definite) system, vs
# the CPU ldlt! (L + D + inertia + solve residual). Correctness only. Run under an env with
# PureSparse + AMDGPU + KernelAbstractions.
using PureSparse, AMDGPU, KernelAbstractions, SparseArrays, LinearAlgebra, Random
ext = Base.get_extension(PureSparse, :PureSparseAMDGPUExt)
@assert ext !== nothing "PureSparseAMDGPUExt did not load"
roczeros(n) = AMDGPU.ROCArray(zeros(Float64, n)); rocarr(a) = AMDGPU.ROCArray(a)
zsud!(x,S) = (for s in 1:S.nsuper; nsc=Int(S.super[s+1])-Int(S.super[s]); nsr=Int(S.rowind_ptr[s+1])-Int(S.rowind_ptr[s]); b=Int(S.px[s]); for j in 1:nsc,i in 1:(j-1); x[b+(j-1)*nsr+(i-1)]=0.0 end end; x)
rng = MersenneTwister(0x1D1)
kkt(n1,n2,f) = (H=sprand(rng,n1,n1,f); H=H+H'+2n1*I; Ac=sprand(rng,n2,n1,f); D=sprand(rng,n2,n2,f); D=D+D'+2n2*I; ([H Ac'; Ac -D], n1, n2))

function test(K, n1, n2, label)
    n = size(K,1); signs_orig = Int8[i ≤ n1 ? 1 : -1 for i in 1:n]
    G0 = ext.gpu_symbolic(K; ordering=PureSparse.AMDOrdering(), frontier_cutoff=0.0)
    snf = sort([sum(Float64(G0.cpu.colcount[j])^2 for j in G0.cpu.super[s]:(G0.cpu.super[s+1]-1)) for s in 1:G0.cpu.nsuper])
    G = ext.gpu_symbolic(K; ordering=PureSparse.AMDOrdering(), frontier_cutoff=snf[cld(G0.cpu.nsuper,2)])  # genuine hybrid
    F = PureSparse.ldlt(G.cpu, K; signs=signs_orig); @assert PureSparse.issuccess(F)
    M = ext.mf_symbolic(G.cpu)
    xh=Vector{Float64}(undef,G.xlen); ha=Vector{Float64}(undef,max(M.arena_peak,1))
    da=roczeros(max(M.arena_peak,1)); dz=roczeros(G.xlen)
    dvec=Vector{Float64}(undef,n); d_dvec=roczeros(n)
    ok,fc,st = ext.gpu_multifrontal_ldlt_hybrid!(xh,dz,ha,da,dvec,d_dvec,M,G,K,F.signs; d2h=true)
    @assert ok "$label non-SPD"
    relL = norm(zsud!(copy(xh),G.cpu)-zsud!(copy(F.x),G.cpu))/norm(zsud!(copy(F.x),G.cpu))
    relD = norm(dvec-F.d)/norm(F.d)
    ii = (st.n_pos,st.n_neg,st.n_zero) == (F.stats.n_pos,F.stats.n_neg,F.stats.n_zero)

    # make-solve-ready + device LDLᵀ solve
    ext.gpu_upload_cpu_panels!(dz, xh, G); copyto!(d_dvec, 1, dvec, 1, n)
    b = randn(rng, n); perm = G.cpu.perm; mer = max(G.cpu.max_extend_rows,1)
    d_y = rocarr(b[perm]); d_upd=roczeros(mer); d_gath=roczeros(mer)
    ext.gpu_solve_ldlt!(d_y, dz, d_dvec, G, d_upd, d_gath)
    x = Vector{Float64}(undef,n); x[perm] = Array(d_y)
    res = norm(K*x - b)/norm(b)

    println(rpad(label,13)," GPU=",count(G.on_gpu),"/",G.cpu.nsuper," relL=",round(relL,sigdigits=2),
            " relD=",round(relD,sigdigits=2)," inertia✓=",ii," solve-res=",round(res,sigdigits=2))
    @assert relL<1e-9 && relD<1e-9 && ii && res<1e-8 "$label FAIL"
end

test(kkt(150,80,0.04)..., "kkt_150_80")
test(kkt(300,150,0.02)..., "kkt_300_150")
test(kkt(400,200,0.02)..., "kkt_400_200")
println("\n✅ END-TO-END LDLᵀ ON AMD (ROCm) — hybrid factor (L+D+inertia) + device solve match ldlt!")
