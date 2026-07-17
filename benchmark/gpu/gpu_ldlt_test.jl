# M6b: all-GPU multifrontal LDLᵀ (blocked device-LDL) vs ldlt! on SQD/KKT (design_gpu.md §6/§M).
using PureSparse, CUDA, KernelAbstractions, SparseArrays, LinearAlgebra, Random
ext = Base.get_extension(PureSparse, :PureSparseCUDAExt)
zsud!(x,S)= (for s in 1:S.nsuper; nsc=Int(S.super[s+1])-Int(S.super[s]); nsr=Int(S.rowind_ptr[s+1])-Int(S.rowind_ptr[s]); b=Int(S.px[s]); for j in 1:nsc,i in 1:(j-1); x[b+(j-1)*nsr+(i-1)]=0.0 end end; x)
rng=MersenneTwister(0x1D1)
kkt(n1,n2,f)= (H=sprand(rng,n1,n1,f); H=H+H'+2n1*I; Ac=sprand(rng,n2,n1,f); D=sprand(rng,n2,n2,f); D=D+D'+2n2*I; ([H Ac'; Ac -D], n1, n2))

function test(K, n1, n2, label)
    n = size(K,1)
    G = ext.gpu_symbolic(K; ordering=PureSparse.AMDOrdering(), frontier_cutoff=0.0)
    signs_orig = Int8[i ≤ n1 ? 1 : -1 for i in 1:n]
    F = PureSparse.ldlt(G.cpu, K; signs=signs_orig); @assert PureSparse.issuccess(F)
    M = ext.mf_symbolic(G.cpu)
    dz=CUDA.zeros(Float64,G.xlen); da=CUDA.zeros(Float64,max(M.arena_peak,1)); dd=CUDA.zeros(Float64,n)
    ok,fc,st = ext.gpu_multifrontal_ldlt!(dz, da, dd, M, G, K, F.signs)
    @assert ok "gpu LDLᵀ non-SPD $label"
    relL = norm(zsud!(Array(dz),G.cpu)-zsud!(copy(F.x),G.cpu))/norm(zsud!(copy(F.x),G.cpu))
    relD = norm(Array(dd)-F.d)/norm(F.d)
    im = (st.n_pos,st.n_neg,st.n_zero); ic = (F.stats.n_pos,F.stats.n_neg,F.stats.n_zero)
    println(rpad(label,14)," relL=",round(relL,sigdigits=3)," relD=",round(relD,sigdigits=3),
            "  inertia gpu=",im," cpu=",ic)
    @assert relL<1e-9 && relD<1e-9 "$label LDLᵀ factor mismatch"
    @assert im==ic "$label inertia mismatch"
end

test(kkt(150,80,0.04)..., "kkt_150_80")
test(kkt(300,150,0.02)..., "kkt_300_150")
test(kkt(80,80,0.08)...,  "kkt_80_80")
test(kkt(400,200,0.015)...,"kkt_400_200")
println("\nGPU MULTIFRONTAL LDLᵀ PASSES — device factor L+D+inertia match ldlt!")
