# Phase 2.2 galen validation: gpu_symbolic builds a device-resident symbolic — pattern arrays
# uploaded once and matching the CPU Symbolic, frontier upward-closed, sizing consistent.
using PureSparse, CUDA, KernelAbstractions, SparseArrays, LinearAlgebra, Random

ext = Base.get_extension(PureSparse, :PureSparseCUDAExt)
ext === nothing && error("PureSparseCUDAExt did not load")

rng = MersenneTwister(7)
A = sprand(rng, 1500, 1500, 0.004); A = A + A' + 3000I    # SPD

S = PureSparse.symbolic(A; ordering = PureSparse.AMDOrdering())
snflop = [sum(Float64(S.colcount[j])^2 for j in S.super[s]:(S.super[s+1]-1)) for s in 1:S.nsuper]
cut = sort(snflop)[cld(S.nsuper, 2)]     # ~half on GPU

G = ext.gpu_symbolic(A; ordering = PureSparse.AMDOrdering(), frontier_cutoff = cut)
println("nsuper=", S.nsuper, "  on GPU=", count(G.on_gpu), "  boundary=", length(G.boundary))

# 1. device pattern arrays match the CPU Symbolic exactly (upload correctness)
@assert Array(G.d_rowind)     == S.rowind      "d_rowind mismatch"
@assert Array(G.d_rowind_ptr) == S.rowind_ptr  "d_rowind_ptr mismatch"
@assert Array(G.d_super)      == S.super       "d_super mismatch"
@assert Array(G.d_snode_of)   == S.snode_of    "d_snode_of mismatch"
println("device pattern arrays match CPU Symbolic ✓")

# 2. frontier upward-closed (constructor asserts, re-check here)
@assert ext.frontier_invariant_holds(G.on_gpu, S.nsuper, S.rowind, S.rowind_ptr, S.snode_of)
for s in 1:S.nsuper
    (G.on_gpu[s] && S.sparent[s] != 0) && @assert G.on_gpu[S.sparent[s]] "closure broken at $s"
end
println("frontier upward-closed ✓")

# 3. gpu_order = ascending GPU supernodes; boundary ⊆ CPU
@assert G.gpu_order == [s for s in 1:S.nsuper if G.on_gpu[s]]
@assert issorted(G.gpu_order)
@assert all(s -> !G.on_gpu[s], G.boundary)
println("gpu_order + boundary consistent ✓")

# 4. sizing consistent
@assert G.bytes.nzval == S.nnzL * sizeof(Float64)
@assert G.bytes.total == G.bytes.nzval + G.bytes.cbuf + G.bytes.boundbuf
println("device-memory budget: total=", round(G.bytes.total/1e6, digits=1), " MB ",
        "(nzval=", round(G.bytes.nzval/1e6,digits=1), " boundbuf=", round(G.bytes.boundbuf/1e6,digits=1), ")")

println("\nALL GPUSymbolic TESTS PASS")
