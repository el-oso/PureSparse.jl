# Host-side GPU/CPU frontier partition of the supernodal etree (design_gpu.md §5.2).
# PURE — no CUDA/KernelAbstractions dependency — so it is both included by PureSparseCUDAExt
# and unit-testable standalone against a CPU `Symbolic` (frontier logic needs a GPU nowhere).
#
# The GPU set is the UPWARD CLOSURE (in the supernode etree `sparent`) of the supernodes whose
# factor+update flops ≥ `cutoff`. Upward closure is what guarantees no device→host update edge:
# a supernode's below-diagonal rows map only to etree ancestors (row-subtree property), and all
# ancestors of a GPU node are GPU — so every cross-frontier edge points CPU→GPU (design_gpu.md
# §5.2). Boundary CPU supernodes (those with a GPU ancestor) are the panels persisted
# device-resident and uploaded once (§5.3).

"""
    frontier_partition!(on_gpu, nsuper, super, sparent, colcount, cutoff) -> on_gpu

Fill `on_gpu[s]` (length `nsuper`) with the upward-closed GPU supernode set: seed = supernodes
with Σ_{j∈s} colcount[j]² ≥ `cutoff` (the factor+update flop weight, matching `Symbolic.flops`),
then close upward over `sparent` (0 = root). Order-independent (walks each seed's ancestor
chain; O(nsuper) amortized).
"""
function frontier_partition!(on_gpu::AbstractVector{Bool}, nsuper::Integer,
                             super::AbstractVector, sparent::AbstractVector,
                             colcount::AbstractVector, cutoff::Float64)
    fill!(on_gpu, false)
    @inbounds for s in 1:nsuper                       # seeds
        f = 0.0
        for j in super[s]:(super[s + 1] - 1)
            c = Float64(colcount[j]); f += c * c
        end
        on_gpu[s] = f ≥ cutoff
    end
    @inbounds for s in 1:nsuper                        # upward closure (ancestor walk)
        if on_gpu[s]
            t = sparent[s]
            while t != 0 && !on_gpu[t]
                on_gpu[t] = true
                t = sparent[t]
            end
        end
    end
    return on_gpu
end

"""
    boundary_supernodes(on_gpu, nsuper, rowind, rowind_ptr, snode_of) -> Vector

CPU supernodes (`!on_gpu[s]`) with ≥1 below-diagonal row updating a GPU supernode — the panels
that must be uploaded once and persisted device-resident (design_gpu.md §5.3).
"""
function boundary_supernodes(on_gpu::AbstractVector{Bool}, nsuper::Integer,
                             rowind::AbstractVector{Ti}, rowind_ptr::AbstractVector,
                             snode_of::AbstractVector) where {Ti}
    b = Ti[]
    @inbounds for s in 1:nsuper
        on_gpu[s] && continue
        for p in rowind_ptr[s]:(rowind_ptr[s + 1] - 1)
            if on_gpu[snode_of[rowind[p]]]
                push!(b, s); break
            end
        end
    end
    return b
end

"""
    frontier_invariant_holds(on_gpu, nsuper, rowind, rowind_ptr, snode_of) -> Bool

Executable check of the §10.2 upward-closure invariant: no GPU supernode has an update edge to
a CPU supernode (`∀ GPU s, ∀ r ∈ rowind(s): on_gpu[snode_of[r]]`). True by construction for a
correct upward closure; the test asserts it.
"""
function frontier_invariant_holds(on_gpu::AbstractVector{Bool}, nsuper::Integer,
                                  rowind::AbstractVector, rowind_ptr::AbstractVector,
                                  snode_of::AbstractVector)
    @inbounds for s in 1:nsuper
        on_gpu[s] || continue
        for p in rowind_ptr[s]:(rowind_ptr[s + 1] - 1)
            on_gpu[snode_of[rowind[p]]] || return false
        end
    end
    return true
end
