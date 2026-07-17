# Multifrontal Cholesky symbolic layer (design_gpu.md §M, amendment F). PURE — no CUDA — so it
# is both included by the ext AND unit-testable standalone against a CPU `Symbolic`. Front =
# supernode; front tree = `sparent`. Produces, at symbolic time (pattern-only):
#   - the front-tree children (CSC),
#   - per-child extend-add map `emap` (ascending; c's below-diagonal rows -> parent's row list)
#     + the `k1` prefix split (emap ≤ parent nscol -> panel region; rest -> U region),
#   - the update-matrix arena offsets `uoff` + exact peak (one postorder stack simulation;
#     single source of truth for order+offsets+peak, §M.3).

struct MFSymbolic{Ti}
    children_ptr::Vector{Ti}   # front-tree children of supernode s: children[children_ptr[s]:..]
    children::Vector{Ti}
    emap::Vector{Ti}           # child c's extend-add map into parent's row list (ascending)
    emap_ptr::Vector{Ti}       #   emap[emap_ptr[c]:emap_ptr[c+1]-1], length nsuper+1
    k1::Vector{Ti}             # per child c: #rows targeting parent's PIVOT cols (panel prefix)
    uoff::Vector{Ti}           # arena offset (1-based) of each front's update matrix U_s
    usize::Vector{Ti}          # size of U_s = (nsrow-nscol)^2  (0 if no below-diagonal rows)
    arena_peak::Int            # exact max arena occupancy (= the allocation, §M.3)
end

"""
    mf_symbolic(sym) -> MFSymbolic

Build the multifrontal symbolic layer from a CPU `Symbolic` (design_gpu.md §M.2/§M.3). Asserts
the pattern-containment `rowind(c)\\cols(c) ⊆ rowind(sparent(c))` that makes every `emap` lookup hit.
"""
function mf_symbolic(sym) where {}
    Ti = eltype(sym.super)
    ns = sym.nsuper
    super = sym.super; sparent = sym.sparent; rowind = sym.rowind; rowind_ptr = sym.rowind_ptr
    nscolf(s) = Int(super[s + 1]) - Int(super[s])
    nsrowf(s) = Int(rowind_ptr[s + 1]) - Int(rowind_ptr[s])

    # --- front-tree children (CSC over sparent) ---
    nchild = zeros(Int, ns)
    @inbounds for c in 1:ns
        p = Int(sparent[c]); p != 0 && (nchild[p] += 1)
    end
    children_ptr = Vector{Ti}(undef, ns + 1); children_ptr[1] = 1
    @inbounds for s in 1:ns; children_ptr[s + 1] = children_ptr[s] + nchild[s]; end
    children = Vector{Ti}(undef, Int(children_ptr[ns + 1]) - 1)
    cursor = [Int(children_ptr[s]) for s in 1:ns]
    @inbounds for c in 1:ns
        p = Int(sparent[c])
        if p != 0; children[cursor[p]] = Ti(c); cursor[p] += 1; end
    end

    # --- emap + k1 (per child) ---
    emap_ptr = Vector{Ti}(undef, ns + 1); emap_ptr[1] = 1
    @inbounds for c in 1:ns
        below = nsrowf(c) - nscolf(c)
        emap_ptr[c + 1] = emap_ptr[c] + (Int(sparent[c]) != 0 ? below : 0)
    end
    emap = Vector{Ti}(undef, Int(emap_ptr[ns + 1]) - 1)
    k1 = zeros(Ti, ns)
    relmap = zeros(Ti, sym.n)
    @inbounds for p in 1:ns                       # build each parent's row->local map, then its children
        rp0 = Int(rowind_ptr[p]); nsr_p = nsrowf(p); nsc_p = nscolf(p)
        for k in 1:nsr_p; relmap[Int(rowind[rp0 + k - 1])] = Ti(k); end
        for ci in Int(children_ptr[p]):(Int(children_ptr[p + 1]) - 1)
            c = Int(children[ci])
            crp0 = Int(rowind_ptr[c]); nsc_c = nscolf(c); below_c = nsrowf(c) - nsc_c
            base = Int(emap_ptr[c]); kk = 0
            for i in 1:below_c
                t = Int(relmap[Int(rowind[crp0 + nsc_c + i - 1])])
                t == 0 && error("mf_symbolic: containment violated (child $c row not in parent $p)")
                emap[base + i - 1] = Ti(t)
                t ≤ nsc_p && (kk += 1)             # ascending emap => this is a prefix count
            end
            k1[c] = Ti(kk)
        end
    end

    # --- arena: postorder stack simulation -> offsets + peak (§M.3) ---
    usize = Vector{Ti}(undef, ns)
    @inbounds for s in 1:ns; b = nsrowf(s) - nscolf(s); usize[s] = Ti(b * b); end
    uoff = zeros(Ti, ns)
    offstack = Int[]; top = 1; peak = 0
    @inbounds for s in 1:ns
        nc = Int(children_ptr[s + 1]) - Int(children_ptr[s])   # s's children are the top nc slots
        for _ in 1:nc; top = pop!(offstack); end               # free them (LIFO, contiguous)
        uoff[s] = Ti(top); push!(offstack, top)
        top += Int(usize[s]); peak = max(peak, top - 1)
    end

    return MFSymbolic{Ti}(children_ptr, children, emap, emap_ptr, k1, uoff, usize, peak)
end
