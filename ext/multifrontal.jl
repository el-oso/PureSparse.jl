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

    # --- arena offsets (§M.3) ---
    # CORRECTNESS-FIRST: monotonic no-reuse arena — each front's U gets a permanent slot, so a
    # parent's U never aliases its children's U's (they COEXIST during extend-add: children are
    # read while the parent's U is zeroed+written). Memory = Σ usize. The bounded stack-with-
    # compaction arena (§M.3, parent U assembled above children then copied down over their freed
    # space) is the memory optimization, deferred until the formulation is validated.
    usize = Vector{Ti}(undef, ns)
    @inbounds for s in 1:ns; b = nsrowf(s) - nscolf(s); usize[s] = Ti(b * b); end
    uoff = Vector{Ti}(undef, ns)
    top = 1
    @inbounds for s in 1:ns
        uoff[s] = Ti(top); top += Int(usize[s])
    end
    peak = top - 1

    return MFSymbolic{Ti}(children_ptr, children, emap, emap_ptr, k1, uoff, usize, peak)
end

# --- CPU multifrontal numeric (pure; the oracle + dev vehicle for the GPU engine, §M.6 step 2) ---
using LinearAlgebra: LAPACK, BLAS
@inline _mfpanel(x, off, m, n) = unsafe_wrap(Array, pointer(x, off), (m, n))

"""
    cpu_multifrontal_cholesky!(x_host, arena, M, sym, A) -> (ok, fail_col)

Multifrontal supernodal Cholesky on CPU (design_gpu.md §M). `x_host` (length `px[end]-1`) holds
the factor (panel regions, bit-compatible with `SupernodalFactor`); `arena` (length
`M.arena_peak`) is the update-matrix working memory. Per front (postorder): zero U_s, extend-add
children into panel + U_s, assemble A (done globally), potrf+trsm, then
`U_s = (extend-added trailing block) − L21·L21ᵀ` (syrk β=1). CPU BLAS (correctness-first oracle;
the shipped CPU path is left-looking `cholesky!`, amendment C/F).
"""
function cpu_multifrontal_cholesky!(x_host::Vector{T}, arena::Vector{T}, M::MFSymbolic,
                                    sym, A) where {T}
    ns = sym.nsuper; super = sym.super; rowind_ptr = sym.rowind_ptr; px = sym.px; amap = sym.amap
    fill!(x_host, zero(T))                                   # global assembly: A into panel regions
    @inbounds for p in eachindex(A.nzval)
        m = Int(amap[p]); m != 0 && (x_host[m] = A.nzval[p])
    end
    ok = true; failcol = 0
    GC.@preserve x_host arena begin
    @inbounds for s in 1:ns
        nscol = Int(super[s + 1]) - Int(super[s]); nsrow = Int(rowind_ptr[s + 1]) - Int(rowind_ptr[s])
        below_s = nsrow - nscol
        panel = _mfpanel(x_host, Int(px[s]), nsrow, nscol)
        uo = Int(M.uoff[s]); us = Int(M.usize[s])
        for i in uo:(uo + us - 1); arena[i] = zero(T); end   # zero U_s (pitfall #4)
        U_s = below_s > 0 ? _mfpanel(arena, uo, below_s, below_s) : nothing

        for ci in Int(M.children_ptr[s]):(Int(M.children_ptr[s + 1]) - 1)   # extend-add children
            c = Int(M.children[ci])
            nsc_c = Int(super[c + 1]) - Int(super[c])
            below_c = Int(rowind_ptr[c + 1]) - Int(rowind_ptr[c]) - nsc_c
            below_c == 0 && continue
            U_c = _mfpanel(arena, Int(M.uoff[c]), below_c, below_c)
            eb = Int(M.emap_ptr[c])
            for b in 1:below_c
                rb = Int(M.emap[eb + b - 1])
                for a in b:below_c                            # lower triangle of U_c (a ≥ b)
                    ra = Int(M.emap[eb + a - 1]); v = U_c[a, b]
                    if rb ≤ nscol
                        panel[ra, rb] += v
                    else
                        U_s[ra - nscol, rb - nscol] += v
                    end
                end
            end
        end

        diag = view(panel, 1:nscol, 1:nscol)
        _, info = LAPACK.potrf!('L', diag)
        info != 0 && (ok = false; failcol = Int(super[s]); break)
        if below_s > 0
            L21 = view(panel, (nscol + 1):nsrow, 1:nscol)
            BLAS.trsm!('R', 'L', 'T', 'N', one(T), diag, L21)
            BLAS.syrk!('L', 'N', -one(T), L21, one(T), U_s)   # U_s = children − L21·L21ᵀ
        end
    end
    end
    return (ok, failcol)
end
