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

    # --- BOUNDED arena: work slot + stack (§M.3) ---
    # Layout: arena[1 : max_usize] = a WORK slot where each front builds its U (children are read
    # from the STACK above, so no aliasing), then compact-copies into arena[uoff[s]:] on the stack
    # (reusing consumed children's freed space). Non-overlapping by construction (work slot < stack).
    # `uoff[s]` = the front's FINAL stack offset (where its parent reads it). Postorder stack
    # simulation gives uoff + the exact peak — Σ live U's, far smaller than the monotonic Σ all U's.
    usize = Vector{Ti}(undef, ns)
    @inbounds for s in 1:ns; b = nsrowf(s) - nscolf(s); usize[s] = Ti(b * b); end
    max_us = max(maximum(Int, usize), 1)                   # work-slot size
    uoff = Vector{Ti}(undef, ns)
    offstack = Int[]; top = max_us + 1; peak = top - 1      # stack starts above the work slot
    @inbounds for s in 1:ns
        nc = Int(children_ptr[s + 1]) - Int(children_ptr[s]); us = Int(usize[s])
        cbase = nc > 0 ? offstack[length(offstack) - nc + 1] : top   # deepest child's offset
        for _ in 1:nc; pop!(offstack); end                 # free children
        top = cbase                                        # compact: U_s lands where children were
        uoff[s] = Ti(top); push!(offstack, top); top += us
        peak = max(peak, top - 1)
    end

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
        for i in 1:us; arena[i] = zero(T); end               # zero U_s in the WORK slot (§M.3)
        U_s = below_s > 0 ? _mfpanel(arena, 1, below_s, below_s) : nothing

        for ci in Int(M.children_ptr[s]):(Int(M.children_ptr[s + 1]) - 1)   # extend-add children
            c = Int(M.children[ci])
            nsc_c = Int(super[c + 1]) - Int(super[c])
            below_c = Int(rowind_ptr[c + 1]) - Int(rowind_ptr[c]) - nsc_c
            below_c == 0 && continue
            U_c = _mfpanel(arena, Int(M.uoff[c]), below_c, below_c)   # child on the STACK
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
        try                                                   # PureBLAS potrf! throws on non-PD (amendment C)
            PureSparse.potrf!(diag; uplo = 'L')
        catch e
            e isa PosDefException || rethrow()
            ok = false; failcol = Int(super[s]); break
        end
        if below_s > 0
            L21 = view(panel, (nscol + 1):nsrow, 1:nscol)
            PureSparse.trsm!(L21, diag; side = 'R', uplo = 'L', transA = 'T', diag = 'N', alpha = one(T))
            PureSparse.syrk!(U_s, L21; uplo = 'L', trans = 'N', alpha = -one(T), beta = one(T))  # U_s = children − L21·L21ᵀ
            copyto!(arena, uo, arena, 1, us)                  # compact work slot → STACK at uoff[s]
        end
    end
    end
    return (ok, failcol)
end

# --- CPU multifrontal LDLᵀ (design_gpu.md §6/§M, amendment E) — the M6b oracle + dev vehicle ---
# Adapts cpu_multifrontal_cholesky!: same extend-add/arena/tree, but the diagonal factorization is
# the signed-regularization LDL loop (ported from ldlt.jl, ORDER-FREE per-supernode-local dmax per
# amendment E), and the update is U_s = children − L21·D·L21ᵀ (D-scaled gemm, not syrk). Returns
# inertia. The factor (unit-lower L + signed D) is dmax-independent → matches ldlt! bit-for-bit;
# only the inertia COUNTS can diverge in a scale-band on heterogeneous KKT (amendment E).
function cpu_multifrontal_ldlt!(x_host::Vector{T}, arena::Vector{T}, dvec::Vector{T},
                                M::MFSymbolic, sym, A, signs::Vector{Int8}) where {T}
    ns = sym.nsuper; super = sym.super; rowind_ptr = sym.rowind_ptr; px = sym.px; amap = sym.amap
    fill!(x_host, zero(T)); ascale = zero(T)                  # assembly + ‖A‖-scale
    @inbounds for p in eachindex(A.nzval)
        m = Int(amap[p]); m == 0 && continue
        v = A.nzval[p]; x_host[m] = v; a = abs(v); a > ascale && (ascale = a)
    end
    delta = T(PureSparse.LDLT_DELTA) * (iszero(ascale) ? one(T) : ascale); zeta = eps(real(T))
    n_pos = 0; n_neg = 0; n_zero = 0; n_perturbed = 0; max_pert = 0.0
    GC.@preserve x_host arena begin
    @inbounds for s in 1:ns
        nscol = Int(super[s + 1]) - Int(super[s]); nsrow = Int(rowind_ptr[s + 1]) - Int(rowind_ptr[s])
        below_s = nsrow - nscol; j0 = Int(super[s])
        panel = _mfpanel(x_host, Int(px[s]), nsrow, nscol)
        uo = Int(M.uoff[s]); us = Int(M.usize[s])
        for i in 1:us; arena[i] = zero(T); end                # zero U_s in the WORK slot (§M.3)
        U_s = below_s > 0 ? _mfpanel(arena, 1, below_s, below_s) : nothing
        for ci in Int(M.children_ptr[s]):(Int(M.children_ptr[s + 1]) - 1)   # extend-add children
            c = Int(M.children[ci]); nsc_c = Int(super[c + 1]) - Int(super[c])
            below_c = Int(rowind_ptr[c + 1]) - Int(rowind_ptr[c]) - nsc_c; below_c == 0 && continue
            U_c = _mfpanel(arena, Int(M.uoff[c]), below_c, below_c); eb = Int(M.emap_ptr[c])
            for b in 1:below_c
                rb = Int(M.emap[eb + b - 1])
                for a in b:below_c
                    ra = Int(M.emap[eb + a - 1]); v = U_c[a, b]
                    rb ≤ nscol ? (panel[ra, rb] += v) : (U_s[ra - nscol, rb - nscol] += v)
                end
            end
        end
        dmax_local = zero(T)                                  # order-free local dmax (amendment E)
        for j in 1:nscol
            jg = j0 + j - 1; dj = panel[j, j]; adj = abs(dj)
            if adj ≤ zeta * max(dmax_local, delta)            # zero test, order-free (amendment E)
                n_zero += 1
            elseif dj > zero(T); n_pos += 1
            else; n_neg += 1 end
            sg = signs[jg]
            wrongsign = (sg == Int8(1) && !(dj > zero(T))) || (sg == Int8(-1) && !(dj < zero(T)))
            if wrongsign || adj < delta                       # signed regularization (dmax-independent)
                target = sg == Int8(0) ? (signbit(dj) ? -one(T) : one(T)) : T(sg)
                newd = target * max(delta, adj); n_perturbed += 1
                pert = Float64(abs(newd - dj)); pert > max_pert && (max_pert = pert); dj = newd
            end
            dvec[jg] = dj; adf = abs(dj); adf > dmax_local && (dmax_local = adf)
            panel[j, j] = one(T); invd = inv(dj)
            for i in (j + 1):nsrow; panel[i, j] *= invd; end
            if j < nscol
                lcol = view(panel, (j + 1):nsrow, j); lrow = view(panel, (j + 1):nscol, j)
                trail = view(panel, (j + 1):nsrow, (j + 1):nscol)
                PureSparse.ger!(-dj, lcol, lrow, trail)
            end
        end
        if below_s > 0                                        # U_s = children − L21·D·L21ᵀ
            L21 = view(panel, (nscol + 1):nsrow, 1:nscol)
            W = Matrix{T}(undef, below_s, nscol)
            for jj in 1:nscol
                d = dvec[j0 + jj - 1]
                for ii in 1:below_s; W[ii, jj] = L21[ii, jj] * d; end
            end
            PureSparse.gemm!(U_s, W, Matrix(L21); transA = 'N', transB = 'T', alpha = -one(T), beta = one(T))
            copyto!(arena, uo, arena, 1, us)                  # compact work slot → STACK at uoff[s]
        end
    end
    end
    return (true, 0, (; n_pos, n_neg, n_zero, n_perturbed, max_pert))
end

# Dense signed-regularization LDLᵀ of a small nscol×nscol block (the blocked device-LDL's
# sequential part, run on CPU per GPU front — design_gpu.md §6, amendment E). In place:
# `block` ← unit-lower L11; returns D + inertia deltas. `sg` = permuted signs for this block.
function _ldl_block!(block::AbstractMatrix{T}, sg, delta::T, zeta::T) where {T}
    nscol = size(block, 1); dvals = Vector{T}(undef, nscol)
    np = 0; nn = 0; nz = 0; npert = 0; maxp = 0.0; dmax = zero(T)
    @inbounds for j in 1:nscol
        dj = block[j, j]; adj = abs(dj)
        if adj ≤ zeta * max(dmax, delta); nz += 1
        elseif dj > zero(T); np += 1
        else; nn += 1 end
        s = sg[j]
        wrong = (s == Int8(1) && !(dj > zero(T))) || (s == Int8(-1) && !(dj < zero(T)))
        if wrong || adj < delta
            target = s == Int8(0) ? (signbit(dj) ? -one(T) : one(T)) : T(s)
            newd = target * max(delta, adj); npert += 1
            p = Float64(abs(newd - dj)); p > maxp && (maxp = p); dj = newd
        end
        dvals[j] = dj; ad = abs(dj); ad > dmax && (dmax = ad)
        block[j, j] = one(T); invd = inv(dj)
        for i in (j + 1):nscol; block[i, j] *= invd; end
        if j < nscol
            lc = view(block, (j + 1):nscol, j); tr = view(block, (j + 1):nscol, (j + 1):nscol)
            PureSparse.ger!(-dj, lc, lc, tr)
        end
    end
    return dvals, np, nn, nz, npert, maxp
end
