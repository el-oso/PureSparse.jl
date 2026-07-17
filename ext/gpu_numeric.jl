# Synchronous all-on-GPU supernodal Cholesky numeric loop (design_gpu.md §4), CORRECTNESS-FIRST.
# A faithful port of `src/numeric/llt.jl`: the left-looking schedule (head/next/dptr/relmap —
# pattern-only, host-driven) is identical; the panel numeric ops go to device (pure gemm for the
# trailing update, cuSOLVER potrf/trsm for the diagonal factor + panel solve). No hybrid CPU/GPU
# split and no async streams yet — this establishes a CORRECT device factor (validated against the
# CPU factor by the §10.1 oracle) before the frontier/stream optimization (§5.4) layers on top.
#
# Per descendant d updating ancestor s: the whole ctot×k1 trailing-update block is one gemm
# C = −A_block·L1ᵀ (A_block = panel_d[q:q+ctot-1,:], L1 = panel_d[q:q+k1-1,:]), scattered into the
# panel as panel[ir[a], ir[b]] += C[a,b] (ir = the resolved target rows; ir[1:k1] are the ancestor
# columns, so they double as the column targets). Each (a,b) hits a distinct cell → no atomics.

using CUDA.CUSOLVER: potrf!
using CUDA.CUBLAS: trsm!
using LinearAlgebra: LAPACK, BLAS, mul!

@kernel unsafe_indices = true function _scatter_add!(panel, @Const(C), @Const(ir), k1, ctot)
    idx = @index(Global)
    if idx ≤ ctot * k1
        a = (idx - 1) % ctot + 1
        b = (idx - 1) ÷ ctot + 1
        @inbounds begin
            ra = ir[a]; rb = ir[b]
            # LOWER-only, matching llt.jl's syrk uplo='L' + gemm: write lower-triangle of the
            # k1×k1 diagonal sub-block + all off-diagonal rows (ra>nscol≥rb there), skip the
            # never-read strict-upper diagonal cells (they must stay 0, as in the CPU factor).
            (ra ≥ rb) && (panel[ra, rb] += C[a, b])     # distinct cells; no atomic needed
        end
    end
end

# panel view (column-major nsrow×nscol) into the packed device factor storage at px[s]
@inline function _dpanel(dx, px, s, nsrow, nscol)
    off = Int(px[s])
    return reshape(view(dx, off:(off + nsrow * nscol - 1)), nsrow, nscol)
end

"""
    gpu_cholesky_sync!(dx, G, A) -> (ok, fail_col)

Factor `A` into the device-resident packed storage `dx` (length `G.xlen`), synchronous all-GPU
(design_gpu.md §4, correctness-first). Returns `(ok::Bool, fail_col::Int)` — `ok=false` on a
non-SPD pivot (cuSOLVER `potrf` info>0). Host-drives `llt.jl`'s schedule; panels are device views.
"""
function gpu_cholesky_sync!(dx, G::GPUSymbolic{Ti}, A) where {Ti}
    sym = G.cpu
    T = eltype(dx)
    nsuper = sym.nsuper
    super = sym.super; rowind = sym.rowind; rowind_ptr = sym.rowind_ptr
    snode_of = sym.snode_of; px = sym.px

    gpu_assemble!(dx, CuArray(A.nzval), G.d_amap)        # step 1: assemble A into dx

    # host scheduling state (all pattern-only, matches llt.jl ws)
    relmap = zeros(Ti, sym.n)
    head = zeros(Ti, nsuper); nxt = zeros(Ti, nsuper); dptr = zeros(Ti, nsuper)
    ir = Vector{Ti}(undef, sym.max_extend_rows + 1)
    mer = sym.max_extend_rows
    d_cbuf = CUDA.zeros(T, max(mer, 1), max(mer, 1))
    backend = get_backend(dx)

    _row(rp0, k) = Int(rowind[rp0 + k - 1])

    @inbounds for s in 1:nsuper
        j0 = Int(super[s]); j1 = Int(super[s + 1]) - 1; nscol = j1 - j0 + 1
        rp0 = Int(rowind_ptr[s]); nsrow = Int(rowind_ptr[s + 1]) - rp0
        for k in 1:nsrow
            relmap[_row(rp0, k)] = Ti(k)
        end
        panel = _dpanel(dx, px, s, nsrow, nscol)

        # step 2: apply pending descendant updates queued on head[s]
        d = head[s]
        while d != zero(Ti)
            dInt = Int(d)
            dnext = nxt[dInt]
            q = Int(dptr[dInt])
            drp0 = Int(rowind_ptr[dInt]); nsrow_d = Int(rowind_ptr[dInt + 1]) - drp0
            ncol_d = Int(super[dInt + 1]) - Int(super[dInt])
            panel_d = _dpanel(dx, px, dInt, nsrow_d, ncol_d)

            k1 = 0
            while q + k1 ≤ nsrow_d && _row(drp0, q + k1) ≤ j1
                k1 += 1
            end
            k2 = nsrow_d - (q + k1 - 1)
            ctot = k1 + k2
            if k1 > 0
                for a in 1:ctot
                    ir[a] = relmap[_row(drp0, q + a - 1)]
                end
                A_block = view(panel_d, q:(q + ctot - 1), 1:ncol_d)   # ctot × ncol_d
                L1 = view(panel_d, q:(q + k1 - 1), 1:ncol_d)          # k1 × ncol_d
                C = view(d_cbuf, 1:ctot, 1:k1)
                gpu_gemm_nt!(C, A_block, L1, -one(T), zero(T))        # C = −A_block·L1ᵀ
                d_ir = CuArray(view(ir, 1:ctot))
                _scatter_add!(backend, 256)(panel, C, d_ir, k1, ctot; ndrange = cld(ctot * k1, 256) * 256)
            end

            newq = q + k1
            if newq ≤ nsrow_d
                dptr[dInt] = Ti(newq)
                s2 = Int(snode_of[_row(drp0, newq)])
                nxt[dInt] = head[s2]; head[s2] = d
            end
            d = dnext
        end

        # step 3: factor the diagonal block (cuSOLVER lower Cholesky)
        diagblk = view(panel, 1:nscol, 1:nscol)
        _, info = potrf!('L', diagblk)
        if info != 0
            return (false, j0)
        end
        # step 4: panel solve  L21 = A21 · L11⁻ᵀ
        if nsrow > nscol
            sub = view(panel, (nscol + 1):nsrow, 1:nscol)
            trsm!('R', 'L', 'T', 'N', one(T), diagblk, sub)
        end

        # requeue THIS supernode onto its first ancestor (left-looking, llt.jl step 5)
        if nsrow > nscol
            dptr[s] = Ti(nscol + 1)
            s2 = Int(snode_of[_row(rp0, nscol + 1)])
            nxt[s] = head[s2]; head[s2] = Ti(s)
        end
    end
    CUDA.synchronize()
    return (true, 0)
end

# ---------------------------------------------------------------------------------------
# HYBRID Cholesky (design_gpu.md §4/§5) — CPU supernodes factored on CPU (into x_host), GPU
# supernodes (the upward-closed frontier) factored on device (d_nzval). Correctness-first,
# synchronous (no streams yet). The key invariant: every descendant of a GPU supernode is in
# d_nzval — GPU-native, or a boundary CPU panel uploaded once after its CPU factor — so the GPU
# update reads d_nzval uniformly (no GPU→CPU edge, by upward closure). GPU panels are also D2H'd
# into x_host so it holds the complete factor (for the oracle + a CPU solve; the device-resident
# / device-solve realization of amendment B is a later optimization). CPU dense ops use
# LinearAlgebra here (correctness-first); the shipped path uses PureBLAS (amendment C).
@inline _hpanel(x, off, nsrow, nscol) = unsafe_wrap(Array, pointer(x, off), (nsrow, nscol))

"""
    gpu_cholesky_hybrid!(x_host, d_nzval, G, A) -> (ok, fail_col)

Hybrid CPU/GPU supernodal Cholesky. `x_host` (length `G.xlen`) holds the complete factor on
return; `d_nzval` holds the device (GPU + uploaded-boundary) panels. `G.on_gpu` is the frontier.
"""
function gpu_cholesky_hybrid!(x_host::Vector{T}, d_nzval, G::GPUSymbolic{Ti}, A;
                             d2h::Bool = true, d_Anz = nothing) where {T,Ti}
    sym = G.cpu
    nsuper = sym.nsuper
    super = sym.super; rowind = sym.rowind; rowind_ptr = sym.rowind_ptr
    snode_of = sym.snode_of; px = sym.px; amap = sym.amap
    on_gpu = G.on_gpu
    is_boundary = falses(nsuper)
    @inbounds for s in G.boundary; is_boundary[s] = true; end

    fill!(x_host, zero(T))                                   # assemble A into x_host (CPU)
    @inbounds for p in eachindex(A.nzval)
        m = Int(amap[p]); m != 0 && (x_host[m] = A.nzval[p])
    end
    d_Anz === nothing && (d_Anz = CuArray(A.nzval))          # pre-alloc'd across refactors if given
    gpu_assemble!(d_nzval, d_Anz, G.d_amap)                  # assemble A into d_nzval (device)

    relmap = zeros(Ti, sym.n)
    head = zeros(Ti, nsuper); nxt = zeros(Ti, nsuper); dptr = zeros(Ti, nsuper)
    ir = Vector{Ti}(undef, sym.max_extend_rows + 1)
    mer = max(sym.max_extend_rows, 1)
    d_cbuf = CUDA.zeros(T, mer, mer); cbuf_h = Matrix{T}(undef, mer, mer)
    d_ir_buf = CUDA.zeros(Ti, mer + 1)                        # reused per update (no per-update alloc)
    backend = get_backend(d_nzval)
    _row(rp0, k) = Int(rowind[rp0 + k - 1])
    ok = true; failcol = 0

    GC.@preserve x_host cbuf_h begin
    @inbounds for s in 1:nsuper
        j0 = Int(super[s]); j1 = Int(super[s + 1]) - 1; nscol = j1 - j0 + 1
        rp0 = Int(rowind_ptr[s]); nsrow = Int(rowind_ptr[s + 1]) - rp0
        for k in 1:nsrow; relmap[_row(rp0, k)] = Ti(k); end
        gpu_s = on_gpu[s]; off_s = Int(px[s]); len_s = Int(px[s + 1]) - off_s

        d = head[s]
        while d != zero(Ti)
            dInt = Int(d); dnext = nxt[dInt]; q = Int(dptr[dInt])
            drp0 = Int(rowind_ptr[dInt]); nsrow_d = Int(rowind_ptr[dInt + 1]) - drp0
            ncol_d = Int(super[dInt + 1]) - Int(super[dInt])
            k1 = 0
            while q + k1 ≤ nsrow_d && _row(drp0, q + k1) ≤ j1; k1 += 1; end
            k2 = nsrow_d - (q + k1 - 1); ctot = k1 + k2
            if k1 > 0
                for a in 1:ctot; ir[a] = relmap[_row(drp0, q + a - 1)]; end
                if gpu_s
                    pd = _dpanel(d_nzval, px, dInt, nsrow_d, ncol_d)
                    Ablk = view(pd, q:(q + ctot - 1), 1:ncol_d); L1 = view(pd, q:(q + k1 - 1), 1:ncol_d)
                    C = view(d_cbuf, 1:ctot, 1:k1)
                    gpu_gemm_nt!(C, Ablk, L1, -one(T), zero(T))
                    copyto!(d_ir_buf, 1, ir, 1, ctot)        # reuse buffer (no per-update alloc)
                    d_ir = view(d_ir_buf, 1:ctot)
                    panel_g = _dpanel(d_nzval, px, s, nsrow, nscol)
                    _scatter_add!(backend, 256)(panel_g, C, d_ir, k1, ctot; ndrange = cld(ctot * k1, 256) * 256)
                else
                    pd = _hpanel(x_host, Int(px[dInt]), nsrow_d, ncol_d)
                    Ablk = view(pd, q:(q + ctot - 1), 1:ncol_d); L1 = view(pd, q:(q + k1 - 1), 1:ncol_d)
                    C = view(cbuf_h, 1:ctot, 1:k1)
                    mul!(C, Ablk, L1', -one(T), zero(T))     # C = −A_block·L1ᵀ
                    panel_h = _hpanel(x_host, off_s, nsrow, nscol)
                    for b in 1:k1, a in 1:ctot
                        ra = Int(ir[a]); rb = Int(ir[b])
                        (ra ≥ rb) && (panel_h[ra, rb] += C[a, b])   # lower-only
                    end
                end
            end
            newq = q + k1
            if newq ≤ nsrow_d
                dptr[dInt] = Ti(newq); s2 = Int(snode_of[_row(drp0, newq)])
                nxt[dInt] = head[s2]; head[s2] = d
            end
            d = dnext
        end

        if gpu_s
            panel_g = _dpanel(d_nzval, px, s, nsrow, nscol)
            diag = view(panel_g, 1:nscol, 1:nscol)
            _, info = potrf!('L', diag)                       # cuSOLVER (amendment C interim)
            info != 0 && (ok = false; failcol = j0; break)
            nsrow > nscol && trsm!('R', 'L', 'T', 'N', one(T), diag, view(panel_g, (nscol + 1):nsrow, 1:nscol))
            d2h && copyto!(x_host, off_s, d_nzval, off_s, len_s)   # D2H → full factor (skip for perf)
        else
            panel_h = _hpanel(x_host, off_s, nsrow, nscol)
            diag = view(panel_h, 1:nscol, 1:nscol)
            _, info = LAPACK.potrf!('L', diag)
            info != 0 && (ok = false; failcol = j0; break)
            nsrow > nscol && BLAS.trsm!('R', 'L', 'T', 'N', one(T), diag, view(panel_h, (nscol + 1):nsrow, 1:nscol))
            is_boundary[s] && copyto!(d_nzval, off_s, x_host, off_s, len_s)   # H2D boundary panel
        end
        if nsrow > nscol
            dptr[s] = Ti(nscol + 1); s2 = Int(snode_of[_row(rp0, nscol + 1)])
            nxt[s] = head[s2]; head[s2] = Ti(s)
        end
    end
    end # GC.@preserve
    CUDA.synchronize()
    return (ok, failcol)
end

# =======================================================================================
# MULTIFRONTAL GPU Cholesky (design_gpu.md §M, amendment F) — Path B. Replaces the launch-bound
# left-looking per-descendant updates with front assembly: per front, one extend-add scatter per
# child + one potrf + trsm + syrk. Reuses the CPU-validated formulation (multifrontal.jl); panels
# in d_nzval (in place, bit-compatible), update matrices U_s in a device arena.
@inline _dslab(dx, off, m, n) = reshape(view(dx, off:(off + m * n - 1)), m, n)

# Extend-add: scatter the lower triangle of child update U_c into the parent — panel cells
# (emap ≤ nscol) or U_s cells (emap > nscol) — via the ascending per-child emap (§M.2). Distinct
# target cells per (a,b) → no atomics (children applied sequentially).
@kernel unsafe_indices = true function _mf_extend_add!(panel, U_s, @Const(U_c), @Const(emap_c),
                                                       nscol, below_c)
    idx = @index(Global)
    if idx ≤ below_c * below_c
        a = (idx - 1) % below_c + 1
        b = (idx - 1) ÷ below_c + 1
        if a ≥ b
            @inbounds begin
                ra = Int(emap_c[a]); rb = Int(emap_c[b]); v = U_c[a, b]
                if rb ≤ nscol
                    panel[ra, rb] += v
                else
                    U_s[ra - nscol, rb - nscol] += v
                end
            end
        end
    end
end

"""
    gpu_multifrontal_cholesky!(d_nzval, d_arena, Msym, G, A) -> (ok, fail_col)

All-GPU multifrontal supernodal Cholesky (design_gpu.md §M). `d_nzval` holds the factor panels
(length `G.xlen`); `d_arena` the update matrices (length `Msym.arena_peak`). Correctness-first
(monotonic arena, per-call `d_emap` upload).
"""
function gpu_multifrontal_cholesky!(d_nzval, d_arena, Msym::MFSymbolic{Ti}, G::GPUSymbolic, A) where {Ti}
    sym = G.cpu; ns = sym.nsuper; super = sym.super; rowind_ptr = sym.rowind_ptr; px = sym.px
    T = eltype(d_nzval)
    gpu_assemble!(d_nzval, CuArray(A.nzval), G.d_amap)     # A into panel regions
    d_emap = CuArray(Msym.emap)
    backend = get_backend(d_nzval)
    d_dummy = CUDA.zeros(T, 1, 1)
    ok = true; failcol = 0
    @inbounds for s in 1:ns
        nscol = Int(super[s + 1]) - Int(super[s]); nsrow = Int(rowind_ptr[s + 1]) - Int(rowind_ptr[s])
        below_s = nsrow - nscol
        panel = _dpanel(d_nzval, px, s, nsrow, nscol)
        uo = Int(Msym.uoff[s]); us = Int(Msym.usize[s])
        us > 0 && CUDA.fill!(view(d_arena, uo:(uo + us - 1)), zero(T))   # zero U_s (pitfall #4)
        U_s = below_s > 0 ? _dslab(d_arena, uo, below_s, below_s) : d_dummy

        for ci in Int(Msym.children_ptr[s]):(Int(Msym.children_ptr[s + 1]) - 1)
            c = Int(Msym.children[ci])
            below_c = Int(rowind_ptr[c + 1]) - Int(rowind_ptr[c]) - (Int(super[c + 1]) - Int(super[c]))
            below_c == 0 && continue
            U_c = _dslab(d_arena, Int(Msym.uoff[c]), below_c, below_c)
            emc = view(d_emap, Int(Msym.emap_ptr[c]):(Int(Msym.emap_ptr[c + 1]) - 1))
            _mf_extend_add!(backend, 256)(panel, U_s, U_c, emc, nscol, below_c;
                                          ndrange = cld(below_c * below_c, 256) * 256)
        end

        diag = view(panel, 1:nscol, 1:nscol)
        _, info = potrf!('L', diag)
        info != 0 && (ok = false; failcol = Int(super[s]); break)
        if below_s > 0
            L21 = view(panel, (nscol + 1):nsrow, 1:nscol)
            trsm!('R', 'L', 'T', 'N', one(T), diag, L21)
            gpu_syrk_nt!(U_s, L21, -one(T), one(T))          # U_s = children − L21·L21ᵀ
        end
    end
    CUDA.synchronize()
    return (ok, failcol)
end
