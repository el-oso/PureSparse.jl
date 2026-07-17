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
using LinearAlgebra: LAPACK, BLAS, mul!, PosDefException

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
function gpu_multifrontal_cholesky!(d_nzval, d_arena, Msym::MFSymbolic{Ti}, G::GPUSymbolic, A;
                                    d_emap = nothing, d_dummy = nothing, d_Anz = nothing,
                                    ws = nothing) where {Ti}
    sym = G.cpu; ns = sym.nsuper; super = sym.super; rowind_ptr = sym.rowind_ptr; px = sym.px
    T = eltype(d_nzval)
    # persistent device buffers reused across refactors (amendment A: 0 pattern-H2D, 0 device-pool)
    gpu_assemble!(d_nzval, isnothing(d_Anz) ? CuArray(A.nzval) : copyto!(d_Anz, A.nzval), G.d_amap)
    isnothing(d_emap) && (d_emap = CuArray(Msym.emap))
    backend = get_backend(d_nzval)
    isnothing(d_dummy) && (d_dummy = CUDA.zeros(T, 1, 1))
    mnc = 0; @inbounds for s in 1:ns; c = Int(super[s + 1]) - Int(super[s]); c > mnc && (mnc = c); end
    isnothing(ws) && (ws = FrontWS(T, cld(mnc, 64)))     # fused-front workspace (amendment C)
    CUDA.fill!(ws.info, Int32(0))                        # deferred devinfo (amendment D)
    ok = true; failcol = 0
    @inbounds for s in 1:ns
        nscol = Int(super[s + 1]) - Int(super[s]); nsrow = Int(rowind_ptr[s + 1]) - Int(rowind_ptr[s])
        below_s = nsrow - nscol
        panel = _dpanel(d_nzval, px, s, nsrow, nscol)
        uo = Int(Msym.uoff[s]); us = Int(Msym.usize[s])
        us > 0 && CUDA.fill!(view(d_arena, 1:us), zero(T))   # zero U_s in the WORK slot (§M.3)
        U_s = below_s > 0 ? _dslab(d_arena, 1, below_s, below_s) : d_dummy

        for ci in Int(Msym.children_ptr[s]):(Int(Msym.children_ptr[s + 1]) - 1)
            c = Int(Msym.children[ci])
            below_c = Int(rowind_ptr[c + 1]) - Int(rowind_ptr[c]) - (Int(super[c + 1]) - Int(super[c]))
            below_c == 0 && continue
            U_c = _dslab(d_arena, Int(Msym.uoff[c]), below_c, below_c)   # child on the STACK
            emc = view(d_emap, Int(Msym.emap_ptr[c]):(Int(Msym.emap_ptr[c + 1]) - 1))
            _mf_extend_add!(backend, 256)(panel, U_s, U_c, emc, nscol, below_c;
                                          ndrange = cld(below_c * below_c, 256) * 256)
        end

        gpu_front!(panel, nscol, ws)                         # fused pure potrf(diag)+solve(L21) (deferred devinfo)
        if below_s > 0
            L21 = view(panel, (nscol + 1):nsrow, 1:nscol)
            gpu_syrk_nt!(U_s, L21, -one(T), one(T))          # U_s = children − L21·L21ᵀ
            copyto!(d_arena, uo, d_arena, 1, us)             # compact work slot → STACK at uoff[s]
        end
    end
    CUDA.synchronize()
    fc = Int(CUDA.@allowscalar ws.info[1]); fc != 0 && (ok = false; failcol = fc)
    return (ok, failcol)
end

# =======================================================================================
# HYBRID MULTIFRONTAL (design_gpu.md §M.4, amendment F) — per-front on_gpu[] dispatch. Small
# fronts factor on CPU (host arena/panels), the upward-closed GPU crown on device. A crossing CPU
# front (CPU with a GPU parent) uploads its U to the device arena at the SAME offset, so a GPU
# parent's extend-add reads all children's U's from the device arena uniformly. No U downloads
# (upward closure). One uoff layout, two physical arenas (host + device).
function gpu_multifrontal_hybrid!(x_host::Vector{T}, d_nzval, host_arena::Vector{T}, device_arena,
                                  Msym::MFSymbolic{Ti}, G::GPUSymbolic, A; d2h::Bool = true,
                                  d_emap = nothing, d_dummy = nothing, d_Anz = nothing,
                                  ws = nothing) where {T,Ti}
    sym = G.cpu; ns = sym.nsuper; super = sym.super; rowind_ptr = sym.rowind_ptr
    px = sym.px; sparent = sym.sparent; amap = sym.amap; on_gpu = G.on_gpu
    fill!(x_host, zero(T))
    @inbounds for p in eachindex(A.nzval); m = Int(amap[p]); m != 0 && (x_host[m] = A.nzval[p]); end
    gpu_assemble!(d_nzval, isnothing(d_Anz) ? CuArray(A.nzval) : copyto!(d_Anz, A.nzval), G.d_amap)
    isnothing(d_emap) && (d_emap = CuArray(Msym.emap)); backend = get_backend(d_nzval)
    isnothing(d_dummy) && (d_dummy = CUDA.zeros(T, 1, 1))
    mnc = 0; @inbounds for s in 1:ns; c = Int(super[s + 1]) - Int(super[s]); c > mnc && (mnc = c); end
    isnothing(ws) && (ws = FrontWS(T, cld(mnc, 64)))     # fused-front workspace (amendment C)
    CUDA.fill!(ws.info, Int32(0))                        # deferred devinfo (amendment D)
    ok = true; failcol = 0
    GC.@preserve x_host host_arena begin
    @inbounds for s in 1:ns
        nscol = Int(super[s + 1]) - Int(super[s]); nsrow = Int(rowind_ptr[s + 1]) - Int(rowind_ptr[s])
        below_s = nsrow - nscol; uo = Int(Msym.uoff[s]); us = Int(Msym.usize[s])
        c0 = Int(Msym.children_ptr[s]); c1 = Int(Msym.children_ptr[s + 1]) - 1
        childbelow(c) = Int(rowind_ptr[c + 1]) - Int(rowind_ptr[c]) - (Int(super[c + 1]) - Int(super[c]))

        if on_gpu[s]
            panel = _dpanel(d_nzval, px, s, nsrow, nscol)
            us > 0 && CUDA.fill!(view(device_arena, 1:us), zero(T))   # zero U_s in the device WORK slot
            U_s = below_s > 0 ? _dslab(device_arena, 1, below_s, below_s) : d_dummy
            for ci in c0:c1
                c = Int(Msym.children[ci]); bc = childbelow(c); bc == 0 && continue
                U_c = _dslab(device_arena, Int(Msym.uoff[c]), bc, bc)   # child on the device STACK
                emc = view(d_emap, Int(Msym.emap_ptr[c]):(Int(Msym.emap_ptr[c + 1]) - 1))
                _mf_extend_add!(backend, 256)(panel, U_s, U_c, emc, nscol, bc; ndrange = cld(bc * bc, 256) * 256)
            end
            gpu_front!(panel, nscol, ws)                     # fused pure potrf(diag)+solve(L21)
            if below_s > 0
                L21 = view(panel, (nscol + 1):nsrow, 1:nscol)
                gpu_syrk_nt!(U_s, L21, -one(T), one(T))              # U_s = children − L21·L21ᵀ
                copyto!(device_arena, uo, device_arena, 1, us)       # compact → device STACK
            end
            d2h && copyto!(x_host, Int(px[s]), d_nzval, Int(px[s]), nsrow * nscol)   # panel D2H (oracle)
        else
            panel = _hpanel(x_host, Int(px[s]), nsrow, nscol)
            for i in 1:us; host_arena[i] = zero(T); end          # zero U_s in the host WORK slot
            U_s = below_s > 0 ? _hpanel(host_arena, 1, below_s, below_s) : nothing
            for ci in c0:c1
                c = Int(Msym.children[ci]); bc = childbelow(c); bc == 0 && continue
                U_c = _hpanel(host_arena, Int(Msym.uoff[c]), bc, bc); eb = Int(Msym.emap_ptr[c])   # child on the host STACK
                for b in 1:bc
                    rb = Int(Msym.emap[eb + b - 1])
                    for a in b:bc
                        ra = Int(Msym.emap[eb + a - 1]); v = U_c[a, b]
                        rb ≤ nscol ? (panel[ra, rb] += v) : (U_s[ra - nscol, rb - nscol] += v)
                    end
                end
            end
            diag = view(panel, 1:nscol, 1:nscol)
            try; PureSparse.potrf!(diag; uplo = 'L')           # PureBLAS on CPU fronts (design §M.4)
            catch e; e isa PosDefException || rethrow(); ok = false; failcol = Int(super[s]); break; end
            if below_s > 0
                L21 = view(panel, (nscol + 1):nsrow, 1:nscol)
                PureSparse.trsm!(L21, diag; side = 'R', uplo = 'L', transA = 'T', diag = 'N', alpha = one(T))
                PureSparse.syrk!(U_s, L21; uplo = 'L', trans = 'N', alpha = -one(T), beta = one(T))
                copyto!(host_arena, uo, host_arena, 1, us)        # compact work slot → host STACK
            end
            # crossing (CPU front, GPU parent): upload its U to the device arena (same STACK offset)
            (Int(sparent[s]) != 0 && on_gpu[Int(sparent[s])] && us > 0) &&
                copyto!(device_arena, uo, host_arena, uo, us)
        end
    end
    end
    CUDA.synchronize()
    fc = Int(CUDA.@allowscalar ws.info[1])                # GPU-front non-PD (deferred, amendment D)
    fc != 0 && (ok && (failcol = fc); ok = false)
    return (ok, failcol)
end

# =======================================================================================
# Device supernodal triangular SOLVE (design_gpu.md §7, amendment B) — factor stays device-
# resident; only b/x vectors transfer. Forward L·y=b then backward Lᵀ·x=y, per supernode:
# trsv on the diagonal block + gemv for the below-diagonal + scatter/gather via rowind.
using CUDA.CUBLAS: trsv!, gemv!

@kernel function _scatter_y!(y, @Const(upd), @Const(rowind), rp0, nscol, below)
    k = @index(Global)
    k ≤ below && (@inbounds y[Int(rowind[rp0 + nscol + k - 1])] += upd[k])
end
@kernel function _gather_y!(g, @Const(y), @Const(rowind), rp0, nscol, below)
    k = @index(Global)
    k ≤ below && (@inbounds g[k] = y[Int(rowind[rp0 + nscol + k - 1])])
end

"""
    gpu_solve!(d_y, d_nzval, G, d_upd, d_gath)

In-place supernodal solve `A·x = b` on device (`A = L·Lᵀ` permuted): `d_y` holds the permuted
RHS on entry, the permuted solution on exit. Factor panels are read from `d_nzval` (must be the
FULL device-resident factor). `d_upd`/`d_gath` are `max_extend_rows` scratch vectors.
"""
function gpu_solve!(d_y, d_nzval, G::GPUSymbolic, d_upd, d_gath)
    sym = G.cpu; ns = sym.nsuper; super = sym.super; rowind_ptr = sym.rowind_ptr; px = sym.px
    d_rowind = G.d_rowind; backend = get_backend(d_y); T = eltype(d_y)
    @inbounds for s in 1:ns                                   # forward L·y = b
        j0 = Int(super[s]); nscol = Int(super[s + 1]) - j0
        rp0 = Int(rowind_ptr[s]); nsrow = Int(rowind_ptr[s + 1]) - rp0; below = nsrow - nscol
        panel = _dpanel(d_nzval, px, s, nsrow, nscol)
        yblk = view(d_y, j0:(j0 + nscol - 1)); Ldiag = view(panel, 1:nscol, 1:nscol)
        trsv!('L', 'N', 'N', Ldiag, yblk)
        if below > 0
            Lbelow = view(panel, (nscol + 1):nsrow, 1:nscol); upd = view(d_upd, 1:below)
            gemv!('N', -one(T), Lbelow, yblk, zero(T), upd)
            _scatter_y!(backend, 256)(d_y, upd, d_rowind, rp0, nscol, below; ndrange = cld(below, 256) * 256)
        end
    end
    @inbounds for s in ns:-1:1                                # backward Lᵀ·x = y
        j0 = Int(super[s]); nscol = Int(super[s + 1]) - j0
        rp0 = Int(rowind_ptr[s]); nsrow = Int(rowind_ptr[s + 1]) - rp0; below = nsrow - nscol
        panel = _dpanel(d_nzval, px, s, nsrow, nscol)
        yblk = view(d_y, j0:(j0 + nscol - 1)); Ldiag = view(panel, 1:nscol, 1:nscol)
        if below > 0
            Lbelow = view(panel, (nscol + 1):nsrow, 1:nscol); gath = view(d_gath, 1:below)
            _gather_y!(backend, 256)(gath, d_y, d_rowind, rp0, nscol, below; ndrange = cld(below, 256) * 256)
            gemv!('T', -one(T), Lbelow, gath, one(T), yblk)
        end
        trsv!('L', 'T', 'N', Ldiag, yblk)
    end
    return d_y
end

# Upload all CPU-front panels to d_nzval so the whole factor is device-resident for the solve
# (make-solve-ready for the hybrid; §M.4 — only CPU-front panels move, GPU fronts already there).
function gpu_upload_cpu_panels!(d_nzval, x_host, G::GPUSymbolic)
    sym = G.cpu; px = sym.px
    @inbounds for s in 1:sym.nsuper
        G.on_gpu[s] && continue
        off = Int(px[s]); len = Int(px[s + 1]) - off
        copyto!(d_nzval, off, x_host, off, len)
    end
    return d_nzval
end

# =======================================================================================
# MULTIFRONTAL GPU LDLᵀ (design_gpu.md §6/§M, amendment E) — blocked device-LDL. The small
# nscol×nscol diagonal block's signed-regularization LDL runs on CPU (D2H → _ldl_block! → H2D);
# the tall parts (L21 panel solve, U_s update) run on device. Reuses the Cholesky multifrontal
# structure with potrf→block-LDL, trsm→unit-trsm+D⁻¹-scale, syrk→D-scaled-gemm.
@kernel function _col_scale!(out, @Const(inp), @Const(dvec), base, invflag, m, n)
    idx = @index(Global)
    if idx ≤ m * n
        i = (idx - 1) % m + 1; j = (idx - 1) ÷ m + 1
        @inbounds begin
            d = dvec[base + j]
            out[i, j] = inp[i, j] * (invflag ? inv(d) : d)
        end
    end
end

function gpu_multifrontal_ldlt!(d_nzval, d_arena, d_dvec, Msym::MFSymbolic{Ti}, G::GPUSymbolic,
                                A, signs::Vector{Int8};
                                d_emap = nothing, d_dummy = nothing, d_W = nothing, d_Anz = nothing,
                                d_signs = nothing, ldlws = nothing) where {Ti}
    sym = G.cpu; ns = sym.nsuper; super = sym.super; rowind_ptr = sym.rowind_ptr; px = sym.px
    T = eltype(d_nzval)
    gpu_assemble!(d_nzval, isnothing(d_Anz) ? CuArray(A.nzval) : copyto!(d_Anz, A.nzval), G.d_amap)
    ascale = zero(T); @inbounds for v in A.nzval; a = abs(v); a > ascale && (ascale = a); end
    delta = T(PureSparse.LDLT_DELTA) * (iszero(ascale) ? one(T) : ascale); zeta = eps(real(T))
    isnothing(d_emap) && (d_emap = CuArray(Msym.emap)); backend = get_backend(d_nzval)
    isnothing(d_dummy) && (d_dummy = CUDA.zeros(T, 1, 1))
    mer = max(sym.max_extend_rows, 1); isnothing(d_W) && (d_W = CUDA.zeros(T, mer, mer))
    isnothing(d_signs) && (d_signs = CuArray(signs))     # signs on device (front-local slices)
    isnothing(ldlws) && (ldlws = LDLFrontWS(T)); CUDA.fill!(ldlws.stats, zero(T))   # inertia accum on device
    ok = true; failcol = 0
    @inbounds for s in 1:ns
        nscol = Int(super[s + 1]) - Int(super[s]); nsrow = Int(rowind_ptr[s + 1]) - Int(rowind_ptr[s])
        below_s = nsrow - nscol; j0 = Int(super[s])
        panel = _dpanel(d_nzval, px, s, nsrow, nscol)
        uo = Int(Msym.uoff[s]); us = Int(Msym.usize[s])
        us > 0 && CUDA.fill!(view(d_arena, 1:us), zero(T))   # zero U_s in the WORK slot (§M.3)
        U_s = below_s > 0 ? _dslab(d_arena, 1, below_s, below_s) : d_dummy
        for ci in Int(Msym.children_ptr[s]):(Int(Msym.children_ptr[s + 1]) - 1)
            c = Int(Msym.children[ci])
            bc = Int(rowind_ptr[c + 1]) - Int(rowind_ptr[c]) - (Int(super[c + 1]) - Int(super[c]))
            bc == 0 && continue
            U_c = _dslab(d_arena, Int(Msym.uoff[c]), bc, bc)   # child on the STACK
            emc = view(d_emap, Int(Msym.emap_ptr[c]):(Int(Msym.emap_ptr[c + 1]) - 1))
            _mf_extend_add!(backend, 256)(panel, U_s, U_c, emc, nscol, bc; ndrange = cld(bc * bc, 256) * 256)
        end
        # fused pure signed-LDL front: diag LDL (signs+reg+inertia) + panel solve, one kernel (amendment C/E)
        sgn_v = view(d_signs, j0:(j0 + nscol - 1)); dv_v = view(d_dvec, j0:(j0 + nscol - 1))
        gpu_ldlt_front!(panel, nscol, sgn_v, dv_v, delta, zeta, ldlws)   # L11 unit, D→d_dvec, L21=W·D⁻¹
        if below_s > 0
            L21 = view(panel, (nscol + 1):nsrow, 1:nscol)
            W2 = view(d_W, 1:below_s, 1:nscol)
            _col_scale!(backend, 256)(W2, L21, d_dvec, j0 - 1, false, below_s, nscol; ndrange = cld(below_s * nscol, 256) * 256)  # W2 = L21·D
            gpu_gemm_nt!(U_s, W2, L21, -one(T), one(T))             # U_s −= L21·D·L21ᵀ
            copyto!(d_arena, uo, d_arena, 1, us)                    # compact work slot → STACK at uoff[s]
        end
    end
    CUDA.synchronize()
    st = Array(ldlws.stats)                                # inertia accumulated on device
    return (ok, failcol, (; n_pos = Int(st[1]), n_neg = Int(st[2]), n_zero = Int(st[3]),
                          n_perturbed = Int(st[4]), max_pert = Float64(st[5])))
end

# =======================================================================================
# HYBRID MULTIFRONTAL LDLᵀ (design_gpu.md §6/§M.4, amendments E/F) — per-front dispatch. CPU
# fronts: CPU LDLᵀ (host panel/arena). GPU fronts: blocked device-LDL. Crossing-U uploads as in
# the Cholesky hybrid. `dvec` (host) holds all D; `d_dvec` holds GPU-front D (device D-scales);
# make-solve-ready uploads host dvec + CPU panels for the device solve.
function gpu_multifrontal_ldlt_hybrid!(x_host::Vector{T}, d_nzval, host_arena::Vector{T}, device_arena,
                                       dvec::Vector{T}, d_dvec, Msym::MFSymbolic{Ti}, G::GPUSymbolic,
                                       A, signs::Vector{Int8}; d2h::Bool=true,
                                       d_emap = nothing, d_dummy = nothing, d_W = nothing, d_Anz = nothing,
                                       d_signs = nothing, ldlws = nothing) where {T,Ti}
    sym = G.cpu; ns = sym.nsuper; super = sym.super; rowind_ptr = sym.rowind_ptr
    px = sym.px; sparent = sym.sparent; amap = sym.amap; on_gpu = G.on_gpu
    isnothing(d_signs) && (d_signs = CuArray(signs))     # signs on device (GPU-front slices)
    isnothing(ldlws) && (ldlws = LDLFrontWS(T)); CUDA.fill!(ldlws.stats, zero(T))   # GPU-front inertia
    fill!(x_host, zero(T)); ascale = zero(T)
    @inbounds for p in eachindex(A.nzval)
        m = Int(amap[p]); m == 0 && continue
        v = A.nzval[p]; x_host[m] = v; a = abs(v); a > ascale && (ascale = a)
    end
    gpu_assemble!(d_nzval, isnothing(d_Anz) ? CuArray(A.nzval) : copyto!(d_Anz, A.nzval), G.d_amap)
    delta = T(PureSparse.LDLT_DELTA) * (iszero(ascale) ? one(T) : ascale); zeta = eps(real(T))
    isnothing(d_emap) && (d_emap = CuArray(Msym.emap)); backend = get_backend(d_nzval)
    isnothing(d_dummy) && (d_dummy = CUDA.zeros(T, 1, 1))
    mer = max(sym.max_extend_rows, 1); isnothing(d_W) && (d_W = CUDA.zeros(T, mer, mer))
    np = 0; nn = 0; nz = 0; npert = 0; maxp = 0.0; ok = true; failcol = 0
    GC.@preserve x_host host_arena begin
    @inbounds for s in 1:ns
        nscol = Int(super[s + 1]) - Int(super[s]); nsrow = Int(rowind_ptr[s + 1]) - Int(rowind_ptr[s])
        below_s = nsrow - nscol; j0 = Int(super[s]); uo = Int(Msym.uoff[s]); us = Int(Msym.usize[s])
        c0 = Int(Msym.children_ptr[s]); c1 = Int(Msym.children_ptr[s + 1]) - 1
        cbelow(c) = Int(rowind_ptr[c + 1]) - Int(rowind_ptr[c]) - (Int(super[c + 1]) - Int(super[c]))

        if on_gpu[s]
            panel = _dpanel(d_nzval, px, s, nsrow, nscol)
            us > 0 && CUDA.fill!(view(device_arena, 1:us), zero(T))   # zero U_s in the device WORK slot
            U_s = below_s > 0 ? _dslab(device_arena, 1, below_s, below_s) : d_dummy
            for ci in c0:c1
                c = Int(Msym.children[ci]); bc = cbelow(c); bc == 0 && continue
                U_c = _dslab(device_arena, Int(Msym.uoff[c]), bc, bc)   # child on the device STACK
                emc = view(d_emap, Int(Msym.emap_ptr[c]):(Int(Msym.emap_ptr[c + 1]) - 1))
                _mf_extend_add!(backend, 256)(panel, U_s, U_c, emc, nscol, bc; ndrange = cld(bc * bc, 256) * 256)
            end
            # fused pure signed-LDL front (diag LDL + panel solve, one kernel — amendment C/E)
            sgn_v = view(d_signs, j0:(j0 + nscol - 1)); dv_v = view(d_dvec, j0:(j0 + nscol - 1))
            gpu_ldlt_front!(panel, nscol, sgn_v, dv_v, delta, zeta, ldlws)   # L11 unit, D→d_dvec, L21=W·D⁻¹
            copyto!(dvec, j0, d_dvec, j0, nscol)               # mirror GPU-front D to host dvec (make-solve-ready)
            if below_s > 0
                L21 = view(panel, (nscol + 1):nsrow, 1:nscol)
                W2 = view(d_W, 1:below_s, 1:nscol)
                _col_scale!(backend, 256)(W2, L21, d_dvec, j0 - 1, false, below_s, nscol;
                                          ndrange = cld(below_s * nscol, 256) * 256)   # W2 = L21·D
                gpu_gemm_nt!(U_s, W2, L21, -one(T), one(T))          # U_s −= L21·D·L21ᵀ
                copyto!(device_arena, uo, device_arena, 1, us)       # compact → device STACK
            end
            d2h && copyto!(x_host, Int(px[s]), d_nzval, Int(px[s]), nsrow * nscol)
        else
            panel = _hpanel(x_host, Int(px[s]), nsrow, nscol)
            for i in 1:us; host_arena[i] = zero(T); end          # zero U_s in the host WORK slot
            U_s = below_s > 0 ? _hpanel(host_arena, 1, below_s, below_s) : nothing
            for ci in c0:c1
                c = Int(Msym.children[ci]); bc = cbelow(c); bc == 0 && continue
                U_c = _hpanel(host_arena, Int(Msym.uoff[c]), bc, bc); eb = Int(Msym.emap_ptr[c])   # child on the host STACK
                for b in 1:bc
                    rb = Int(Msym.emap[eb + b - 1])
                    for a in b:bc
                        ra = Int(Msym.emap[eb + a - 1]); v = U_c[a, b]
                        rb ≤ nscol ? (panel[ra, rb] += v) : (U_s[ra - nscol, rb - nscol] += v)
                    end
                end
            end
            dmax_local = zero(T)
            for j in 1:nscol
                jg = j0 + j - 1; dj = panel[j, j]; adj = abs(dj)
                if adj ≤ zeta * max(dmax_local, delta); nz += 1
                elseif dj > zero(T); np += 1 else nn += 1 end
                sg = signs[jg]
                wrong = (sg == Int8(1) && !(dj > zero(T))) || (sg == Int8(-1) && !(dj < zero(T)))
                if wrong || adj < delta
                    target = sg == Int8(0) ? (signbit(dj) ? -one(T) : one(T)) : T(sg)
                    newd = target * max(delta, adj); npert += 1
                    p = Float64(abs(newd - dj)); p > maxp && (maxp = p); dj = newd
                end
                dvec[jg] = dj; adf = abs(dj); adf > dmax_local && (dmax_local = adf)
                panel[j, j] = one(T); invd = inv(dj)
                for i in (j + 1):nsrow; panel[i, j] *= invd; end
                if j < nscol
                    PureSparse.ger!(-dj, view(panel, (j + 1):nsrow, j), view(panel, (j + 1):nscol, j),
                                    view(panel, (j + 1):nsrow, (j + 1):nscol))
                end
            end
            if below_s > 0
                L21 = view(panel, (nscol + 1):nsrow, 1:nscol); W = Matrix{T}(undef, below_s, nscol)
                for jj in 1:nscol
                    d = dvec[j0 + jj - 1]; for ii in 1:below_s; W[ii, jj] = L21[ii, jj] * d; end
                end
                PureSparse.gemm!(U_s, W, Matrix(L21); transA = 'N', transB = 'T', alpha = -one(T), beta = one(T))
                copyto!(host_arena, uo, host_arena, 1, us)        # compact work slot → host STACK
            end
            # crossing (CPU front, GPU parent): upload its U to the device arena (same STACK offset)
            (Int(sparent[s]) != 0 && on_gpu[Int(sparent[s])] && us > 0) &&
                copyto!(device_arena, uo, host_arena, uo, us)
        end
    end
    end
    CUDA.synchronize()
    st = Array(ldlws.stats)                                # add GPU-front inertia to the CPU-front totals
    np += Int(st[1]); nn += Int(st[2]); nz += Int(st[3]); npert += Int(st[4]); maxp = max(maxp, Float64(st[5]))
    return (ok, failcol, (; n_pos = np, n_neg = nn, n_zero = nz, n_perturbed = npert, max_pert = maxp))
end

# Device LDLᵀ solve: forward L·z=b (unit), D⁻¹ scale, backward Lᵀ·x=w (unit). L is unit-lower.
function gpu_solve_ldlt!(d_y, d_nzval, d_dvec, G::GPUSymbolic, d_upd, d_gath)
    sym = G.cpu; ns = sym.nsuper; super = sym.super; rowind_ptr = sym.rowind_ptr; px = sym.px
    d_rowind = G.d_rowind; backend = get_backend(d_y); T = eltype(d_y)
    @inbounds for s in 1:ns                                   # forward L·z = b (unit lower)
        j0 = Int(super[s]); nscol = Int(super[s + 1]) - j0
        rp0 = Int(rowind_ptr[s]); nsrow = Int(rowind_ptr[s + 1]) - rp0; below = nsrow - nscol
        panel = _dpanel(d_nzval, px, s, nsrow, nscol); yblk = view(d_y, j0:(j0 + nscol - 1))
        trsv!('L', 'N', 'U', view(panel, 1:nscol, 1:nscol), yblk)
        if below > 0
            Lbelow = view(panel, (nscol + 1):nsrow, 1:nscol); upd = view(d_upd, 1:below)
            gemv!('N', -one(T), Lbelow, yblk, zero(T), upd)
            _scatter_y!(backend, 256)(d_y, upd, d_rowind, rp0, nscol, below; ndrange = cld(below, 256) * 256)
        end
    end
    d_y ./= d_dvec                                            # D⁻¹ scale (w = D⁻¹·z)
    @inbounds for s in ns:-1:1                                # backward Lᵀ·x = w (unit lower)
        j0 = Int(super[s]); nscol = Int(super[s + 1]) - j0
        rp0 = Int(rowind_ptr[s]); nsrow = Int(rowind_ptr[s + 1]) - rp0; below = nsrow - nscol
        panel = _dpanel(d_nzval, px, s, nsrow, nscol); yblk = view(d_y, j0:(j0 + nscol - 1))
        if below > 0
            Lbelow = view(panel, (nscol + 1):nsrow, 1:nscol); gath = view(d_gath, 1:below)
            _gather_y!(backend, 256)(gath, d_y, d_rowind, rp0, nscol, below; ndrange = cld(below, 256) * 256)
            gemv!('T', -one(T), Lbelow, gath, one(T), yblk)
        end
        trsv!('L', 'T', 'U', view(panel, 1:nscol, 1:nscol), yblk)
    end
    return d_y
end
