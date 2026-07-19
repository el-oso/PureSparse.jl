# ==============================  GPU NUMERIC ENGINES — ORIENTATION  ==============================
# Backend-generic (design_gpu_multibackend.md §B1): every device op on the shipped :auto path goes
# through the shim in gpu_shared.jl (_dev_zeros / _dev_upload / fill! / KernelAbstractions.synchronize
# / Array), so this file compiles + runs on any KA backend (CUDA, ROCm, oneAPI). The ONE exception is
# the `frontmode=:vendor` branch (gate arm 4), which calls cuBLAS `trsm!` — guarded by
# `_vendor_available()` (CUDA-only) and never reached on :auto. Same exception in gpu_dense.jl.
#
#   SHIPPED (the M6 gate path — multifrontal, design_gpu.md §M, amendment F):
#     gpu_multifrontal_cholesky!  / gpu_multifrontal_hybrid!        (Cholesky, all-GPU / hybrid)
#     gpu_multifrontal_ldlt!      / gpu_multifrontal_ldlt_hybrid!   (LDLᵀ,     all-GPU / hybrid)
#     gpu_solve! / gpu_solve_ldlt! / gpu_upload_cpu_panels!         (device solves + make-ready)
#     Shared front helpers: _dslab, _dpanel, _hpanel, _mf_extend_add! (kernel), _col_scale! (kernel),
#     and _extend_add_cpu! / _cpu_ldl_front! (in multifrontal.jl).
#
#   REFERENCE, SUPERSEDED (§4 left-looking) + VENDOR (§8 arm 4): CUDA-only (cuSOLVER/cuBLAS), split
#   out to gpu_leftlooking_reference.jl + gpu_vendor_solve.jl, included ONLY by the CUDA ext.
# ================================================================================================
using LinearAlgebra: PosDefException

# panel view (column-major nsrow×nscol) into the packed device factor storage at px[s]
@inline function _dpanel(dx, px, s, nsrow, nscol)
    off = Int(px[s])
    return reshape(view(dx, off:(off + nsrow * nscol - 1)), nsrow, nscol)
end

# host panel view (column-major) into x_host at a byte offset — CPU-front / D2H staging
@inline _hpanel(x, off, nsrow, nscol) = unsafe_wrap(Array, pointer(x, off), (nsrow, nscol))

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
    T = eltype(d_nzval); backend = get_backend(d_nzval)
    # persistent device buffers reused across refactors (amendment A: 0 pattern-H2D, 0 device-pool)
    gpu_assemble!(d_nzval, isnothing(d_Anz) ? _dev_upload(backend, A.nzval) : copyto!(d_Anz, A.nzval), G.d_amap)
    isnothing(d_emap) && (d_emap = _dev_upload(backend, Msym.emap))
    isnothing(d_dummy) && (d_dummy = _dev_zeros(backend, T, 1, 1))
    mnc = 0; @inbounds for s in 1:ns; c = Int(super[s + 1]) - Int(super[s]); c > mnc && (mnc = c); end
    isnothing(ws) && (ws = FrontWS(backend, T, cld(mnc, 64)))     # fused-front workspace (amendment C)
    fill!(ws.info, Int32(0))                        # deferred devinfo (amendment D)
    ok = true; failcol = 0
    @inbounds for s in 1:ns
        nscol = Int(super[s + 1]) - Int(super[s]); nsrow = Int(rowind_ptr[s + 1]) - Int(rowind_ptr[s])
        below_s = nsrow - nscol
        panel = _dpanel(d_nzval, px, s, nsrow, nscol)
        uo = Int(Msym.uoff[s]); us = Int(Msym.usize[s])
        us > 0 && fill!(view(d_arena, 1:us), zero(T))   # zero U_s in the WORK slot (§M.3)
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
    KernelAbstractions.synchronize(backend)
    fc = Int(Array(ws.info)[1]); fc != 0 && (ok = false; failcol = fc)
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
                                  ws = nothing, frontmode::Symbol = :auto) where {T,Ti}
    sym = G.cpu; ns = sym.nsuper; super = sym.super; rowind_ptr = sym.rowind_ptr
    px = sym.px; sparent = sym.sparent; amap = sym.amap; on_gpu = G.on_gpu
    backend = get_backend(d_nzval)
    fill!(x_host, zero(T))
    @inbounds for p in eachindex(A.nzval); m = Int(amap[p]); m != 0 && (x_host[m] = A.nzval[p]); end
    gpu_assemble!(d_nzval, isnothing(d_Anz) ? _dev_upload(backend, A.nzval) : copyto!(d_Anz, A.nzval), G.d_amap)
    isnothing(d_emap) && (d_emap = _dev_upload(backend, Msym.emap))
    isnothing(d_dummy) && (d_dummy = _dev_zeros(backend, T, 1, 1))
    mnc = 0; @inbounds for s in 1:ns; c = Int(super[s + 1]) - Int(super[s]); c > mnc && (mnc = c); end
    isnothing(ws) && (ws = FrontWS(backend, T, cld(mnc, 64)))     # fused-front workspace (amendment C)
    fill!(ws.info, Int32(0))                        # deferred devinfo (amendment D)
    ok = true; failcol = 0
    GC.@preserve x_host host_arena begin
    @inbounds for s in 1:ns
        nscol = Int(super[s + 1]) - Int(super[s]); nsrow = Int(rowind_ptr[s + 1]) - Int(rowind_ptr[s])
        below_s = nsrow - nscol; uo = Int(Msym.uoff[s]); us = Int(Msym.usize[s])
        c0 = Int(Msym.children_ptr[s]); c1 = Int(Msym.children_ptr[s + 1]) - 1
        childbelow(c) = Int(rowind_ptr[c + 1]) - Int(rowind_ptr[c]) - (Int(super[c + 1]) - Int(super[c]))

        if on_gpu[s]
            panel = _dpanel(d_nzval, px, s, nsrow, nscol)
            us > 0 && fill!(view(device_arena, 1:us), zero(T))   # zero U_s in the device WORK slot
            U_s = below_s > 0 ? _dslab(device_arena, 1, below_s, below_s) : d_dummy
            for ci in c0:c1
                c = Int(Msym.children[ci]); bc = childbelow(c); bc == 0 && continue
                U_c = _dslab(device_arena, Int(Msym.uoff[c]), bc, bc)   # child on the device STACK
                emc = view(d_emap, Int(Msym.emap_ptr[c]):(Int(Msym.emap_ptr[c + 1]) - 1))
                _mf_extend_add!(backend, 256)(panel, U_s, U_c, emc, nscol, bc; ndrange = cld(bc * bc, 256) * 256)
            end
            gpu_front!(panel, nscol, ws; mode = frontmode)   # fused pure potrf(diag)+solve(L21)
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
                _extend_add_cpu!(panel, U_s, U_c, Msym.emap, eb, bc, nscol)
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
    KernelAbstractions.synchronize(backend)
    fc = Int(Array(ws.info)[1])                # GPU-front non-PD (deferred, amendment D)
    fc != 0 && (ok && (failcol = fc); ok = false)
    return (ok, failcol)
end

# =======================================================================================
# Device supernodal triangular SOLVE (design_gpu.md §7/§8, amendment B) — factor stays device-
# resident; only b/x vectors transfer. LEVEL-SCHEDULED batched kernels (ext/gpu_solve.jl, pure
# KA): one launch per elimination level instead of per-supernode trsv/gemv/scatter (which was
# launch-bound — ~63k launches ≈ the whole factor time on SQD 40³).
solve_schedule(G::GPUSymbolic) = solve_schedule(G.cpu.sparent, G.cpu.px, G.d_rowind)

"""
    gpu_solve!(d_y, d_nzval, G[, d_upd, d_gath]; sched=solve_schedule(G))

In-place supernodal solve `A·x = b` on device (`A = L·Lᵀ` permuted): `d_y` holds the permuted
RHS on entry, the permuted solution on exit. Factor panels are read from `d_nzval` (must be the
FULL device-resident factor). Pass a prebuilt `sched` to amortize the schedule upload across
solves ("analyze once, solve many"). `d_upd`/`d_gath` are accepted for call compatibility but
unused (the batched kernels need no scratch).
"""
function gpu_solve!(d_y, d_nzval, G::GPUSymbolic, d_upd = nothing, d_gath = nothing;
                    sched::SolveSchedule = solve_schedule(G))
    return batched_solve!(d_y, d_nzval, G.d_rowind, G.d_rowind_ptr, G.d_super, sched,
                          false, nothing)
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
    T = eltype(d_nzval); backend = get_backend(d_nzval)
    gpu_assemble!(d_nzval, isnothing(d_Anz) ? _dev_upload(backend, A.nzval) : copyto!(d_Anz, A.nzval), G.d_amap)
    ascale = zero(T); @inbounds for v in A.nzval; a = abs(v); a > ascale && (ascale = a); end
    delta = T(PureSparse.LDLT_DELTA) * (iszero(ascale) ? one(T) : ascale); zeta = eps(real(T))
    isnothing(d_emap) && (d_emap = _dev_upload(backend, Msym.emap))
    isnothing(d_dummy) && (d_dummy = _dev_zeros(backend, T, 1, 1))
    mer = max(sym.max_extend_rows, 1); isnothing(d_W) && (d_W = _dev_zeros(backend, T, mer, mer))
    isnothing(d_signs) && (d_signs = _dev_upload(backend, signs))     # signs on device (front-local slices)
    isnothing(ldlws) && (ldlws = LDLFrontWS(backend, T)); fill!(ldlws.stats, zero(T))   # inertia accum on device
    ok = true; failcol = 0
    @inbounds for s in 1:ns
        nscol = Int(super[s + 1]) - Int(super[s]); nsrow = Int(rowind_ptr[s + 1]) - Int(rowind_ptr[s])
        below_s = nsrow - nscol; j0 = Int(super[s])
        panel = _dpanel(d_nzval, px, s, nsrow, nscol)
        uo = Int(Msym.uoff[s]); us = Int(Msym.usize[s])
        us > 0 && fill!(view(d_arena, 1:us), zero(T))   # zero U_s in the WORK slot (§M.3)
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
    KernelAbstractions.synchronize(backend)
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
                                       d_signs = nothing, ldlws = nothing,
                                       frontmode::Symbol = :auto) where {T,Ti}
    sym = G.cpu; ns = sym.nsuper; super = sym.super; rowind_ptr = sym.rowind_ptr
    px = sym.px; sparent = sym.sparent; amap = sym.amap; on_gpu = G.on_gpu
    backend = get_backend(d_nzval)
    isnothing(d_signs) && (d_signs = _dev_upload(backend, signs))     # signs on device (GPU-front slices)
    isnothing(ldlws) && (ldlws = LDLFrontWS(backend, T)); fill!(ldlws.stats, zero(T))   # GPU-front inertia
    fill!(x_host, zero(T)); ascale = zero(T)
    @inbounds for p in eachindex(A.nzval)
        m = Int(amap[p]); m == 0 && continue
        v = A.nzval[p]; x_host[m] = v; a = abs(v); a > ascale && (ascale = a)
    end
    gpu_assemble!(d_nzval, isnothing(d_Anz) ? _dev_upload(backend, A.nzval) : copyto!(d_Anz, A.nzval), G.d_amap)
    delta = T(PureSparse.LDLT_DELTA) * (iszero(ascale) ? one(T) : ascale); zeta = eps(real(T))
    isnothing(d_emap) && (d_emap = _dev_upload(backend, Msym.emap))
    isnothing(d_dummy) && (d_dummy = _dev_zeros(backend, T, 1, 1))
    mer = max(sym.max_extend_rows, 1); isnothing(d_W) && (d_W = _dev_zeros(backend, T, mer, mer))
    np = 0; nn = 0; nz = 0; npert = 0; maxp = 0.0; ok = true; failcol = 0
    GC.@preserve x_host host_arena begin
    @inbounds for s in 1:ns
        nscol = Int(super[s + 1]) - Int(super[s]); nsrow = Int(rowind_ptr[s + 1]) - Int(rowind_ptr[s])
        below_s = nsrow - nscol; j0 = Int(super[s]); uo = Int(Msym.uoff[s]); us = Int(Msym.usize[s])
        c0 = Int(Msym.children_ptr[s]); c1 = Int(Msym.children_ptr[s + 1]) - 1
        cbelow(c) = Int(rowind_ptr[c + 1]) - Int(rowind_ptr[c]) - (Int(super[c + 1]) - Int(super[c]))

        if on_gpu[s]
            panel = _dpanel(d_nzval, px, s, nsrow, nscol)
            us > 0 && fill!(view(device_arena, 1:us), zero(T))   # zero U_s in the device WORK slot
            U_s = below_s > 0 ? _dslab(device_arena, 1, below_s, below_s) : d_dummy
            for ci in c0:c1
                c = Int(Msym.children[ci]); bc = cbelow(c); bc == 0 && continue
                U_c = _dslab(device_arena, Int(Msym.uoff[c]), bc, bc)   # child on the device STACK
                emc = view(d_emap, Int(Msym.emap_ptr[c]):(Int(Msym.emap_ptr[c + 1]) - 1))
                _mf_extend_add!(backend, 256)(panel, U_s, U_c, emc, nscol, bc; ndrange = cld(bc * bc, 256) * 256)
            end
            if frontmode == :vendor
                # vendor reference front (gate arm 4, CUDA-only) — CPU signed-LDL of the diag block
                # (D2H → _ldl_block! → H2D) + cuBLAS unit-trsm + D⁻¹ scale. `trsm!` resolves only in
                # the CUDA ext (gpu_leftlooking_reference.jl); never taken on the shipped :auto path.
                _vendor_available() || error("frontmode=:vendor is a CUDA-only reference arm " *
                    "(cuBLAS trsm); it is not available on $(nameof(typeof(get_backend(d_nzval)))). Use :auto.")
                blk_h = Array(view(panel, 1:nscol, 1:nscol))
                dvals, dnp, dnn, dnz, dnpe, dmp = _ldl_block!(blk_h, view(signs, j0:(j0 + nscol - 1)), delta, zeta)
                np += dnp; nn += dnn; nz += dnz; npert += dnpe; dmp > maxp && (maxp = dmp)
                copyto!(view(panel, 1:nscol, 1:nscol), blk_h)
                @views dvec[j0:(j0 + nscol - 1)] .= dvals; copyto!(d_dvec, j0, dvals, 1, nscol)
                if below_s > 0
                    L21 = view(panel, (nscol + 1):nsrow, 1:nscol)
                    trsm!('R', 'L', 'T', 'U', one(T), view(panel, 1:nscol, 1:nscol), L21)   # cuBLAS
                    nd = cld(below_s * nscol, 256) * 256
                    _col_scale!(backend, 256)(L21, L21, d_dvec, j0 - 1, true, below_s, nscol; ndrange = nd)   # L21 = W·D⁻¹
                    W2 = view(d_W, 1:below_s, 1:nscol)
                    _col_scale!(backend, 256)(W2, L21, d_dvec, j0 - 1, false, below_s, nscol; ndrange = nd)   # W2 = L21·D
                    gpu_gemm_nt!(U_s, W2, L21, -one(T), one(T))          # U_s −= L21·D·L21ᵀ
                    copyto!(device_arena, uo, device_arena, 1, us)       # compact → device STACK
                end
            else
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
            end
            d2h && copyto!(x_host, Int(px[s]), d_nzval, Int(px[s]), nsrow * nscol)
        else
            panel = _hpanel(x_host, Int(px[s]), nsrow, nscol)
            for i in 1:us; host_arena[i] = zero(T); end          # zero U_s in the host WORK slot
            U_s = below_s > 0 ? _hpanel(host_arena, 1, below_s, below_s) : nothing
            for ci in c0:c1
                c = Int(Msym.children[ci]); bc = cbelow(c); bc == 0 && continue
                U_c = _hpanel(host_arena, Int(Msym.uoff[c]), bc, bc); eb = Int(Msym.emap_ptr[c])   # child on the host STACK
                _extend_add_cpu!(panel, U_s, U_c, Msym.emap, eb, bc, nscol)
            end
            np_s, nn_s, nz_s, npert_s, maxp_s =
                _cpu_ldl_front!(panel, U_s, dvec, signs, j0, nscol, nsrow, below_s, delta, zeta)
            np += np_s; nn += nn_s; nz += nz_s; npert += npert_s; maxp_s > maxp && (maxp = maxp_s)
            below_s > 0 && copyto!(host_arena, uo, host_arena, 1, us)  # compact work slot → host STACK
            # crossing (CPU front, GPU parent): upload its U to the device arena (same STACK offset)
            (Int(sparent[s]) != 0 && on_gpu[Int(sparent[s])] && us > 0) &&
                copyto!(device_arena, uo, host_arena, uo, us)
        end
    end
    end
    KernelAbstractions.synchronize(backend)
    st = Array(ldlws.stats)                                # add GPU-front inertia to the CPU-front totals
    np += Int(st[1]); nn += Int(st[2]); nz += Int(st[3]); npert += Int(st[4]); maxp = max(maxp, Float64(st[5]))
    return (ok, failcol, (; n_pos = np, n_neg = nn, n_zero = nz, n_perturbed = npert, max_pert = maxp))
end

# Device LDLᵀ solve: forward L·z=b (unit), D⁻¹ scale, backward Lᵀ·x=w (unit). L is unit-lower.
# Level-scheduled batched kernels (ext/gpu_solve.jl); d_upd/d_gath compat-only, unused.
function gpu_solve_ldlt!(d_y, d_nzval, d_dvec, G::GPUSymbolic, d_upd = nothing, d_gath = nothing;
                         sched::SolveSchedule = solve_schedule(G))
    return batched_solve!(d_y, d_nzval, G.d_rowind, G.d_rowind_ptr, G.d_super, sched,
                          true, d_dvec)
end
