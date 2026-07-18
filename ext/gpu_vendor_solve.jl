# Split from gpu_numeric.jl (T3.1): the VENDOR (cuBLAS trsv/gemv) device solves — the §8-gate
# reference arm 4, retained verbatim, CUDA-only, NEVER on the shipped path. Uses `_dpanel` from
# gpu_numeric.jl (module-wide; function refs resolve at call time so include order is free).

# =======================================================================================
# VENDOR device solve (cuBLAS trsv/gemv per supernode) — the pre-batched solve, retained
# verbatim (commit 9aef65d) as the §8-gate reference arm 4. CUDA-only, never on the shipped
# path.
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
    gpu_solve_vendor!(d_y, d_nzval, G, d_upd, d_gath)

Vendor (cuBLAS) supernodal solve `A·x = b` on device — per-supernode trsv/gemv/scatter.
`d_upd`/`d_gath` are `max_extend_rows` scratch vectors.
"""
function gpu_solve_vendor!(d_y, d_nzval, G::GPUSymbolic, d_upd, d_gath)
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

"""
    gpu_solve_ldlt_vendor!(d_y, d_nzval, d_dvec, G, d_upd, d_gath)

Vendor (cuBLAS) LDLᵀ solve: forward unit-L·z=b, D⁻¹ scale, backward unit-Lᵀ·x=w.
"""
function gpu_solve_ldlt_vendor!(d_y, d_nzval, d_dvec, G::GPUSymbolic, d_upd, d_gath)
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
