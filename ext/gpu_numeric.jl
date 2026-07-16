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
