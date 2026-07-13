# Supernodal LLᵀ numeric factorization (design.md §4). Left-looking scheme (Ng–Peyton
# 1993; Rothberg–Gupta 1993 — never CHOLMOD source, CLAUDE.md req 1). The update-loop
# scheduling (relmap, linked-list descendant tracking) was hand-traced and verified sound
# by an adversarial review before implementation (design.md §0 "N1"); every PureBLAS
# kernel call below matches that review's verified mapping exactly.

@inline _row(rowind::Vector{Ti}, base::Int, local_idx::Int) where {Ti} = Int(rowind[base + local_idx - 1])

# A plain `Matrix{T}` view of `v[off:off+nrow*ncol-1]` (column-major, off 1-based).
# `reshape(view(v, off:...), nrow, ncol)` is semantically identical but produces a
# `ReshapedArray{T,2,SubArray{...}}` — an exotic type PureBLAS's kernels apparently
# haven't been exercised against, which triggered a catastrophic ~90s-PER-KERNEL first-
# call LLVM compile (measured directly: potrf! alone took 93s on that type, 1.3s on the
# `unsafe_wrap`'d plain Matrix below — a ~70x difference). `x`/`cbuf` are kept alive by
# the caller's `GC.@preserve` for the whole factorization call, so this is safe: the
# wrapped array never outlives the vector it points into.
@inline function _panelview(v::Vector{T}, off::Int, nrow::Int, ncol::Int) where {T}
    return unsafe_wrap(Array, pointer(v, off), (nrow, ncol))
end

# Build every supernode's panel wrapper ONCE, at factor-construction time — reused across
# every subsequent `cholesky!` call on this factor (design.md §0 follow-up: this is what
# removes the per-call `unsafe_wrap` allocation from the hot refactorize path; only the
# variable-shaped update-block buffer still allocates fresh per call, see `_panelview`'s
# remaining uses in `cholesky!` below).
function _build_panels(x::Vector{T}, sym::Symbolic{Ti}) where {T,Ti<:Integer}
    nsuper = sym.nsuper
    panels = Vector{Matrix{T}}(undef, nsuper)
    @inbounds for s in 1:nsuper
        nrow = Int(sym.rowind_ptr[s + 1] - sym.rowind_ptr[s])
        ncol = Int(sym.super[s + 1] - sym.super[s])
        panels[s] = _panelview(x, Int(sym.px[s]), nrow, ncol)
    end
    return panels
end

"""
    cholesky(sym::Symbolic{Ti}, A::SparseMatrixCSC{T}) where {T,Ti} -> SupernodalFactor{T,Ti}

Allocate a new factor sharing `sym` and factor `A` into it (design.md §6). For repeated
factorization of matrices sharing `sym`'s sparsity pattern, prefer [`cholesky!`](@ref) on
a factor obtained once from this — it never allocates.
"""
function cholesky(sym::Symbolic{Ti}, A::SparseMatrixCSC{T}) where {T,Ti<:Integer}
    xsize = Int(sym.px[sym.nsuper + 1]) - 1
    x = Vector{T}(undef, xsize)
    panels = _build_panels(x, sym)
    F = SupernodalFactor{T,Ti}(sym, x, panels, Workspace{T,Ti}(sym), FactorStats(), true)
    cholesky!(F, A)
    return F
end

"""
    cholesky(A::SparseMatrixCSC; ordering=AMDOrdering()) -> SupernodalFactor

One-shot symbolic analysis + numeric factorization (design.md §6).
"""
function cholesky(A::SparseMatrixCSC{T}; ordering::AbstractOrdering = AMDOrdering()) where {T}
    sym = symbolic(A; ordering)
    return cholesky(sym, A)
end

"""
    cholesky!(F::SupernodalFactor, A::SparseMatrixCSC) -> F

Refactor `A` (same sparsity pattern as `F.sym`, new numeric values) into `F` in place.
Zero allocations after warmup (CLAUDE.md req 5). Sets `F.ok = false` and records the
failing column in `F.stats` on a non-SPD pivot rather than throwing — callers that want
`PosDefException` semantics should check `issuccess(F)` or use `\\`/`ldiv!`.
"""
function cholesky!(F::SupernodalFactor{T,Ti}, A::SparseMatrixCSC{T}) where {T,Ti<:Integer}
    sym = F.sym
    nsuper = sym.nsuper
    super = sym.super
    rowind_ptr = sym.rowind_ptr
    rowind = sym.rowind
    snode_of = sym.snode_of
    px = sym.px
    amap = sym.amap
    x = F.x
    ws = F.ws
    relmap = ws.relmap
    head = ws.head
    next = ws.next
    dptr = ws.dptr
    cbuf = ws.c

    fill!(x, zero(T))
    @inbounds for p in eachindex(A.nzval)
        m = Int(amap[p])
        m == 0 && continue
        x[m] = A.nzval[p]
    end
    @inbounds fill!(head, zero(Ti))

    F.ok = true
    fail_col = 0

    GC.@preserve x cbuf begin
    @inbounds for s in 1:nsuper
        j0 = Int(super[s])
        j1 = Int(super[s + 1]) - 1
        nscol = j1 - j0 + 1
        rp0 = Int(rowind_ptr[s])
        nsrow = Int(rowind_ptr[s + 1]) - rp0

        for k in 1:nsrow
            relmap[_row(rowind, rp0, k)] = Ti(k)
        end

        panel = F.panels[s]

        # ---- 2. apply pending updates from descendants queued on head[s] ----
        d = head[s]
        while d != zero(Ti)
            dInt = Int(d)
            dnext_link = next[dInt]
            q = Int(dptr[dInt])
            drp0 = Int(rowind_ptr[dInt])
            nsrow_d = Int(rowind_ptr[dInt + 1]) - drp0
            ncol_d = Int(super[dInt + 1]) - Int(super[dInt])
            panel_d = F.panels[dInt]

            k1 = 0
            while q + k1 <= nsrow_d && _row(rowind, drp0, q + k1) <= j1
                k1 += 1
            end
            k2 = nsrow_d - (q + k1 - 1)

            if k1 > 0
                L1 = view(panel_d, q:(q + k1 - 1), 1:ncol_d)
                ctot = k1 + k2
                C = _panelview(cbuf, 1, ctot, k1)
                C1 = view(C, 1:k1, :)
                syrk!(C1, L1; uplo = 'L', trans = 'N', alpha = -one(T), beta = zero(T))
                # scatter LOWER triangle of C1 only (syrk leaves the strict-upper stale)
                for a in 1:k1
                    ra = _row(rowind, drp0, q + a - 1)
                    lra = Int(relmap[ra])
                    for b in 1:a
                        rb = _row(rowind, drp0, q + b - 1)
                        lrb = Int(relmap[rb])
                        panel[lra, lrb] += C1[a, b]
                    end
                end
                if k2 > 0
                    L2 = view(panel_d, (q + k1):nsrow_d, 1:ncol_d)
                    C2 = view(C, (k1 + 1):ctot, :)
                    gemm!(C2, L2, L1; transA = 'N', transB = 'T', alpha = -one(T), beta = zero(T))
                    for a in 1:k2
                        ra = _row(rowind, drp0, q + k1 + a - 1)
                        lra = Int(relmap[ra])
                        for b in 1:k1
                            rb = _row(rowind, drp0, q + b - 1)
                            lrb = Int(relmap[rb])
                            panel[lra, lrb] += C2[a, b]
                        end
                    end
                end
            end

            newq = q + k1
            if newq <= nsrow_d
                dptr[dInt] = Ti(newq)
                s2 = Int(snode_of[_row(rowind, drp0, newq)])
                next[dInt] = head[s2]
                head[s2] = d
            end
            d = dnext_link
        end

        # ---- 3. factor diagonal block ----
        Ldiag = view(panel, 1:nscol, 1:nscol)
        ok = true
        try
            potrf!(Ldiag; uplo = 'L')
        catch e
            e isa LinearAlgebra.PosDefException || rethrow()
            ok = false
        end
        if !ok
            F.ok = false
            # PureBLAS's Float64-lower fast path doesn't thread the exact failing pivot
            # column through PosDefException (always .info==1 there — see its
            # lapack.jl:537 `# ponytail: faer returns Bool; exact pivot column not
            # threaded`), so we can only report "somewhere at or after column j0", not
            # the precise column. j0 is the honest, non-overclaiming value.
            F.stats.fail_col = j0
            F.stats.nnzL = Int(sym.nnzL)
            F.stats.flops = sym.flops
            return F
        end

        # ---- 4. panel solve ----
        if nsrow > nscol
            Lbelow = view(panel, (nscol + 1):nsrow, 1:nscol)
            trsm!(Lbelow, Ldiag; side = 'R', uplo = 'L', transA = 'T', diag = 'N', alpha = one(T))

            # ---- 5. schedule s onto its first ancestor ----
            dptr[s] = Ti(nscol + 1)
            s2 = Int(snode_of[_row(rowind, rp0, nscol + 1)])
            next[s] = head[s2]
            head[s2] = Ti(s)
        end
    end
    end # GC.@preserve x cbuf

    F.stats.nnzL = Int(sym.nnzL)
    F.stats.flops = sym.flops
    return F
end
