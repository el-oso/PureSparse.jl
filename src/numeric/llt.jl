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

# Staged-update scatter-add: fold columns 1..k1 of the staged block cbuf[1:ctot, 1:k1]
# (lower triangle of the k1×k1 head + all of the (ctot-k1)-row tail — one fused pass,
# they are vertically adjacent rows of the same cbuf columns) into `panel` at the target
# rows ir[1:ctot], whose maximal-consecutive-run structure is rs[1:nr+1] (built by the
# caller's contiguity-check walk). Column-outer so the inner loop walks both `panel` and
# `cbuf` contiguously down one column (ir ascends — rowind is sorted); the row-outer
# order was measured at ~28% of total cholesky! wall time before the swap. Target rows
# come from the ir hoist (one sequential load per element) rather than a per-element
# relmap[_row(...)] double indirection — that recomputation was measured at ~1.5
# ns/element (≈12 of 65 ms at n=2048 on galen/5900X) across k1 identical re-resolutions.
#
# Two strategies, picked per update: run-based contiguous SIMD adds when runs are long
# on average (66% of scattered elements at n=2048 live in runs ≥ 9 rows, 38% in runs
# > 32 — measured), element-based otherwise. Run-based scattering pays a per-run visit
# cost of O(nr) PER COLUMN (r0 only skips runs wholly above the triangle start), so
# applying it unconditionally was a measured net LOSS (57.98 → 59.81 ms at n=2048 on
# galen) — below a mean run length of 8, an ≤8-wide SIMD add plus loop setup does not
# beat 8 plain indexed adds. Shared by cholesky! and ldlt! (identical scatter). Kept
# @noinline: inlining this 3-branch body into the factorization loop was a measured
# 30–95% regression at n=16..64 on wintermute (7640U/Zen4) — the call is once per
# staged update, so call cost is noise while the outer loop stays compact.
@noinline function _scatter_update!(
        panel::Matrix{T}, cbuf::Matrix{T}, ir::Vector{Ti}, rs::Vector{Ti},
        nr::Int, k1::Int, ctot::Int,
) where {T,Ti<:Integer}
    @inbounds if ctot >= 8 * nr
        r0 = 1
        for b in 1:k1
            lrb = Int(ir[b])
            while Int(rs[r0 + 1]) <= b   # run ends before row b: skip forever
                r0 += 1
            end
            for r in r0:nr
                a0 = Int(rs[r])
                aend = Int(rs[r + 1]) - 1
                astart = a0 < b ? b : a0
                lr = Int(ir[astart])
                @simd for t in 0:(aend - astart)
                    panel[lr + t, lrb] += cbuf[astart + t, b]
                end
            end
        end
    else
        for b in 1:k1
            lrb = Int(ir[b])
            for a in b:ctot
                panel[Int(ir[a]), lrb] += cbuf[a, b]
            end
        end
    end
    return nothing
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
    check_refactor_shape(A, sym.n, sym.n, "cholesky!")
    check_refactor_nnz(A, length(sym.amap), "cholesky!")
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
    ir = ws.ir
    rs = ws.rs

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
                # Contiguity fast path: when the descendant's remaining rows occupy a
                # contiguous run of the ancestor's row list (lr(a) = lr0 + a - 1 — the
                # identity-shift scatter, and the overwhelmingly common case in the dense
                # trailing part of the factor where the bulk of the update volume lives),
                # the "compute into cbuf, then scatter-add" pair collapses to syrk!/gemm!
                # accumulating straight into the panel with beta = 1: same arithmetic,
                # same schedule, but no staging write+read of C and no per-element
                # relmap[_row(...)] lookups at all. Profiled at n = 2048 the staged
                # scatter was ~22–28% of total cholesky! wall time; this removes it for
                # the contiguous case. `panel` (ancestor s) and `panel_d` (descendant
                # d < s) never alias — distinct px ranges of x.
                # The check loop doubles as the scatter's index hoist: every
                # relmap[_row(...)] target row it visits is stored into ws.ir (the
                # lookup result is IDENTICAL for every column b of the update block, so
                # resolving it once removes a per-element double indirection that was
                # measured at ~1.5 ns/element — ≈12 of 65 ms at n=2048 on galen/5900X),
                # and simultaneously records the maximal-consecutive-run structure of
                # those targets in ws.rs (contig ⟺ a single run). rowind is sorted, so
                # ir ascends and runs are well defined.
                lr0 = Int(relmap[_row(rowind, drp0, q)])
                ir[1] = Ti(lr0)
                nr = 1
                rs[1] = Ti(1)
                for a in 2:ctot
                    lra = relmap[_row(rowind, drp0, q + a - 1)]
                    ir[a] = lra
                    if lra != ir[a - 1] + one(Ti)
                        nr += 1
                        rs[nr] = Ti(a)
                    end
                end
                rs[nr + 1] = Ti(ctot + 1)
                contig = nr == 1
                if contig
                    # rows q..q+k1-1 are ancestor columns (their local indices are ≤ nscol
                    # because the ancestor's first nscol rows ARE its columns j0..j1), so
                    # the k1×k1 target sits on the panel's diagonal: a syrk with uplo='L'
                    # updates exactly the lower triangle the staged scatter updated.
                    D1 = view(panel, lr0:(lr0 + k1 - 1), lr0:(lr0 + k1 - 1))
                    syrk!(D1, L1; uplo = 'L', trans = 'N', alpha = -one(T), beta = one(T))
                    if k2 > 0
                        L2 = view(panel_d, (q + k1):nsrow_d, 1:ncol_d)
                        B1 = view(panel, (lr0 + k1):(lr0 + ctot - 1), lr0:(lr0 + k1 - 1))
                        gemm!(B1, L2, L1; transA = 'N', transB = 'T', alpha = -one(T), beta = one(T))
                    end
                else
                    C = view(cbuf, 1:ctot, 1:k1)   # zero-alloc: view of a pre-existing Matrix (types.jl)
                    C1 = view(C, 1:k1, :)
                    syrk!(C1, L1; uplo = 'L', trans = 'N', alpha = -one(T), beta = zero(T))
                    if k2 > 0
                        L2 = view(panel_d, (q + k1):nsrow_d, 1:ncol_d)
                        C2 = view(C, (k1 + 1):ctot, :)
                        gemm!(C2, L2, L1; transA = 'N', transB = 'T', alpha = -one(T), beta = zero(T))
                    end
                    # Fold the staged block into the panel (see _scatter_update! above
                    # for the full strategy/measurement notes).
                    _scatter_update!(panel, cbuf, ir, rs, nr, k1, ctot)
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
        # Width-1 fast path: a 1×1 Cholesky is L₁₁ = √A₁₁ and the panel solve is a
        # column scale by 1/L₁₁ (textbook base case — generic small-block algebra, no
        # CHOLMOD content). Done inline because at fragmented transitional sizes the
        # FIXED per-call kernel overhead (dispatch, LAPACK marshaling on the OpenBLAS
        # diagnostic arm, PureBLAS's own dispatch) dominates the ~1 flop of real work:
        # the n=64 sweep matrix has nine width-1..3 supernodes, and in-context timing
        # of every real kernel call (ROADMAP "n=64" addendum) showed potrf!+trsm! call
        # overhead — not arithmetic — as the bulk of the OB-arm gap there. Lives in the
        # shared scheduling code, so BOTH kernel backends benefit. Semantics match
        # potrf!'s 1×1 case exactly: pivot check `real(A₁₁) > 0` (NaN fails → not
        # positive definite, same as LAPACK/PureBLAS), sqrt stored back; the below-
        # diagonal scale multiplies by inv(L₁₁), same as PureBLAS trsm!'s dense-R base
        # (`_scal_simd_ptr!(…, inv(A[j,j]))`) and reference-BLAS trsm.
        ok = true
        if nscol == 1
            d1 = real(panel[1, 1])
            if d1 > 0
                panel[1, 1] = sqrt(d1)
            else
                ok = false
            end
        elseif nscol == 2
            # Width-2 fast path: 2×2 lower Cholesky, same textbook recurrence as
            # PureBLAS's generic unblocked base (`_potf2_lower!`): l11=√d1, scale by
            # inv(l11), Hermitian downdate, l22=√d2. Failure check per pivot, same
            # `real(d) > 0` rule (NaN fails) at either column.
            d1 = real(panel[1, 1])
            if d1 > 0
                l11 = sqrt(d1)
                panel[1, 1] = l11
                i11 = inv(l11)
                l21 = panel[2, 1] * i11
                panel[2, 1] = l21
                d2 = real(panel[2, 2]) - real(l21 * conj(l21))
                if d2 > 0
                    panel[2, 2] = sqrt(d2)
                else
                    ok = false
                end
            else
                ok = false
            end
        else
            Ldiag = view(panel, 1:nscol, 1:nscol)
            try
                potrf!(Ldiag; uplo = 'L')
            catch e
                e isa LinearAlgebra.PosDefException || rethrow()
                ok = false
            end
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
            if nscol == 1
                # width-1 fast path (see step 3's comment): trsm! side='R' on a 1×1
                # diagonal block is a column scale by inv(L₁₁).
                invl = inv(panel[1, 1])
                for i in 2:nsrow
                    panel[i, 1] *= invl
                end
            elseif nscol == 2
                # width-2 fast path: forward substitution per row against the 2×2
                # diagonal block (B := B·inv(Ldiagᵀ), transA='T' — unconjugated
                # coefficient, same semantics as the else-branch trsm! call).
                i11 = inv(panel[1, 1])
                i22 = inv(panel[2, 2])
                l21 = panel[2, 1]
                for i in 3:nsrow
                    x1 = panel[i, 1] * i11
                    panel[i, 1] = x1
                    panel[i, 2] = muladd(x1, -l21, panel[i, 2]) * i22
                end
            else
                Ldiag = view(panel, 1:nscol, 1:nscol)
                Lbelow = view(panel, (nscol + 1):nsrow, 1:nscol)
                trsm!(Lbelow, Ldiag; side = 'R', uplo = 'L', transA = 'T', diag = 'N', alpha = one(T))
            end

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
    check_finite(F.x, "cholesky!")
    return F
end
