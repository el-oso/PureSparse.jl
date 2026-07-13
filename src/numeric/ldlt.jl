# Supernodal LDLᵀ for symmetric quasi-definite systems (design.md §5). Same left-looking
# structure and relmap/linked-list scheduling as `llt.jl` (design.md §4.3), with three
# changes (design.md §5.1):
#   1. Descendant updates carry D — the gemm operand is an L·D column-scaled copy staged
#      in `Workspace.cd` (a trivial scaling loop, no new PureBLAS kernel).
#   2. The diagonal-block factorization is a hand-rolled dense unit-LDLᵀ right-looking
#      column loop (standard textbook algorithm — Golub & Van Loan, *Matrix
#      Computations*, symmetric indefinite factorization without pivoting; no
#      CHOLMOD-specific content), with PureBLAS `ger!` for the rank-1 trailing updates.
#      The same rank-1 update covers the below-diagonal panel rows, so there is no
#      separate trsm panel-solve step (the LLᵀ potrf!+trsm! pair is replaced by this one
#      loop over the full panel height).
#   3. Signed regularization, QDLDL/Clarabel-style (Vanderbei 1995 for SQD strong
#      factorizability; Stellato et al. OSQP/QDLDL and the Clarabel solver for the
#      forced-sign fixed-pivot-order scheme — deliberately NOT MA57's Bunch–Kaufman
#      pivoting, CLAUDE.md requirement 8 / design.md §0 D3). Inertia is accounted on the
#      pre-perturbation pivots, for free, inside the column loop.
#
# Efficiency note (documented opportunity, not a defect): the descendant update computes
# the FULL |R|×|R1| block C with one gemm! even though its top |R1|×|R1| part is
# symmetric (L·D·Lᵀ) — a symmetric-aware rank-k-with-diagonal kernel would halve that
# part's flops. Design.md §5.1 explicitly allows the plain-gemm formulation; revisit only
# if the M2 benchmark pass shows the diagonal-part flops matter.

"""
    ldlt(sym::Symbolic{Ti}, A::SparseMatrixCSC{T}; signs=nothing) -> LDLFactor{T,Ti}

Allocate a new LDLᵀ factor sharing `sym` and factor `A` into it (design.md §5/§6).
`signs` are the expected pivot signs (`+1`/`-1`, `0` = no sign expectation) **in the
ORIGINAL column order of `A`** — they are reindexed through `sym`'s fill-reducing
permutation internally, so the caller never needs to know the factor ordering.
`signs = nothing` means all-free: only the magnitude floor `|d_j| ≥ ldlt_delta·max|A|`
is enforced, never a sign flip. For repeated factorization of matrices sharing `sym`'s
pattern, prefer [`ldlt!`](@ref) on a factor obtained once from this — it never
recomputes the symbolic analysis (CLAUDE.md requirement 7).
"""
function ldlt(
        sym::Symbolic{Ti}, A::SparseMatrixCSC{T};
        signs::Union{Nothing,AbstractVector{<:Integer}} = nothing,
) where {T,Ti<:Integer}
    n = sym.n
    psigns = Vector{Int8}(undef, n)
    if isnothing(signs)
        fill!(psigns, Int8(0))
    else
        length(signs) == n || throw(DimensionMismatch("ldlt: length(signs) = $(length(signs)), need n = $n"))
        @inbounds for k in 1:n
            sg = signs[Int(sym.perm[k])]
            (sg == 1 || sg == -1 || sg == 0) || throw(ArgumentError("ldlt: signs entries must be -1, 0, or +1 (got $sg)"))
            psigns[k] = Int8(sg)      # signs given in ORIGINAL order; store permuted (factor order)
        end
    end
    xsize = Int(sym.px[sym.nsuper + 1]) - 1
    x = Vector{T}(undef, xsize)
    panels = _build_panels(x, sym)
    F = LDLFactor{T,Ti}(
        sym, x, panels, Vector{T}(undef, n), psigns,
        Workspace{T,Ti}(sym), FactorStats(), true,
    )
    ldlt!(F, A)
    return F
end

"""
    ldlt(A::SparseMatrixCSC; signs=nothing, n_pos=nothing, n_neg=nothing,
         ordering=AMDOrdering()) -> LDLFactor

One-shot symbolic analysis + numeric LDLᵀ factorization (design.md §6). Either pass
`signs` (expected pivot signs in `A`'s ORIGINAL column order — see the `Symbolic` method
above), or, for a block-structured KKT matrix `[H Aᵀ; A -C]` stored with the `H` block
leading, the convenience pair `n_pos`/`n_neg`: it builds `signs` as `n_pos` leading `+1`s
followed by `n_neg` trailing `-1`s **in original column order** (which composes cleanly
with any fill-reducing ordering, because the permutation is applied to `signs`
internally, never by the caller). Passing both `signs` and `n_pos`/`n_neg` is an error.
"""
function ldlt(
        A::SparseMatrixCSC{T};
        signs::Union{Nothing,AbstractVector{<:Integer}} = nothing,
        n_pos::Union{Nothing,Int} = nothing, n_neg::Union{Nothing,Int} = nothing,
        ordering::AbstractOrdering = AMDOrdering(),
) where {T}
    if !isnothing(n_pos) || !isnothing(n_neg)
        isnothing(signs) || throw(ArgumentError("ldlt: pass either signs or n_pos/n_neg, not both"))
        np = something(n_pos, 0)
        nn = something(n_neg, 0)
        (np >= 0 && nn >= 0 && np + nn == size(A, 1)) ||
            throw(ArgumentError("ldlt: n_pos + n_neg = $(np + nn) must equal n = $(size(A, 1))"))
        signs = vcat(fill(Int8(1), np), fill(Int8(-1), nn))
    end
    sym = symbolic(A; ordering)
    return ldlt(sym, A; signs)
end

"""
    ldlt!(F::LDLFactor, A::SparseMatrixCSC) -> F

Refactor `A` (same sparsity pattern as `F.sym`, new numeric values) into `F` in place,
reusing `F.signs`. Never throws on SQD input: signed regularization (design.md §5.1)
forces every pivot to the expected sign and above the magnitude floor
`δ = ldlt_delta · max|A|`, so `F.ok` stays `true`; `F.stats` records the observed
pre-perturbation inertia (`n_pos`/`n_neg`/`n_zero`) and how much forcing occurred
(`n_perturbed`/`max_perturbation`).
"""
function ldlt!(F::LDLFactor{T,Ti}, A::SparseMatrixCSC{T}) where {T,Ti<:Integer}
    sym = F.sym
    nsuper = sym.nsuper
    super = sym.super
    rowind_ptr = sym.rowind_ptr
    rowind = sym.rowind
    snode_of = sym.snode_of
    amap = sym.amap
    x = F.x
    dvec = F.d
    signs = F.signs
    stats = F.stats
    ws = F.ws
    relmap = ws.relmap
    head = ws.head
    next = ws.next
    dptr = ws.dptr
    cbuf = ws.c
    cdbuf = ws.cd
    ir = ws.ir
    rs = ws.rs

    # ---- 1. assembly (same amap replay as cholesky!, design §4.2) + ‖A‖ scale ----
    # δ's ‖A‖-scale (design §5.1) is max|assembled entry| — our own choice of scale
    # (any norm-equivalent works; max-abs is O(nnz), free inside the load loop).
    fill!(x, zero(T))
    ascale = zero(T)
    @inbounds for p in eachindex(A.nzval)
        m = Int(amap[p])
        m == 0 && continue
        v = A.nzval[p]
        x[m] = v
        a = abs(v)
        a > ascale && (ascale = a)
    end
    delta = T(LDLT_DELTA) * (iszero(ascale) ? one(T) : ascale)
    # Zero-pivot classification threshold ζ (design §5.1 step 1 leaves ζ free): machine
    # epsilon relative to the running max |d| — the standard "numerically zero at this
    # scale" cut; our own choice, no external provenance.
    zeta = eps(real(T))
    @inbounds fill!(head, zero(Ti))

    stats.n_pos = 0
    stats.n_neg = 0
    stats.n_zero = 0
    stats.n_perturbed = 0
    stats.max_perturbation = 0.0
    dmax = zero(T)
    dmin = T(Inf)
    F.ok = true

    GC.@preserve x cbuf cdbuf begin
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

        # ---- 2. apply pending descendant updates: C = -L_d[R,:]·D_d·L_d[R1,:]ᵀ ----
        d = head[s]
        while d != zero(Ti)
            dInt = Int(d)
            dnext_link = next[dInt]
            q = Int(dptr[dInt])
            drp0 = Int(rowind_ptr[dInt])
            nsrow_d = Int(rowind_ptr[dInt + 1]) - drp0
            ncol_d = Int(super[dInt + 1]) - Int(super[dInt])
            dj0 = Int(super[dInt])
            panel_d = F.panels[dInt]

            k1 = 0
            while q + k1 <= nsrow_d && _row(rowind, drp0, q + k1) <= j1
                k1 += 1
            end
            k2 = nsrow_d - (q + k1 - 1)

            if k1 > 0
                ctot = k1 + k2
                # Contiguity fast path (same as llt.jl's): when the descendant's rows
                # occupy a contiguous run of the ancestor's row list (identity-shift
                # scatter), the chunked gemm accumulates straight into the panel
                # (beta = 1 from the first chunk) and the staged scatter disappears.
                # The gemm then also writes the never-stored/never-read strict-upper of
                # the k1×k1 diagonal-block part — harmlessly, and with its symmetric
                # mirror values (C's top block is L·D·Lᵀ-symmetric), not garbage; that
                # region was already junk-written by ger!'s trailing rectangle below.
                # Both view branches are the same SubArray type, so `C` stays concrete.
                # Check loop doubles as the scatter's index hoist into ws.ir plus the
                # run-structure build into ws.rs (identical for every column b of the
                # update block; contig ⟺ a single run — see llt.jl's comment).
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
                C = contig ? view(panel, lr0:(lr0 + ctot - 1), lr0:(lr0 + k1 - 1)) :
                    view(cbuf, 1:ctot, 1:k1)   # zero-alloc: view of a pre-existing Matrix (types.jl)
                # L·D scaled copy of the R1 rows, staged in cdbuf (design §5.1). The
                # natural staging shape (k1, ncol_d) is NOT bounded by max_extend_rows
                # on the column axis (a wide descendant with a short update block), so
                # the gemm is chunked over descendant columns with width capped at
                # cdbuf's own column capacity (= max_extend_rows): each chunk's
                # view(cdbuf, 1:k1, 1:wk) is then in-bounds by construction (k1 ≤
                # max_extend_rows by the same containment as `c`; wk ≤ size(cdbuf, 2)
                # by the cap — full derivation in types.jl's Workspace docstring) and,
                # being a view of a pre-existing Matrix, costs zero allocation — the
                # old flat max_update_size-sized cdbuf needed a fresh `_panelview`
                # unsafe_wrap per chunk, ldlt!'s last per-call allocation source. In
                # the common case (ncol_d ≤ max_extend_rows) one chunk covers all of
                # ncol_d; chunking over the contraction axis only adds gemm calls
                # (beta = 1 accumulation from the second chunk on), never extra flops.
                w = min(ncol_d, size(cdbuf, 2))
                c0 = 1
                while c0 <= ncol_d
                    wk = min(w, ncol_d - c0 + 1)
                    CD = view(cdbuf, 1:k1, 1:wk)
                    for kk in 1:wk
                        dv = dvec[dj0 + c0 + kk - 2]
                        for a in 1:k1
                            CD[a, kk] = panel_d[q + a - 1, c0 + kk - 1] * dv
                        end
                    end
                    Lblk = view(panel_d, q:(q + ctot - 1), c0:(c0 + wk - 1))
                    gemm!(C, Lblk, CD; transA = 'N', transB = 'T', alpha = -one(T),
                        beta = ((contig || c0 > 1) ? one(T) : zero(T)))
                    c0 += wk
                end
                if !contig
                    # Scatter-add: full C for the below-block rows, lower triangle only
                    # inside the diagonal block (its strict-upper is never stored/read,
                    # same convention as llt.jl's syrk scatter). Shared llt.jl helper —
                    # identical scatter semantics; in this branch C IS
                    # view(cbuf, 1:ctot, 1:k1), so the helper's direct cbuf[a, b]
                    # indexing addresses the same elements without the view wrapper.
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

        # ---- 3+4. base-case unit-LDLᵀ column loop over the FULL panel height ----
        # Right-looking within the block (Golub & Van Loan); the rank-1 ger! covers
        # rows j+1:nsrow, i.e. the diagonal block AND the below-diagonal panel — the
        # LDLᵀ replacement for llt.jl's potrf! + trsm! pair.
        for j in 1:nscol
            jg = j0 + j - 1
            dj = panel[j, j]
            adj = abs(dj)

            # inertia accounting, PRE-perturbation (design §5.1 step 1)
            if adj <= zeta * dmax
                stats.n_zero += 1
            elseif dj > zero(T)
                stats.n_pos += 1
            else
                stats.n_neg += 1
            end

            # signed regularization (design §5.1 step 2)
            sg = signs[jg]
            wrongsign = (sg == Int8(1) && !(dj > zero(T))) ||
                (sg == Int8(-1) && !(dj < zero(T)))
            if wrongsign || adj < delta
                target = sg == Int8(0) ? (signbit(dj) ? -one(T) : one(T)) : T(sg)
                newd = target * max(delta, adj)
                stats.n_perturbed += 1
                pert = Float64(abs(newd - dj))
                pert > stats.max_perturbation && (stats.max_perturbation = pert)
                dj = newd
            end

            dvec[jg] = dj
            adf = abs(dj)
            adf > dmax && (dmax = adf)
            adf < dmin && (dmin = adf)

            panel[j, j] = one(T)
            invd = inv(dj)
            for i in (j + 1):nsrow
                panel[i, j] *= invd
            end
            if j < nscol
                lcol = view(panel, (j + 1):nsrow, j)          # L[j+1:end, j], full height
                lrow = view(panel, (j + 1):nscol, j)          # L entries inside the diag block
                trail = view(panel, (j + 1):nsrow, (j + 1):nscol)
                ger!(-dj, lcol, lrow, trail)                   # A ← A - d_j·l·l_blkᵀ
            end
        end

        # ---- 5. schedule s onto its first ancestor (identical to llt.jl) ----
        if nsrow > nscol
            dptr[s] = Ti(nscol + 1)
            s2 = Int(snode_of[_row(rowind, rp0, nscol + 1)])
            next[s] = head[s2]
            head[s2] = Ti(s)
        end
    end
    end # GC.@preserve

    stats.nnzL = Int(sym.nnzL)
    stats.flops = sym.flops
    stats.rcond_est = dmax > zero(T) ? Float64(dmin / dmax) : Inf
    F.ok = true    # regularization guarantees completion (design §5.1)
    return F
end
