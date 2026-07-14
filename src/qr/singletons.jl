# Column-singleton pre-elimination, design_qr.md §2.3 (M5a task 9). SPQR paper §2.1,
# reimplemented from the paper's description (never its source): a column singleton is
# a column with exactly one LIVE nonzero whose magnitude exceeds a threshold; permute
# it (and its row) to the front, delete both, repeat. Breadth-first peeling on a
# value-aware row-form of A, O(|A|) total.

"""
    _row_form_values(m, n, colptr, rowval, nzval) -> (rowptr, rowidx, rowpos)

Value-aware row-form of `A` (like [`csc_transpose`](@ref), but also returns `rowpos`:
for each entry, its ORIGINAL position in `colptr`/`rowval`/`nzval`, so callers can
fetch `nzval[rowpos[k]]` without a separate value copy). Column-ascending scatter
order keeps each row's entries sorted by column, same as `csc_transpose`.
"""
function _row_form_values(m::Int, n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}) where {Ti<:Integer}
    deg = zeros(Ti, m)
    @inbounds for p in eachindex(rowval)
        deg[rowval[p]] += one(Ti)
    end
    rowptr = Vector{Ti}(undef, m + 1)
    rowptr[1] = one(Ti)
    @inbounds for i in 1:m
        rowptr[i + 1] = rowptr[i] + deg[i]
    end
    cursor = copy(rowptr)
    rowidx = Vector{Ti}(undef, length(rowval))
    rowpos = Vector{Ti}(undef, length(rowval))
    @inbounds for j in 1:n, p in colptr[j]:(colptr[j + 1] - 1)
        i = rowval[p]
        rowidx[cursor[i]] = Ti(j)
        rowpos[cursor[i]] = Ti(p)
        cursor[i] += one(Ti)
    end
    return rowptr, rowidx, rowpos
end

"""
    peel_column_singletons(A::SparseMatrixCSC{T,Ti}, threshold::T) -> (peel_col, peel_row, collive, rowlive)

Breadth-first column-singleton peeling (design_qr.md §2.3): repeatedly find a LIVE
column with exactly one LIVE row entry whose magnitude exceeds `threshold`, peel both
(column and row) to the front, and repeat. `peel_col[k]`/`peel_row[k]` (original
indices) are the `k`-th peeled column/row — this directly gives `R11`/`R12`: row `k`'s
entries are exactly the ORIGINAL row `peel_row[k]`'s stored entries (no numerical work,
no fill — the "reflector" for a length-1 pivot vector is a pure sign/scale, §2.3).
`collive`/`rowlive` mark the surviving (`A22`) columns/rows. A magnitude test failure
(a structural singleton whose value is too small) leaves the column live for the main
symbolic/numeric pipeline to handle — not a bug, §2.3's "values, not just pattern"
policy point.
"""
function peel_column_singletons(A::SparseMatrixCSC{T,Ti}, threshold::T) where {T<:Real,Ti<:Integer}
    m, n = size(A)
    rowptr, rowidx, rowpos = _row_form_values(m, n, A.colptr, A.rowval)

    coldeg = Vector{Ti}(undef, n)
    @inbounds for j in 1:n
        coldeg[j] = A.colptr[j + 1] - A.colptr[j]
    end
    rowlive = trues(m)
    collive = trues(n)
    peel_col = Ti[]
    peel_row = Ti[]

    queue = Ti[j for j in 1:n if coldeg[j] == 1]
    qi = 1
    @inbounds while qi <= length(queue)
        j = queue[qi]
        qi += 1
        (!collive[j] || coldeg[j] != 1) && continue
        r = zero(Ti)
        rp = zero(Ti)
        for p in A.colptr[j]:(A.colptr[j + 1] - 1)
            i = A.rowval[p]
            if rowlive[i]
                r = i
                rp = Ti(p)
                break
            end
        end
        r == 0 && continue                     # bookkeeping says live but none found: skip defensively
        abs(A.nzval[rp]) <= threshold && continue   # §2.3: magnitude test failed, leave it live

        collive[j] = false
        rowlive[r] = false
        push!(peel_col, j)
        push!(peel_row, r)
        for p in rowptr[r]:(rowptr[r + 1] - 1)
            jj = rowidx[p]
            collive[jj] || continue
            coldeg[jj] -= one(Ti)
            coldeg[jj] == one(Ti) && push!(queue, jj)
        end
    end
    return peel_col, peel_row, collive, rowlive
end

"""
    _qr_compose_singletons(A, peel_col, peel_row, collive, rowlive, ordering, tol) -> QRFactor

Assemble a full `QRFactor` (`sym.n1 > 0`) from a peeled singleton block plus the
factorization of the surviving `A22` submatrix (design_qr.md §2.3).

**R11/R12 need no numerical work** (own derivation, verified algebraically): a
length-1 Householder reflector has two valid sign conventions (`H=+1` or `H=-1`,
either gives a valid `QR` pair for a scalar). Choosing `H=+1` (`Q=I`) makes `R11`/`R12`
literal copies of `A`'s own values at the peeled rows — no transformation needed — and
makes the OVERALL `Q` block-diagonal (identity ⊕ `Q_block`). Proof sketch this relies
on: a column can only be peeled when its one remaining live row is not shared by any
row that survives into `A22` (else that column would have had degree ≥ 2 at peel
time, contradicting readiness) — so the bottom-left block (`A22`'s rows, peeled
columns) is provably all-zero, and `A`(permuted) already equals `[R11 R12; 0 A22]`
exactly, with no elimination required. `apply_Q!`/`apply_Qt!` therefore need no
special-casing for the singleton rows at all — only `solve!`'s back-substitution does
(§6.2's "PREPENDED" singleton-block solve).
"""
function _qr_compose_singletons(
        A::SparseMatrixCSC{T,Ti}, peel_col::Vector{Ti}, peel_row::Vector{Ti},
        collive::BitVector, rowlive::BitVector,
        ordering::AbstractOrdering, tol::Union{Nothing,Real},
) where {T,Ti<:Integer}
    m, n = size(A)
    n1 = length(peel_col)

    surv_rows = Ti[i for i in 1:m if rowlive[i]]
    surv_cols = Ti[j for j in 1:n if collive[j]]
    A22 = A[surv_rows, surv_cols]

    F22 = _qr_block(A22; ordering, tol)
    sym22 = F22.sym
    nb = length(sym22.parent)

    cperm = Vector{Ti}(undef, n)
    @inbounds for k in 1:n1
        cperm[k] = peel_col[k]
    end
    @inbounds for k in 1:nb
        cperm[n1 + k] = surv_cols[sym22.cperm[k]]
    end
    ciperm = Vector{Ti}(undef, n)
    @inbounds for (k, p) in enumerate(cperm)
        ciperm[p] = Ti(k)
    end

    rperm = Vector{Ti}(undef, m)
    @inbounds for k in 1:n1
        rperm[k] = peel_row[k]
    end
    @inbounds for k in 1:(m - n1)
        rperm[n1 + k] = surv_rows[sym22.rperm[k]]
    end
    riperm = Vector{Ti}(undef, m)
    @inbounds for (k, p) in enumerate(rperm)
        riperm[p] = Ti(k)
    end

    sym = QRSymbolic{Ti}(
        m, n, n1, sym22.mb,
        cperm, ciperm, rperm, riperm,
        sym22.parent, sym22.sptr, sym22.sind,
        sym22.rcount, sym22.rptr, sym22.vptr, sym22.vrowind, sym22.pivotslot,
        sym22.max_rrow, sym22.max_vcol, sym22.nnzR, sym22.nnzV, sym22.flops,
    )

    # R11/R12: gather each peeled row's ORIGINAL entries (value-aware row-form of the
    # FULL A, not A22), mapped to FINAL column position via ciperm, sorted ascending
    # (upper-triangular: a row's entries only ever land at columns >= its own index,
    # by the no-numerical-work argument above).
    rowptr, rowidx, rowpos = _row_form_values(m, n, A.colptr, A.rowval)
    r1ptr = Vector{Ti}(undef, n1 + 1)
    r1ptr[1] = one(Ti)
    @inbounds for k in 1:n1
        r = peel_row[k]
        r1ptr[k + 1] = r1ptr[k] + (rowptr[r + 1] - rowptr[r])
    end
    r1colind = Vector{Ti}(undef, Int(r1ptr[n1 + 1] - 1))
    r1val = Vector{T}(undef, Int(r1ptr[n1 + 1] - 1))
    @inbounds for k in 1:n1
        r = peel_row[k]
        pairs = Tuple{Ti,T}[]
        for p in rowptr[r]:(rowptr[r + 1] - 1)
            push!(pairs, (ciperm[rowidx[p]], A.nzval[rowpos[p]]))
        end
        sort!(pairs; by = first)
        c = r1ptr[k]
        for (fc, v) in pairs
            r1colind[c] = fc
            r1val[c] = v
            c += one(Ti)
        end
    end

    # F22.stats only covers A22's own block (it has no knowledge n1 singleton columns
    # even exist) — every singleton is a genuine LIVE pivot by construction (it passed
    # the magnitude threshold before being peeled, §2.3), contributing to rank/nnzR
    # but never to n_dead/dropped_norm (no dropping ever happens in the singleton
    # block) and no flops (§2.3: "no numerical work, no fill").
    stats = QRStats(
        F22.stats.nnzR + length(r1colind),
        F22.stats.nnzV,
        F22.stats.flops,
        F22.stats.rank + n1,
        F22.stats.n_dead,
        F22.stats.dropped_norm,
    )

    return QRFactor{T,Ti}(
        sym, F22.rcolind, F22.rval, F22.vval, F22.beta,
        r1ptr, r1colind, r1val, F22.ws, stats, F22.ok,
    )
end
