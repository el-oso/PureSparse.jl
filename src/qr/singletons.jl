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
    _restrict_ordering(ordering, collive, surv_cols) -> AbstractOrdering

Adapt `ordering` (chosen against the FULL, pre-peel `A`) so it can be handed to
`A22`'s own `order_columns` call, which only ever sees `A22`'s `n - n1` surviving
columns. `AMDOrdering`/`COLAMDOrdering`/`NaturalOrdering` need no adaptation — their
`order_columns` methods recompute a fresh permutation from whatever pattern they're
given, so they already produce a correctly-sized result for `A22` unchanged. Only
`GivenOrdering` carries a FIXED, externally-sized permutation vector (task #50: this
is what forced the M5 gate's same-permutation arm to disable singleton peeling
entirely, `singletons=false`, rather than crash on `order_columns`'s length check) —
for that case, restrict `ordering.perm` to just the entries naming a surviving
column (dropping peeled-column entries), preserving relative order, then relabel each
from its ORIGINAL column index to its LOCAL index within `A22` (its position in
`surv_cols`). This is the natural generalization of "same permutation": the peeled
columns are a PureSparse-specific optimization layer the external ordering has no
opinion on (SPQR applies its own equivalent structural-singleton trick internally
regardless of arm — ROADMAP.md's own account), so honoring the GIVEN ordering's
relative column order among the columns it actually still has a say over is what
"same ordering, modulo our own preprocessing" means here.
"""
_restrict_ordering(ordering::AbstractOrdering, ::BitVector, ::Vector) = ordering
function _restrict_ordering(ordering::GivenOrdering{Ti}, collive::BitVector, surv_cols::Vector{Ti}) where {Ti<:Integer}
    n = length(collive)
    local_of = Vector{Ti}(undef, n)
    @inbounds for (k, j) in enumerate(surv_cols)
        local_of[j] = Ti(k)
    end
    restricted = Vector{Ti}(undef, length(surv_cols))
    c = 0
    @inbounds for j in ordering.perm
        collive[j] || continue
        c += 1
        restricted[c] = local_of[j]
    end
    return GivenOrdering(restricted)
end

"""
    _insort_row!(colind, val, c0, deg)

In-place insertion sort of `colind[c0+1:c0+deg]` (and `val` in lockstep) by ascending
`colind`. A standalone function so its `while`-inside-`for` compiles as its own small
LLVM unit — see the call site's comment in [`_qr_compose_singletons`](@ref).
"""
function _insort_row!(colind::Vector{Ti}, val::Vector{T}, c0::Ti, deg::Ti) where {T,Ti<:Integer}
    @inbounds for i in Ti(2):deg
        cj = colind[c0 + i - one(Ti)]
        vj = val[c0 + i - one(Ti)]
        jj = i - one(Ti)
        while jj >= one(Ti) && colind[c0 + jj - one(Ti)] > cj
            colind[c0 + jj] = colind[c0 + jj - one(Ti)]
            val[c0 + jj] = val[c0 + jj - one(Ti)]
            jj -= one(Ti)
        end
        colind[c0 + jj] = cj
        val[c0 + jj] = vj
    end
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
    # A22 = A[surv_rows, surv_cols], built directly (not via getindex) so we ALSO get
    # a22map: each A22 nzval slot's position in A's own nzval. The map is pattern-
    # invariant (surv_rows/surv_cols are structural, §2.3 warm-refactor update), so
    # `qr!` on the composed factor can refresh A22's values from a new A2 with a plain
    # zero-alloc gather into this same buffer. Rows stay sorted per column because A's
    # are and the live-row relabeling is monotone.
    row_local = Vector{Ti}(undef, m)
    li = zero(Ti)
    @inbounds for i in 1:m
        if rowlive[i]
            li += one(Ti)
            row_local[i] = li
        end
    end
    colptr22 = Vector{Ti}(undef, length(surv_cols) + 1)
    colptr22[1] = one(Ti)
    nnz22 = 0
    @inbounds for (jj, j) in enumerate(surv_cols)
        for p in A.colptr[j]:(A.colptr[j + 1] - 1)
            rowlive[A.rowval[p]] && (nnz22 += 1)
        end
        colptr22[jj + 1] = Ti(nnz22 + 1)
    end
    rowval22 = Vector{Ti}(undef, nnz22)
    nzval22 = Vector{T}(undef, nnz22)
    a22map = Vector{Ti}(undef, nnz22)
    k22 = 0
    @inbounds for j in surv_cols
        for p in A.colptr[j]:(A.colptr[j + 1] - 1)
            i = A.rowval[p]
            rowlive[i] || continue
            k22 += 1
            rowval22[k22] = row_local[i]
            nzval22[k22] = A.nzval[p]
            a22map[k22] = Ti(p)
        end
    end
    A22 = SparseMatrixCSC(length(surv_rows), length(surv_cols), colptr22, rowval22, nzval22)

    ordering22 = _restrict_ordering(ordering, collive, surv_cols)
    F22 = _qr_block(A22; ordering = ordering22, tol)
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
    r1srcpos = Vector{Ti}(undef, Int(r1ptr[n1 + 1] - 1))
    r1val = Vector{T}(undef, Int(r1ptr[n1 + 1] - 1))
    # Write each row's (final-column, source-position) pairs directly into their final
    # r1colind/r1srcpos slice (already exactly sized via r1ptr above), then insertion-sort
    # that slice in place by column — task #51: the ORIGINAL version allocated a fresh
    # `Tuple{Ti,T}[]` (grown via `push!`) per peeled row and `sort!`ed it, which on
    # lp_slack-shaped matrices (n1 in the hundreds, one alloc+sort per row) dominated
    # this function's own cold-call time (measured: ~2500 allocations total for
    # lp_slack_n800x150's 800 peeled rows, ~59% of the full `qr()` call's wall time).
    # Insertion sort (not `sort!`) is deliberate: each row's degree is small (LP-slack's
    # own shape — a handful of structural-column entries per constraint row, §2.3's
    # motivating class), where O(deg²) comparisons cost far less than one heap
    # allocation; correctness is unaffected either way (both produce the same sorted
    # column-ascending order this array's own contract requires, verified against the
    # original `sort!`-based version on the gate matrix set before landing).
    # `_insort_row!` is its OWN function (a function-barrier split, not inlined here)
    # because fusing this insertion sort's `while`-inside-`for` into `_qr_compose_
    # singletons`'s own triple-nested loop caused a pathological LLVM compile hang
    # (LoopStrengthReduce/SCEV, minutes+, confirmed via a hung stack trace pointing at
    # `jl_parallel_gc_threadfun`/`SCEVExpander` during THIS function's first
    # compilation) — a known LLVM class for deeply-nested generic loops, not a runtime
    # bug; isolating it as a small standalone function gives LLVM a much smaller unit
    # to analyze and resolved it.
    # Sort (colind, SOURCE POSITION) in lockstep, then fill r1val through r1srcpos —
    # same result as sorting (colind, value) directly (final columns within a row are
    # distinct, so the order is unique), but keeps r1srcpos as the persistent
    # slot→A.nzval map the §2.3 warm refactor's zero-alloc re-harvest needs.
    @inbounds for k in 1:n1
        r = peel_row[k]
        c0 = r1ptr[k]
        deg = zero(Ti)
        for p in rowptr[r]:(rowptr[r + 1] - 1)
            r1colind[c0 + deg] = ciperm[rowidx[p]]
            r1srcpos[c0 + deg] = rowpos[p]
            deg += one(Ti)
        end
        _insort_row!(r1colind, r1srcpos, c0, deg)
    end
    @inbounds for p in eachindex(r1val)
        r1val[p] = A.nzval[r1srcpos[p]]
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

    # F22.ws's rblk/n1a/n1b scratch (task 10, solve!/solve_minnorm! zero-alloc) is
    # sized from F22's OWN symbolic, which by construction always has n1==0 (it never
    # peeled anything itself) — the composed `sym` above is the first point n1 is
    # actually known, so this is also the first point those two buffers can be
    # correctly sized. F22.ws's OTHER fields must be reused, not rebuilt: `rcursor` in
    # particular holds REAL populated state from A22's own numeric factorization (each
    # row's actual live-entry end position) that solve_R!/solve_Rt! depend on —
    # rebuilding a fresh QRWorkspace from `sym` would silently replace it with
    # uninitialized garbage (confirmed: caused a real segfault via an out-of-bounds
    # `ws.rcursor[k]`-driven loop bound in solve_R!, caught by testing this task's
    # zero-alloc changes before committing).
    ws = QRWorkspace{T,Ti}(
        F22.ws.x, F22.ws.stamp, F22.ws.tsub, F22.ws.pack, F22.ws.rcursor,
        Vector{T}(undef, max(nb, 1)),
        Vector{T}(undef, max(n1, 1)),
        Vector{T}(undef, max(n1, 1)),
    )

    return QRFactor{T,Ti}(
        sym, F22.rcolind, F22.rval, F22.vval, F22.beta,
        r1ptr, r1colind, r1val,
        # §2.3 warm-refactor state: bsym is F22's OWN symbolic (block-local
        # cperm/riperm over A22, sharing every other field with the composed `sym`
        # above by reference); a22buf is the very A22 the cold block factorization
        # just consumed — its colptr/rowval are fixed, its nzval becomes the warm
        # refresh target.
        F22.sym, A22, a22map, r1srcpos,
        ws, stats, F22.ok,
    )
end
