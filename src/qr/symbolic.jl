# Sparse QR symbolic analysis, design_qr.md §3 (M5a tasks 4-6: star pattern builder,
# V/R row structure, and the full driver). Drives the EXISTING etree.jl/counts.jl
# functions unchanged — no reimplementation (design_qr.md §1.5's module-layout note).

"""
    row_leftcol(m, n, colptr, rowval, ciperm) -> (rowptr, rowidx, leftcol)

The row-form of `A` (`rowptr`/`rowidx` = `csc_transpose`'s output, reused by
[`star_pattern`](@ref) so the row-form is only built once) plus, for every row `r`,
`leftcol[r]` = the PERMUTED column of `r`'s leftmost (permuted) nonzero, or `0` if row
`r` is entirely null. This is the same quantity both [`star_pattern`](@ref) (§3.2,
where it is called the star-center) and the V/R row-structure builder (§3.4, where it
is called `leftcol(r)`) need — computed once here to avoid duplicating (and risking
divergent copies of) the min-finding loop.
"""
function row_leftcol(m::Int, n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}, ciperm::Vector{Ti}) where {Ti<:Integer}
    rowptr, rowidx = csc_transpose(m, n, colptr, rowval)
    leftcol = zeros(Ti, m)
    @inbounds for i in 1:m
        lo, hi = rowptr[i], rowptr[i + 1] - 1
        lo > hi && continue                      # null row: no leftmost column
        kmin = typemax(Ti)
        for p in lo:hi
            j = ciperm[rowidx[p]]
            j < kmin && (kmin = j)
        end
        leftcol[i] = kmin
    end
    return rowptr, rowidx, leftcol
end

"""
    star_pattern(m, n, colptr, rowval, ciperm) -> (colptr2, rowval2)

Build the star matrix `S` (Gilbert–Li–Ng–Peyton 2001, via the survey's §7.1
presentation — the primary paper is unavailable, design_qr.md §3.2) for `A` (`m×n`,
CSC) under the column permutation `ciperm` (`ciperm[jorig]` = permuted position of
original column `jorig`, the `symmetrized_upper`/`full_symmetric_pattern` convention).
Column `k` of `S` (in PERMUTED column space) is the union, over every row `i` of `A`
whose leftmost nonzero in permuted column order is column `k`, of that row's *other*
permuted-column entries — a "star" centered at `k`. `|S| ≤ |A|`.

Returned as a ONE-triangle pattern (only column `k`'s own star edges are stored, not
yet mirrored into row/column `j`'s list) — feed the result through
[`symmetrized_upper`](@ref) with an identity permutation to symmetrize before handing
to [`etree`](@ref)/[`column_counts`](@ref) (design_qr.md §3.2: "feed the strict-upper
part through the existing `symmetrized_upper`-shaped entry points unchanged").

**Correctness (H1, design_qr.md §3.2):** `G(S)` and `G(AᵀA)` have the same FILLED
graph (proved via the fill-path theorem, Rose–Tarjan), so `etree(S)` = the column
elimination tree of `A` and `column_counts(S)` = `rcount` (row sizes of `R`) —
without ever forming `AᵀA`.
"""
function star_pattern(m::Int, n::Int, colptr::Vector{Ti}, rowval::Vector{Ti}, ciperm::Vector{Ti}) where {Ti<:Integer}
    rowptr, rowidx, leftcol = row_leftcol(m, n, colptr, rowval, ciperm)

    # Bucket rows by their star-center (= leftcol[i]) via a head/next intrusive linked
    # list (the same idiom `Workspace.head`/`next` uses for descendant lists, design.md
    # §4.3) — REQUIRED, not a style choice: a single marker array tagged by the star
    # center is only a valid dedup check while all rows sharing that center are
    # processed contiguously. Rows are naturally encountered in ROW order (1:m), and
    # unrelated rows sharing a DIFFERENT center can touch the same target column in
    # between — that would silently overwrite `mark[j]`'s tag before an earlier
    # center's own dedup check runs again, producing duplicate entries. Grouping by
    # center first (this bucketing pass) and then processing star-column k's entire row
    # bucket before moving to k+1 is what keeps the "one shared marker array, no
    # re-zeroing" idiom (`ata_pattern`'s own style) sound here — `ata_pattern` avoids
    # the hazard for a different reason (its outer loop already IS the target column),
    # which does not carry over to a row-outer-loop construction.
    head = zeros(Ti, n)
    next = zeros(Ti, m)
    @inbounds for i in 1:m
        kmin = leftcol[i]
        kmin == 0 && continue                     # empty row: no star membership
        next[i] = head[kmin]
        head[kmin] = Ti(i)
    end

    mark = zeros(Ti, n)
    # Pass 1: count nnz per star-column, one kmin bucket at a time.
    deg = zeros(Ti, n)
    @inbounds for k in 1:n
        i = head[k]
        while i != 0
            lo, hi = rowptr[i], rowptr[i + 1] - 1
            for p in lo:hi
                j = ciperm[rowidx[p]]
                (j == k || mark[j] == k) && continue
                mark[j] = k
                deg[k] += one(Ti)
            end
            i = next[i]
        end
    end
    colptr2 = Vector{Ti}(undef, n + 1)
    colptr2[1] = one(Ti)
    @inbounds for k in 1:n
        colptr2[k + 1] = colptr2[k] + deg[k]
    end
    # Pass 2: scatter (same tag idiom and bucket order; reset once, safe for the same
    # reason as pass 1 — each k's bucket is exhausted before k+1 starts).
    rowval2 = Vector{Ti}(undef, colptr2[n + 1] - 1)
    cursor = copy(colptr2)
    fill!(mark, zero(Ti))
    @inbounds for k in 1:n
        i = head[k]
        while i != 0
            lo, hi = rowptr[i], rowptr[i + 1] - 1
            for p in lo:hi
                j = ciperm[rowidx[p]]
                (j == k || mark[j] == k) && continue
                mark[j] = k
                rowval2[cursor[k]] = j
                cursor[k] += one(Ti)
            end
            i = next[i]
        end
    end
    @inbounds for k in 1:n
        sort!(view(rowval2, colptr2[k]:(colptr2[k + 1] - 1)))
    end
    return colptr2, rowval2
end

"""
    qr_row_structure(m, n, parent, leftcol) -> (rperm, riperm, mb, vptr, vrowind, pivotslot, vcount)

The physical row permutation, `V` column structure, and pivot-row assignment
(design_qr.md §3.4, hotspot H2, v2 with the B1/B2 fixes). `parent` is the POSTORDERED
column elimination tree (children always precede their parent, i.e. `parent[k] > k`
or `parent[k] == 0`, the same convention `etree`/`postorder` produce elsewhere in this
package); `leftcol` is [`row_leftcol`](@ref)'s per-row output (`0` for a null row).

- **Row assignment.** `a[k]` = number of rows with `leftcol[r] == k`.
- **Physical row numbering (`rperm`/`riperm`), decided independently of pivot
  selection (B2 fix).** Assigned rows get physical numbers `1..mb` grouped by
  ascending `leftcol` (a contiguous block per column, exactly like a CSC `colptr`
  over `a`) then ascending original row index within each block — a simple, already-
  sorted-by-construction canonical order, no separate sort needed. Null rows fill the
  remaining slots `mb+1..m`, by original index. `mb = Σ a[k]` (B2: at most `mb`
  columns can ever be live — every live column retires exactly one DISTINCT physical
  row, and there are only `mb` of them; the `m < n` case falls out with no
  special-casing).
- **`vcount`** (B1 fix): `vcount[k] = a[k] + Σ_{c: parent[c]==k} max(vcount[c]-1, 0)`,
  computed in a single ASCENDING pass over the postordered `parent` (every child's
  contribution to its parent is folded in before that parent's own index is reached,
  since postordering guarantees `parent[k] > k`) — the `max(·,0)` clamp is required:
  a structurally dead child (`vcount[c]==0`, empty `S_c`) retires no pivot and must
  contribute 0, not `vcount[c]-1 = -1` (design_qr.md §3.4's worked examples).
- **`S_k` materialization and `pivotslot`.** For each column `k` (ascending), gather
  the assigned physical-number block (already sorted) plus every child's non-pivot
  survivor rows (each child's own list is complete and sorted by the time `k` is
  reached, by the same postorder-ascending argument as `vcount`), and take the
  smallest physical number as `pivotslot[k]` — matching the design's deterministic
  tie-break ("smallest physical row number"). The remaining rows are passed to
  `parent[k]`. A structurally dead column (`vcount[k]==0`) leaves
  `pivotslot[k] == 0` (sentinel) and contributes nothing to its parent, matching
  `vcount`'s own clamp.

**Implementation note (2026-07-16 rewrite):** each `S_k` is written DIRECTLY into its
final `vrowind[vptr[k]:vptr[k+1]-1]` segment with no sort and no per-column
accumulator — the earlier per-column `Vector` + `sort!` version was measured (gate
matrix `banded_ls_n1500x500_bw15`, `Profile.Allocs`) to be the dominant cost of the
whole symbolic analysis (~2350 allocations / ~8 MB churned per call from `push!`/
`append!` growth), exactly the "revisit if the gate shows symbolic cost matters"
condition the original note deferred on. No sort is needed, by two invariants this
function itself establishes (own derivation, no external reference):

1. Physical row numbers are grouped by ascending `leftcol` block (`aptr` above), so
   for columns `j1 < j2`, every physical number in block `j1` precedes every one in
   block `j2`.
2. `parent` is postordered, so each child subtree is a contiguous column interval
   `[first_descendant(c), c]`, distinct children's intervals are disjoint and
   ascending in `c`, and all lie strictly below `k`. By induction, every row in
   child `c`'s survivor list has `leftcol` inside `subtree(c)` (assigned there, or
   passed up from `c`'s own subtree), hence its physical number lies in
   `subtree(c)`'s block range.

Therefore "child survivors in ascending child order, then column `k`'s own assigned
block (itself consecutive by construction)" is already globally sorted, the segment's
first element is the minimum (= `pivotslot[k]`), and each child's survivor sublist
`vrowind[vptr[c]+1 : vptr[c]+vcount[c]-1]` is final (postorder: `c < k`) and sorted
by the same induction. The `vcount` recurrence guarantees the pieces fill the segment
exactly.
"""
function qr_row_structure(m::Int, n::Int, parent::Vector{Ti}, leftcol::Vector{Ti};
        build_v::Bool = true) where {Ti<:Integer}
    a = zeros(Ti, n)
    @inbounds for r in 1:m
        k = leftcol[r]
        k != 0 && (a[k] += one(Ti))
    end
    aptr = Vector{Ti}(undef, n + 1)
    aptr[1] = one(Ti)
    @inbounds for k in 1:n
        aptr[k + 1] = aptr[k] + a[k]
    end
    mb = Int(aptr[n + 1] - 1)

    rperm = Vector{Ti}(undef, m)
    riperm = Vector{Ti}(undef, m)
    cursor = copy(aptr)
    nullcursor = Ti(mb + 1)
    @inbounds for r in 1:m
        k = leftcol[r]
        if k == 0
            rperm[nullcursor] = Ti(r)
            riperm[r] = nullcursor
            nullcursor += one(Ti)
        else
            p = cursor[k]
            rperm[p] = Ti(r)
            riperm[r] = p
            cursor[k] += one(Ti)
        end
    end

    vcount = copy(a)
    @inbounds for k in 1:n
        p = parent[k]
        p != 0 && (vcount[p] += max(vcount[k] - one(Ti), zero(Ti)))
    end
    vptr = Vector{Ti}(undef, n + 1)
    vptr[1] = one(Ti)
    @inbounds for k in 1:n
        vptr[k + 1] = vptr[k] + vcount[k]
    end
    nnzV = Int(vptr[n + 1] - 1)

    # `build_v=false`: the caller only needs `rperm`/`riperm`/`mb` (row-space physical
    # numbering) plus `vcount`/`vptr`/`nnzV` (cheap, O(n), feed `flops`/`max_vcol` in
    # `symbolic_qr` below) — NOT the actual V-pattern contents. This is the M5b
    # (`:frontal`) path's own case: confirmed by grep across src/qr/frontal*.jl that
    # NOTHING there ever reads `sym.vrowind`/`sym.pivotslot`/`sym.vptr` (the frontal
    # numeric loop builds its own front-local V storage via `symbolic_qr_frontal`'s
    # `fsym.nnzVF`, entirely independent of this). `vrowind` is the single largest
    # allocation in the whole cold `qr_frontal(A)` call on some matrices (measured:
    # 4.87 MiB / 45x the input's own nnz on `grid_ls_70x50`, `Profile.Allocs`) — pure
    # waste for a caller that never reads it. Skipping it removes both that
    # allocation AND the O(nnzV) child-merge loop that fills it.
    if !build_v
        return rperm, riperm, mb, vptr, Ti[], Ti[], vcount
    end

    # Child lists via the same head/next linked-list idiom `postorder` uses; head
    # insertion in descending k yields per-parent lists in ASCENDING child order —
    # exactly the order the no-sort merge argument above requires.
    chead = zeros(Ti, n)
    cnext = zeros(Ti, n)
    @inbounds for k in n:-1:1
        p = parent[k]
        if p != 0
            cnext[k] = chead[p]
            chead[p] = Ti(k)
        end
    end

    pivotslot = zeros(Ti, n)
    vrowind = Vector{Ti}(undef, nnzV)
    @inbounds for k in 1:n
        vcount[k] == 0 && continue               # dead column: pivotslot[k] stays 0
        idx = vptr[k]
        c = chead[k]
        while c != 0
            for q in (vptr[c] + one(Ti)):(vptr[c] + vcount[c] - one(Ti))
                vrowind[idx] = vrowind[q]        # child survivors, already final+sorted
                idx += one(Ti)
            end
            c = cnext[c]
        end
        for p in aptr[k]:(aptr[k + 1] - 1)       # own assigned block, consecutive
            vrowind[idx] = p
            idx += one(Ti)
        end
        pivotslot[k] = vrowind[vptr[k]]          # segment is sorted: first = smallest
    end
    return rperm, riperm, mb, vptr, vrowind, pivotslot, vcount
end

"""
    symbolic_qr(A::SparseMatrixCSC; ordering::AbstractOrdering) -> QRSymbolic

Full symbolic analysis pipeline for sparse QR (design_qr.md §2-§3): column ordering →
star pattern (§3.2, H1) → postorder → R structure (§3.3) → V/row structure (§3.4, H2).
No amalgamation priority is needed (M5a has no supernodes/fronts, design_qr.md §3.2),
so a single default `postorder` pass suffices — unlike [`symbolic`](@ref)'s two-pass
merge-aware version. Column-singleton pre-elimination (§2.3, `n1 > 0`) is not yet
implemented (M5a task 9); `n1` is always `0` here and `m == mb` whenever `A` has no
fully-null rows.

No default `ordering` yet — the design's stated default is `COLAMDOrdering()`
(§2.1), landing in a later task; callers must pass one explicitly for now (e.g.
`AMDOrdering()`, already implemented, §2.2.6).

`build_v=false` skips materializing `vrowind`/`pivotslot` (they come back as empty
`Ti[]` placeholders) — `nnzV`/`max_vcol`/`flops` stay fully accurate either way
(computed from `vptr`/`vcount`, not `vrowind`/`pivotslot` themselves). Only for
callers that provably never read `sym.vrowind`/`sym.pivotslot`/`sym.vptr` afterward
— currently just `qr_frontal`'s own internal `symbolic_qr` call (confirmed by grep:
nothing in `src/qr/frontal*.jl` touches those fields; the `:frontal` numeric loop
uses its own `symbolic_qr_frontal`-built `fsym.nnzVF`/front storage instead). Do NOT
default this to `false` — the `:column` path (`numeric.jl`, `singletons.jl`,
`QRWorkspace`) needs the real `vrowind`/`pivotslot` contents.
"""
function symbolic_qr(A::SparseMatrixCSC{T,Ti}; ordering::AbstractOrdering,
        build_v::Bool = true) where {T,Ti<:Integer}
    m, n = size(A)
    fcperm = order_columns(ordering, m, n, A.colptr, A.rowval)
    fciperm = Vector{Ti}(undef, n)
    @inbounds for (k, p) in enumerate(fcperm)
        fciperm[p] = Ti(k)
    end

    scolptr, srowval = star_pattern(m, n, A.colptr, A.rowval, fciperm)
    idp = collect(Ti, 1:n)
    ucolptr, urowval = symmetrized_upper(n, scolptr, srowval, idp, idp)
    parent0 = etree(n, ucolptr, urowval)
    post, postinv = postorder(n, parent0)
    sptr, sind = relabel_pattern(n, ucolptr, urowval, postinv)
    parent = etree(n, sptr, sind)
    rcount = column_counts(n, sptr, sind, parent)

    cperm = Vector{Ti}(undef, n)
    @inbounds for orig in 1:n
        cperm[postinv[fciperm[orig]]] = Ti(orig)
    end
    ciperm = Vector{Ti}(undef, n)
    @inbounds for (k, p) in enumerate(cperm)
        ciperm[p] = Ti(k)
    end

    _, _, leftcol = row_leftcol(m, n, A.colptr, A.rowval, ciperm)
    rperm0, riperm0, mb, vptr, vrowind, pivotslot, vcount =
        qr_row_structure(m, n, parent, leftcol; build_v)
    # QRSymbolic's rperm/riperm are FULL-space (length m) fields (design_qr.md §1.4);
    # qr_row_structure already returns exactly that shape when n1 == 0 (no singleton
    # rows to prepend) — direct assignment, no further translation needed yet (M5a
    # task 9 adds the singleton-block composition on top of this).
    rperm = rperm0
    riperm = riperm0

    rptr = Vector{Ti}(undef, n + 1)
    rptr[1] = one(Ti)
    @inbounds for k in 1:n
        rptr[k + 1] = rptr[k] + rcount[k]
    end
    nnzR = Int(rptr[n + 1] - 1)
    nnzV = Int(vptr[n + 1] - 1)
    max_rrow = n == 0 ? 0 : Int(maximum(rcount))
    max_vcol = n == 0 ? 0 : Int(maximum(vcount))
    flops = 0.0
    @inbounds for i in 1:n
        flops += 4.0 * vcount[i] * (rcount[i] - 1) + 3.0 * vcount[i]
    end

    return QRSymbolic{Ti}(
        m, n, 0, mb,
        cperm, ciperm, rperm, riperm,
        parent, sptr, sind,
        rcount, rptr, vptr, vrowind, pivotslot,
        max_rrow, max_vcol, nnzR, nnzV, flops,
    )
end
