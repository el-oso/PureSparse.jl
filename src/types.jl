# Core types, design.md §1.2. `Symbolic` is the pattern-only analysis, shared by
# reference across many numeric factorizations of the same sparsity pattern (the
# "analyze once, factorize many times" split — CLAUDE.md requirement 7).

"""
    Symbolic{Ti<:Integer}

Fill-reducing permutation, elimination tree, column counts, and supernode partition of a
sparse symmetric pattern. Computed once by [`symbolic`](@ref) and reused across many
numeric factorizations of matrices sharing that pattern. Never holds numeric values.
"""
struct Symbolic{Ti<:Integer}
    n::Int
    perm::Vector{Ti}              # fill-reducing permutation
    iperm::Vector{Ti}
    parent::Vector{Ti}            # elimination tree (postordered), per-column
    colcount::Vector{Ti}          # nnz per column of L

    # --- supernode partition ---
    nsuper::Int
    super::Vector{Ti}             # super[s] = first column of supernode s (length nsuper+1)
    rowind_ptr::Vector{Ti}        # CSC-style pointers into rowind, length nsuper+1
    rowind::Vector{Ti}            # concatenated row patterns, one block per supernode
    snode_of::Vector{Ti}          # column -> supernode, length n
    sparent::Vector{Ti}           # supernode elimination tree, length nsuper (0 = root)
    px::Vector{Ti}                # per-supernode offset into x, length nsuper+1 (design §4.1)

    # --- assembly ---
    amap::Vector{Ti}              # nnz(tril(permuted A)) -> destination offset in x (design §4.2)

    # --- workspace sizing, derived independently from the §4.3 schedule (B1: these are
    # NOT CHOLMOD's maxcsize/maxesize fields — renamed and re-derived per design.md §0 B1) ---
    max_update_size::Int          # max over (descendant,ancestor) pairs of |R|*|R1| (design §3.6)
    max_extend_rows::Int          # max over supernodes of below-diagonal row count (design §3.6)

    nnzL::Int
    flops::Float64                # sum(colcount[j]^2), the GFlops diagnostic denominator
end

"""
    FactorStats

Reported statistics from a numeric factorization: fill/flop counts, and (for LDLᵀ) the
observed pivot inertia and regularization record (design.md §5.1/§5.2). Public API — a
downstream IPOPT-style consumer reads `n_pos`/`n_neg`/`n_zero` to run its own
regularization loop on top of PureSparse (design.md §5.2).
"""
mutable struct FactorStats
    nnzL::Int
    flops::Float64
    n_pos::Int
    n_neg::Int
    n_zero::Int
    n_perturbed::Int
    max_perturbation::Float64
    rcond_est::Float64
    fail_col::Int                 # 0 = ok; else 1-based column of the first non-SPD pivot (LLᵀ)
end

FactorStats() = FactorStats(0, 0.0, 0, 0, 0, 0, 0.0, Inf, 0)

"""
    Workspace{T,Ti}

Preallocated scratch buffers reused across every `cholesky!`/`ldlt!`/`solve!` call on a
factor sharing one `Symbolic` — sized once from `Symbolic.max_update_size`/
`max_extend_rows` so the numeric phase never allocates (CLAUDE.md requirement 5).
"""
struct Workspace{T,Ti<:Integer}
    c::Matrix{T}                  # update block buffer, (max_extend_rows, max_extend_rows) —
                                   # see llt.jl's `_row` comment: for any (descendant d, ancestor
                                   # s) pair, ctot = |R| and k1 = |R1| are BOTH ≤ d's own
                                   # below-diagonal row count ≤ max_extend_rows (R1 ⊆ R ⊆ d's
                                   # below-diagonal rows, by construction of the update schedule,
                                   # design §4.3/§0 B1) — so `view(c, 1:ctot, 1:k1)` is always
                                   # in-bounds and, being a view of an already-allocated Matrix
                                   # (not a fresh `unsafe_wrap`/`reshape`), costs zero allocation
                                   # (measured: unsafe_wrap 80 B/call, reshape 48 B/call, view of
                                   # a pre-existing Matrix 0 B/call). Real matrices show this is a
                                   # modest 1.0–4.4x memory increase over the old flat
                                   # `max_update_size` sizing (measured on the M1 gate set), a
                                   # one-time Workspace cost, not a hot-path one.
    cd::Vector{T}                  # L*D scaled-copy buffer for LDLᵀ updates, length max_update_size —
                                   # STILL allocates per-call via `_panelview` (M1 task 7 follow-up,
                                   # not covered by this pass: its (k1, wk) chunk shape isn't bounded
                                   # by max_extend_rows the way `c`'s is — see ROADMAP.md)
    relmap::Vector{Ti}            # length n, no clearing needed between supernodes (design §4.3)
    head::Vector{Ti}              # length nsuper+1, intrusive linked-list heads (0 = empty)
    next::Vector{Ti}              # length nsuper, intrusive linked-list next pointers
    dptr::Vector{Ti}              # length nsuper, per-descendant progress cursor into rowind
    rhs::Vector{T}                # permuted-RHS scratch for solve!, length n — zero-alloc single-
                                   # vector solves (multi-RHS still allocates: nrhs is unbounded,
                                   # not knowable ahead of time)
end

function Workspace{T,Ti}(sym::Symbolic) where {T,Ti<:Integer}
    mer = max(sym.max_extend_rows, 1)
    Workspace{T,Ti}(
        Matrix{T}(undef, mer, mer),
        Vector{T}(undef, sym.max_update_size),
        Vector{Ti}(undef, sym.n),
        zeros(Ti, sym.nsuper + 1),
        zeros(Ti, sym.nsuper),
        Vector{Ti}(undef, sym.nsuper),
        Vector{T}(undef, sym.n),
    )
end

"""
    AbstractSparseFactor{T}

Supertype for every factor PureSparse produces (`SupernodalFactor`, `LDLFactor`,
`SimplicialLDLFactor`) — the common surface shared across them: `sym::Symbolic`,
`stats::FactorStats`, `ok::Bool` (queried via [`issuccess`](@ref)), and the generic
[`solve!`](@ref)/[`refine!`](@ref) entry points.
"""
abstract type AbstractSparseFactor{T} end

"""
    SupernodalFactor{T,Ti} <: AbstractSparseFactor{T}

Supernodal LLᵀ factor (SPD path, design.md §4). Dense column-major panels for every
supernode, stored contiguously in `x`. Produced by [`cholesky`](@ref)/refactored in place
by [`cholesky!`](@ref) (zero allocations after the first call).
"""
mutable struct SupernodalFactor{T,Ti<:Integer} <: AbstractSparseFactor{T}
    sym::Symbolic{Ti}              # sym.px gives per-supernode offsets into x (shared, not duplicated)
    x::Vector{T}                   # dense column-major storage, one block per supernode
    panels::Vector{Matrix{T}}      # panels[s] = unsafe_wrap(x @ px[s], nrow_s x ncol_s), built ONCE
                                    # (design §0 follow-up: caching these — fixed shape per supernode
                                    # across every cholesky! call on this factor, since px/rowind_ptr
                                    # come from the unchanging Symbolic — is what makes cholesky!
                                    # allocation-free; a fresh unsafe_wrap per call, while still far
                                    # cheaper than the reshape(view(...)) compile-tax trap it replaced,
                                    # still allocates a small Array header each time)
    ws::Workspace{T,Ti}
    stats::FactorStats
    ok::Bool
end

"""
    LDLFactor{T,Ti} <: AbstractSparseFactor{T}

Supernodal LDLᵀ factor for symmetric quasi-definite systems (design.md §5). Same panel
layout as [`SupernodalFactor`](@ref) (unit-lower L, explicit 1s on the diagonal of each
panel) plus a separate diagonal `d` and the caller's expected pivot `signs`.
"""
mutable struct LDLFactor{T,Ti<:Integer} <: AbstractSparseFactor{T}
    sym::Symbolic{Ti}              # sym.px gives per-supernode offsets into x (shared, not duplicated)
    x::Vector{T}
    panels::Vector{Matrix{T}}      # built-once panel wrappers, same caching as SupernodalFactor.panels
    d::Vector{T}                   # diagonal of D, permuted, length n
    signs::Vector{Int8}            # expected pivot signs (+1/-1/0=free), permuted, length n
    ws::Workspace{T,Ti}
    stats::FactorStats
    ok::Bool
end

"""
    SimplicialLDLFactor{T,Ti} <: AbstractSparseFactor{T}

Simplicial (one-column-at-a-time) unit-lower LDLᵀ factor — the representation the
Davis–Hager rank-1 update/downdate operates on (design.md §1.2/§7; Davis & Hager,
*Modifying a Sparse Cholesky Factorization*, SIAM J. Matrix Anal. Appl. 20(3), 1999).
Produced by [`simplicial`](@ref)`(F::LDLFactor)`; modified in place by
[`updowndate!`](@ref); solved by the same exported split solves as the supernodal
factors (design.md §6/§0 N7).

Storage layout (everything in FACTOR order, i.e. permuted by `sym.perm`):

- Column `j` of `L` owns the fixed slot range `colptr[j] : colptr[j+1]-1` in
  `rowval`/`nzval`, of which the first `colnnz[j]` slots hold its current
  **strictly-lower** pattern (sorted, ascending) and values; the diagonal of `L` is an
  implicit `1` and is never stored. The remaining slots are *slack*: `updowndate!`
  grows a column's pattern in place (paper §4.1 Case 1 / §6, Algorithm 6a — an update
  can add fill along the etree path) without any reallocation; a column whose growth
  would exceed its slack makes `updowndate!` return `:refactor_required` instead
  (design.md §7). Slack is sized by the `simplicial_grow` Preference (see `tuning.jl`).
- `d` is the diagonal of `D` (1×1 pivots, may be negative for SQD factors).
- `parent[j] = min(pattern of column j)` (`0` at a root) — the elimination tree **of
  the stored pattern** (Davis–Hager §2: `π(j) = min Lⱼ \\ {j}`). The stored pattern is
  the supernodal one including amalgamation padding, a closed superset of the true
  L-pattern (see [`simplicial`](@ref)'s provenance note), so this parent map can be a
  refinement of `sym.parent`; `updowndate!` keeps it consistent as patterns grow
  (paper Algorithm 3: `π̄(j) = min L̄ⱼ \\ {j}`).
- `wval`/`wpat` are preallocated `updowndate!` workspace (scattered `w` values and its
  sorted support). `wval` is kept all-zero between calls so no O(n) clearing is ever
  needed — `updowndate!` re-zeroes exactly the entries it touched (O(changed nnz)).
- `ok` is `false` after a failed `updowndate!` (`:refactor_required` or
  `:not_definite`) — the numeric contents are then partially modified and the factor
  must be rebuilt via a fresh `ldlt`/`simplicial`. `stats.fail_col` records the
  factor-order column at which the failure was detected.
"""
mutable struct SimplicialLDLFactor{T,Ti<:Integer} <: AbstractSparseFactor{T}
    sym::Symbolic{Ti}              # shared symbolic (perm/iperm/n); parent below is the factor's own
    colptr::Vector{Ti}             # column j owns slots colptr[j]:colptr[j+1]-1, length n+1
    colnnz::Vector{Ti}             # slots of column j currently in use, length n
    rowval::Vector{Ti}             # strictly-lower row indices, sorted per column
    nzval::Vector{T}               # matching L values (unit diagonal implicit)
    d::Vector{T}                   # diagonal of D, factor order, length n
    parent::Vector{Ti}             # etree of the STORED pattern: min of column pattern, 0 = root
    wval::Vector{T}                # updowndate! workspace: scattered w, all-zero between calls
    wpat::Vector{Ti}               # updowndate! workspace: sorted support of w, length n
    stats::FactorStats
    ok::Bool
end

"""
    issuccess(F::AbstractSparseFactor) -> Bool

Whether the last `cholesky!`/`ldlt!` call on `F` succeeded (design.md §4.3 step 3, §5.1 —
LDLᵀ factors always succeed via regularization; LLᵀ factors fail on a non-SPD pivot).
"""
issuccess(F::AbstractSparseFactor) = F.ok
