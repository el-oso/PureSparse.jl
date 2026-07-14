# Sparse QR core types, design_qr.md §1.4/§4.5/§6.5 (M5a task 1). Field names follow
# this package's own established conventions (`rowind`-style naming, `px`-style
# pointer arrays from `types.jl`); `beta` is the survey §7.3 pseudocode's own name for
# the Householder coefficients. None are copied from any SuiteSparse internal, which we
# have never seen (design_qr.md §11).

"""
    QRSymbolic{Ti<:Integer}

Column ordering, singleton block, column elimination tree, R/V structure, and
workspace-sizing scalars for a sparse QR factorization pattern. Computed once by
[`symbolic_qr`](@ref) and reused across every numeric refactorization sharing that
pattern (design_qr.md §1.4, mirroring [`Symbolic`](@ref)'s "analyze once" role).

**Index-space convention (design_qr.md D5):** two spaces exist. *Full* space has size
`n` for columns / `m` for rows (the original problem). *Block* space has size `n-n1`
for columns / `mb` for rows (the non-singleton block A22 that §3/§4 actually operate
on). `cperm`/`ciperm`/`rperm`/`riperm` translate between the two (singleton entries
first, block entries after); every other field below is block-local. An implementer
must never index `rcount`/`vptr`/etc. by a full-space column number without first
translating through `ciperm`.
"""
struct QRSymbolic{Ti<:Integer}
    m::Int                         # full row count
    n::Int                         # full column count
    # --- singleton block (§2.3); n1 == 0 when disabled or none found ---
    n1::Int                        # number of pre-eliminated column singletons
    mb::Int                        # block row count m′ — the size of the PHYSICAL
                                    #   permuted-row space every block-local structure
                                    #   below indexes into; can be LESS than n-n1 (the
                                    #   m < n-n1 case, §3.4 B2 fix)
    # --- permutations (FULL space, size n / m) ---
    cperm::Vector{Ti}              # column permutation (singletons first, then
    ciperm::Vector{Ti}             #   fill-reducing ∘ postorder on the rest), length n
    rperm::Vector{Ti}              # row permutation (singleton rows first, then the
    riperm::Vector{Ti}             #   block's own staircase permutation, §3.4), length m
    # --- column elimination tree of the block (postordered; BLOCK space, size n-n1) ---
    parent::Vector{Ti}             # length n-n1; 0 = root
    # --- star matrix S's strict-upper pattern, in the FINAL postordered column space
    # (M5a task 6 addition — not in the original design_qr.md §1.4 field list, added
    # once the numeric loop's §4.1 step 2 turned out to need it: "for each j in
    # pattern(S[:,k])" requires S's own pattern to seed the row-subtree ancestor climb,
    # and nothing else in QRSymbolic determines it — rcount/rptr size R's ROW
    # structure, an entirely different axis from "which prior columns does column k's
    # apply step read from"). Same CSC-of-strict-upper-triangle shape `symmetrized_upper`
    # already produces; column k's entries are exactly the seed set for T^k. ---
    sptr::Vector{Ti}                # length n-n1+1
    sind::Vector{Ti}                # length sptr[end]-1
    # --- factor structure (BLOCK space throughout) ---
    rcount::Vector{Ti}             # nnz of row k of R (= colcount of L(AᵀA)), length n-n1
    rptr::Vector{Ti}               # row-of-R pointers (CSC of Rᵀ), length n-n1+1
    vptr::Vector{Ti}               # V column pointers, length n-n1+1
    vrowind::Vector{Ti}            # V row patterns, physical (block-permuted, 1..mb)
                                    #   row numbers — §3.4; pivot row for column k is
                                    #   NOT assumed to be numbered k (B2) — see pivotslot
    pivotslot::Vector{Ti}          # B2 fix: pivotslot[k] = the physical row number
                                    #   (1..mb) that is column k's designated pivot row,
                                    #   for a LIVE column k; 0 for a structurally dead
                                    #   column (vcount[k]==0). Decouples "row k of R"
                                    #   (a logical index, always 1..n-n1, live or dead)
                                    #   from "physical row number" (1..mb, only live
                                    #   columns consume one) — see design_qr.md §3.4
                                    #   worked examples. Chosen at symbolic time
                                    #   (pattern-only, static).
    # --- workspace sizing ---
    max_rrow::Int                  # max rcount — sizes the row-subtree gather buffer
    max_vcol::Int                  # max V column length — sizes the packed reflector buffer
    nnzR::Int
    nnzV::Int                      # Σ vcount (D6: an upper bound on true numeric V
                                    #   nonzeros, exact as a STRUCTURAL/allocation count)
    flops::Float64                 # §3.5 — exact when rank detection is off
end

"""
    QRStats

Reported statistics from a numeric QR factorization: fill/flop counts and (after §5
rank-deficiency handling) the observed rank and the Foster–Davis phase-1 dropped-tail
error certificate. Public API, mirroring [`FactorStats`](@ref)'s role for Cholesky/LDLᵀ.
"""
mutable struct QRStats
    nnzR::Int
    nnzV::Int
    flops::Float64
    rank::Int                      # live pivots after §5 dead-column handling
    n_dead::Int                    # dropped columns
    dropped_norm::Float64          # ‖dropped tails‖_F (§5.2); 0.0 when full rank
end

QRStats() = QRStats(0, 0, 0.0, 0, 0, 0.0)

"""
    QRWorkspace{T,Ti<:Integer}

Preallocated scratch buffers reused across every `qr!`/`solve!` call on a factor
sharing one `QRSymbolic`, sized once from `QRSymbolic`'s workspace-sizing scalars so the
numeric phase never allocates (CLAUDE.md requirement 5; design_qr.md §4.5).
"""
struct QRWorkspace{T,Ti<:Integer}
    x::Vector{T}                   # length mb (physical/block row space, §3.4),
                                    #   zero-kept between columns (§4.1)
    stamp::Vector{Ti}               # row-subtree stamp array, indexed by BLOCK COLUMN
                                    #   (ancestor-climb marker, §4.1 step 2 — task 6
                                    #   correction: the design's §4.5 prose groups this
                                    #   under "max_rrow" alongside tsub, but stamp[node]
                                    #   is looked up by column index (1..n-n1), not by
                                    #   position-within-T^k — sizing it max_rrow would
                                    #   under-allocate whenever max_rrow < n-n1), length n-n1
    tsub::Vector{Ti}                # gathered/sorted T^k (row subtree), length n-n1 — task
                                    #   7 correction: `max_rrow = max(rcount)` bounds R's
                                    #   ROW sizes (column sizes of the star matrix's implied
                                    #   Cholesky factor), but `T^k = {i<k : R[i,k]≠0}` is a
                                    #   COLUMN-of-R quantity (⟺ a ROW of that same Cholesky
                                    #   factor) — a genuinely different, uncomputed bound;
                                    #   confirmed by a real BoundsError under
                                    #   `--check-bounds=yes` (`max_rrow` undersized `tsub`
                                    #   on a case where |T^k| exceeded it). `|T^k| < k ≤ n-n1`
                                    #   always holds trivially, so this is the simple correct
                                    #   fix rather than deriving the tight row-count bound.
    pack::Vector{T}                 # packed reflector staging buffer, length max_vcol (§4.4)
    rcursor::Vector{Ti}              # per-row append cursor into rcolind/rval, length n-n1
    rhs::Vector{T}                   # solve scratch over the caller's original-shaped
                                    #   RHS (full row space), length m
end

function QRWorkspace{T,Ti}(sym::QRSymbolic) where {T,Ti<:Integer}
    nb = length(sym.parent)        # block column count, n-n1
    QRWorkspace{T,Ti}(
        zeros(T, max(sym.mb, 1)),
        zeros(Ti, nb),
        Vector{Ti}(undef, max(nb, 1)),
        Vector{T}(undef, max(sym.max_vcol, 1)),
        Vector{Ti}(undef, nb),
        Vector{T}(undef, sym.m),
    )
end

"""
    QRFactor{T<:Real,Ti<:Integer} <: AbstractSparseFactor{T}

Sparse QR factor: `R` stored row-wise (CSC of `Rᵀ`) and `Q` implicit as stored
Householder vectors `V`/`beta` (design_qr.md §1.4/§4.1). Produced by
[`qr`](@ref)/refactored in place by [`qr!`](@ref) (zero allocations after the first
call, matching [`cholesky!`](@ref)/[`ldlt!`](@ref)'s contract).
"""
mutable struct QRFactor{T<:Real,Ti<:Integer} <: AbstractSparseFactor{T}
    sym::QRSymbolic{Ti}
    # R stored ROW-wise (CSC of Rᵀ): row k of R owns slots rptr[k]:rptr[k+1]-1.
    rcolind::Vector{Ti}
    rval::Vector{T}
    # Q implicit: V column-wise on sym.vptr/vrowind; beta[k] == 0 ⇒ dead/trivial
    # reflector (H_k = I), which makes §5's dead-column skip a plain no-op.
    vval::Vector{T}
    beta::Vector{T}
    ws::QRWorkspace{T,Ti}          # §4.5
    stats::QRStats
    ok::Bool
end
