# Machine-independent tunables, design.md §1.4/§3.5. All are free heuristics (no CHOLMOD
# provenance, see design.md §11 B2) exposed via Preferences.jl so they are compile-time
# consts under juliac --trim (no runtime dispatch cost) yet still calibratable per box.
# Every default here carries a derivation comment (design.md §3.5) rather than being a
# bare literal — required even though these aren't hardware-cache-derived like PureBLAS's
# `cpuinfo.jl` consts, because "where did this number come from" must always have an
# answer that isn't "I remembered CHOLMOD's default" (CLAUDE.md requirement 1).

using Preferences: @load_preference

# Relaxed-amalgamation merged-width tiers (design.md §3.5). Narrower than
# `amalg_cols[1]`: PureBLAS's Float64 microkernel register tile is ~8 columns wide, so
# blocks below that waste the kernel regardless of density — merge nearly always
# (amalg_zmax[1] high). Between tiers 1-2: update flops scale ~quadratically in width, so
# a zero-fraction z inflates flops by roughly 1/(1-z)^2 on the padded block. Above tier 2:
# wide panels already run near peak; padding is pure loss, and panel growth starts
# pressuring the update-buffer's cache residency, so only near-nesting qualifies.
#
# **Empirically recalibrated (2026-07-13, ROADMAP task 7b') against the exact
# union-height row estimate**, superseding the original starting-point numbers above —
# the previous defaults (`(8,32,128)`/`(0.9,0.15,0.03)`) were picked before
# `relaxed_amalgamation` could iterate to a fixpoint (design §3.5's old single-pass
# version); once the estimate feeding the z-test is exact instead of a proxy (see the
# `relaxed_amalgamation` docstring, src/symbolic/supernodes.jl), those thresholds turned
# out to systematically UNDER-merge on the gate matrices (M1 task 8's actual benchmark
# pass): a Chairmarks sweep of the M1 wall-time gate's warm-refactor arm over
# `amalg_zmax ∈ {(0.9,0.15,0.03), (0.95,0.3,0.08), (0.97,0.35,0.08), (0.98,0.4,0.1)}` ×
# `amalg_cols ∈ {(8,32,128), (16,64,128), (16,64,256)}` found the ORIGINAL tiers gave
# 4/14 gate passes (regressed from the prior single-pass algorithm's 6/14 — over-tight
# thresholds fragmented already-good structure), while doubling the column tiers to
# `(16,64,128)` and loosening z to `(0.97,0.35,0.08)` (deliberately the SAME zmax point
# already probed and reported as "far more permissive" in the prior child-ordering
# session, reused here rather than re-derived from scratch) restored solid PASSes on
# every banded and Laplacian gate row while leaving small unstructured-random cases
# (n<=1000) at a noise-level tie — those were already at or near CHOLMOD's per-call-
# overhead floor before any of this work (design's fixed cost is unrelated to supernode
# shape at that scale). Doubling `amalg_cols` is the free-tunable half of this
# recalibration: the exact height estimate now lets multi-child cascaded merges reach
# useful widths that the original single-microkernel-tile-anchored cap would truncate
# mid-cascade, so a wider cap gives the fixpoint loop room to actually converge on a
# BLAS-3-efficient block before hitting the next tier's zero-fraction wall. No
# correctness weight attaches to any of these numbers — only to the §3.4 superset
# invariant (checked independently in tests), and this remains a swept starting point,
# not a hardware- or paper-derived constant (see the numbered rationale in design §3.5's
# table, unchanged conceptually — only the calibration below it moved).

# `@load_preference` returns a plain `Vector` for a TOML array (Preferences.jl has no
# tuple representation), so an `::NTuple{3,T}` typeassert on the raw result throws the
# moment anyone actually overrides these via Preferences — `Tuple(...)` converts either
# the Vector default or a loaded Vector override uniformly.
const AMALG_COLS = Tuple(@load_preference("amalg_cols", [16, 64, 128]))::NTuple{3,Int}
const AMALG_ZMAX = Tuple(@load_preference("amalg_zmax", [0.97, 0.35, 0.08]))::NTuple{3,Float64}

# AMD dense-row multiplier (design.md §2.2 pt 6). Attribution: the AMD *package's*
# documented user-guide default (`AMD_DENSE = 10`), not the 1996 paper's algorithm text
# (the paper's algorithm has no dense-row special case). The `16` floor is ours, so tiny
# problems never strip anything.
const AMD_DENSE_MULT = @load_preference("amd_dense_mult", 10.0)::Float64
const AMD_DENSE_FLOOR = 16

# Signed-regularization floor for LDLᵀ/SQD factorization (design.md §5.1), relative to
# the matrix's diagonal scale (`δ_j = LDLT_DELTA * ‖A‖-scale`). QDLDL/Clarabel-style
# default magnitude.
const LDLT_DELTA = @load_preference("ldlt_delta", 1e-12)::Float64

# Per-supernode flop threshold above which the GPU path engages (design.md §8, M3).
const GPU_FLOP_THRESHOLD = @load_preference("gpu_flop_threshold", 2e9)::Float64

# Per-column slack grow factor for simplicial storage (design.md §7: "simplicial()
# allocates each column with slack (grow-factor Preference)"). Column j gets
# `min(n - j, max(len_j, ceil(grow * (len_j + 1))))` row slots for `len_j` initial
# entries — the `+ 1` counts the implicit unit diagonal so that empty/short columns
# (exactly where an update's new fill lands first, since fill enters at the low end of
# the etree path) still receive slack instead of `ceil(grow * 0) = 0`. The 1.5 default
# is our own elbow-room starting point (Davis–Hager 1999 §7 sizes columns from a known
# worst-case factor instead, which we don't have for a general update stream); no
# external-implementation provenance, free tunable, to be calibrated in the M2
# benchmark pass. Exceeding a column's slack is not an error: `updowndate!` returns
# `:refactor_required` (design.md §7's documented overflow contract).
const SIMPLICIAL_GROW = @load_preference("simplicial_grow", 1.5)::Float64

# Sparse QR tunables (design_qr.md §1.6). Rank threshold τ = qr_tol_mult *
# max(m,n) * eps(T) * max_j‖A[:,j]‖₂ (design_qr.md §5.3) — own derivation, free
# tunable, no external provenance (design.md's B2 discipline, distinct from
# design_qr.md's own B2, §0).
const QR_TOL_MULT = @load_preference("qr_tol_mult", 8.0)::Float64

# Column-singleton pre-elimination magnitude threshold = this × τ (design_qr.md §2.3).
const QR_SINGLETON_MULT = @load_preference("qr_singleton_mult", 1.0)::Float64

# `qr(A; method=:auto)`'s :column-vs-:frontal split (design_qr_m5b.md §A5.6): predictor
# is `sym.flops / sym.nnzR` (both already computed by `symbolic_qr` before either
# numeric path runs, no extra work to obtain), `:frontal` when the ratio exceeds this.
# Provenance: the PREDICTOR (not this number) is faer 0.24.1's own choice for its
# simplicial-vs-supernodal QR dispatch (`flops/nnz`, sparse/linalg/qr.rs — faer is
# MIT-licensed, freely readable, unlike the CHOLMOD/SuiteSparse GPL prohibition,
# CLAUDE.md req 1); faer's own threshold constant is `40.0`
# (`QR_SUPERNODAL_RATIO_FACTOR`, sparse/linalg/mod.rs). Independently verified here,
# not blindly copied: measuring this ratio against measured :column-vs-:frontal cold
# times on the M5 gate set (task 16e, galen, 2026-07-15) showed a clean, wide
# separation — every :column-winning matrix sat at ratio ≤ 7.0, every :frontal-winning
# matrix at ratio ≥ 863 — so any threshold in that gap works on this data; faer's own
# 40.0 is kept rather than picking an arbitrary round number in the gap, since it's
# the one with independent (non-PureSparse) grounding.
const QR_AUTO_METHOD_RATIO = @load_preference("qr_auto_method_ratio", 40.0)::Float64

# Frontal path's small-front scalar fallback threshold (design_qr_m5b.md §A5.3 —
# added post-close, 2026-07-15): a front with `m_f*n_f` below this uses a pure
# column-by-column scalar Householder pass (no `wy_t!`/`wy_apply!` calls at all)
# instead of the staircase-blocked WY-group loop. Provenance: faer 0.24.1's own
# `qr_in_place`/`qr_in_place_blocked` (src/linalg/qr/no_pivoting/factor.rs:137-158)
# recursively drops to `qr_in_place_unblocked` (a pure scalar loop, zero BLAS-3 calls)
# whenever the REMAINING sub-problem's element count falls below
# `QrParams::auto().blocking_threshold = 48*48 = 2304` (factor.rs:131) — i.e. faer
# itself never calls a blocked kernel on a problem this small. Root-caused via direct
# profiling (`banded_ls_n1500x500_bw15`'s fronts: max 25×65=1625 elements, well under
# faer's own threshold) that PureSparse was paying `wy_t!`/`wy_apply!`'s BLAS-3
# dispatch cost on problems faer itself would never block. Threshold kept at faer's
# own value rather than re-derived, matching this project's precedent
# (`QR_AUTO_METHOD_RATIO`'s own reasoning) — re-verify empirically before trusting
# it to extrapolate past the matrices it was found on.
const QR_FRONTAL_UNBLOCKED_THRESHOLD = @load_preference("qr_frontal_unblocked_threshold", 2304)::Int

# COLAMD dense-row/column withholding multipliers (design_qr.md §2.2 pt 5). D1: this is
# a REUSE of the existing AMD dense-row heuristic shape (`max(16, mult*sqrt(n))`,
# ultimately sourced to the AMD package User Guide, design.md §2.2 pt 6), not an
# independently derived constant — the COLAMD paper's own default is a flat 50%
# density ("probably too high for most matrices", paper p.362), deliberately not used
# here so the whole ordering layer shares one dense-threshold convention.
const COLAMD_DENSE_ROW_MULT = @load_preference("colamd_dense_row_mult", 10.0)::Float64
const COLAMD_DENSE_COL_MULT = @load_preference("colamd_dense_col_mult", 10.0)::Float64
const COLAMD_DENSE_FLOOR = 16

# Drop-in activation (design.md §10 M4): whether `src/dropin.jl` — which extends
# `LinearAlgebra.cholesky`/`ldlt` for `SparseMatrixCSC`, a genuine stdlib method-table
# overwrite — is even `include`d. This MUST be a compile-time Preference (not a runtime
# `Ref{Bool}` toggle checked inside an unconditionally-defined method): Julia's method
# table has no "temporarily shadow a method" primitive, and defining the override
# unconditionally then branching on a runtime flag INSIDE it would still constitute
# "silently extending stdlib the moment PureSparse loads" (CLAUDE.md's explicit
# concern, `PureSparse.jl`'s `import LinearAlgebra` comment) even when the flag is off
# — the override would already exist, just be a no-op. Making the `include` itself
# conditional on a compile-time const is the only way the override genuinely doesn't
# exist until opted in, and it is what keeps this trim-compatible (no runtime
# `eval`/`invokelatest`, CLAUDE.md requirement 4) — the tradeoff, unlike PureBLAS's
# `activate()`/`deactivate()` (which forwards through `libblastrampoline`, a C-ABI
# indirection layer BLAS calls already go through and that supports true runtime
# hot-swapping), is that PureSparse's `activate!()`/`deactivate!()` (`dropin_toggle.jl`)
# set this Preference and require a Julia restart to take effect — an honest
# consequence of pure-Julia multiple dispatch having no equivalent trampoline layer,
# not a corner cut.
const DROPIN_ACTIVE = @load_preference("dropin_active", false)::Bool
