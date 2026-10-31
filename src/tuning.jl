# Machine-independent tunables, design.md §1.4/§3.5. All are free heuristics (no CHOLMOD
# provenance, see design.md §11 B2) exposed via Preferences.jl so they are compile-time
# consts under juliac --trim (no runtime dispatch cost) yet still calibratable per box.
# Every default here carries a derivation comment (design.md §3.5) rather than being a
# bare literal — required even though these aren't hardware-cache-derived like PureBLAS's
# `cpuinfo.jl` consts, because "where did this number come from" must always have an
# answer that isn't "I remembered CHOLMOD's default" (CONTRIBUTING.md requirement 1).

using Preferences: @load_preference

# Relaxed-amalgamation merged-width tiers (design.md §3.5). Narrower than
# `amalg_cols[1]`: PureBLAS's Float64 microkernel register tile is ~8 columns wide, so
# blocks below that waste the kernel regardless of density — merge nearly always
# (amalg_zmax[1] high). Between tiers 1-2: update flops scale ~quadratically in width, so
# a zero-fraction z inflates flops by roughly 1/(1-z)^2 on the padded block. Above tier 2:
# wide panels already run near peak; padding is pure loss, and panel growth starts
# pressuring the update-buffer's cache residency, so only near-nesting qualifies.
#
# **Empirically recalibrated against the exact
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
const AMALG_COLS = Tuple(@load_preference("amalg_cols", [16, 64, 128]))::NTuple{3, Int}
const AMALG_ZMAX = Tuple(@load_preference("amalg_zmax", [0.97, 0.35, 0.08]))::NTuple{3, Float64}

# AMD dense-row multiplier (design.md §2.2 pt 6). Attribution: the AMD *package's*
# documented user-guide default (`AMD_DENSE = 10`), not the 1996 paper's algorithm text
# (the paper's algorithm has no dense-row special case). The `16` floor is ours, so tiny
# problems never strip anything.
const AMD_DENSE_MULT = @load_preference("amd_dense_mult", 10.0)::Float64
const AMD_DENSE_FLOOR = 16

# Signed-regularization floor for LDLᵀ/SQD factorization (design.md §5.1), relative to
# the matrix's diagonal scale (`δ_j = LDLT_DELTA * ‖A‖-scale`). QDLDL/Clarabel-style
# default magnitude.
const LDLT_DELTA = @load_preference("ldlt_delta", 1.0e-12)::Float64

# Simplicial-versus-supernodal switch (design.md §4.5): `symbolic(A)` takes the simplicial
# path when `flops / nnzL` (= Σ colcount² / Σ colcount, the mean multiply-adds per stored
# entry of L, both already computed from the column counts) is at or below this value.
# Provenance: the CHOLMOD *User Guide* documents that it switches on such a ratio, which is
# the only thing taken from it; the number here is PureSparse's own measurement
# (benchmark/simplicial_threshold.jl, Zen 5c core pinned, single thread, four runs on a
# shared machine; simplicial-to-supernodal time ratios of the warm refactorization,
# the warm single-vector solve, and their sum, medians; the sum is the quantity the
# switch is set on, since the target users refactorize many times per analysis):
#
#   matrix              flops/nnzL   refactor   solve      refactor+solve
#   banded bw 2..4        3..5       0.17-0.59  0.23-0.89  0.19-0.64
#   banded bw 8           9          0.70-0.99  0.47-1.04  0.74-0.83   (one run 1.5)
#   banded bw 12         13          1.36       1.28       1.35
#   laplacian2d 20x20    13          0.57       0.62       0.58
#   banded bw 16         17          0.84-2.1   0.36-1.43  0.73-2.0    (wins in 1 of 4)
#   laplacian2d 30x30    18          0.57-1.0   0.38-0.78  0.58-0.95
#   laplacian2d 40x40    25          1.05       0.86       1.03
#   laplacian2d 50x50    29          1.1-1.2    0.56-0.88  1.0-1.11
#   banded bw 24..32    25..33       2.4-4.1    0.98-1.54  2.3-3.7
#   laplacian2d 70x70    40          1.55-1.66  0.61-0.82  1.4-1.5
#   laplacian3d 8^3      43          1.3-1.65   0.59-0.75  1.25-1.5
#   laplacian2d 100x100  55          1.85-2.03  0.58-1.04  1.66-1.92
#   banded bw 64         65          3.6-5.1    1.27-2.08  3.4-4.7
#   laplacian2d 140x140  71          2.2-2.7    0.67-0.89  2.0-2.5
#   laplacian3d, random 102-649      4-12       1.0-3.3    3.8-11
#
# The up-looking simplicial factorization streams every column of L once per nonzero
# of its row structure (Σ colcount² of traffic) where the supernodal panels reuse
# cache, so its refactorization falls behind early; the simplicial solve, with no
# padding and no per-supernode kernel calls, stays ahead longer on grids than on
# banded matrices (long chain supernodes suit the supernodal solve). Refactor+solve is
# faster on the simplicial path for every matrix with a predictor of at most 9 in
# every run, grids keep winning to about 18, and the first losses are banded bw 12
# (13) and bw 16 (17); from 25 on every matrix loses. The default is the integer
# above the last predictor that wins in every run (9) and below the first observed
# loss (13): 10. A solve-dominated workload (one factorization, many right-hand
# sides) can raise this preference — the simplicial solve alone beats the supernodal
# one on 2D grids up to a predictor of about 70 — or pass `method = :simplicial`.
const SIMPLICIAL_RATIO = @load_preference("simplicial_ratio", 10.0)::Float64

# Per-supernode flop threshold above which the GPU path engages (design.md §8, M3).
const GPU_FLOP_THRESHOLD = @load_preference("gpu_flop_threshold", 2.0e9)::Float64

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
# CONTRIBUTING.md req 1); faer's own threshold constant is `40.0`
# (`QR_SUPERNODAL_RATIO_FACTOR`, sparse/linalg/mod.rs). Independently verified here,
# not blindly copied: measuring this ratio against measured :column-vs-:frontal cold
# times on the M5 gate set on a Ryzen 9 5900X showed a clean, wide
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
# "silently extending stdlib the moment PureSparse loads" (an explicit
# concern, `PureSparse.jl`'s `import LinearAlgebra` comment) even when the flag is off
# — the override would already exist, just be a no-op. Making the `include` itself
# conditional on a compile-time const is the only way the override genuinely doesn't
# exist until opted in, and it is what keeps this trim-compatible (no runtime
# `eval`/`invokelatest`, CONTRIBUTING.md requirement 4) — the tradeoff, unlike PureBLAS's
# `activate()`/`deactivate()` (which forwards through `libblastrampoline`, a C-ABI
# indirection layer BLAS calls already go through and that supports true runtime
# hot-swapping), is that PureSparse's `activate!()`/`deactivate!()` (`dropin_toggle.jl`)
# set this Preference and require a Julia restart to take effect — an honest
# consequence of pure-Julia multiple dispatch having no equivalent trampoline layer,
# not a corner cut.
const DROPIN_ACTIVE = @load_preference("dropin_active", false)::Bool
