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
