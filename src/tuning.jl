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
# a zero-fraction z inflates flops by roughly 1/(1-z)^2 on the padded block — z=0.15 caps
# inflation near 1.38x. Above tier 2: wide panels already run near peak; padding is pure
# loss, and panel growth starts pressuring the update-buffer's cache residency, so only
# near-perfect nesting (z<=0.03) qualifies. These are STARTING POINTS ONLY, swept and
# recalibrated per matrix class in M1's benchmark pass (ROADMAP.md M1 task 8) — no
# correctness weight attaches to the exact numbers, only to satisfying the §3.4 superset
# invariant (checked independently in tests).
const AMALG_COLS = @load_preference("amalg_cols", (8, 32, 128))::NTuple{3,Int}
const AMALG_ZMAX = @load_preference("amalg_zmax", (0.9, 0.15, 0.03))::NTuple{3,Float64}

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
