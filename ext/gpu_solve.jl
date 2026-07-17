# ext/gpu_solve.jl — LEVEL-SCHEDULED batched device triangular solve (design_gpu.md §7/§8).
# BACKEND-GENERIC (pure KernelAbstractions + Atomix, no CUDA/cuBLAS) — included by the CUDA ext
# AND standalone by the AMD probe (benchmark/gpu/amd_solve_test.jl), like gpu_dense.jl.
#
# Why: the per-supernode trsv/gemv/scatter solve was launch-bound (~63k microscopic launches on
# SQD 40³ ≈ the whole factor time). Fix: level-schedule on the supernodal etree — level[s] =
# 1 + max(level[children]); all supernodes in a level are mutually independent (a supernode's
# below-panel rows target only its etree ANCESTORS, which sit at strictly higher levels), so one
# level = ONE kernel launch with one workgroup per supernode. #launches = 2·#levels (+1 D-scale)
# instead of ~6·#supernodes.
#
# Forward (L·y=b, ascending levels): per workgroup, column-sweep trsv on the diagonal block +
# below-panel gemv + scatter into y. Siblings in a level can scatter into a SHARED ancestor row
# → BARE atomic add (return value never used — gfx1151's compiler segfaults on a used
# atomic-rmw return, see _front_fused64!'s history in gpu_dense.jl).
# Backward (Lᵀ·x=y, descending levels): per workgroup, fused gather+gemvᵀ (reads only finished
# ancestor entries) + right-looking column-sweep trsvᵀ. Writes only its own y-block → no atomics.
#
# The schedule is analysis-once data (built on CPU from sparent, uploaded once) — pass a
# prebuilt `SolveSchedule` to solve many times with zero setup cost.

# Workgroup width. Wide on purpose: near-root levels have 1–4 supernodes with MB-sized
# below-panels, and a level gets ONE workgroup per supernode — width is the only latency
# hiding there (galen sweep on SQD 40³: 64→99 ms, 128→56, 256→36, 512→27, 1024→26).
# Leaf levels (thousands of tiny supernodes) measured insensitive to the width.
const SOLVE_GROUP = 1024

"""
    SolveSchedule(lev_ptr, d_nodes, d_px)

Level schedule for the batched device solve: level ℓ's supernodes are
`d_nodes[lev_ptr[ℓ] : lev_ptr[ℓ+1]-1]` (device), `lev_ptr` stays on host (drives launches),
`d_px` is the per-supernode panel offset (device). Build once via [`solve_schedule`](@ref).
"""
struct SolveSchedule{VI}
    lev_ptr::Vector{Int}
    d_nodes::VI
    d_px::VI
end

"""
    solve_schedule(sparent, px, ref) -> SolveSchedule

Compute the elimination-level schedule (CPU, pure) from the supernodal etree `sparent` and
upload it to the device of `ref` (any device integer array — used only for `similar`).
"""
function solve_schedule(sparent::AbstractVector, px::AbstractVector, ref)
    Ti = eltype(sparent)
    ns = length(sparent)
    level = zeros(Int, ns)
    @inbounds for s in 1:ns
        p = Int(sparent[s])
        p == 0 && continue
        @assert p > s "solve_schedule: sparent not topologically ordered (parent $p ≤ child $s)"
        level[p] ≤ level[s] && (level[p] = level[s] + 1)
    end
    nlev = ns == 0 ? 0 : maximum(level) + 1
    cnt = zeros(Int, nlev)
    @inbounds for s in 1:ns; cnt[level[s] + 1] += 1; end
    lev_ptr = Vector{Int}(undef, nlev + 1); lev_ptr[1] = 1
    @inbounds for l in 1:nlev; lev_ptr[l + 1] = lev_ptr[l] + cnt[l]; end
    nodes = Vector{Ti}(undef, ns); cur = lev_ptr[1:nlev]
    @inbounds for s in 1:ns
        l = level[s] + 1; nodes[cur[l]] = Ti(s); cur[l] += 1
    end
    d_nodes = similar(ref, Ti, ns); copyto!(d_nodes, nodes)
    d_px = similar(ref, Ti, length(px)); copyto!(d_px, px)
    return SolveSchedule{typeof(d_nodes)}(lev_ptr, d_nodes, d_px)
end

# One workgroup per supernode in the level. trsv is a column sweep (nscol is group-uniform →
# barriers are uniform); the below-panel gemv+scatter strides rows over the group's threads.
@kernel unsafe_indices = true function _solve_fwd_level!(y, @Const(xv), @Const(rowind),
        @Const(rowind_ptr), @Const(super), @Const(px), @Const(nodes), base, unitdiag)
    li = @index(Local, Linear)                 # 1..SOLVE_GROUP
    g = @index(Group, Linear)
    T = eltype(y)
    @inbounds begin
        s = Int(nodes[base + g - 1])
        j0 = Int(super[s]); nscol = Int(super[s + 1]) - j0
        rp0 = Int(rowind_ptr[s]); nsrow = Int(rowind_ptr[s + 1]) - rp0
        off = Int(px[s]) - 1                   # panel[i,j] = xv[off + (j-1)*nsrow + i]
        for j in 1:nscol                       # trsv L·y=b, left-looking column sweep
            jj = j0 + j - 1
            if !unitdiag && li == 1
                y[jj] /= xv[off + (j - 1) * nsrow + j]
            end
            @synchronize
            yj = y[jj]
            i = j + li
            while i ≤ nscol
                y[j0 + i - 1] = muladd(-xv[off + (j - 1) * nsrow + i], yj, y[j0 + i - 1])
                i += SOLVE_GROUP
            end
            @synchronize
        end
        below = nsrow - nscol
        k = li
        while k ≤ below                        # gemv row k of the below panel + atomic scatter
            acc = zero(T)
            r = off + nscol + k
            for j in 1:nscol
                acc = muladd(xv[r], y[j0 + j - 1], acc)
                r += nsrow
            end
            @atomic y[Int(rowind[rp0 + nscol + k - 1])] += -acc   # bare atomic add (AMD-safe)
            k += SOLVE_GROUP
        end
    end
end

@kernel unsafe_indices = true function _solve_bwd_level!(y, @Const(xv), @Const(rowind),
        @Const(rowind_ptr), @Const(super), @Const(px), @Const(nodes), base, unitdiag)
    li = @index(Local, Linear)
    g = @index(Group, Linear)
    T = eltype(y)
    @inbounds begin
        s = Int(nodes[base + g - 1])
        j0 = Int(super[s]); nscol = Int(super[s + 1]) - j0
        rp0 = Int(rowind_ptr[s]); nsrow = Int(rowind_ptr[s + 1]) - rp0
        off = Int(px[s]) - 1
        below = nsrow - nscol
        j = li
        while j ≤ nscol                        # fused gather+gemvᵀ: y_j −= L21[:,j]·x[anc rows]
            acc = zero(T)
            r = off + (j - 1) * nsrow + nscol
            for k in 1:below
                acc = muladd(xv[r + k], y[Int(rowind[rp0 + nscol + k - 1])], acc)
            end
            y[j0 + j - 1] -= acc
            j += SOLVE_GROUP
        end
        @synchronize
        for j in nscol:-1:1                    # trsv Lᵀ·x=w, right-looking column sweep
            jj = j0 + j - 1
            if !unitdiag && li == 1
                y[jj] /= xv[off + (j - 1) * nsrow + j]
            end
            @synchronize
            yj = y[jj]
            i = li
            while i ≤ j - 1                    # w_i −= L[j,i]·x_j
                y[j0 + i - 1] = muladd(-xv[off + (i - 1) * nsrow + j], yj, y[j0 + i - 1])
                i += SOLVE_GROUP
            end
            @synchronize
        end
    end
end

"""
    batched_solve!(y, xv, d_rowind, d_rowind_ptr, d_super, sched, unitdiag, dvec) -> y

Level-scheduled supernodal triangular solve on device: forward `L·z=b` (ascending levels),
optional `z ./= dvec` (LDLᵀ D⁻¹ scale, pass `nothing` for Cholesky), backward `Lᵀ·x=z`
(descending levels). `unitdiag=true` for the LDLᵀ unit-lower factor. One launch per level.
"""
function batched_solve!(y, xv, d_rowind, d_rowind_ptr, d_super, sched::SolveSchedule,
                        unitdiag::Bool, dvec)
    backend = get_backend(y)
    fwd = _solve_fwd_level!(backend, SOLVE_GROUP)
    bwd = _solve_bwd_level!(backend, SOLVE_GROUP)
    lp = sched.lev_ptr; nlev = length(lp) - 1
    args = (xv, d_rowind, d_rowind_ptr, d_super, sched.d_px, sched.d_nodes)
    for l in 1:nlev
        ng = lp[l + 1] - lp[l]
        ng == 0 && continue
        fwd(y, args..., lp[l], unitdiag; ndrange = ng * SOLVE_GROUP)
    end
    isnothing(dvec) || (y ./= dvec)
    for l in nlev:-1:1
        ng = lp[l + 1] - lp[l]
        ng == 0 && continue
        bwd(y, args..., lp[l], unitdiag; ndrange = ng * SOLVE_GROUP)
    end
    return y
end
