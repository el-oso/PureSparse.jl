# pure_ldlt_front.jl — M6 LDLᵀ SCRATCH: fused pure-device SIGNED-LDL front (the LDLᵀ
# analogue of gpu_front! / _front_fused64_v2! in ext/gpu_dense.jl). Ports the CPU reference
# column loop of cpu_multifrontal_ldlt! (ext/multifrontal.jl, amendment E) — per-column
# fixed-pivot signed regularization + inertia + running per-supernode dmax — into ONE fused
# kernel per 64-wide panel: in-shared signed-LDL of the diag block (redundant per group,
# sequential over columns — the pivot chain is inherently serial) + the tall-panel solve
# L21 = A21·L11⁻ᵀ·D⁻¹ in the same launch. nscol > 64 blocks right-looking with a D-scaled
# trapezoidal rank-64 trailing update (one launch per panel). Replaces the current vendor
# LDLᵀ front (CPU _ldl_block! D2H/H2D round-trip + cuBLAS trsm + colscale), which cannot
# fuse and pins the KKT path at 3.41× vs the SPD path's 5.04×.
#
# Key semantics ported verbatim from the reference (the numeric caveats that matter):
#   • classification (n_pos/n_neg/n_zero) uses the RAW pivot, zero test is the order-free
#     `adj ≤ zeta·max(dmax_local, delta)` (amendment E);
#   • regularization test `wrongsign || adj < delta` is dmax-INDEPENDENT ⇒ the factor
#     (unit-lower L, signed D) is identical no matter how dmax is carried — only the
#     inertia COUNTS need dmax, so only ONE group (gi==1) tracks it and the stats;
#   • the PERTURBED pivot (not the raw one) scales the column, feeds the rank-1 update,
#     and feeds dmax_local;
#   • dmax_local is a RUNNING per-supernode max: carried across 64-blocks through a device
#     stats slot (stream-ordered launches), reset per front via the firstblk flag.
#
# Clean-room: textbook fixed-pivot signed-regularization LDL (QDLDL/Clarabel-style, as in
# src/numeric/ldlt.jl), G&VL blocked right-looking structure, unit-triangular forward-
# substitution inverse + MAGMA-style block-inverse multiply (published technique) — no
# CHOLMOD/cuSOLVER source. Reciprocal-once for FP64 division.
#
# Zero device-pool alloc per driver call: caller-provided LDLFrontWS (arrival counter +
# 6-slot stats buffer), views + @localmem/@private only.
#
# Run (galen): julia --project=$HOME/Documents/claude/gpu_probe benchmark/gpu/pure_ldlt_front.jl

using CUDA
using KernelAbstractions
using KernelAbstractions: @kernel, @index, @localmem, @private, @synchronize, get_backend
using Base.Cartesian: @nexprs
using LinearAlgebra
using Random
using Printf
using CUDA.CUBLAS: trsm!
using PureSparse                       # PureSparse.ger! + LDLT_DELTA (faithful vendor arm)

# =======================================================================================
# Kernels
# =======================================================================================

# FUSED signed-LDL front kernel v2 (register-tiled multiply — the _front_fused64_v2!
# analogue): every group stages lower(A[n×n]) (n ≤ 64) into shared, redundantly runs the
# signed-LDL pivot loop (sequential over columns; trailing 64×64 update parallel across
# the 256 threads), inverts the UNIT-lower L11 in the strict upper triangle of the same
# tile (thread-per-column forward substitution — race-free, each thread reads lower + its
# own row), then computes its OWN 64-row tile of the tall panel as the dependency-free
# multiply L21[i,j] = (Σ_{k≤j} A21[i,k]·invL[j,k])·(1/d_j), 4×4 register tiles. Group 1
# writes dvec + (thread 1) the inertia/dmax stats; the LAST-arriving group (monotonic
# atomic counter — same ordering proof as _front_fused64!) writes the unit-lower factored
# diagonal back.
@kernel unsafe_indices = true function _ldl_front_fused64_v2!(A, B, n, m, @Const(sgn), dv,
                                                              stats, delta, zeta, firstblk,
                                                              cnt, target)
    T = eltype(A)
    li = @index(Local, NTuple)            # (16,16)
    gi = @index(Group, Linear)
    tx = li[1]; ty = li[2]
    tid = (ty - 1) * 16 + tx              # 1..256
    Ls = @localmem T (64, 64)
    Ds = @localmem T (64,)                # signed D (perturbed pivots)
    Dinv = @localmem T (64,)              # 1/d_j — reciprocal once, FP64 div is slow
    Wc = @localmem T (64,)                # d_j·l_cc, hoisted per column (ger!'s `a` — keeps
    As = @localmem T (64, 8)              # the trailing sweep at ONE fma/element)
    lastf = @localmem Int32 (1,)
    # zero-padded staging of lower(A) into the fixed 64×64 tile (bit-ops only)
    q = tid
    @inbounds while q <= 4096
        r = ((q - 1) & 63) + 1
        c = ((q - 1) >> 6) + 1
        Ls[r, c] = (r <= n && c <= n && r >= c) ? A[r, c] : zero(T)
        q += 256
    end
    @synchronize
    # in-shared right-looking signed LDL, redundant per group. Counts live in thread-1
    # registers; only gi==1's are real (they alone see the carried dmax + write stats).
    np = 0; nn = 0; nz = 0; npert = 0
    maxp = zero(T); dmax = zero(T)
    @inbounds if tid == 1 && gi == 1 && firstblk == Int32(0)
        dmax = stats[6]                   # per-front running dmax, carried across blocks
    end
    for j in 1:n
        if tid == 1
            @inbounds begin
                dj = Ls[j, j]; adj = abs(dj)
                if adj <= zeta * max(dmax, delta)     # zero test, order-free (amendment E)
                    nz += 1
                elseif dj > zero(T)
                    np += 1
                else
                    nn += 1
                end
                sg = sgn[j]
                wrongsign = (sg == Int8(1) && !(dj > zero(T))) ||
                            (sg == Int8(-1) && !(dj < zero(T)))
                if wrongsign || adj < delta           # signed regularization (dmax-independent)
                    tgt = sg == Int8(0) ? (signbit(dj) ? -one(T) : one(T)) : T(sg)
                    newd = tgt * max(delta, adj); npert += 1
                    pert = abs(newd - dj); pert > maxp && (maxp = pert)
                    dj = newd                         # PERTURBED pivot from here on
                end
                Ds[j] = dj; Dinv[j] = one(T) / dj
                adf = abs(dj); adf > dmax && (dmax = adf)
                Ls[j, j] = one(T)                     # unit-lower L11
            end
        end
        @synchronize
        djv = Ds[j]
        r2 = j + tid
        @inbounds if r2 <= n
            lv = Ls[r2, j] * Dinv[j]                  # scale column below diag …
            Ls[r2, j] = lv
            Wc[r2] = djv * lv                         # … and hoist d_j·l (ger!'s `a`)
        end
        @synchronize
        q = tid
        @inbounds while q <= 4096                     # rank-1 trailing update, −d_j·l·lᵀ
            rr = j + ((q - 1) & 63) + 1
            cc = j + ((q - 1) >> 6) + 1
            if rr <= n && cc <= n && rr >= cc
                Ls[rr, cc] = muladd(-Wc[cc], Ls[rr, j], Ls[rr, cc])
            end
            q += 256
        end
        @synchronize
    end
    # stats + D write (group 1 only; single thread, launches are stream-ordered)
    @inbounds if gi == 1 && tid == 1
        stats[1] += T(np); stats[2] += T(nn); stats[3] += T(nz); stats[4] += T(npert)
        maxp > stats[5] && (stats[5] = maxp)
        stats[6] = dmax
    end
    @inbounds if gi == 1 && tid <= n
        dv[tid] = Ds[tid]
    end
    # invert the UNIT-lower L11: thread c writes column c of L⁻¹ TRANSPOSED into row c of
    # the strict upper triangle (reads only lower entries + its own row ⇒ race-free)
    if tid <= n
        c = tid
        @inbounds begin
            for i2 in (c + 1):n
                s = Ls[i2, c]                         # ·invL[c,c] = 1 (unit)
                for k in (c + 1):(i2 - 1)
                    s = muladd(Ls[i2, k], Ls[c, k], s)
                end
                Ls[c, i2] = -s                        # unit diag ⇒ no scaling
            end
        end
    end
    @synchronize
    # panel multiply: this group's 64-row tile, L21 := (A21·L11⁻ᵀ)·D⁻¹, 4×4 register tiles
    br = (gi - 1) * 64
    acc = @private T (4, 4)
    @inbounds for i in 1:4, j in 1:4
        acc[i, j] = zero(T)
    end
    k0 = 0
    nt = div(n + 7, 8)
    for _ in 1:nt
        @inbounds for t in 1:2
            p = (tid - 1) + (t - 1) * 256
            ml = p & 63; kl = p >> 6
            gr = br + ml; gk = k0 + kl
            As[ml + 1, kl + 1] = (gr < m && gk < n) ? B[gr + 1, gk + 1] : zero(T)
        end
        @synchronize
        @inbounds for kk in 1:8
            k = k0 + kk                   # 1-based k
            @nexprs 4 i -> (a_i = As[(tx - 1) * 4 + i, kk])
            @nexprs 4 j -> begin
                col_j = (ty - 1) * 4 + j
                w_j = k < col_j ? Ls[k, col_j] : (k == col_j ? one(T) : zero(T))
            end
            @nexprs 4 i -> @nexprs 4 j -> (acc[i, j] = muladd(a_i, w_j, acc[i, j]))
        end
        @synchronize
        k0 += 8
    end
    @inbounds for i in 1:4, j in 1:4
        gr = br + (tx - 1) * 4 + (i - 1)
        gc = (ty - 1) * 4 + (j - 1)
        if gr < m && gc < n
            B[gr + 1, gc + 1] = acc[i, j] * Dinv[gc + 1]   # D⁻¹ fold at write-back
        end
    end
    # last-arriving group writes the factored unit-lower diagonal back (ordering proof:
    # see _front_fused64! — every group's A-loads happen-before its fenced arrival)
    @synchronize
    if tid == 1
        CUDA.threadfence()
        old = CUDA.atomic_add!(pointer(cnt), Int64(1))
        lastf[1] = old == target - 1 ? Int32(1) : Int32(0)
    end
    @synchronize
    if lastf[1] == Int32(1)
        q = tid
        @inbounds while q <= 4096
            r = ((q - 1) & 63) + 1
            c = ((q - 1) >> 6) + 1
            if r <= n && c <= n && r >= c
                A[r, c] = Ls[r, c]
            end
            q += 256
        end
    end
end

# FUSED signed-LDL front kernel v1 (row-solve — the _front_fused64! analogue, for tall
# panels where v2's per-group redundant invert stops paying, cf. FUSE_M_MAX). Same factor
# loop; then the strict upper triangle is filled with the D-SCALED transposed factor
# Lds[k,j] = L11[j,k]·d_k, and each thread forward-substitutes one row of
# L21·(D·L11ᵀ) = A21:  L21[i,j] = (A21[i,j] − Σ_{k<j} L21[i,k]·Lds[k,j])·(1/d_j).
@kernel unsafe_indices = true function _ldl_front_fused64_v1!(A, B, n, m, @Const(sgn), dv,
                                                              stats, delta, zeta, firstblk,
                                                              cnt, target)
    T = eltype(A)
    li = @index(Local, Linear)            # 1..256
    gi = @index(Group, Linear)
    Ls = @localmem T (64, 64)
    Ds = @localmem T (64,)
    Dinv = @localmem T (64,)
    Wc = @localmem T (64,)                # d_j·l_cc, hoisted per column (see v2)
    lastf = @localmem Int32 (1,)
    q = li
    @inbounds while q <= 4096
        r = ((q - 1) & 63) + 1
        c = ((q - 1) >> 6) + 1
        Ls[r, c] = (r <= n && c <= n && r >= c) ? A[r, c] : zero(T)
        q += 256
    end
    @synchronize
    np = 0; nn = 0; nz = 0; npert = 0
    maxp = zero(T); dmax = zero(T)
    @inbounds if li == 1 && gi == 1 && firstblk == Int32(0)
        dmax = stats[6]
    end
    for j in 1:n
        if li == 1
            @inbounds begin
                dj = Ls[j, j]; adj = abs(dj)
                if adj <= zeta * max(dmax, delta)
                    nz += 1
                elseif dj > zero(T)
                    np += 1
                else
                    nn += 1
                end
                sg = sgn[j]
                wrongsign = (sg == Int8(1) && !(dj > zero(T))) ||
                            (sg == Int8(-1) && !(dj < zero(T)))
                if wrongsign || adj < delta
                    tgt = sg == Int8(0) ? (signbit(dj) ? -one(T) : one(T)) : T(sg)
                    newd = tgt * max(delta, adj); npert += 1
                    pert = abs(newd - dj); pert > maxp && (maxp = pert)
                    dj = newd
                end
                Ds[j] = dj; Dinv[j] = one(T) / dj
                adf = abs(dj); adf > dmax && (dmax = adf)
                Ls[j, j] = one(T)
            end
        end
        @synchronize
        djv = Ds[j]
        r2 = j + li
        @inbounds if r2 <= n
            lv = Ls[r2, j] * Dinv[j]
            Ls[r2, j] = lv
            Wc[r2] = djv * lv
        end
        @synchronize
        q = li
        @inbounds while q <= 4096
            rr = j + ((q - 1) & 63) + 1
            cc = j + ((q - 1) >> 6) + 1
            if rr <= n && cc <= n && rr >= cc
                Ls[rr, cc] = muladd(-Wc[cc], Ls[rr, j], Ls[rr, cc])
            end
            q += 256
        end
        @synchronize
    end
    @inbounds if gi == 1 && li == 1
        stats[1] += T(np); stats[2] += T(nn); stats[3] += T(nz); stats[4] += T(npert)
        maxp > stats[5] && (stats[5] = maxp)
        stats[6] = dmax
    end
    @inbounds if gi == 1 && li <= n
        dv[li] = Ds[li]
    end
    # strict upper ← D-scaled transposed factor: Ls[k,j] = L11[j,k]·d_k (writes upper,
    # reads lower + Ds — disjoint cells, no barrier needed within the pass)
    q = li
    @inbounds while q <= 4096
        r = ((q - 1) & 63) + 1
        c = ((q - 1) >> 6) + 1
        if r < c && c <= n
            Ls[r, c] = Ls[c, r] * Ds[r]
        end
        q += 256
    end
    @synchronize
    i = (gi - 1) * 256 + li
    if i <= m
        @inbounds for j in 1:n
            s = B[i, j]
            for k in 1:(j - 1)
                s = muladd(-B[i, k], Ls[k, j], s)
            end
            B[i, j] = s * Dinv[j]
        end
    end
    @synchronize
    if li == 1
        CUDA.threadfence()
        old = CUDA.atomic_add!(pointer(cnt), Int64(1))
        lastf[1] = old == target - 1 ? Int32(1) : Int32(0)
    end
    @synchronize
    if lastf[1] == Int32(1)
        q = li
        @inbounds while q <= 4096
            r = ((q - 1) & 63) + 1
            c = ((q - 1) >> 6) + 1
            if r <= n && c <= n && r >= c
                A[r, c] = Ls[r, c]
            end
            q += 256
        end
    end
end

# D-SCALED trapezoidal syrk: C[M×N] += α·(A·diag(d))·Ãᵀ, Ã = first N rows of A[M×K],
# keeping gr ≥ gc — the _syrk_trap_4x4! of ext/gpu_dense.jl with the block's signed D
# folded into the A-side shared load (one extra multiply per staged element). This is the
# right-looking trailing update U −= L̂·D̂·L̂ᵀ restricted to the remaining panel columns.
@kernel unsafe_indices = true function _syrk_trap_d_4x4!(C, @Const(A), @Const(d), alpha,
                                                         beta, M, N, K)
    T = eltype(C)
    li = @index(Local, NTuple)
    gi = @index(Group, NTuple)
    tx = li[1]; ty = li[2]
    tid = (ty - 1) * 16 + (tx - 1)
    br = (gi[1] - 1) * 64
    bc = (gi[2] - 1) * 64
    As = @localmem T (64, 8)
    Bs = @localmem T (64, 8)
    acc = @private T (4, 4)
    if br + 64 > bc                       # group-uniform: skip tiles strictly above diagonal
        @inbounds for i in 1:4, j in 1:4
            acc[i, j] = zero(T)
        end
        k0 = 0
        nt = div(K + 7, 8)
        for _ in 1:nt
            @inbounds for t in 1:2
                p = tid + (t - 1) * 256
                ml = p & 63; kl = p >> 6
                gr = br + ml; gk = k0 + kl
                As[ml + 1, kl + 1] = (gr < M && gk < K) ? A[gr + 1, gk + 1] * d[gk + 1] :
                                     zero(T)
                gc = bc + ml
                Bs[ml + 1, kl + 1] = (gc < N && gk < K) ? A[gc + 1, gk + 1] : zero(T)
            end
            @synchronize
            @inbounds for kk in 1:8
                @nexprs 4 i -> (a_i = As[(tx - 1) * 4 + i, kk])
                @nexprs 4 j -> (b_j = Bs[(ty - 1) * 4 + j, kk])
                @nexprs 4 i -> @nexprs 4 j -> (acc[i, j] = muladd(a_i, b_j, acc[i, j]))
            end
            @synchronize
            k0 += 8
        end
        @inbounds for i in 1:4, j in 1:4
            gr = br + (tx - 1) * 4 + (i - 1)
            gc = bc + (ty - 1) * 4 + (j - 1)
            if gr < M && gc < N && gr >= gc
                C[gr + 1, gc + 1] = beta == zero(T) ? alpha * acc[i, j] :
                                    muladd(alpha, acc[i, j], beta * C[gr + 1, gc + 1])
            end
        end
    end
end

function gpu_syrk_trap_d!(C, A, d, N, alpha, beta)
    M, K = size(A)
    backend = get_backend(C)
    kern = _syrk_trap_d_4x4!(backend, (16, 16))
    kern(C, A, d, alpha, beta, M, N, K; ndrange = (cld(M, 64) * 16, cld(N, 64) * 16))
    return C
end

# Vendor arm's device column scale (copied verbatim from ext/gpu_numeric.jl _col_scale!).
@kernel function _col_scale!(out, @Const(inp), @Const(dvec), base, invflag, m, n)
    idx = @index(Global)
    if idx ≤ m * n
        i = (idx - 1) % m + 1; j = (idx - 1) ÷ m + 1
        @inbounds begin
            d = dvec[base + j]
            out[i, j] = inp[i, j] * (invflag ? inv(d) : d)
        end
    end
end

# =======================================================================================
# Driver
# =======================================================================================

"""
Caller-provided workspace: `cnt` + `arrivals` (device Int64 arrival counter + host mirror,
MONOTONIC — same pattern as FrontWS), `stats` (Float64[6] device buffer:
[n_pos, n_neg, n_zero, n_perturbed, max_pert, dmax_carry] — counts accumulate across
calls, zero slots 1:5 per factorization and read back once at the end; slot 6 is the
per-front running dmax, reset inside the driver via the firstblk flag). Zero device-pool
allocation per driver call.
"""
mutable struct LDLFrontWS{TC, TS}
    cnt::TC
    stats::TS
    arrivals::Int64
end

LDLFrontWS(::Type{T}) where {T} = LDLFrontWS(CUDA.zeros(Int64, 1), CUDA.zeros(T, 6), 0)

# v1/v2 crossover — MEASURED (galen, this file's bench, post Wc-hoist): same window as
# the Cholesky front's FUSE_M_MAX. v2 (invert + register-tiled multiply) wins for panel
# rows ≲ 2500; above that its per-group redundant factor+invert rivals the per-group
# solve work and the v1 row-solve (cld(m,256) groups) is faster: (615,4231)/(859,3587)
# → v1 8.87/14.5 ms vs v2 10.1/16.2 ms; every other mix shape → v2 wins by 3–6%.
const LDL_FUSE_M_MAX = Ref(2500)

"""
    gpu_ldlt_front!(P, nscol, sgn, dv, delta, zeta, ws; nb=64, mode=:auto) -> P

Whole signed-LDL GPU front in device kernels only: factors `P[1:nscol,1:nscol]` to
unit-lower L11 (1's stored on the diagonal) with fixed-pivot signed regularization
(`sgn`: Int8 ∈ {+1,−1,0} per column, front-local; `delta`,`zeta` as in
cpu_multifrontal_ldlt!), writes signed D to `dv` (front-local device vector), solves
`P[nscol+1:end,1:nscol]` to L21, and accumulates inertia stats into `ws.stats`.
Right-looking over 64-wide panels: [fused factor+solve, 1 launch] → [D-scaled trapezoidal
trailing update, 1 launch]. The caller does the U_s Schur update separately.
`mode`: `:fused` (v1 row-solve) | `:fused2` (v2 block-inverse multiply) | `:auto`
(v2 iff panel rows ≤ LDL_FUSE_M_MAX[]).
"""
function gpu_ldlt_front!(P, nscol::Int, sgn, dv, delta, zeta, ws::LDLFrontWS;
                         nb::Int = 64, mode::Symbol = :auto)
    nsrow = size(P, 1)
    n = nscol
    @assert size(P, 2) == n && nsrow >= n
    n == 0 && return P
    T = eltype(P)
    backend = get_backend(P)
    f1 = _ldl_front_fused64_v1!(backend, 256)
    f2 = _ldl_front_fused64_v2!(backend, (16, 16))
    for j0 in 1:nb:n
        jb = min(nb, n - j0 + 1)
        j1 = j0 + jb - 1
        m = nsrow - j1
        D = view(P, j0:j1, j0:j1)
        Bv = view(P, (j1 + 1):nsrow, j0:j1)
        sv = view(sgn, j0:j1)
        dvv = view(dv, j0:j1)
        fb = j0 == 1 ? Int32(1) : Int32(0)            # reset the per-front dmax carry
        md = mode == :auto ? (m <= LDL_FUSE_M_MAX[] ? :fused2 : :fused) : mode
        if md == :fused2 || m == 0
            G = max(cld(m, 64), 1)
            f2(D, Bv, jb, m, sv, dvv, ws.stats, T(delta), T(zeta), fb, ws.cnt,
               ws.arrivals + G; ndrange = (G * 16, 16))
            ws.arrivals += G
        else                                          # :fused (v1)
            G = max(cld(m, 256), 1)
            f1(D, Bv, jb, m, sv, dvv, ws.stats, T(delta), T(zeta), fb, ws.cnt,
               ws.arrivals + G; ndrange = G * 256)
            ws.arrivals += G
        end
        if j1 < n                         # trailing trapezoid: rows j1+1..nsrow, cols j1+1..n
            C = view(P, (j1 + 1):nsrow, (j1 + 1):n)
            A2 = view(P, (j1 + 1):nsrow, j0:j1)
            gpu_syrk_trap_d!(C, A2, dvv, n - j1, -one(T), one(T))
        end
    end
    return P
end

# =======================================================================================
# CPU oracle — the reference column loop of cpu_multifrontal_ldlt! (ext/multifrontal.jl
# lines 202–224), dense, verbatim semantics.
# =======================================================================================

function cpu_ldl_front!(P::Matrix{Float64}, signs::Vector{Int8}, delta::Float64,
                        zeta::Float64)
    nsrow, nscol = size(P)
    dvec = zeros(nscol)
    np = 0; nn = 0; nz = 0; npert = 0; maxp = 0.0; dmax = 0.0
    @inbounds for j in 1:nscol
        dj = P[j, j]; adj = abs(dj)
        if adj <= zeta * max(dmax, delta)
            nz += 1
        elseif dj > 0.0
            np += 1
        else
            nn += 1
        end
        sg = signs[j]
        wrongsign = (sg == Int8(1) && !(dj > 0.0)) || (sg == Int8(-1) && !(dj < 0.0))
        if wrongsign || adj < delta
            tgt = sg == Int8(0) ? (signbit(dj) ? -1.0 : 1.0) : Float64(sg)
            newd = tgt * max(delta, adj); npert += 1
            p = abs(newd - dj); p > maxp && (maxp = p); dj = newd
        end
        dvec[j] = dj; adf = abs(dj); adf > dmax && (dmax = adf)
        P[j, j] = 1.0; invd = 1.0 / dj
        for i in (j + 1):nsrow
            P[i, j] *= invd
        end
        if j < nscol                      # ger!(-dj, lcol, lrow, trail) on the FULL trail
            for cc in (j + 1):nscol
                a = -dj * P[cc, j]
                for rr in (j + 1):nsrow
                    P[rr, cc] = muladd(a, P[rr, j], P[rr, cc])
                end
            end
        end
    end
    return dvec, (np, nn, nz, npert, maxp)
end

# Vendor arm's CPU diag factor — verbatim port of _ldl_block! (ext/multifrontal.jl),
# including its PureSparse.ger! dense rank-1 (the actual op the shipped path pays for).
function ldl_block_cpu!(block::AbstractMatrix{T}, sg, delta::T, zeta::T) where {T}
    nscol = size(block, 1); dvals = Vector{T}(undef, nscol)
    np = 0; nn = 0; nz = 0; npert = 0; maxp = 0.0; dmax = zero(T)
    @inbounds for j in 1:nscol
        dj = block[j, j]; adj = abs(dj)
        if adj <= zeta * max(dmax, delta); nz += 1
        elseif dj > zero(T); np += 1
        else; nn += 1 end
        s = sg[j]
        wrong = (s == Int8(1) && !(dj > zero(T))) || (s == Int8(-1) && !(dj < zero(T)))
        if wrong || adj < delta
            target = s == Int8(0) ? (signbit(dj) ? -one(T) : one(T)) : T(s)
            newd = target * max(delta, adj); npert += 1
            p = Float64(abs(newd - dj)); p > maxp && (maxp = p); dj = newd
        end
        dvals[j] = dj; ad = abs(dj); ad > dmax && (dmax = ad)
        block[j, j] = one(T); invd = inv(dj)
        for i in (j + 1):nscol; block[i, j] *= invd; end
        if j < nscol
            lc = view(block, (j + 1):nscol, j); tr = view(block, (j + 1):nscol, (j + 1):nscol)
            PureSparse.ger!(-dj, lc, lc, tr)
        end
    end
    return dvals, np, nn, nz, npert, maxp
end

# =======================================================================================
# Path test — exact-arithmetic coverage of the pivot paths random SQD blocks can't hit
# deterministically: exact-zero pivot (n_zero), tiny pivot < delta, wrong-sign flip,
# sg==0 column, and the dmax carry ACROSS 64-blocks (col 65's zero-classification flips
# if the carry is broken). A purely DIAGONAL A11 makes every pivot exactly its input
# value (all rank-1 updates are ×0 ⇒ exact), so CPU/GPU must agree bit-for-bit.
# =======================================================================================

function path_test(mode::Symbol, ws::LDLFrontWS)
    n = 100; below = 200; nsrow = n + below
    dl = ones(n); sg = Int8[k % 3 == 0 ? -1 : 1 for k in 1:n]
    for k in 1:n; dl[k] = sg[k] == Int8(1) ? 1.0 : -1.0; end
    dl[1] = 1.0e6;  sg[1] = Int8(1)       # sets dmax carry for block 2
    dl[2] = -2.0;   sg[2] = Int8(-1)
    dl[5] = 0.0;    sg[5] = Int8(1)       # exact zero → n_zero + perturbed (wrongsign)
    dl[7] = 1.0e-14; sg[7] = Int8(1)      # ≤ zeta·dmax → n_zero; < delta → perturbed
    dl[65] = 1.0e-11; sg[65] = Int8(1)    # zero-class ONLY via carried dmax (block 2!)
    dl[66] = -3.0;  sg[66] = Int8(1)      # wrong sign → perturbed (raw counts n_neg)
    dl[70] = -0.5;  sg[70] = Int8(0)      # sg==0: no perturbation, counts by value
    P = zeros(nsrow, n)
    for k in 1:n; P[k, k] = dl[k]; end
    P[(n + 1):nsrow, :] .= randn(below, n)
    delta = PureSparse.LDLT_DELTA * 1.0e6; zeta = eps(Float64)
    Pref = copy(P)
    dref, sref = cpu_ldl_front!(Pref, sg, delta, zeta)
    @assert sref[3] == 3 "path test setup: expected n_zero == 3, got $(sref[3])"
    @assert sref[4] == 4 "path test setup: expected n_perturbed == 4, got $(sref[4])"
    dP = CuArray(P); d_sg = CuArray(sg); d_dv = CUDA.zeros(Float64, n)
    CUDA.fill!(ws.stats, 0.0)
    gpu_ldlt_front!(dP, n, d_sg, d_dv, delta, zeta, ws; mode)
    CUDA.synchronize()
    hp = Array(dP); hd = Array(d_dv); st = Array(ws.stats)
    for j in 2:n, i in 1:(j - 1)
        hp[i, j] = 0.0; Pref[i, j] = 0.0
    end
    relL = norm(hp - Pref) / norm(Pref)
    relD = norm(hd - dref) / norm(dref)
    inert = (Int(st[1]), Int(st[2]), Int(st[3]))
    ok = relL <= 1e-15 && relD <= 1e-15 && inert == (sref[1], sref[2], sref[3]) &&
         Int(st[4]) == sref[4] && abs(st[5] - sref[5]) <= 1e-9 * max(1.0, sref[5])
    @printf("path test mode=%-7s relL=%.1e relD=%.1e inertia=%s ref=%s npert=%d/%d  %s\n",
            mode, relL, relD, inert, (sref[1], sref[2], sref[3]), Int(st[4]), sref[4],
            ok ? "PASS" : "FAIL")
    foreach(CUDA.unsafe_free!, (dP, d_sg, d_dv))
    return ok
end

# =======================================================================================
# Validation + benchmark on the REAL KKT crown-front shapes
# =======================================================================================

const SHAPES = [(55, 754), (152, 671), (234, 1436), (314, 1464),
                (615, 4231), (859, 3587), (1064, 1273)]
# Count weights: the measured 44³-KKT bucket counts (75/53/19/4/2 by nscol bucket, cf.
# pure_potrf_opt.jl BUCKETS), split evenly across this file's representatives per bucket.
const COUNTW = Dict((55, 754) => 75.0, (152, 671) => 53.0, (234, 1436) => 9.5,
                    (314, 1464) => 9.5, (615, 4231) => 2.0, (859, 3587) => 2.0,
                    (1064, 1273) => 2.0)

front_flops(nscol, below) = nscol^3 / 3 + Float64(below) * nscol^2

med(v) = sort(v)[cld(length(v), 2)]

# Well-scaled SQD block [H Aᵀ; A −D] + tall panel; signs = [+1…; −1…] with (on selected
# shapes) one deliberate wrong-sign flip (forces the regularization path) and one sg=0.
function make_kkt_front(nscol, below; flip::Bool = false)
    nh = round(Int, 0.65 * nscol); nd = nscol - nh
    W = randn(nh, nh)
    H = W * W' / nh + 2.0 * I
    Ac = randn(nd, nh) / sqrt(nh)
    Dd = Diagonal(1.0 .+ rand(nd))
    A11 = [Matrix(H) Ac'; Ac -Matrix(Dd)]
    P = zeros(nscol + below, nscol)
    P[1:nscol, 1:nscol] .= A11
    P[(nscol + 1):end, :] .= randn(below, nscol) ./ sqrt(nscol)
    signs = vcat(fill(Int8(1), nh), fill(Int8(-1), nd))
    if flip
        signs[3] = Int8(-1)               # wrong sign in the H part → forced perturbation
        signs[10] = Int8(0)               # sg==0 column
    end
    return P, signs
end

function timeit(op!, dst, src; reps = 9)
    copyto!(dst, src); CUDA.synchronize()
    op!(dst); CUDA.synchronize()          # warm
    ts = Float64[]
    for _ in 1:reps
        copyto!(dst, src); CUDA.synchronize()
        t0 = time_ns()
        op!(dst); CUDA.synchronize()
        push!(ts, (time_ns() - t0) / 1e9)
    end
    return med(ts)
end

function bench(ws::LDLFrontWS)
    T = Float64
    zeta = eps(Float64)
    @printf("\n== fused pure signed-LDL front vs vendor (CPU _ldl_block! D2H/H2D + cuBLAS trsm + colscale) ==\n")
    @printf("%6s %6s | %7s | %9s %9s | %9s %9s %9s | %8s %8s\n",
            "nscol", "below", "GFlop", "vendor", "venddiag", "fused", "fused2", "auto",
            "vs vend", "(auto)")
    tim = Dict{Tuple{Int, Int}, Dict{Symbol, Float64}}()
    allpass = true
    valrows = String[]
    for (nscol, below) in SHAPES
        nsrow = nscol + below
        flip = (nscol, below) in ((55, 754), (152, 671), (615, 4231))
        P, signs = make_kkt_front(nscol, below; flip)
        delta = PureSparse.LDLT_DELTA * maximum(abs, P)
        # oracle
        Pref = copy(P)
        dref, sref = cpu_ldl_front!(Pref, signs, delta, zeta)
        maskP = copy(Pref)
        for j in 2:nscol, i in 1:(j - 1); maskP[i, j] = 0.0; end
        nrmL = norm(maskP); nrmD = norm(dref)
        src = CuArray{T}(P); dP = similar(src); dV = similar(src)
        d_sg = CuArray(signs); d_dv = CUDA.zeros(T, nscol)
        backend = get_backend(dP)
        # --- arms ---
        pure_auto! = X -> gpu_ldlt_front!(X, nscol, d_sg, d_dv, delta, zeta, ws; mode = :auto)
        pure_f1! = X -> gpu_ldlt_front!(X, nscol, d_sg, d_dv, delta, zeta, ws; mode = :fused)
        pure_f2! = X -> gpu_ldlt_front!(X, nscol, d_sg, d_dv, delta, zeta, ws; mode = :fused2)
        vend_diag! = X -> begin           # just the CPU diag round-trip piece
            blk_dev = view(X, 1:nscol, 1:nscol)
            blk_h = Array(blk_dev)
            dvals, _, _, _, _, _ = ldl_block_cpu!(blk_h, signs, delta, zeta)
            copyto!(blk_dev, blk_h)
            copyto!(d_dv, 1, dvals, 1, nscol)
        end
        vend! = X -> begin                # the full vendor LDLᵀ front it replaces
            vend_diag!(X)
            if below > 0
                L21 = view(X, (nscol + 1):nsrow, 1:nscol)
                trsm!('R', 'L', 'T', 'U', one(T), view(X, 1:nscol, 1:nscol), L21)
                nd = cld(below * nscol, 256) * 256
                _col_scale!(backend, 256)(L21, L21, d_dv, 0, true, below, nscol;
                                          ndrange = nd)
            end
        end
        # --- validation (gate: relL ≤ 1e-9, relD ≤ 1e-9, inertia EXACT) ---
        function validate(op!, name)
            copyto!(dP, src); CUDA.fill!(ws.stats, 0.0)
            op!(dP); CUDA.synchronize()
            hp = Array(dP); hd = Array(d_dv)
            for j in 2:nscol, i in 1:(j - 1); hp[i, j] = 0.0; end
            relL = norm(hp - maskP) / nrmL
            relD = norm(hd - dref) / nrmD
            if name == :vendor            # vendor inertia comes from its CPU block
                copyto!(dV, src)
                blk_h = Array(view(dV, 1:nscol, 1:nscol))
                _, vnp, vnn, vnz, vnpe, vmp = ldl_block_cpu!(blk_h, signs, delta, zeta)
                inert = (vnp, vnn, vnz); npe = vnpe; mp = vmp
            else
                st = Array(ws.stats)
                inert = (Int(st[1]), Int(st[2]), Int(st[3])); npe = Int(st[4]); mp = st[5]
            end
            iok = inert == (sref[1], sref[2], sref[3]) && npe == sref[4]
            mok = abs(mp - sref[5]) <= 1e-9 * max(1.0, sref[5])
            pass = relL <= 1e-9 && relD <= 1e-9 && iok && mok
            push!(valrows,
                  @sprintf("%6d %6d %-7s relL=%.2e relD=%.2e inertia=%s ref=%s npert=%d/%d maxpert=%.3e/%.3e  %s",
                           nscol, below, name, relL, relD, inert,
                           (sref[1], sref[2], sref[3]), npe, sref[4], mp, sref[5],
                           pass ? "PASS" : "FAIL"))
            return pass
        end
        for (op!, nm) in ((pure_auto!, :auto), (pure_f1!, :fused), (pure_f2!, :fused2),
                          (vend!, :vendor))
            allpass &= validate(op!, nm)
        end
        # --- timing ---
        tm = Dict{Symbol, Float64}()
        tm[:vendor] = timeit(vend!, dV, src)
        tm[:venddiag] = timeit(vend_diag!, dV, src)
        tm[:fused] = timeit(pure_f1!, dP, src)
        tm[:fused2] = timeit(pure_f2!, dP, src)
        tm[:auto] = timeit(pure_auto!, dP, src)
        tim[(nscol, below)] = tm
        @printf("%6d %6d | %7.3f | %9.1f %9.1f | %9.1f %9.1f %9.1f | %7.2fx\n",
                nscol, below, front_flops(nscol, below) / 1e9,
                tm[:vendor] * 1e6, tm[:venddiag] * 1e6,
                tm[:fused] * 1e6, tm[:fused2] * 1e6, tm[:auto] * 1e6,
                tm[:vendor] / tm[:auto])
        foreach(CUDA.unsafe_free!, (src, dP, dV, d_sg, d_dv))
    end
    @printf("\n== validation vs cpu_multifrontal_ldlt! column loop (gates: relL ≤ 1e-9, relD ≤ 1e-9, inertia EXACT) ==\n")
    foreach(println, valrows)
    println(allpass ? "ALL PASS" : "SOME FAIL")
    # aggregates
    @printf("\n== aggregates over the shape mix ==\n")
    for (wname, w, s) in (("count-weighted (44³ bucket counts)", sh -> COUNTW[sh], 1.0),
                          ("flop-weighted (normalized)", sh -> front_flops(sh...),
                           sum(front_flops(sh...) for sh in SHAPES) /
                           sum(COUNTW[sh] for sh in SHAPES)))
        tv = sum(w(sh) * tim[sh][:vendor] for sh in SHAPES) / s
        ta = sum(w(sh) * tim[sh][:auto] for sh in SHAPES) / s
        @printf("%-36s vendor=%9.3f ms  pure(auto)=%9.3f ms  speedup=%5.2fx\n",
                wname, tv * 1e3, ta * 1e3, tv / ta)
    end
    return allpass
end

# Zero device-pool allocation per warm call (scratch caller-provided, kernels @localmem/
# @private only).
function check_zero_alloc(ws::LDLFrontWS)
    nscol, below = 234, 1436
    nsrow = nscol + below
    P, signs = make_kkt_front(nscol, below)
    delta = PureSparse.LDLT_DELTA * maximum(abs, P); zeta = eps(Float64)
    src = CuArray{Float64}(P); dP = similar(src)
    d_sg = CuArray(signs); d_dv = CUDA.zeros(Float64, nscol)
    copyto!(dP, src)
    gpu_ldlt_front!(dP, nscol, d_sg, d_dv, delta, zeta, ws); CUDA.synchronize()
    copyto!(dP, src)
    b = CUDA.@allocated begin
        gpu_ldlt_front!(dP, nscol, d_sg, d_dv, delta, zeta, ws)
        CUDA.synchronize()
    end
    println("\ndevice-pool bytes allocated by warm gpu_ldlt_front!(mode=:auto): ", b)
    @assert b == 0
    foreach(CUDA.unsafe_free!, (src, dP, d_sg, d_dv))
    return nothing
end

function main()
    Random.seed!(20260717)
    println("GPU: ", CUDA.name(CUDA.device()), "   CUDA ", CUDA.runtime_version())
    println("LDL_FUSE_M_MAX = ", LDL_FUSE_M_MAX[], "   LDLT_DELTA = ", PureSparse.LDLT_DELTA)
    ws = LDLFrontWS(Float64)
    ok = path_test(:fused2, ws) & path_test(:fused, ws) & path_test(:auto, ws)
    ok &= bench(ws)
    check_zero_alloc(ws)
    println(ok ? "\nALL GATES PASS" : "\nGATE FAILURES — see above")
    println("done.")
end

main()
