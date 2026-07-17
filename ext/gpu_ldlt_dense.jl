# ext/gpu_ldlt_dense.jl — pure device FUSED SIGNED-LDL front (amendment C portability + the
# LDLᵀ analogue of gpu_front!). Fable-authored + galen-validated: matches cpu_multifrontal_ldlt!
# at relL/relD ≤ 1e-15 with EXACT inertia across the real KKT crown shapes + sign patterns;
# flop-weighted 4.42× vs the CPU-diag+cuBLAS-trsm vendor front (removes the nscol³ host round-
# trip). Full harness: benchmark/gpu/pure_ldlt_front.jl. Reuses gpu_gemm_nt! + _col_scale! from
# the ext. Included after gpu_dense.jl.

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
