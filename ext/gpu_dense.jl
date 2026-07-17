# ext/gpu_dense.jl — pure device dense kernels for the multifrontal GPU fronts (amendment C:
# portable, no cuSOLVER/cuBLAS). Fable-authored + galen-validated (≤2.3e-16 vs the vendor pair
# on the real crown-front shape mix; weighted aggregate 1.13× vs cuSOLVER/cuBLAS). Full
# validation harness + per-shape numbers: benchmark/gpu/pure_potrf_opt.jl. Reuses gpu_gemm_nt!
# from PureSparseCUDAExt.jl. Included by the ext after the gemm kernel is defined.

# a front's trsm-updates ride the potrf trailing update in one launch (right-looking
# partial Cholesky, G&VL §4.2). Group-uniform skip of tiles strictly above the diagonal.
@kernel unsafe_indices = true function _syrk_trap_4x4!(C, @Const(A), alpha, beta, M, N, K)
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
                As[ml + 1, kl + 1] = (gr < M && gk < K) ? A[gr + 1, gk + 1] : zero(T)
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

function gpu_syrk_trap!(C, A, N, alpha, beta)
    M, K = size(A)
    backend = get_backend(C)
    kern = _syrk_trap_4x4!(backend, (16, 16))
    kern(C, A, alpha, beta, M, N, K; ndrange = (cld(M, 64) * 16, cld(N, 64) * 16))
    return C
end

gpu_syrk_ln!(C, A, alpha, beta) = gpu_syrk_trap!(C, A, size(C, 1), alpha, beta)

# Dev base Cholesky (32-wide, one (32,32) workgroup) — kept for the OLD-driver baseline arm.
@kernel unsafe_indices = true function _potrf_base32!(A, kb, info, j0)
    T = eltype(A)
    li = @index(Local, NTuple)
    tx = li[1]; ty = li[2]
    As = @localmem T (32, 32)
    piv = @localmem T (1,)
    inb = tx <= kb && ty <= kb
    @inbounds if inb
        As[tx, ty] = tx >= ty ? A[tx, ty] : zero(T)
    end
    @synchronize
    for j in 1:kb
        if tx == j && ty == j
            @inbounds begin
                d = As[j, j]
                if !(d > zero(T)) && info[1] == Int32(0)
                    info[1] = Int32(j0 + j - 1)
                end
                s = sqrt(d)
                As[j, j] = s
                piv[1] = one(T) / s
            end
        end
        @synchronize
        if ty == j && tx > j && tx <= kb
            @inbounds As[tx, j] *= piv[1]
        end
        @synchronize
        if ty > j && tx >= ty && tx <= kb
            @inbounds As[tx, ty] = muladd(-As[tx, j], As[ty, j], As[tx, ty])
        end
        @synchronize
    end
    @inbounds if inb && tx >= ty
        A[tx, ty] = As[tx, ty]
    end
end

# Dev base right-lower-transpose TRSM (1 thread per row, kb ≤ 64) — kept both for the OLD
# arm and as the :split_base panel-solve variant in the sweep.
@kernel unsafe_indices = true function _trsm_rlt_base!(B, @Const(L), m, kb)
    T = eltype(B)
    li = @index(Local, Linear)
    gi = @index(Group, Linear)
    Ls = @localmem T (64, 64)
    Ld = @localmem T (64,)
    p = li
    @inbounds while p <= kb * kb
        r = (p - 1) % kb + 1
        c = (p - 1) ÷ kb + 1
        Ls[r, c] = r >= c ? L[r, c] : zero(T)
        p += 256
    end
    @inbounds if li <= kb
        Ld[li] = one(T) / L[li, li]
    end
    @synchronize
    i = (gi - 1) * 256 + li
    if i <= m
        @inbounds for j in 1:kb
            s = B[i, j]
            for k in 1:(j - 1)
                s = muladd(-B[i, k], Ls[j, k], s)
            end
            B[i, j] = s * Ld[j]
        end
    end
end

# =======================================================================================
# NEW kernels
# =======================================================================================

# FUSED front-panel kernel: potrf(A[n×n]) + B := B·L⁻ᵀ (B m×n) in ONE launch, n ≤ 64.
# Every workgroup loads lower(A) into shared and redundantly runs the in-shared
# right-looking Cholesky (G&VL 4.2.1) — the redundancy (≤ 64³/3 flops, groups run on
# different SMs concurrently) buys grid-wide availability of L without a second launch.
# Each group then solves its 256-row slice of B against the shared factor (rows of
# X·Lᵀ = B are independent; same-thread B[i,k] read-back hits L2, cf. _trsm_rlt_base!).
# Write-back of the factored diagonal goes to the DISJOINT scratch Dout (group 1 writes
# it; lower = L, strict upper = 0) — A is never written, so no group's load of the
# UNfactored A can race it and no cross-group ordering is needed. (Replaces the old
# last-arriving-group election via atomic counter: gfx1151's compiler segfaults on a
# used atomic-rmw return value, and the scratch removes the mechanism entirely.) The
# driver copies Dout into the panel after the launch (stream-ordered).
@kernel unsafe_indices = true function _front_fused64!(A, B, n, m, info, Dout, j0)
    T = eltype(A)
    li = @index(Local, Linear)            # 1..256
    gi = @index(Group, Linear)
    Ls = @localmem T (64, 64)
    Ldi = @localmem T (64,)               # 1/L[j,j] — reciprocal once, FP64 div is slow
    p = li
    @inbounds while p <= n * n
        r = (p - 1) % n + 1
        c = (p - 1) ÷ n + 1
        Ls[r, c] = r >= c ? A[r, c] : zero(T)
        p += 256
    end
    @synchronize
    for j in 1:n
        if li == 1
            @inbounds begin
                d = Ls[j, j]
                if gi == 1 && !(d > zero(T)) && info[1] == Int32(0)
                    info[1] = Int32(j0 + j - 1)
                end
                s = sqrt(d)
                Ls[j, j] = s
                Ldi[j] = one(T) / s
            end
        end
        @synchronize
        r2 = j + li
        @inbounds if r2 <= n
            Ls[r2, j] *= Ldi[j]
        end
        @synchronize
        t = n - j
        q = li
        @inbounds while q <= t * t
            rr = (q - 1) % t + 1 + j
            cc = (q - 1) ÷ t + 1 + j
            if rr >= cc
                Ls[rr, cc] = muladd(-Ls[rr, j], Ls[cc, j], Ls[rr, cc])
            end
            q += 256
        end
        @synchronize
    end
    if gi == 1                            # group 1 has the full factor ⇒ write it to scratch
        p = li
        @inbounds while p <= n * n
            r = (p - 1) % n + 1
            c = (p - 1) ÷ n + 1
            Dout[r, c] = r >= c ? Ls[r, c] : zero(T)
            p += 256
        end
    end
    i = (gi - 1) * 256 + li
    if i <= m
        @inbounds for j in 1:n
            s = B[i, j]
            for k in 1:(j - 1)
                s = muladd(-B[i, k], Ls[j, k], s)
            end
            B[i, j] = s * Ldi[j]
        end
    end
end

# FUSED front-panel kernel v2 — same contract as _front_fused64! but restructured for the
# measured bottlenecks of v1 (v1 at (63,790): ~200μs of its 341μs is the 1-thread-per-row
# solve running on only cld(m,256)=4 SMs at 2 FP64 FMA/cycle/SM, and the factor loop's
# runtime-divisor integer %/÷ (no HW idiv on sm_89) burns ~100 cycles per index):
#   • all indexing in bit ops on a fixed 64-padded tile (zero-filled),
#   • after the redundant in-shared factor, each group also inverts L in shared (textbook
#     per-column forward substitution; column c of L⁻¹ stored TRANSPOSED in row c of the
#     STRICT UPPER triangle of the same 64×64 tile + diag in invDiag — fits the 48 KB
#     static-shared budget, no second tile),
#   • the panel solve becomes the dependency-free multiply y[i,j] = Σ_{k≤j} x[i,k]·invL[j,k]
#     (MAGMA-style block-inverse trick), register-tiled 4×4 per thread over 64-row tiles ⇒
#     cld(m,64) groups (4× the SMs of v1) and gemm-grade per-thread throughput,
#   • in-place safe: each group stages only its OWN 64 rows of B through shared before any
#     write; diag write-back by group 1 into the disjoint scratch Dout (see _front_fused64!).
@kernel unsafe_indices = true function _front_fused64_v2!(A, B, n, m, info, Dout, j0)
    T = eltype(A)
    li = @index(Local, NTuple)            # (16,16)
    gi = @index(Group, Linear)
    tx = li[1]; ty = li[2]
    tid = (ty - 1) * 16 + tx              # 1..256
    Ls = @localmem T (64, 64)
    Ldi = @localmem T (64,)               # 1/L[j,j]
    invDiag = @localmem T (64,)           # diag of L⁻¹ (== Ldi, kept separate for clarity)
    As = @localmem T (64, 8)
    # zero-padded staging of lower(A) into the fixed 64×64 tile (bit-ops only)
    q = tid
    @inbounds while q <= 4096
        r = ((q - 1) & 63) + 1
        c = ((q - 1) >> 6) + 1
        Ls[r, c] = (r <= n && c <= n && r >= c) ? A[r, c] : zero(T)
        q += 256
    end
    @synchronize
    # in-shared right-looking Cholesky (G&VL 4.2.1), redundant per group
    for j in 1:n
        if tid == 1
            @inbounds begin
                d = Ls[j, j]
                if gi == 1 && !(d > zero(T)) && info[1] == Int32(0)
                    info[1] = Int32(j0 + j - 1)
                end
                s = sqrt(d)
                Ls[j, j] = s
                Ldi[j] = one(T) / s
            end
        end
        @synchronize
        r2 = j + tid
        @inbounds if r2 <= n
            Ls[r2, j] *= Ldi[j]
        end
        @synchronize
        q = tid
        @inbounds while q <= 4096         # fixed 64×64 sweep of the trailing block, bit ops
            rr = j + ((q - 1) & 63) + 1
            cc = j + ((q - 1) >> 6) + 1
            if rr <= n && cc <= n && rr >= cc
                Ls[rr, cc] = muladd(-Ls[rr, j], Ls[cc, j], Ls[rr, cc])
            end
            q += 256
        end
        @synchronize
    end
    if gi == 1                            # group 1 → scratch (lower = L, rest 0); reads only
        q = tid                           # lower(Ls), disjoint from the strict-upper inversion
        @inbounds while q <= 4096
            r = ((q - 1) & 63) + 1
            c = ((q - 1) >> 6) + 1
            if r <= n && c <= n
                Dout[r, c] = r >= c ? Ls[r, c] : zero(T)
            end
            q += 256
        end
    end
    # invert L in shared: thread c computes column c of L⁻¹ into ROW c of the strict upper
    # triangle (reads only lower entries + its own already-written row ⇒ race-free, no sync)
    if tid <= n
        c = tid
        @inbounds begin
            xc = Ldi[c]
            invDiag[c] = xc
            for i2 in (c + 1):n
                s = Ls[i2, c] * xc
                for k in (c + 1):(i2 - 1)
                    s = muladd(Ls[i2, k], Ls[c, k], s)
                end
                Ls[c, i2] = -s * Ldi[i2]
            end
        end
    end
    @synchronize
    # panel multiply: this group's 64-row tile, y := x·L⁻ᵀ, gemm-style 4×4 register tiling
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
                w_j = k < col_j ? Ls[k, col_j] : (k == col_j ? invDiag[col_j] : zero(T))
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
            B[gr + 1, gc + 1] = acc[i, j]
        end
    end
end

# FUSED front-panel kernel v3 — same contract as _front_fused64! but fully REGISTER-
# RESIDENT (measured on galen: v2's in-shared factor sweep is shared-traffic bound at
# ~92µs vs its 18µs FMA-issue floor, the in-shared L-inversion adds ~120µs, and a
# row-serial B solve is 2-warp latency-bound at ~100µs regardless of m; probe:
# benchmark/gpu/pivot_variant_probe.jl + front_v3_check.jl):
#   • each thread of the (16,16) group owns TWO 4×4 register tiles: `acc` = its patch of
#     the (redundantly factored) 64×64 diag block, `bcc` = its patch of the group's OWN
#     64-row slice of B (rows (gi-1)*64+1..). All tile indices are compile-time literals
#     (@nexprs) — one runtime index demotes the tile to local memory (measured 3× slower),
#   • ONE right-looking rank-1 sweep (G&VL 4.2.1) drives factor AND solve: at step j the
#     owner threads scale+publish L's column j (colbuf) and the B-tile's column j
#     (colbufB = B[:,j]·(1/L[j,j])); every thread then updates its trailing A-cells
#     (−colbuf[r]·colbuf[c]) and its B-cells for c > j (−colbufB[r]·colbuf[c]) — the
#     solve reuses the very column the factor just published, so the panel solve costs
#     16 extra register FMAs per thread per step and zero extra barriers (2 per column),
#   • shared usage is 3 column buffers (~1.5 KB, no 64×64 tile) ⇒ several groups
#     co-resident per SM,
#   • diag write-back by group 1 into the disjoint scratch Dout (see _front_fused64!).
@kernel unsafe_indices = true function _front_fused64_v3!(A, B, n, m, info, Dout, j0)
    T = eltype(A)
    li = @index(Local, NTuple)            # (16,16)
    gi = @index(Group, Linear)
    tx = li[1]; ty = li[2]
    Ldi = @localmem T (64,)               # 1/L[j,j]
    colbuf = @localmem T (64,)            # column j of L (rows > j; zero-padded n<64)
    colbufB = @localmem T (64,)           # solved column j of this group's B tile
    acc = @private T (4, 4)               # diag-block patch (redundant per group)
    bcc = @private T (4, 4)               # B patch: rows br+1..br+64, cols 1..n
    br = (gi - 1) * 64
    @inbounds @nexprs 4 qq -> @nexprs 4 ii -> begin
        r = 4 * (tx - 1) + ii
        c = 4 * (ty - 1) + qq
        acc[ii, qq] = (r <= n && c <= n && r >= c) ? A[r, c] : zero(T)
        bcc[ii, qq] = (br + r <= m && c <= n) ? B[br + r, c] : zero(T)
    end
    for j in 1:n
        jt = ((j - 1) >> 2) + 1           # owning tile index of column j
        jr = ((j - 1) & 3) + 1            # local index of j inside its tile
        if tx == jt && ty == jt
            @inbounds @nexprs 4 qq -> begin
                if jr == qq
                    d = acc[qq, qq]
                    if gi == 1 && !(d > zero(T)) && info[1] == Int32(0)
                        info[1] = Int32(j0 + j - 1)
                    end
                    s = sqrt(d)
                    acc[qq, qq] = s
                    Ldi[j] = one(T) / s
                end
            end
        end
        @synchronize                      # Ldi[j] ready; also fences colbuf reuse from j-1
        if ty == jt                       # scale+publish column j of L and of B
            @inbounds begin
                rl = Ldi[j]
                @nexprs 4 qq -> begin
                    if jr == qq
                        @nexprs 4 ii -> begin
                            r = 4 * (tx - 1) + ii
                            if r > j
                                v = acc[ii, qq] * rl
                                acc[ii, qq] = v
                                colbuf[r] = v
                            end
                            vb = bcc[ii, qq] * rl
                            bcc[ii, qq] = vb
                            colbufB[r] = vb
                        end
                    end
                end
            end
        end
        @synchronize
        @inbounds @nexprs 4 qq -> begin   # rank-1 update of trailing A and B, registers
            c = 4 * (ty - 1) + qq
            if c > j
                bc = colbuf[c]
                @nexprs 4 ii -> begin
                    r = 4 * (tx - 1) + ii
                    if r >= c
                        acc[ii, qq] = muladd(-colbuf[r], bc, acc[ii, qq])
                    end
                    bcc[ii, qq] = muladd(-colbufB[r], bc, bcc[ii, qq])
                end
            end
        end
        # no barrier here: next step's pivot only writes Ldi[j+1] (fresh slot); the
        # barrier before the next scale phase fences the colbuf/colbufB reuse
    end
    if gi == 1                            # group 1 → scratch (lower = L, rest 0)
        @inbounds @nexprs 4 qq -> @nexprs 4 ii -> begin
            r = 4 * (tx - 1) + ii
            c = 4 * (ty - 1) + qq
            Dout[r, c] = r >= c ? acc[ii, qq] : zero(T)
        end
    end
    @inbounds @nexprs 4 qq -> @nexprs 4 ii -> begin
        r = br + 4 * (tx - 1) + ii
        c = 4 * (ty - 1) + qq
        if r <= m && c <= n
            B[r, c] = bcc[ii, qq]
        end
    end
end

# Single-group diag kernel: factor A[n×n] (n ≤ 64) in shared, write lower back (single
# group ⇒ no write-back hazard), and — if doinv==1 — also write L⁻¹ to invD (64×64 view):
# thread c forward-substitutes column c of L·X = I (textbook), zeros above the diagonal,
# so the panel solve becomes the in-place rank-n multiply `_trmm_rnt64!` (MAGMA-style
# block-inverse trick — turns the latency-bound row solve into a throughput-bound gemm).
@kernel unsafe_indices = true function _potrf_diag64!(A, invD, n, doinv, info, j0)
    T = eltype(A)
    li = @index(Local, Linear)            # 1..256, single group
    Ls = @localmem T (64, 64)
    Ldi = @localmem T (64,)
    p = li
    @inbounds while p <= n * n
        r = (p - 1) % n + 1
        c = (p - 1) ÷ n + 1
        Ls[r, c] = r >= c ? A[r, c] : zero(T)
        p += 256
    end
    @synchronize
    for j in 1:n
        if li == 1
            @inbounds begin
                d = Ls[j, j]
                if !(d > zero(T)) && info[1] == Int32(0)
                    info[1] = Int32(j0 + j - 1)
                end
                s = sqrt(d)
                Ls[j, j] = s
                Ldi[j] = one(T) / s
            end
        end
        @synchronize
        r2 = j + li
        @inbounds if r2 <= n
            Ls[r2, j] *= Ldi[j]
        end
        @synchronize
        t = n - j
        q = li
        @inbounds while q <= t * t
            rr = (q - 1) % t + 1 + j
            cc = (q - 1) ÷ t + 1 + j
            if rr >= cc
                Ls[rr, cc] = muladd(-Ls[rr, j], Ls[cc, j], Ls[rr, cc])
            end
            q += 256
        end
        @synchronize
    end
    p = li
    @inbounds while p <= n * n
        r = (p - 1) % n + 1
        c = (p - 1) ÷ n + 1
        if r >= c
            A[r, c] = Ls[r, c]
        end
        p += 256
    end
    if doinv == Int32(1) && li <= n
        x = @private T (64,)
        c = li
        @inbounds begin
            x[c] = Ldi[c]
            for i2 in (c + 1):n
                s = zero(T)
                for k in c:(i2 - 1)
                    s = muladd(Ls[i2, k], x[k], s)
                end
                x[i2] = -s * Ldi[i2]
            end
            for r2 in 1:n
                invD[r2, c] = r2 < c ? zero(T) : x[r2]
            end
        end
    end
end

# Invert every 64-wide diagonal block of an ALREADY-FACTORED lower L (n×n) into
# invD[:,:,b] — one workgroup per block, one launch for the whole matrix. Used by the
# standalone optimized trsm.
@kernel unsafe_indices = true function _invert_lblocks!(invD, @Const(L), n)
    T = eltype(invD)
    li = @index(Local, Linear)            # 1..256
    b = @index(Group, Linear)
    Ls = @localmem T (64, 64)
    Ldi = @localmem T (64,)
    base = (b - 1) * 64
    jb = min(64, n - base)
    p = li
    @inbounds while p <= jb * jb
        r = (p - 1) % jb + 1
        c = (p - 1) ÷ jb + 1
        Ls[r, c] = r >= c ? L[base + r, base + c] : zero(T)
        p += 256
    end
    @inbounds if li <= jb
        Ldi[li] = one(T) / L[base + li, base + li]
    end
    @synchronize
    if li <= jb
        x = @private T (64,)
        c = li
        @inbounds begin
            x[c] = Ldi[c]
            for i2 in (c + 1):jb
                s = zero(T)
                for k in c:(i2 - 1)
                    s = muladd(Ls[i2, k], x[k], s)
                end
                x[i2] = -s * Ldi[i2]
            end
            for r2 in 1:jb
                invD[r2, c, b] = r2 < c ? zero(T) : x[r2]
            end
        end
    end
end

# In-place C := C·Wᵀ, W = invD block (K×K, K ≤ 64): register-tiled like _gemm_nt_4x4! but
# reading C itself (deliberately NOT @Const — C aliases the output). Safe in place: K ≤ 64
# ⇒ a single column-group, each row-group reads only its OWN 64 rows of C (all reads happen
# before the last barrier of the k-loop, writes after it), and no group touches another's
# rows.
@kernel unsafe_indices = true function _trmm_rnt64!(C, @Const(W), M, K)
    T = eltype(C)
    li = @index(Local, NTuple)
    gi = @index(Group, NTuple)
    tx = li[1]; ty = li[2]
    tid = (ty - 1) * 16 + (tx - 1)
    br = (gi[1] - 1) * 64
    As = @localmem T (64, 8)
    Bs = @localmem T (64, 8)
    acc = @private T (4, 4)
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
            As[ml + 1, kl + 1] = (gr < M && gk < K) ? C[gr + 1, gk + 1] : zero(T)
            Bs[ml + 1, kl + 1] = (ml < K && gk < K) ? W[ml + 1, gk + 1] : zero(T)
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
        gc = (ty - 1) * 4 + (j - 1)
        if gr < M && gc < K
            C[gr + 1, gc + 1] = acc[i, j]
        end
    end
end

# =======================================================================================
# OLD drivers (verbatim from pure_potrf_dev.jl — the regressing baseline arm)
# =======================================================================================

function gpu_trsm_rlt!(B, L)
    m, n = size(B)
    (m == 0 || n == 0) && return B
    T = eltype(B)
    backend = get_backend(B)
    tk = _trsm_rlt_base!(backend, 256)
    nb = 64
    for j0 in 1:nb:n
        jb = min(nb, n - j0 + 1)
        Bj = view(B, :, j0:(j0 + jb - 1))
        if j0 > 1
            gpu_gemm_nt!(Bj, view(B, :, 1:(j0 - 1)), view(L, j0:(j0 + jb - 1), 1:(j0 - 1)),
                         -one(T), one(T))
        end
        tk(Bj, view(L, j0:(j0 + jb - 1), j0:(j0 + jb - 1)), m, jb; ndrange = cld(m, 256) * 256)
    end
    return B
end

function gpu_potrf!(A, d_info; nb::Int = 32)
    n = size(A, 1)
    n == 0 && return A
    T = eltype(A)
    backend = get_backend(A)
    pk = _potrf_base32!(backend, (32, 32))
    tk = _trsm_rlt_base!(backend, 256)
    for k0 in 1:nb:n
        kb = min(nb, n - k0 + 1)
        Akk = view(A, k0:(k0 + kb - 1), k0:(k0 + kb - 1))
        for i0 in 1:32:kb
            ib = min(32, kb - i0 + 1)
            D = view(Akk, i0:(i0 + ib - 1), i0:(i0 + ib - 1))
            pk(D, ib, d_info, k0 + i0 - 1; ndrange = (32, 32))
            q0 = i0 + ib
            if q0 <= kb
                Pn = view(Akk, q0:kb, i0:(i0 + ib - 1))
                tk(Pn, D, kb - q0 + 1, ib; ndrange = cld(kb - q0 + 1, 256) * 256)
                gpu_syrk_ln!(view(Akk, q0:kb, q0:kb), Pn, -one(T), one(T))
            end
        end
        r0 = k0 + kb
        if r0 <= n
            A21 = view(A, r0:n, k0:(k0 + kb - 1))
            gpu_trsm_rlt!(A21, Akk)
            gpu_syrk_ln!(view(A, r0:n, r0:n), A21, -one(T), one(T))
        end
    end
    return A
end

# =======================================================================================
# NEW drivers
# =======================================================================================

"""
Caller-provided workspace: `info` ([1], Int32, zeroed by the caller once per
factorization — non-zero after sync = 1-based column of first non-positive pivot),
`invD` (64×64×nblk Float scratch for the inverted diagonal blocks; block 1 doubles as
the fused kernels' factored-diagonal scratch `Dout` — the fused modes never use the
inverse). One `FrontWS` per factorization stream; zero device-pool allocation on any
driver call.
"""
struct FrontWS{TI, TD}
    info::TI
    invD::TD
end

# backend-generic (KernelAbstractions.zeros) so the same workspace serves CUDA/AMDGPU/oneAPI
FrontWS(backend, ::Type{T}, maxblk::Int) where {T} =
    FrontWS(KernelAbstractions.zeros(backend, Int32, 1),
            KernelAbstractions.zeros(backend, T, 64, 64, maxblk))

# Panel-rows threshold for :auto — MEASURED (galen, front_v3_check.jl): v3 (register-
# resident fused factor+solve, cld(m,64) groups) is ~134µs/launch flat for m ≤ 2500 and
# still beats v1 at m=5000 (216 vs 341µs); at m=10000 the per-group redundant factor
# (cld(m,64) groups stacking on 46 SMs) loses to v1's cld(m,256)-group row solve
# (418 vs 343µs). Crossover between the two measured points.
const FUSE_M_MAX = Ref(6000)

"""
    gpu_front!(P, nscol, ws; nb=64, mode=:auto) -> P

Whole GPU-front dense op in one call: potrf('L') of `P[1:nscol,1:nscol]` + the below-panel
solve `P[nscol+1:end,1:nscol] := · L11⁻ᵀ` (≡ CUSOLVER.potrf! + CUBLAS.trsm! on the same
views), as a right-looking partial Cholesky over 64-wide panels:
  per panel: [fused factor+solve over the FULL remaining height — 1 launch]
             (or, `mode=:split_gemm`, diag factor+inverse (1 group) → in-place trmm)
             → [one trapezoidal trailing update of the remaining columns].
`mode`: `:fused` | `:split_gemm` | `:split_base` | `:auto` (fused iff panel rows ≤
FUSE_M_MAX[]).
"""
function gpu_front!(P, nscol::Int, ws::FrontWS; nb::Int = 64, mode::Symbol = :auto)
    nsrow = size(P, 1)
    n = nscol
    @assert size(P, 2) == n && nsrow >= n
    n == 0 && return P
    T = eltype(P)
    backend = get_backend(P)
    fk = _front_fused64!(backend, 256)
    f2 = _front_fused64_v2!(backend, (16, 16))
    f3 = _front_fused64_v3!(backend, (16, 16))
    dk = _potrf_diag64!(backend, 256)
    tb = _trsm_rlt_base!(backend, 256)
    tm = _trmm_rnt64!(backend, (16, 16))
    for j0 in 1:nb:n
        jb = min(nb, n - j0 + 1)
        j1 = j0 + jb - 1
        m = nsrow - j1
        D = view(P, j0:j1, j0:j1)
        md = mode == :auto ? (m <= FUSE_M_MAX[] ? :fused3 : :fused) : mode
        if md == :fused3 || (md ∉ (:fused, :fused2) && m == 0)
            G = max(cld(m, 64), 1)
            Bv = view(P, (j1 + 1):nsrow, j0:j1)
            Dout = view(ws.invD, :, :, 1)
            f3(D, Bv, jb, m, ws.info, Dout, j0; ndrange = (G * 16, 16))
            D .= view(ws.invD, 1:jb, 1:jb, 1)     # scratch → panel diag (stream-ordered)
        elseif md == :fused2 || (md != :fused && m == 0)
            G = max(cld(m, 64), 1)
            Bv = view(P, (j1 + 1):nsrow, j0:j1)
            Dout = view(ws.invD, :, :, 1)
            f2(D, Bv, jb, m, ws.info, Dout, j0; ndrange = (G * 16, 16))
            D .= view(ws.invD, 1:jb, 1:jb, 1)     # scratch → panel diag (stream-ordered)
        elseif md == :fused || m == 0
            G = max(cld(m, 256), 1)
            Bv = view(P, (j1 + 1):nsrow, j0:j1)
            Dout = view(ws.invD, :, :, 1)
            fk(D, Bv, jb, m, ws.info, Dout, j0; ndrange = G * 256)
            D .= view(ws.invD, 1:jb, 1:jb, 1)     # scratch → panel diag (stream-ordered)
        else
            Bv = view(P, (j1 + 1):nsrow, j0:j1)
            iD = view(ws.invD, :, :, 1)
            if md == :split_gemm
                dk(D, iD, jb, Int32(1), ws.info, j0; ndrange = 256)
                tm(Bv, view(ws.invD, 1:jb, 1:jb, 1), m, jb; ndrange = (cld(m, 64) * 16, 16))
            else                          # :split_base
                dk(D, iD, jb, Int32(0), ws.info, j0; ndrange = 256)
                tb(Bv, D, m, jb; ndrange = cld(m, 256) * 256)
            end
        end
        if j1 < n                         # trailing trapezoid: rows j1+1..nsrow, cols j1+1..n
            C = view(P, (j1 + 1):nsrow, (j1 + 1):n)
            A2 = view(P, (j1 + 1):nsrow, j0:j1)
            gpu_syrk_trap!(C, A2, n - j1, -one(T), one(T))
        end
    end
    return P
end

"""
    gpu_potrf_opt!(A, ws; nb=64, mode=:auto) -> A

Optimized standalone potrf('L') — `gpu_front!` on a square panel.
"""
function gpu_potrf_opt!(A, ws::FrontWS; nb::Int = 64, mode::Symbol = :auto)
    @assert size(A, 1) == size(A, 2)
    return gpu_front!(A, size(A, 2), ws; nb, mode)
end

"""
    gpu_trsm_rlt_opt!(B, L, ws) -> B

Optimized standalone `B := B·L⁻ᵀ`: ONE launch inverts every 64-wide diagonal block of the
already-factored `L` into `ws.invD`, then per 64-column block: one gemm update (left-
looking) + one in-place trmm against the inverted block — every launch throughput-bound.
"""
function gpu_trsm_rlt_opt!(B, L, ws::FrontWS)
    m, n = size(B)
    @assert size(L, 1) == n && size(L, 2) == n
    (m == 0 || n == 0) && return B
    T = eltype(B)
    backend = get_backend(B)
    nblk = cld(n, 64)
    @assert size(ws.invD, 3) >= nblk
    ik = _invert_lblocks!(backend, 256)
    tm = _trmm_rnt64!(backend, (16, 16))
    ik(ws.invD, L, n; ndrange = nblk * 256)
    for (b, j0) in enumerate(1:64:n)
        jb = min(64, n - j0 + 1)
        Bj = view(B, :, j0:(j0 + jb - 1))
        if j0 > 1
            gpu_gemm_nt!(Bj, view(B, :, 1:(j0 - 1)), view(L, j0:(j0 + jb - 1), 1:(j0 - 1)),
                         -one(T), one(T))
        end
        tm(Bj, view(ws.invD, 1:jb, 1:jb, b), m, jb; ndrange = (cld(m, 64) * 16, 16))
    end
    return B
end
