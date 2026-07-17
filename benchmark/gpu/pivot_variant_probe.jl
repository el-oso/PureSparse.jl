# Verify WHERE the ~280us/launch of _front_fused64_v2! goes: pivot sqrt/div vs sweep vs invert.
using PureSparse, CUDA, KernelAbstractions, LinearAlgebra, Random, Statistics, Printf
using KernelAbstractions: @kernel, @index, @localmem, @private, @Const, get_backend
import KernelAbstractions
using Base.Cartesian: @nexprs
ext = Base.get_extension(PureSparse, :PureSparseCUDAExt)
backend = CUDA.CUDABackend()

tb(f, N=50) = (f(); CUDA.synchronize(); CUDA.@elapsed(begin
    for _ in 1:N; f(); end end) / N)

# --- variant A: v2 factor loop only (no invert, no solve, no writeback) ---
@kernel unsafe_indices = true function _fac_only!(@Const(A), n, sink)
    T = eltype(A)
    li = @index(Local, NTuple); gi = @index(Group, Linear)
    tx = li[1]; ty = li[2]; tid = (ty - 1) * 16 + tx
    Ls = @localmem T (64, 64); Ldi = @localmem T (64,)
    q = tid
    @inbounds while q <= 4096
        r = ((q - 1) & 63) + 1; c = ((q - 1) >> 6) + 1
        Ls[r, c] = (r <= n && c <= n && r >= c) ? A[r, c] : zero(T)
        q += 256
    end
    @synchronize
    for j in 1:n
        if tid == 1
            @inbounds begin
                d = Ls[j, j]; s = sqrt(d); Ls[j, j] = s; Ldi[j] = one(T) / s
            end
        end
        @synchronize
        r2 = j + tid
        @inbounds if r2 <= n
            Ls[r2, j] *= Ldi[j]
        end
        @synchronize
        q = tid
        @inbounds while q <= 4096
            rr = j + ((q - 1) & 63) + 1; cc = j + ((q - 1) >> 6) + 1
            if rr <= n && cc <= n && rr >= cc
                Ls[rr, cc] = muladd(-Ls[rr, j], Ls[cc, j], Ls[rr, cc])
            end
            q += 256
        end
        @synchronize
    end
    if gi == 1 && tid == 1
        @inbounds sink[1] = Ls[n, n]
    end
end

# --- variant B: same but pivot without sqrt/div (WRONG math, timing only) ---
@kernel unsafe_indices = true function _fac_nosqrt!(@Const(A), n, sink)
    T = eltype(A)
    li = @index(Local, NTuple); gi = @index(Group, Linear)
    tx = li[1]; ty = li[2]; tid = (ty - 1) * 16 + tx
    Ls = @localmem T (64, 64); Ldi = @localmem T (64,)
    q = tid
    @inbounds while q <= 4096
        r = ((q - 1) & 63) + 1; c = ((q - 1) >> 6) + 1
        Ls[r, c] = (r <= n && c <= n && r >= c) ? A[r, c] : zero(T)
        q += 256
    end
    @synchronize
    for j in 1:n
        if tid == 1
            @inbounds begin
                d = Ls[j, j]; Ls[j, j] = d * T(0.5); Ldi[j] = d * T(0.25)  # fake pivot
            end
        end
        @synchronize
        r2 = j + tid
        @inbounds if r2 <= n
            Ls[r2, j] *= Ldi[j]
        end
        @synchronize
        q = tid
        @inbounds while q <= 4096
            rr = j + ((q - 1) & 63) + 1; cc = j + ((q - 1) >> 6) + 1
            if rr <= n && cc <= n && rr >= cc
                Ls[rr, cc] = muladd(-Ls[rr, j], Ls[cc, j], Ls[rr, cc])
            end
            q += 256
        end
        @synchronize
    end
    if gi == 1 && tid == 1
        @inbounds sink[1] = Ls[n, n]
    end
end

# --- variant C: pivot via Float32-seed rsqrt + FP64 Newton + Markstein sqrt correction ---
@inline function _fast_pivot(d::Float64)
    y = Float64(1.0f0 / sqrt(Float32(d)))          # fast FP32 seed (~2^-12)
    y = y * muladd(-0.5 * d * y, y, 1.5)           # Newton -> ~2^-24
    y = y * muladd(-0.5 * d * y, y, 1.5)           # Newton -> ~2^-48
    y = y * muladd(-0.5 * d * y, y, 1.5)           # Newton -> ~1 ulp
    s0 = d * y
    s = muladd(0.5 * y, muladd(-s0, s0, d), s0)    # Markstein: faithful sqrt
    yi = muladd(y, muladd(-s, y, 1.0), y)          # refine reciprocal 1/s
    return s, yi
end
@kernel unsafe_indices = true function _fac_fastpiv!(@Const(A), n, sink)
    T = eltype(A)
    li = @index(Local, NTuple); gi = @index(Group, Linear)
    tx = li[1]; ty = li[2]; tid = (ty - 1) * 16 + tx
    Ls = @localmem T (64, 64); Ldi = @localmem T (64,)
    q = tid
    @inbounds while q <= 4096
        r = ((q - 1) & 63) + 1; c = ((q - 1) >> 6) + 1
        Ls[r, c] = (r <= n && c <= n && r >= c) ? A[r, c] : zero(T)
        q += 256
    end
    @synchronize
    for j in 1:n
        if tid == 1
            @inbounds begin
                d = Ls[j, j]
                s, yi = _fast_pivot(d)
                Ls[j, j] = s; Ldi[j] = yi
            end
        end
        @synchronize
        r2 = j + tid
        @inbounds if r2 <= n
            Ls[r2, j] *= Ldi[j]
        end
        @synchronize
        q = tid
        @inbounds while q <= 4096
            rr = j + ((q - 1) & 63) + 1; cc = j + ((q - 1) >> 6) + 1
            if rr <= n && cc <= n && rr >= cc
                Ls[rr, cc] = muladd(-Ls[rr, j], Ls[cc, j], Ls[rr, cc])
            end
            q += 256
        end
        @synchronize
    end
    if gi == 1 && tid == 1
        @inbounds sink[1] = Ls[n, n]
    end
end

# --- variant D: factor-only, no trailing sweep (pivot+scale only; WRONG, timing) ---
@kernel unsafe_indices = true function _fac_nosweep!(@Const(A), n, sink)
    T = eltype(A)
    li = @index(Local, NTuple); gi = @index(Group, Linear)
    tx = li[1]; ty = li[2]; tid = (ty - 1) * 16 + tx
    Ls = @localmem T (64, 64); Ldi = @localmem T (64,)
    q = tid
    @inbounds while q <= 4096
        r = ((q - 1) & 63) + 1; c = ((q - 1) >> 6) + 1
        Ls[r, c] = (r <= n && c <= n && r >= c) ? A[r, c] : zero(T)
        q += 256
    end
    @synchronize
    for j in 1:n
        if tid == 1
            @inbounds begin
                d = Ls[j, j]; s = sqrt(d); Ls[j, j] = s; Ldi[j] = one(T) / s
            end
        end
        @synchronize
        r2 = j + tid
        @inbounds if r2 <= n
            Ls[r2, j] *= Ldi[j]
        end
        @synchronize
    end
    if gi == 1 && tid == 1
        @inbounds sink[1] = Ls[n, n]
    end
end

rng = MersenneTwister(1)
Mm = randn(rng, 64, 64); A = CuArray(Mm'*Mm + 64.0*I)
sink = CUDA.zeros(Float64, 1)
G = 16   # like a typical fused launch
for (nm, k) in (("A fac_only (sqrt+div) ", _fac_only!), ("B fac_nosqrt          ", _fac_nosqrt!),
                ("C fac_fastpiv         ", _fac_fastpiv!), ("D fac_nosweep         ", _fac_nosweep!))
    kern = k(backend, (16,16))
    t1 = tb(() -> kern(A, 64, sink; ndrange=(16,16)))
    tG = tb(() -> kern(A, 64, sink; ndrange=(16*G,16)))
    @printf("%s  G=1 %7.1f us   G=%d %7.1f us\n", nm, t1*1e6, G, tG*1e6)
end

# accuracy of fast pivot vs sqrt/1/sqrt on host-representative values
maxrel_s = 0.0; maxrel_y = 0.0
for _ in 1:100000
    d = exp(40*(rand()-0.5))
    y = Float64(1.0f0/sqrt(Float32(d)))
    y = y*muladd(-0.5*d*y, y, 1.5); y = y*muladd(-0.5*d*y, y, 1.5); y = y*muladd(-0.5*d*y, y, 1.5)
    s0 = d*y; s = muladd(0.5*y, muladd(-s0,s0,d), s0)
    yi = muladd(y, muladd(-s,y,1.0), y)
    global maxrel_s = max(maxrel_s, abs(s - sqrt(d))/sqrt(d))
    global maxrel_y = max(maxrel_y, abs(yi - 1/sqrt(d))*sqrt(d))
end
@printf("fast pivot accuracy: max rel err sqrt %.2e, recip %.2e\n", maxrel_s, maxrel_y)
