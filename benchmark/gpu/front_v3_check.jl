# v3 fused-front correctness vs CPU Cholesky + per-launch timing (scratch probe).
using PureSparse, CUDA, KernelAbstractions, LinearAlgebra, Random, Statistics, Printf
ext = Base.get_extension(PureSparse, :PureSparseCUDAExt)
backend = CUDA.CUDABackend()
tb(f, N=50) = (f(); CUDA.synchronize(); CUDA.@elapsed(begin
    for _ in 1:N; f(); end end) / N)

function check(nscol, below)
    rng = MersenneTwister(2); nsrow = nscol + below
    Mm = randn(rng, nscol, nscol); P11 = Mm'*Mm + nscol*I
    A21 = randn(rng, below, nscol)
    P = zeros(nsrow, nscol); P[1:nscol,1:nscol] = P11; P[(nscol+1):nsrow,1:nscol] = A21
    # CPU reference: chol of P11, then A21 * L^-T
    L = LinearAlgebra.cholesky(Symmetric(P11, :L)).L
    Ref_ = zeros(nsrow, nscol)
    Ref_[1:nscol,1:nscol] = Matrix(L)
    below > 0 && (Ref_[(nscol+1):nsrow,1:nscol] = A21 / L')
    ws = ext.FrontWS(backend, Float64, cld(nscol,64))
    dP = CuArray(P)
    ext.gpu_front!(dP, nscol, ws)
    G = Array(dP)
    # mask never-read strict upper of the diag block
    for c in 2:nscol, r in 1:c-1
        G[r,c] = 0.0; Ref_[r,c] = 0.0
    end
    relerr = norm(G - Ref_) / norm(Ref_)
    info = Array(ws.info)[1]
    @printf("nscol=%5d below=%5d  relerr %.2e  info=%d %s\n", nscol, below, relerr, info,
            relerr <= 1e-14 ? "OK" : "FAIL")
    return relerr
end

for (nc, bl) in ((64,0),(64,186),(128,186),(192,50),(256,186),(384,1000),(512,186),
                 (1024,186),(1536,186),(1536,1000),(100,77),(129,257))
    check(nc, bl)
end

# per-launch v3 timing at representative m
rng = MersenneTwister(1)
Mm = randn(rng, 64, 64); A = CuArray(Mm'*Mm + 64.0*I)
ws = ext.FrontWS(backend, Float64, 1)
f3 = ext._front_fused64_v3!(backend, (16,16))
f2 = ext._front_fused64_v2!(backend, (16,16))
Dout = view(ws.invD, :, :, 1)
for m in (0, 186, 500, 1000, 2500, 5000, 10000)
    B = CUDA.rand(Float64, max(m,1), 64)
    G = max(cld(m,64),1)
    t3 = tb(() -> f3(A, B, 64, m, ws.info, Dout, 1; ndrange=(G*16,16)))
    G2 = G
    t2 = tb(() -> f2(A, B, 64, m, ws.info, Dout, 1; ndrange=(G2*16,16)))
    fk = ext._front_fused64!(backend, 256)
    G1 = max(cld(m,256),1)
    t1 = tb(() -> fk(A, B, 64, m, ws.info, Dout, 1; ndrange=G1*256))
    @printf("m=%6d  v3 %7.1f us   v2 %7.1f us   v1 %7.1f us\n", m, t3*1e6, t2*1e6, t1*1e6)
end
