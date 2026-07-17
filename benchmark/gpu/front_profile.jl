# Micro-profile the pieces of gpu_front! at the failing shapes (scratch probe).
using PureSparse, CUDA, KernelAbstractions, LinearAlgebra, Random, Statistics, Printf
using CUDA.CUSOLVER: potrf!
using CUDA.CUBLAS: trsm!
ext = Base.get_extension(PureSparse, :PureSparseCUDAExt)
backend = CUDA.CUDABackend()

# batch timing: run f() N times inside one @elapsed, divide
tb(f, N=30) = (f(); CUDA.synchronize(); CUDA.@elapsed(begin
    for _ in 1:N; f(); end end) / N)

function profile_front(nscol, below)
    rng = MersenneTwister(1); nsrow = nscol + below
    Mm = randn(rng, nscol, nscol); P11 = Mm'*Mm + nscol*I
    P = zeros(nsrow, nscol); P[1:nscol,1:nscol] = P11
    P[(nscol+1):nsrow,1:nscol] = randn(rng, below, nscol)
    dP0 = CuArray(P)
    dP = copy(dP0)
    ws = ext.FrontWS(backend, Float64, cld(nscol,64))
    T = Float64
    f2 = ext._front_fused64_v2!(backend, (16,16))
    fk = ext._front_fused64!(backend, 256)
    dk = ext._potrf_diag64!(backend, 256)

    @printf("== nscol=%d below=%d ==\n", nscol, below)
    t_restore = tb(() -> copyto!(dP, dP0))
    @printf("  (restore copy             %8.1f us)\n", t_restore*1e6)
    t_tot = tb(() -> (copyto!(dP, dP0); ext.gpu_front!(dP, nscol, ws))) - t_restore
    @printf("  gpu_front! total          %8.1f us\n", t_tot*1e6)

    # pieces, on pristine dP (fused kernels read D, write only B/Dout -> D stays SPD)
    copyto!(dP, dP0)
    t_f2 = 0.0; t_f1 = 0.0; t_cp = 0.0; t_sy = 0.0; t_dk = 0.0
    dDsave = CuArray(zeros(64,64))
    for j0 in 1:64:nscol
        jb = min(64, nscol - j0 + 1); j1 = j0 + jb - 1
        m = nsrow - j1
        D = view(dP, j0:j1, j0:j1)
        Bv = view(dP, (j1+1):nsrow, j0:j1)
        Dout = view(ws.invD, :, :, 1)
        G = max(cld(m,64),1)
        t_f2 += tb(() -> f2(D, Bv, jb, m, ws.info, Dout, j0; ndrange=(G*16,16)))
        G1 = max(cld(m,256),1)
        t_f1 += tb(() -> fk(D, Bv, jb, m, ws.info, Dout, j0; ndrange=G1*256))
        # diag kernel writes D back: restore D each iter (cost of that copy ~ t_cp/n, subtracted via reporting)
        copyto!(view(dDsave,1:jb,1:jb), D)
        t_dk += tb(() -> (D .= view(dDsave,1:jb,1:jb);
                          dk(D, view(ws.invD,:,:,1), jb, Int32(1), ws.info, j0; ndrange=256)))
        t_cp += tb(() -> (D .= view(ws.invD, 1:jb, 1:jb, 1)))
        if j1 < nscol
            C = view(dP, (j1+1):nsrow, (j1+1):nscol)
            A2 = view(dP, (j1+1):nsrow, j0:j1)
            t_sy += tb(() -> ext.gpu_syrk_trap!(C, A2, nscol-j1, -one(T), one(T)))
        end
    end
    @printf("  sum fused_v2 launches     %8.1f us\n", t_f2*1e6)
    @printf("  sum fused_v1 launches     %8.1f us\n", t_f1*1e6)
    @printf("  sum diag64(fac+inv)+Dcp   %8.1f us\n", t_dk*1e6)
    @printf("  sum D .= copies           %8.1f us\n", t_cp*1e6)
    @printf("  sum syrk_trap (K=64)      %8.1f us   [syrk here reads unfactored data - timing only]\n", t_sy*1e6)

    # vendor pieces (restore diag before potrf each iter)
    dV = copy(dP0)
    diag = view(dV, 1:nscol, 1:nscol)
    diag0 = CuArray(P11)
    t_vre = tb(() -> copyto!(diag, diag0))
    t_po = tb(() -> (copyto!(diag, diag0); potrf!('L', diag))) - t_vre
    t_tr = below > 0 ? tb(() -> trsm!('R','L','T','N',1.0,diag,view(dV,(nscol+1):nsrow,1:nscol))) : 0.0
    @printf("  vendor potrf              %8.1f us\n", t_po*1e6)
    @printf("  vendor trsm               %8.1f us\n", t_tr*1e6)

    # one big syrk_trap at K=768 (what an LL-outer / bigger-NB update runs at)
    if nscol >= 1024
        C = view(dP, 769:nsrow, 769:min(nscol,1024))
        A = view(dP, 769:nsrow, 1:768)
        t_big = tb(() -> ext.gpu_syrk_trap!(C, A, size(C,2), -one(T), one(T)))
        gf = 2 * size(C,1)*size(C,2)*768 / t_big / 1e9
        @printf("  syrk_trap K=768 M=%d N=%d  %8.1f us (%.0f GF incl. wasted upper-skip)\n",
                size(C,1), size(C,2), t_big*1e6, gf)
    end
end

profile_front(128, 186)
profile_front(1536, 186)
profile_front(1536, 1000)
