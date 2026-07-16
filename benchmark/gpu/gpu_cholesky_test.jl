# Phase 2.3 correctness oracle: the synchronous all-GPU Cholesky factor must match the CPU
# factor (design_gpu.md §10.1). Same Symbolic → identical packed-storage (px) layout, so the
# device `dx` is compared directly to the CPU factor's `F.x`.
using PureSparse, CUDA, KernelAbstractions, SparseArrays, LinearAlgebra, Random

ext = Base.get_extension(PureSparse, :PureSparseCUDAExt)
ext === nothing && error("PureSparseCUDAExt did not load")

# Zero the strict-upper triangle of each supernode's diagonal block — those cells are NEVER
# read (solve + descendant updates use only lower-tri diagonal + off-diagonal panels), and the
# CPU factor leaves inconsistent update garbage there (syrk-'L' vs full-scatter) while the GPU
# cleanly leaves 0. The MEANINGFUL factor is everything else; compare that.
function zero_strict_upper_diag!(x, S)
    for s in 1:S.nsuper
        nsc = Int(S.super[s+1]) - Int(S.super[s])
        nsr = Int(S.rowind_ptr[s+1]) - Int(S.rowind_ptr[s])
        base = Int(S.px[s])
        for j in 1:nsc, i in 1:(j-1)        # strict upper of the nsc×nsc diagonal block
            x[base + (j-1)*nsr + (i-1)] = 0.0
        end
    end
    return x
end

function test_one(A, label)
    G = ext.gpu_symbolic(A; ordering = PureSparse.AMDOrdering(), frontier_cutoff = 0.0)
    F = PureSparse.cholesky(G.cpu, A)          # CPU factor, SAME symbolic → same px layout
    @assert PureSparse.issuccess(F) "CPU factor not SPD for $label"

    dx = CUDA.zeros(Float64, G.xlen)
    ok, failc = ext.gpu_cholesky_sync!(dx, G, A)
    @assert ok "GPU factor reported non-SPD (fail_col=$failc) for $label"

    xg = zero_strict_upper_diag!(Array(dx), G.cpu)
    xc = zero_strict_upper_diag!(copy(F.x), G.cpu)
    relerr = norm(xg - xc) / norm(xc)
    println(rpad(label, 22), "  nsuper=", G.cpu.nsuper, "  xlen=", G.xlen,
            "  ‖L_gpu-L_cpu‖/‖L_cpu‖ = ", relerr)
    if relerr ≥ 1e-10
        d = abs.(xg .- xc); p = argmax(d)
        px = G.cpu.px; s = findlast(k -> Int(px[k]) ≤ p, 1:G.cpu.nsuper)
        off = p - Int(px[s]); nsc = Int(G.cpu.super[s+1]) - Int(G.cpu.super[s])
        nsr = Int(G.cpu.rowind_ptr[s+1]) - Int(G.cpu.rowind_ptr[s])
        println("   MAX DIFF ", d[p], " at x[$p]  s=$s panel[", off % nsr + 1, ",", off ÷ nsr + 1,
                "] (nscol=$nsc nsrow=$nsr)  gpu=", xg[p], " cpu=", xc[p])
        println("   #entries |diff|>1e-8: ", count(>(1e-8), d), " / ", length(d))
    end
    @assert relerr < 1e-10 "$label: GPU factor mismatch after masking unused cells (relerr=$relerr)"
    return relerr
end

rng = MersenneTwister(11)
# small → medium SPD, varied structure (each factored on both CPU and GPU, compared)
test_one((let n=200; A=sprand(rng,n,n,0.02); A+A'+n*I end), "rand_n200")
test_one((let n=600; A=sprand(rng,n,n,0.01); A+A'+2n*I end), "rand_n600")
test_one((let nx=25,ny=25          # 2-D grid Laplacian
    n=nx*ny; A=spzeros(n,n)
    for j in 1:ny, i in 1:nx
        k=(j-1)*nx+i; A[k,k]=4.0
        i<nx && (A[k,k+1]=A[k+1,k]=-1.0); j<ny && (A[k,k+nx]=A[k+nx,k]=-1.0)
    end; A+0.05I
end), "grid_25x25")
test_one((let n=1200; A=sprand(rng,n,n,0.006); A+A'+3n*I end), "rand_n1200")

println("\nALL GPU CHOLESKY ORACLE TESTS PASS — device factor matches CPU factor")
