# End-to-end Cholesky on Apple Metal via PureSparseMetalExt — the Metal mirror of
# benchmark/gpu/amd_end2end_test.jl. Runs the wired solver (gpu_symbolic → multifrontal hybrid FACTOR →
# device SOLVE) and checks it against the CPU factor and a residual gate at every frontier cutoff.
# Float32 only: Apple GPUs have no FP64. Run under an env with PureSparse + Metal + KernelAbstractions.
using PureSparse, Metal, KernelAbstractions, SparseArrays, LinearAlgebra, Random
ext = Base.get_extension(PureSparse, :PureSparseMetalExt)
@assert ext !== nothing "PureSparseMetalExt did not load (need Metal + KernelAbstractions present)"
println("PureSparseMetalExt loaded; device = ", Metal.device())

mtlzeros(::Type{T}, n) where {T} = MtlArray(zeros(T, n)); mtlarr(a) = MtlArray(a)
zsud!(x, S) = (
    for s in 1:S.nsuper
        nsc = Int(S.super[s + 1]) - Int(S.super[s]); nsr = Int(S.rowind_ptr[s + 1]) - Int(S.rowind_ptr[s]); b = Int(S.px[s])
        for j in 1:nsc, i in 1:(j - 1)
            x[b + (j - 1) * nsr + (i - 1)] = zero(eltype(x))
        end
    end; x
)

function test_chol(A::SparseMatrixCSC{T}, label) where {T}
    S0 = PureSparse.symbolic(A); F = PureSparse.cholesky(S0, A); @assert PureSparse.issuccess(F)
    xc = zsud!(copy(F.x), S0)
    tolf = T == Float32 ? 1.0e-5 : 1.0e-10; tols = T == Float32 ? 1.0e-3 : 1.0e-8
    snf = sort([sum(Float64(S0.colcount[j])^2 for j in S0.super[s]:(S0.super[s + 1] - 1)) for s in 1:S0.nsuper])
    for (tag, cut) in [("all-GPU", 0.0), ("hyb-25%", snf[cld(3 * S0.nsuper, 4)]), ("hyb-75%", snf[cld(S0.nsuper, 4)]), ("all-CPU", Inf)]
        G = ext.gpu_symbolic(A; ordering = PureSparse.AMDOrdering(), frontier_cutoff = cut)  # backend defaults to MetalBackend
        M = ext.mf_symbolic(G.cpu)
        x_host = Vector{T}(undef, G.xlen); ha = Vector{T}(undef, max(M.arena_peak, 1))
        da = mtlzeros(T, max(M.arena_peak, 1)); dz = mtlzeros(T, G.xlen)
        ok, fc = ext.gpu_multifrontal_hybrid!(x_host, dz, ha, da, M, G, A)
        @assert ok "Metal hybrid-mf non-SPD $label/$tag (failcol=$fc)"
        relerr = norm(zsud!(x_host, S0) - xc) / norm(xc)
        msg = rpad("$label/$tag", 24) * " GPU=$(lpad(count(G.on_gpu), 4))/$(S0.nsuper)  relerr=$relerr"
        # device Cholesky SOLVE on the all-GPU cutoff (dz holds the full device factor there)
        if tag == "all-GPU"
            n = S0.n; b = randn(MersenneTwister(7), T, n); perm = S0.perm
            d_y = mtlarr(b[perm]); ext.gpu_solve!(d_y, dz, G)
            x = Vector{T}(undef, n); x[perm] = Array(d_y)
            res = norm(A * x - b) / norm(b); msg *= "  solve-res=$res"
            @assert res < tols "$label solve res=$res"
        end
        println(msg)
        @assert relerr < tolf "$label/$tag mismatch $relerr"
    end
    return
end

rng = MersenneTwister(41)
grid3d(d) = (
    let n = d^3
        A = spzeros(n, n); lin(i, j, k) = ((k - 1) * d + (j - 1)) * d + i
        for k in 1:d, j in 1:d, i in 1:d
            p = lin(i, j, k); A[p, p] = 6.0
            i < d&&(A[p, lin(i + 1, j, k)] = A[lin(i + 1, j, k), p] = -1.0); j < d&&(A[p, lin(i, j + 1, k)] = A[lin(i, j + 1, k), p] = -1.0)
            k < d&&(A[p, lin(i, j, k + 1)] = A[lin(i, j, k + 1), p] = -1.0)
        end; A + 0.1I
    end
)
test_chol(
    (
        let n = 300
            A = sprand(rng, n, n, 0.02); A + A' + n * I
        end
    ) |> SparseMatrixCSC{Float32, Int}, "rand_n300"
)
test_chol(
    (
        let nx = 25, ny = 25
            n = nx * ny; A = spzeros(n, n)
            for j in 1:ny, i in 1:nx
                k = (j - 1) * nx + i; A[k, k] = 4.0
                i < nx&&(A[k, k + 1] = A[k + 1, k] = -1.0); j < ny&&(A[k, k + nx] = A[k + nx, k] = -1.0)
            end; A + 0.05I
        end
    ) |> SparseMatrixCSC{Float32, Int}, "grid_25x25"
)
test_chol(SparseMatrixCSC{Float32, Int}(grid3d(12)), "grid3d_12")
println("\n✅ END-TO-END CHOLESKY ON METAL — Float32 factor+solve match CPU at every cutoff")
