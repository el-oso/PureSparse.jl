# GPU frontier partition (design_gpu.md §5.2) — CPU-only tests. The frontier logic is pure
# host-side graph work on a `Symbolic`; it needs no GPU, so it runs in the normal suite.
# ext/frontier.jl is included directly (it has no CUDA/KA dependency).

@testitem "GPU frontier: upward closure + invariant + boundary (design_gpu.md §5.2)" begin
    using PureSparse, SparseArrays, LinearAlgebra, Random
    include(joinpath(@__DIR__, "..", "ext", "frontier.jl"))

    # p-quantile of a vector (avoids a Statistics dep); defined inside the isolated test item
    quantile_sorted(v, p) = (s = sort(v); isempty(s) ? 0.0 : s[clamp(round(Int, p * length(s)), 1, length(s))])

    rng = MersenneTwister(0xF1)
    # A spread of SPD patterns so the supernode etree has real structure.
    mats = [
        (let n = 400; A = sprand(rng, n, n, 0.01); A = A + A' + n * I; A end),
        (let nx = 30, ny = 20  # 2-D grid Laplacian-ish
            n = nx * ny
            A = spzeros(n, n)
            for j in 1:ny, i in 1:nx
                k = (j - 1) * nx + i
                A[k, k] = 4.0
                i < nx && (A[k, k + 1] = A[k + 1, k] = -1.0)
                j < ny && (A[k, k + nx] = A[k + nx, k] = -1.0)
            end
            A + 0.1I
        end),
        (let n = 800; A = sprand(rng, n, n, 0.005); A = A + A' + 2n * I; A end),
    ]

    for A in mats
        S = PureSparse.symbolic(A)
        ns = S.nsuper
        on_gpu = Vector{Bool}(undef, ns)

        # sweep cutoffs from "all GPU" to "none GPU"
        snflop = [sum(Float64(S.colcount[j])^2 for j in S.super[s]:(S.super[s+1]-1)) for s in 1:ns]
        cutoffs = [0.0, quantile_sorted(snflop, 0.5), quantile_sorted(snflop, 0.9),
                   maximum(snflop) + 1.0]

        for cut in cutoffs
            frontier_partition!(on_gpu, ns, S.super, S.sparent, S.colcount, cut)

            # (1) upward-closure invariant: no GPU→CPU update edge
            @test frontier_invariant_holds(on_gpu, ns, S.rowind, S.rowind_ptr, S.snode_of)

            # (2) upward closure directly: every GPU node's etree parent is GPU
            for s in 1:ns
                if on_gpu[s] && S.sparent[s] != 0
                    @test on_gpu[S.sparent[s]]
                end
            end

            # (3) seeds are included: any supernode over cutoff is on GPU
            for s in 1:ns
                snflop[s] ≥ cut && @test on_gpu[s]
            end

            # (4) boundary supernodes are exactly the CPU nodes with a GPU-targeting row
            bnd = boundary_supernodes(on_gpu, ns, S.rowind, S.rowind_ptr, S.snode_of)
            @test all(s -> !on_gpu[s], bnd)             # boundary ⊆ CPU
            @test allunique(bnd)
            # a boundary node genuinely has a GPU ancestor row
            for s in bnd
                @test any(p -> on_gpu[S.snode_of[S.rowind[p]]],
                          S.rowind_ptr[s]:(S.rowind_ptr[s+1]-1))
            end
        end

        # edge cases
        frontier_partition!(on_gpu, ns, S.super, S.sparent, S.colcount, 0.0)
        @test all(on_gpu)                                # cutoff 0 → everything on GPU
        @test isempty(boundary_supernodes(on_gpu, ns, S.rowind, S.rowind_ptr, S.snode_of))
        frontier_partition!(on_gpu, ns, S.super, S.sparent, S.colcount, Inf)
        @test !any(on_gpu)                               # cutoff ∞ → nothing on GPU
        @test isempty(boundary_supernodes(on_gpu, ns, S.rowind, S.rowind_ptr, S.snode_of))
    end
end

@testitem "GPU device-memory budget (design_gpu.md §5.3)" begin
    using PureSparse, SparseArrays, LinearAlgebra, Random
    include(joinpath(@__DIR__, "..", "ext", "frontier.jl"))

    rng = MersenneTwister(0xB0)
    A = sprand(rng, 600, 600, 0.008); A = A + A' + 1200I
    S = PureSparse.symbolic(A)
    ns = S.nsuper
    on_gpu = Vector{Bool}(undef, ns)
    elt = sizeof(Float64)

    snflop = [sum(Float64(S.colcount[j])^2 for j in S.super[s]:(S.super[s+1]-1)) for s in 1:ns]
    midcut = sort(snflop)[cld(ns, 2)]
    frontier_partition!(on_gpu, ns, S.super, S.sparent, S.colcount, midcut)
    bnd = boundary_supernodes(on_gpu, ns, S.rowind, S.rowind_ptr, S.snode_of)

    b = gpu_device_bytes(S.super, S.rowind_ptr, bnd, S.nnzL, S.max_extend_rows, elt)
    @test b.nzval == S.nnzL * elt
    @test b.cbuf == S.max_extend_rows^2 * elt
    @test b.boundbuf ≥ 0
    @test b.total == b.nzval + b.cbuf + b.boundbuf

    # cutoff 0 → all GPU → no boundary panels → boundbuf 0
    frontier_partition!(on_gpu, ns, S.super, S.sparent, S.colcount, 0.0)
    bnd0 = boundary_supernodes(on_gpu, ns, S.rowind, S.rowind_ptr, S.snode_of)
    @test gpu_device_bytes(S.super, S.rowind_ptr, bnd0, S.nnzL, S.max_extend_rows, elt).boundbuf == 0

    # capacity check: fits in plenty, not in a sliver (margin enforced)
    @test gpu_capacity_ok(b.total, b.total + 1_000_000, 500_000)
    @test !gpu_capacity_ok(b.total, b.total, 1)          # margin makes an exact fit fail
    @test !gpu_capacity_ok(b.total, b.total ÷ 2, 0)
end
