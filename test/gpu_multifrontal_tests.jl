# Multifrontal Cholesky symbolic layer (design_gpu.md §M) — CPU-only (no GPU). ext/multifrontal.jl
# is pure, included directly. Validates children-CSC, ascending emap + k1 prefix split, and the
# arena simulation's non-aliasing (pitfall #3).

@testitem "Multifrontal symbolic: children/emap/k1/arena (design_gpu.md §M)" begin
    using PureSparse, SparseArrays, LinearAlgebra, Random
    include(joinpath(@__DIR__, "..", "ext", "multifrontal.jl"))

    rng = MersenneTwister(0xACE)
    mats = [
        (let n=300; A=sprand(rng,n,n,0.02); A+A'+n*I end),
        (let nx=22,ny=18; n=nx*ny; A=spzeros(n,n)
            for j in 1:ny, i in 1:nx
                k=(j-1)*nx+i; A[k,k]=4.0
                i<nx && (A[k,k+1]=A[k+1,k]=-1.0); j<ny && (A[k,k+nx]=A[k+nx,k]=-1.0)
            end; A+0.1I end),
        (let n=700; A=sprand(rng,n,n,0.01); A+A'+2n*I end),
    ]
    for A in mats
        S = PureSparse.symbolic(A)
        M = mf_symbolic(S)
        ns = S.nsuper
        nscolf(s) = Int(S.super[s+1]) - Int(S.super[s])
        nsrowf(s) = Int(S.rowind_ptr[s+1]) - Int(S.rowind_ptr[s])

        # (1) children CSC: each non-root supernode appears exactly once, under its sparent
        seen = falses(ns)
        for s in 1:ns, ci in Int(M.children_ptr[s]):(Int(M.children_ptr[s+1])-1)
            c = Int(M.children[ci])
            @test Int(S.sparent[c]) == s
            @test !seen[c]; seen[c] = true
        end
        @test count(seen) == count(c -> Int(S.sparent[c]) != 0, 1:ns)   # all non-roots are children

        # (2) emap ascending + valid range; (3) k1 prefix split
        for c in 1:ns
            Int(S.sparent[c]) == 0 && continue
            p = Int(S.sparent[c]); nsc_p = nscolf(p); nsr_p = nsrowf(p)
            rng_c = Int(M.emap_ptr[c]):(Int(M.emap_ptr[c+1])-1)
            em = [Int(M.emap[i]) for i in rng_c]
            @test length(em) == nsrowf(c) - nscolf(c)
            @test all(1 .≤ em .≤ nsr_p)
            @test issorted(em) && allunique(em)                        # strictly ascending
            k = Int(M.k1[c])
            @test all(em[1:k] .≤ nsc_p)                                # prefix -> parent pivot cols
            @test all(em[k+1:end] .> nsc_p)                            # rest -> U region
        end

        # (4) arena non-aliasing (pitfall #3): replay postorder, assert live U intervals disjoint
        live = Tuple{Int,Int,Int}[]   # (supernode, lo, hi) currently on the stack
        for s in 1:ns
            nc = Int(M.children_ptr[s+1]) - Int(M.children_ptr[s])
            for _ in 1:nc; pop!(live); end
            lo = Int(M.uoff[s]); hi = lo + Int(M.usize[s]) - 1
            for (_, l2, h2) in live
                @test hi < l2 || lo > h2                               # disjoint from every live slot
            end
            (hi ≥ lo) && push!(live, (s, lo, hi))                      # skip 0-size U's
            @test hi ≤ M.arena_peak
        end
        @test M.arena_peak ≥ maximum(Int, M.usize)
    end
end

@testitem "Multifrontal CPU numeric: factor matches cholesky! (design_gpu.md §M, the relay)" begin
    using PureSparse, SparseArrays, LinearAlgebra, Random
    include(joinpath(@__DIR__, "..", "ext", "multifrontal.jl"))

    # mask the never-read strict-upper diagonal cells (see cholesky_test): CPU cholesky! leaves
    # update garbage there; multifrontal cleanly leaves 0. Compare everything else.
    function zsud!(x, S)
        for s in 1:S.nsuper
            nsc=Int(S.super[s+1])-Int(S.super[s]); nsr=Int(S.rowind_ptr[s+1])-Int(S.rowind_ptr[s]); b=Int(S.px[s])
            for j in 1:nsc, i in 1:(j-1); x[b+(j-1)*nsr+(i-1)]=0.0; end
        end; x
    end

    rng = MersenneTwister(0xF00D)
    mats = [
        ("rand_n200",  (let n=200; A=sprand(rng,n,n,0.03); A+A'+n*I end)),
        ("grid_20x20", (let nx=20,ny=20; n=nx*ny; A=spzeros(n,n)
            for j in 1:ny,i in 1:nx; k=(j-1)*nx+i; A[k,k]=4.0
                i<nx&&(A[k,k+1]=A[k+1,k]=-1.0); j<ny&&(A[k,k+nx]=A[k+nx,k]=-1.0) end; A+0.05I end)),
        ("rand_n700",  (let n=700; A=sprand(rng,n,n,0.012); A+A'+2n*I end)),
        ("grid3d_10",  (let d=10; n=d^3; A=spzeros(n,n); lin(i,j,k)=((k-1)*d+(j-1))*d+i
            for k in 1:d,j in 1:d,i in 1:d; p=lin(i,j,k); A[p,p]=6.0
                i<d&&(A[p,lin(i+1,j,k)]=A[lin(i+1,j,k),p]=-1.0); j<d&&(A[p,lin(i,j+1,k)]=A[lin(i,j+1,k),p]=-1.0)
                k<d&&(A[p,lin(i,j,k+1)]=A[lin(i,j,k+1),p]=-1.0) end; A+0.1I end)),
    ]
    for (label, A) in mats
        S = PureSparse.symbolic(A)
        F = PureSparse.cholesky(S, A); @assert PureSparse.issuccess(F)
        M = mf_symbolic(S)
        xlen = Int(S.px[S.nsuper+1]) - 1
        x_host = Vector{Float64}(undef, xlen); arena = Vector{Float64}(undef, max(M.arena_peak,1))
        ok, fc = cpu_multifrontal_cholesky!(x_host, arena, M, S, A)
        @test ok
        relerr = norm(zsud!(x_host, S) - zsud!(copy(F.x), S)) / norm(zsud!(copy(F.x), S))
        @test relerr < 1e-10   # multifrontal factor == left-looking factor (the relay is correct)
    end
end

@testitem "Multifrontal CPU LDLᵀ: L+D+inertia match ldlt! (design_gpu.md §6/§M, amendment E)" begin
    using PureSparse, SparseArrays, LinearAlgebra, Random
    include(joinpath(@__DIR__, "..", "ext", "multifrontal.jl"))
    function zsud!(x, S)
        for s in 1:S.nsuper
            nsc=Int(S.super[s+1])-Int(S.super[s]); nsr=Int(S.rowind_ptr[s+1])-Int(S.rowind_ptr[s]); b=Int(S.px[s])
            for j in 1:nsc, i in 1:(j-1); x[b+(j-1)*nsr+(i-1)]=0.0 end
        end; x
    end
    rng = MersenneTwister(0x1D1)
    kkt(n1,n2,f) = begin      # symmetric quasi-definite [H Aᵀ; A −D], H,D SPD
        H=sprand(rng,n1,n1,f); H=H+H'+2n1*I; Ac=sprand(rng,n2,n1,f)
        D=sprand(rng,n2,n2,f); D=D+D'+2n2*I
        ([H Ac'; Ac -D], n1, n2)
    end
    for (K,n1,n2,label) in [(kkt(150,80,0.04)...,"kkt_150_80"), (kkt(300,150,0.02)...,"kkt_300_150"),
                            (kkt(80,80,0.08)...,"kkt_80_80")]
        F = PureSparse.ldlt(K; n_pos=n1, n_neg=n2); @test PureSparse.issuccess(F)
        S = F.sym; M = mf_symbolic(S)
        xlen = Int(S.px[S.nsuper+1])-1
        xh=Vector{Float64}(undef,xlen); ar=Vector{Float64}(undef,max(M.arena_peak,1)); dv=Vector{Float64}(undef,S.n)
        ok,fc,st = cpu_multifrontal_ldlt!(xh,ar,dv,M,S,K,F.signs)
        @test ok
        relL = norm(zsud!(xh,S)-zsud!(copy(F.x),S))/norm(zsud!(copy(F.x),S))
        relD = norm(dv-F.d)/norm(F.d)
        @test relL < 1e-9      # unit-lower L matches ldlt! (dmax-independent)
        @test relD < 1e-9      # signed D matches
        @test (st.n_pos,st.n_neg,st.n_zero) == (F.stats.n_pos,F.stats.n_neg,F.stats.n_zero)  # well-scaled → inertia matches
    end
end
