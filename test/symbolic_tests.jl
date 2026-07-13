@testsetup module SymbolicHelpers
using Random, SparseArrays
export random_symmetric_matrix, panel_value

# Random symmetric SparseMatrixCSC with a genuine (nonzero) diagonal. `store` controls
# whether both triangles are explicitly stored (`:full`) or only the lower (`:lower`) —
# `symbolic()`'s documented input convention (src/symbolic/driver.jl) must give IDENTICAL
# results either way, since only the lower triangle is ever read.
function random_symmetric_matrix(rng, n::Int, density::Float64; store::Symbol = :full)
    I = Int[]; J = Int[]; V = Float64[]
    for j in 1:n, i in (j + 1):n
        rand(rng) < density || continue
        v = randn(rng)
        push!(I, i); push!(J, j); push!(V, v)
        if store === :full
            push!(I, j); push!(J, i); push!(V, v)
        end
    end
    for j in 1:n
        push!(I, j); push!(J, j); push!(V, randn(rng) + 4.0)  # nonzero diagonal
    end
    return sparse(I, J, V, n, n)
end

# Read the value stored at GLOBAL (row=i, col=j), i>=j, out of a supernodal panel buffer
# `x` (post-`symbolic()` layout: sym.px/sym.rowind_ptr/sym.rowind/sym.super/sym.snode_of),
# or `nothing` if that row isn't part of column j's supernode panel (shouldn't happen for
# any TRUE stored entry, by the superset invariant).
function panel_value(sym, x, i::Int, j::Int)
    s = sym.snode_of[j]
    j0 = sym.super[s]
    nrow = Int(sym.rowind_ptr[s + 1] - sym.rowind_ptr[s])
    local_row = 0
    for (k, r) in enumerate(sym.rowind_ptr[s]:(sym.rowind_ptr[s + 1] - 1))
        if sym.rowind[r] == i
            local_row = k
            break
        end
    end
    local_row == 0 && return nothing
    local_col = j - j0 + 1
    return x[sym.px[s] + (local_col - 1) * nrow + (local_row - 1)]
end
end

@testitem "symbolic(): Symbolic struct is internally consistent" setup = [SymbolicHelpers] begin
    using Random, SparseArrays
    rng = MersenneTwister(51)
    for n in (1, 2, 5, 10, 30, 60), density in (0.0, 0.05, 0.2, 0.5)
        A = random_symmetric_matrix(rng, n, density)
        sym = PureSparse.symbolic(A)

        @test sym.n == n
        @test sort(sym.perm) == collect(1:n)
        @test all(sym.iperm[sym.perm[k]] == k for k in 1:n)

        @test sym.super[1] == 1
        @test sym.super[sym.nsuper + 1] == n + 1
        @test issorted(sym.super)

        @test sym.px[1] == 1
        @test sym.rowind_ptr[1] == 1
        for s in 1:sym.nsuper
            nrow = Int(sym.rowind_ptr[s + 1] - sym.rowind_ptr[s])
            ncol = Int(sym.super[s + 1] - sym.super[s])
            @test sym.px[s + 1] - sym.px[s] == nrow * ncol
        end
        @test sym.nnzL == sum(sym.colcount)
        @test sym.flops == sum(abs2, sym.colcount)
        @test length(sym.amap) == nnz(A)
    end
end

@testitem "symbolic(): identical results for lower-only vs fully-stored input" setup = [SymbolicHelpers] begin
    using Random, SparseArrays
    rng = MersenneTwister(52)
    for n in (5, 20, 50), density in (0.05, 0.2)
        rng2 = MersenneTwister(hash((n, density)))  # same edges/values for both stores
        Afull = random_symmetric_matrix(rng2, n, density; store = :full)
        rng2 = MersenneTwister(hash((n, density)))
        Alower = random_symmetric_matrix(rng2, n, density; store = :lower)

        symf = PureSparse.symbolic(Afull)
        syml = PureSparse.symbolic(Alower)

        @test symf.perm == syml.perm
        @test symf.super == syml.super
        @test symf.rowind == syml.rowind
        @test symf.colcount == syml.colcount
    end
end

@testitem "symbolic(): amap round-trips every stored lower-triangle entry" setup = [SymbolicHelpers] begin
    using Random, SparseArrays
    rng = MersenneTwister(53)
    for n in (1, 2, 5, 10, 30, 60), density in (0.0, 0.05, 0.2, 0.5)
        A = random_symmetric_matrix(rng, n, density)
        sym = PureSparse.symbolic(A)

        xsize = sym.px[sym.nsuper + 1] - 1
        x = zeros(Float64, xsize)
        nfilled = 0
        for p in 1:nnz(A)
            m = sym.amap[p]
            m == 0 && continue
            x[m] = A.nzval[p]
            nfilled += 1
        end
        # Exactly nnz(tril(A)) entries got a nonzero amap slot.
        ntril_true = 0
        for j in 1:n, p in A.colptr[j]:(A.colptr[j + 1] - 1)
            A.rowval[p] >= j && (ntril_true += 1)
        end
        @test nfilled == ntril_true

        # Every original stored lower-triangle entry (i,j), i>=j, must read back correctly
        # from the panel at its PERMUTED position.
        for j in 1:n, p in A.colptr[j]:(A.colptr[j + 1] - 1)
            i = A.rowval[p]
            i >= j || continue
            ni, nj = sym.iperm[i], sym.iperm[j]
            lo, hi = ni < nj ? (ni, nj) : (nj, ni)
            v = panel_value(sym, x, hi, lo)
            @test v !== nothing
            @test v == A.nzval[p]
        end
    end
end

@testitem "full_symmetric_pattern produces a genuinely symmetric full pattern" begin
    using Random
    rng = MersenneTwister(54)
    for n in (1, 2, 5, 10, 30), density in (0.0, 0.1, 0.3)
        # one-triangle-only strict-upper input (reuse the etree-style CSC construction)
        adj = [Set{Int}() for _ in 1:n]
        for j in 1:n, i in 1:(j - 1)
            rand(rng) < density && (push!(adj[i], j); push!(adj[j], i))
        end
        colptr = Vector{Int}(undef, n + 1)
        rowval = Int[]
        colptr[1] = 1
        for j in 1:n
            append!(rowval, sort!([i for i in adj[j] if i < j]))
            colptr[j + 1] = length(rowval) + 1
        end
        cp2, rv2 = PureSparse.full_symmetric_pattern(n, colptr, rowval)
        @test cp2[1] == 1
        @test cp2[n + 1] - 1 == length(rv2)
        for j in 1:n
            rows = rv2[cp2[j]:(cp2[j + 1] - 1)]
            @test issorted(rows)
            @test allunique(rows)
            @test j ∉ rows   # no diagonal
        end
        # symmetry: (i,j) present iff (j,i) present
        edges = Set{Tuple{Int,Int}}()
        for j in 1:n, p in cp2[j]:(cp2[j + 1] - 1)
            push!(edges, (rv2[p], j))
        end
        for (i, j) in edges
            @test (j, i) in edges
        end
    end
end
