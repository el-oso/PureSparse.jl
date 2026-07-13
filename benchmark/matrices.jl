# Synthetic SPD matrix generators for the benchmark harness (design.md §9.4 CI set — "no
# download needed" generators). A curated SuiteSparse Collection download (§9.1 point 6,
# the correctness-test zoo) is a separate, not-yet-built piece of test infrastructure;
# these generators are what the M1 gate (§9.3) runs against for now, matching the design's
# explicit permission to use synthetics for the CI/gate matrix set.

using SparseArrays, Random

"""
    random_spd(n, density; rng) -> SparseMatrixCSC

Random sparse SPD matrix: random off-diagonal entries below the diagonal at the given
density, diagonal set to each row's absolute off-diagonal sum + 1 (diagonally dominant ⟹
SPD). Same construction as `test/llt_tests.jl`'s `LLTHelpers.random_spd_matrix`.
"""
function random_spd(n::Int, density::Float64; rng::AbstractRNG = Random.default_rng())
    I = Int[]; J = Int[]; V = Float64[]
    rowsum = zeros(n)
    for j in 1:n, i in (j + 1):n
        rand(rng) < density || continue
        v = randn(rng)
        push!(I, i); push!(J, j); push!(V, v)
        rowsum[i] += abs(v); rowsum[j] += abs(v)
    end
    for j in 1:n
        push!(I, j); push!(J, j); push!(V, rowsum[j] + 1.0)
    end
    return sparse(I, J, V, n, n)
end

"""
    banded_spd(n, bandwidth; rng) -> SparseMatrixCSC

Random SPD matrix with all nonzeros confined to `bandwidth` below the diagonal (models
FEM/structural stiffness matrices from locally-connected meshes). Diagonally dominant.
"""
function banded_spd(n::Int, bandwidth::Int; rng::AbstractRNG = Random.default_rng())
    I = Int[]; J = Int[]; V = Float64[]
    rowsum = zeros(n)
    for j in 1:n, i in (j + 1):min(j + bandwidth, n)
        v = randn(rng)
        push!(I, i); push!(J, j); push!(V, v)
        rowsum[i] += abs(v); rowsum[j] += abs(v)
    end
    for j in 1:n
        push!(I, j); push!(J, j); push!(V, rowsum[j] + 1.0)
    end
    return sparse(I, J, V, n, n)
end

"""
    laplacian2d(nx, ny) -> SparseMatrixCSC

Standard 5-point graph Laplacian of an `nx × ny` grid (Dirichlet boundary), SPD by
construction. Models the graph-Laplacian/FEM matrix class from design.md §9.4.
"""
function laplacian2d(nx::Int, ny::Int)
    n = nx * ny
    idx(i, j) = (j - 1) * nx + i
    I = Int[]; J = Int[]; V = Float64[]
    for j in 1:ny, i in 1:nx
        p = idx(i, j)
        # Dirichlet boundary: every node has degree 4 (missing grid neighbors are fixed
        # zero-value boundary nodes, absorbed into the diagonal, not stored) — this is what
        # makes the operator strictly diagonally dominant (hence SPD) rather than the
        # singular pure graph Laplacian (row sums to zero, degree = in-grid neighbor count).
        i > 1 && (q = idx(i - 1, j); push!(I, p); push!(J, q); push!(V, -1.0))
        j > 1 && (q = idx(i, j - 1); push!(I, p); push!(J, q); push!(V, -1.0))
        push!(I, p); push!(J, p); push!(V, 4.0)
    end
    return sparse(I, J, V, n, n)
end

"""
    GATE_MATRICES

The M1 gate/CI matrix set (design.md §9.3/§9.4): named `(label, A)` pairs spanning the
KKT/FEM/Laplacian classes at sizes from small to moderately large, all synthetic (no
download).
"""
function gate_matrices()
    rng = MersenneTwister(2026)
    return [
        ("random_n200_d02", random_spd(200, 0.02; rng)),
        ("random_n500_d01", random_spd(500, 0.01; rng)),
        ("random_n1000_d005", random_spd(1000, 0.005; rng)),
        ("banded_n1000_bw20", banded_spd(1000, 20; rng)),
        ("banded_n3000_bw10", banded_spd(3000, 10; rng)),
        ("laplacian2d_40x40", laplacian2d(40, 40)),
        ("laplacian2d_80x80", laplacian2d(80, 80)),
    ]
end

"""
    random_sqd_kkt(npos, nneg, density; rng) -> SparseMatrixCSC

Random symmetric quasi-definite KKT matrix `[H Aᵀ; A −C]` with `H` (`npos × npos`) and
`C` (`nneg × nneg`) diagonally-dominant SPD and a random sparse coupling block `A` —
the M2 SQD gate class (design.md §9.4's synthetic-KKT set), models an interior-point
iterate's KKT system. Same construction as `test/ldlt_tests.jl`'s
`LDLTHelpers.random_sqd_kkt`. Vanderbei 1995: strongly factorizable, inertia exactly
`(npos, nneg, 0)`.
"""
function random_sqd_kkt(npos::Int, nneg::Int, density::Float64; rng::AbstractRNG = Random.default_rng())
    n = npos + nneg
    I = Int[]; J = Int[]; V = Float64[]
    rowsum = zeros(n)
    addsym!(i, j, v) = (push!(I, i); push!(J, j); push!(V, v);
        i != j && (push!(I, j); push!(J, i); push!(V, v));
        rowsum[i] += abs(v); i != j && (rowsum[j] += abs(v)))
    for j in 1:npos, i in (j + 1):npos
        rand(rng) < density && addsym!(i, j, randn(rng))
    end
    for j in 1:nneg, i in (j + 1):nneg
        rand(rng) < density && addsym!(npos + i, npos + j, randn(rng))
    end
    for j in 1:npos, i in 1:nneg
        rand(rng) < density && addsym!(npos + i, j, randn(rng))
    end
    for j in 1:n
        v = (rowsum[j] + 1.0) * (j <= npos ? 1.0 : -1.0)
        push!(I, j); push!(J, j); push!(V, v)
    end
    return sparse(I, J, V, n, n)
end

"""
    sqd_gate_matrices()

The M2 SQD gate matrix set (design.md §9.4): named `(label, A, n_pos, n_neg)` tuples —
`n_pos`/`n_neg` are needed by [`PureSparse.ldlt`](@ref)'s convenience constructor and by
CHOLMOD's own `perm=`/inertia-free sparse `ldlt`.
"""
function sqd_gate_matrices()
    rng = MersenneTwister(2026)
    return [
        ("sqd_n200_d02", random_sqd_kkt(120, 80, 0.02; rng), 120, 80),
        ("sqd_n500_d01", random_sqd_kkt(300, 200, 0.01; rng), 300, 200),
        ("sqd_n1000_d005", random_sqd_kkt(600, 400, 0.005; rng), 600, 400),
        ("sqd_n2000_d002", random_sqd_kkt(1200, 800, 0.002; rng), 1200, 800),
    ]
end
