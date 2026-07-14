# Synthetic sparse QR gate matrix generators (design_qr.md §9.4), stratified into the
# three classes the gate's stated expectation (H4) is measured against: (i) singleton-
# dominated/LP-like, (ii) sparse-R/small-front least-squares, (iii) flop-rich/large-front
# least-squares. All synthetic — design_qr.md §9.4 explicitly permits this for the
# CI/gate matrix set ("sizes capped to CI-tolerable downloads... large stratum-(iii)
# instances live in the performance set, not CI"), same permission `matrices.jl` already
# uses for the M1 Cholesky gate (no SuiteSparse Collection downloader exists yet, for
# either milestone).

using SparseArrays, Random

"""
    lp_slack_matrix(n_constraints, n_vars, density; rng) -> SparseMatrixCSC

Stratum (i): LP-constraint shape `[A_dense | I_slack]` (SPQR paper §2.1's motivating
class) — `n_constraints` slack columns are immediate singletons (each a scaled unit
vector on its own row), plus `n_vars` "structural" columns with random sparse coupling
into the same `n_constraints` rows. Peeling removes every slack column (`n_vars` of the
`n_vars + n_constraints` total) before the structural block is touched at all.
"""
function lp_slack_matrix(n_constraints::Int, n_vars::Int, density::Float64; rng::AbstractRNG = Random.default_rng())
    m = n_constraints
    n = n_vars + n_constraints
    I = Int[]; J = Int[]; V = Float64[]
    for j in 1:n_vars
        for i in 1:m
            rand(rng) < density || continue
            push!(I, i); push!(J, j); push!(V, randn(rng))
        end
    end
    for k in 1:n_constraints
        push!(I, k); push!(J, n_vars + k); push!(V, 1.0 + rand(rng))
    end
    return sparse(I, J, V, m, n)
end

"""
    staircase_singleton_matrix(n; rng) -> SparseMatrixCSC

Stratum (i), the SPQR paper's "matrix becomes entirely singletons" extreme (215/353 of
its LP collection): column `j`'s only entries are its own row `j` and row `j-1` — column
1 is an immediate singleton, and removing its row makes column 2 a singleton in turn,
cascading strictly 1..n. The WHOLE factorization needs zero numerical work (§2.3's own
derivation) — the limit case stratum (i)'s expectation (H4) is about.
"""
function staircase_singleton_matrix(n::Int; rng::AbstractRNG = Random.default_rng())
    I = Int[]; J = Int[]; V = Float64[]
    for j in 1:n
        push!(I, j); push!(J, j); push!(V, 1.0 + rand(rng))
        j > 1 && (push!(I, j - 1); push!(J, j); push!(V, 0.3 * randn(rng)))
    end
    return sparse(I, J, V, n, n)
end

"""
    banded_ls(m, n, bandwidth, density; rng) -> SparseMatrixCSC

Stratum (ii): banded tall rectangular matrix (models a locally-connected LS problem —
e.g. a 1-D surveying/curve-fit design matrix) — small elimination-tree fronts, sparse
`R`. Every column guaranteed at least one entry (its own band center) so no column is
structurally empty.
"""
function banded_ls(m::Int, n::Int, bandwidth::Int, density::Float64; rng::AbstractRNG = Random.default_rng())
    I = Int[]; J = Int[]; V = Float64[]
    for j in 1:n
        center = clamp(round(Int, j * m / n), 1, m)
        push!(I, center); push!(J, j); push!(V, 1.0 + rand(rng))
        for i in max(1, center - bandwidth):min(m, center + bandwidth)
            i == center && continue
            rand(rng) < density || continue
            push!(I, i); push!(J, j); push!(V, randn(rng))
        end
    end
    return sparse(I, J, V, m, n)
end

"""
    grid_ls(nx, ny) -> SparseMatrixCSC

Stratum (ii): 2-D-grid surveying-type LS matrix (design_qr.md §9.4) — each of the
`nx*ny` interior grid points contributes two rows (a horizontal and vertical finite-
difference-style observation against its neighbors), an `m ≈ 2n` overdetermined system
with small, spatially-local fronts.
"""
function grid_ls(nx::Int, ny::Int)
    n = nx * ny
    idx(i, j) = (j - 1) * nx + i
    I = Int[]; J = Int[]; V = Float64[]
    row = 0
    for j in 1:ny, i in 1:nx
        p = idx(i, j)
        if i < nx
            row += 1
            push!(I, row); push!(J, p); push!(V, 1.0)
            push!(I, row); push!(J, idx(i + 1, j)); push!(V, -1.0)
        end
        if j < ny
            row += 1
            push!(I, row); push!(J, p); push!(V, 1.0)
            push!(I, row); push!(J, idx(i, j + 1)); push!(V, -1.0)
        end
    end
    return sparse(I, J, V, row, n)
end

"""
    dense_arrow_ls(m, n, ndense, density; rng) -> SparseMatrixCSC

Stratum (iii): mostly-sparse tall matrix plus `ndense` fully-dense columns — the dense
columns force a single large elimination-tree front covering nearly the whole factor
(the "flop-rich/large-front" class §9.3's H4 stated expectation names as M5a's
plausible loss stratum, since multifrontal BLAS-3 is what earns its keep here).
"""
function dense_arrow_ls(m::Int, n::Int, ndense::Int, density::Float64; rng::AbstractRNG = Random.default_rng())
    I = Int[]; J = Int[]; V = Float64[]
    for j in 1:(n - ndense)
        for i in 1:m
            rand(rng) < density || continue
            push!(I, i); push!(J, j); push!(V, randn(rng))
        end
    end
    for j in (n - ndense + 1):n
        for i in 1:m
            push!(I, i); push!(J, j); push!(V, randn(rng))
        end
    end
    return sparse(I, J, V, m, n)
end

"""
    random_tall_ls(m, n, density; rng) -> SparseMatrixCSC

Stratum (iii): plain random tall sparse LS matrix at moderate-to-high density (design_qr.md
§9.4's "random tall sparse" synthetic) — no exploitable sparsity structure, close to a
worst-case front size for its shape.
"""
function random_tall_ls(m::Int, n::Int, density::Float64; rng::AbstractRNG = Random.default_rng())
    return sprand(rng, m, n, density)
end

"""
    QR_GATE_MATRICES

The M5a gate/CI matrix set (design_qr.md §9.3/§9.4): named `(label, A, stratum)` tuples
spanning the three strata H4 stratifies the gate verdict by.
"""
function qr_gate_matrices()
    rng = MersenneTwister(2026)
    return [
        ("lp_slack_n300x60", lp_slack_matrix(300, 60, 0.05; rng), "i_singleton"),
        ("lp_slack_n800x150", lp_slack_matrix(800, 150, 0.03; rng), "i_singleton"),
        ("staircase_n2000", staircase_singleton_matrix(2000; rng), "i_singleton"),
        ("banded_ls_n1500x500_bw15", banded_ls(1500, 500, 15, 0.3; rng), "ii_sparse_R"),
        ("grid_ls_40x30", grid_ls(40, 30), "ii_sparse_R"),
        ("grid_ls_70x50", grid_ls(70, 50), "ii_sparse_R"),
        ("dense_arrow_n800x200_d8dense", dense_arrow_ls(800, 200, 8, 0.02; rng), "iii_flop_rich"),
        ("random_tall_n1200x300_d05", random_tall_ls(1200, 300, 0.05; rng), "iii_flop_rich"),
    ]
end
