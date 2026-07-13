# juliac entry point for the trimmed executable: the M1 "factor-and-solve" smoke
# (design.md §10, ROADMAP M1 gate). Builds a small SPD 2-D Laplacian, runs
# symbolic → cholesky → solve!, refactorizes in place (cholesky!), then the LDLᵀ path
# (ldlt → solve!) on the same Symbolic. Exits 0 iff every residual ∞-norm is small.
# Run via `julia juliac/build.jl`, then `juliac/build/puresparse_smoke`.
using PureSparse
using SparseArrays

# 2-D Laplacian (5-point stencil) on an m×m grid: SPD, the M1 FEM-class smoke matrix.
# CSC arrays are built directly (each column's stencil rows k−m < k−1 < k < k+1 < k+m are
# emitted in sorted order) rather than via `sparse(I,J,V,…)`, whose coalescing path takes
# an abstract `combine::Function` — an unresolved call under --trim=safe.
function laplacian2d(m::Int)
    n = m * m
    colptr = Vector{Int64}(undef, n + 1)
    rowval = Int64[]
    nzval = Float64[]
    colptr[1] = 1
    for j in 1:m, i in 1:m
        k = Int64((j - 1) * m + i)
        for (i2, j2, v) in ((i, j - 1, -1.0), (i - 1, j, -1.0), (i, j, 4.0),
                (i + 1, j, -1.0), (i, j + 1, -1.0))
            if 1 <= i2 <= m && 1 <= j2 <= m
                push!(rowval, Int64((j2 - 1) * m + i2))
                push!(nzval, v)
            end
        end
        colptr[k + 1] = Int64(length(rowval) + 1)
    end
    return SparseMatrixCSC{Float64, Int64}(n, n, colptr, rowval, nzval)
end

# Hand-rolled ∞-norm residual ‖b − A·x‖∞ over raw CSC storage — keeps the check itself
# trivially trim-safe instead of pulling stdlib SpMV dispatch into the trimmed image.
function residual_inf(A::SparseMatrixCSC{Float64, Int64}, x::Vector{Float64}, b::Vector{Float64})
    r = copy(b)
    colptr = A.colptr; rowval = A.rowval; nzval = A.nzval
    @inbounds for j in 1:(A.n)
        xj = x[j]
        for p in colptr[j]:(colptr[j + 1] - 1)
            r[rowval[p]] -= nzval[p] * xj
        end
    end
    m = 0.0
    @inbounds for k in 1:length(r)
        m = max(m, abs(r[k]))
    end
    return m
end

function (@main)(argv::Vector{String})::Cint
    # Print via the concrete Core.stdout: bare `println` routes through the abstract
    # `Base.stdout::IO` global — an unresolved call under --trim=safe.
    out = Core.stdout
    A = laplacian2d(12)                       # n = 144
    n = size(A, 1)
    b = ones(Float64, n)
    x = zeros(Float64, n)
    tol = 1.0e-10
    ok = true

    sym = symbolic(A)

    F = cholesky(sym, A)
    solve!(x, F, b)
    r1 = residual_inf(A, x, b)
    println(out, "cholesky  residual_inf = ", r1)
    ok &= issuccess(F) && r1 < tol

    cholesky!(F, A)                           # analyze-once / refactorize path
    solve!(x, F, b)
    r2 = residual_inf(A, x, b)
    println(out, "cholesky! residual_inf = ", r2)
    ok &= issuccess(F) && r2 < tol

    G = ldlt(sym, A)
    solve!(x, G, b)
    r3 = residual_inf(A, x, b)
    println(out, "ldlt      residual_inf = ", r3)
    ok &= issuccess(G) && r3 < tol

    println(out, ok ? "PureSparse trim smoke: OK" : "PureSparse trim smoke: FAIL")
    return ok ? Cint(0) : Cint(1)
end
