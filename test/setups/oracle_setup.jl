# Shared test infrastructure (design.md §9.1 point 4c, §9.4): CHOLMOD oracle wrappers
# (observing CHOLMOD's OUTPUT via SparseArrays' public API is black-box and
# clean-room-safe per design.md §11 — its source is never read) and synthetic matrix
# generators for correctness/property tests across every M1+ test file.
@testsetup module OracleSetup
using SparseArrays
using LinearAlgebra
using Random

export cholmod_factor, cholmod_perm, cholmod_nnzL
export banded_spd, rand_spd, laplacian_2d, kkt_sqd
export relerr, upper_pattern

"""
    cholmod_factor(A::SparseMatrixCSC; perm=nothing) -> SparseArrays.CHOLMOD.Factor

Black-box oracle: factor `A` (symmetric, referenced via its lower triangle) with the
stdlib's CHOLMOD wrapper. `perm`, if given, forces that fill-reducing permutation
(1-based) — used for the same-permutation benchmark/comparison arm (design.md §9.3).
"""
function cholmod_factor(A::SparseMatrixCSC; perm = nothing)
    S = Symmetric(A, :L)
    return perm === nothing ? cholesky(S) : cholesky(S; perm = collect(Int, perm))
end

"""
    cholmod_perm(A::SparseMatrixCSC) -> Vector{Int}

CHOLMOD's own fill-reducing permutation for `A` (1-based), for AMD fill-quality
comparison (design.md §2.3 quality gate: our AMD's `nnz(L)` should be within ~1.15x of
CHOLMOD's).
"""
cholmod_perm(A::SparseMatrixCSC) = cholmod_factor(A).p

"""
    cholmod_nnzL(A::SparseMatrixCSC; perm=nothing) -> Int

`nnz(L)` from CHOLMOD's factorization (materializing the sparse `L` factor component).
"""
cholmod_nnzL(A::SparseMatrixCSC; perm = nothing) = nnz(sparse(cholmod_factor(A; perm).L))

"""
    upper_pattern(A::SparseMatrixCSC) -> (n, colptr, rowval)

Extract the STRICT UPPER triangular pattern of a full symmetric `SparseMatrixCSC` `A` in
1-based CSC form — the input contract `etree`/`column_counts`/ordering functions expect.
"""
function upper_pattern(A::SparseMatrixCSC)
    n = size(A, 1)
    colptr = Vector{Int}(undef, n + 1)
    rowval = Int[]
    colptr[1] = 1
    for j in 1:n
        for p in A.colptr[j]:(A.colptr[j + 1] - 1)
            i = A.rowval[p]
            i < j && push!(rowval, i)
        end
        colptr[j + 1] = length(rowval) + 1
    end
    return n, colptr, rowval
end

"""
    banded_spd(n, bw; rng=Random.default_rng(), T=Float64) -> SparseMatrixCSC{T}

Random banded SPD matrix (bandwidth `bw`, diagonally dominant via row-sum + 1 on the
diagonal — guarantees SPD for a symmetric matrix by Gershgorin's theorem).
"""
function banded_spd(n::Int, bw::Int; rng = Random.default_rng(), T::Type = Float64)
    I = Int[]; J = Int[]; V = T[]
    rowsum = zeros(T, n)
    for j in 1:n, i in max(1, j - bw):(j - 1)
        v = T(randn(rng))
        push!(I, i); push!(J, j); push!(V, v)
        push!(I, j); push!(J, i); push!(V, v)
        rowsum[i] += abs(v)
        rowsum[j] += abs(v)
    end
    for j in 1:n
        push!(I, j); push!(J, j); push!(V, rowsum[j] + one(T))
    end
    return sparse(I, J, V, n, n)
end

"""
    rand_spd(n, density; rng=Random.default_rng(), T=Float64) -> SparseMatrixCSC{T}

Random sparse SPD matrix at the given off-diagonal density, made SPD the same
diagonally-dominant way as [`banded_spd`](@ref).
"""
function rand_spd(n::Int, density::Real; rng = Random.default_rng(), T::Type = Float64)
    I = Int[]; J = Int[]; V = T[]
    rowsum = zeros(T, n)
    for j in 1:n, i in 1:(j - 1)
        rand(rng) < density || continue
        v = T(randn(rng))
        push!(I, i); push!(J, j); push!(V, v)
        push!(I, j); push!(J, i); push!(V, v)
        rowsum[i] += abs(v)
        rowsum[j] += abs(v)
    end
    for j in 1:n
        push!(I, j); push!(J, j); push!(V, rowsum[j] + one(T))
    end
    return sparse(I, J, V, n, n)
end

"""
    laplacian_2d(k; shift=1e-2, T=Float64) -> SparseMatrixCSC{T}

5-point graph Laplacian on a `k×k` grid (`n = k^2`), the classic sparse SPD stress
matrix — shifted by `shift·I` to be strictly SPD (the pure Laplacian is only PSD, its
constant vector is a null vector).
"""
function laplacian_2d(k::Int; shift::Real = 1.0e-2, T::Type = Float64)
    n = k * k
    idx(r, c) = (c - 1) * k + r
    I = Int[]; J = Int[]; V = T[]
    for c in 1:k, r in 1:k
        p = idx(r, c)
        deg = zero(T)
        for (dr, dc) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            rr, cc = r + dr, c + dc
            (1 <= rr <= k && 1 <= cc <= k) || continue
            q = idx(rr, cc)
            deg += one(T)
            if q > p   # store each off-diagonal edge once; symmetrize below
                push!(I, p); push!(J, q); push!(V, -one(T))
                push!(I, q); push!(J, p); push!(V, -one(T))
            end
        end
        push!(I, p); push!(J, p); push!(V, deg + T(shift))
    end
    return sparse(I, J, V, n, n)
end

"""
    kkt_sqd(nQ, m, density; rng=Random.default_rng(), T=Float64) -> SparseMatrixCSC{T}

Synthetic symmetric quasi-definite KKT system `[[Q, Aᵀ], [A, -D]]` (`n = nQ+m`): `Q` SPD
`nQ×nQ` (via [`rand_spd`](@ref)), `A` random sparse `m×nQ`, `D` SPD diagonal `m×m`. For M2
(design.md §5) — scaffolded now since it's independent of the LDLᵀ implementation.
"""
function kkt_sqd(nQ::Int, m::Int, density::Real; rng = Random.default_rng(), T::Type = Float64)
    Q = rand_spd(nQ, density; rng, T)
    Ia = Int[]; Ja = Int[]; Va = T[]
    for j in 1:nQ, i in 1:m
        rand(rng) < density || continue
        push!(Ia, i); push!(Ja, j); push!(Va, T(randn(rng)))
    end
    A = sparse(Ia, Ja, Va, m, nQ)
    D = Diagonal(abs.(randn(rng, T, m)) .+ one(T))
    n = nQ + m
    I = Int[]; J = Int[]; V = T[]
    for j in 1:nQ, p in Q.colptr[j]:(Q.colptr[j + 1] - 1)
        push!(I, Q.rowval[p]); push!(J, j); push!(V, Q.nzval[p])
    end
    for j in 1:nQ, p in A.colptr[j]:(A.colptr[j + 1] - 1)
        i = A.rowval[p]
        push!(I, nQ + i); push!(J, j); push!(V, A.nzval[p])
        push!(I, j); push!(J, nQ + i); push!(V, A.nzval[p])
    end
    for i in 1:m
        push!(I, nQ + i); push!(J, nQ + i); push!(V, -D[i, i])
    end
    return sparse(I, J, V, n, n)
end

relerr(a, b) = norm(a .- b) / max(norm(b), eps(Float64))

end
