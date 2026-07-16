# Verify the cuSOLVER potrf! / cuBLAS trsm! API + semantics for the supernode diagonal
# factorization + panel solve (design_gpu.md §4): lower Cholesky in place, then the
# below-diagonal panel solve L21 = A21 * L11⁻ᵀ. Compare to a CPU dense factor.
using CUDA, LinearAlgebra
using CUDA.CUSOLVER: potrf!
using CUDA.CUBLAS: trsm!

# Build an SPD "supernode panel": [A11 (nscol×nscol) ; A21 (below)] stacked, nsrow×nscol.
nscol, nsrow = 40, 130
Random_seed = 0
X = randn(nsrow, nscol)
G = X' * X + nscol * I               # nscol×nscol SPD (the diagonal block Gram)
A21 = randn(nsrow - nscol, nscol)    # arbitrary below-diagonal

# CPU reference: L11 = chol(G).L; L21 = A21 * inv(L11)'  (i.e. solve L21 L11' = A21)
F = cholesky(Matrix(G))
L11_cpu = Matrix(F.L)
L21_cpu = A21 / L11_cpu'             # A21 * inv(L11')

# Device: pack panel as a RESHAPE-OF-VIEW (the exact structure gpu_cholesky_sync! uses — the
# panel is a strided view into the packed factor storage, lda = nsrow ≠ nscol). This exercises
# cuSOLVER's leading-dimension handling on a non-contiguous diagonal block.
storage = CUDA.zeros(Float64, 12000)
off = 211
packed = vcat(Matrix(G), A21)                 # nsrow×nscol column-major
copyto!(view(storage, off:(off + nsrow*nscol - 1)), vec(packed))
panel = reshape(view(storage, off:(off + nsrow*nscol - 1)), nsrow, nscol)
diagblk = view(panel, 1:nscol, 1:nscol)
_, info = potrf!('L', diagblk)                 # in-place lower Cholesky of the diag block
@assert info == 0 "potrf! info=$info"
sub = view(panel, nscol+1:nsrow, 1:nscol)
# solve sub * L11' = A21  →  side=R, uplo=L, transA=T (L11'), diag=N
trsm!('R', 'L', 'T', 'N', 1.0, diagblk, sub)
CUDA.synchronize()

L11_gpu = Array(view(panel, 1:nscol, 1:nscol))
# zero the strict upper (potrf! leaves the original upper untouched; CPU L is lower)
for i in 1:nscol, j in i+1:nscol; L11_gpu[i, j] = 0.0; end
L21_gpu = Array(view(panel, nscol+1:nsrow, 1:nscol))

e11 = norm(L11_gpu - L11_cpu) / norm(L11_cpu)
e21 = norm(L21_gpu - L21_cpu) / norm(L21_cpu)
println("potrf! L11 relerr = ", e11)
println("trsm!  L21 relerr = ", e21)
@assert e11 < 1e-12 "L11 mismatch"
@assert e21 < 1e-12 "L21 mismatch"
println("cuSOLVER potrf! + cuBLAS trsm! API CONFIRMED (lower Cholesky + panel solve match CPU)")
