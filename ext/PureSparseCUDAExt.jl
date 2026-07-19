# PureSparse GPU offload (M6) — CUDA/KernelAbstractions weak-dep extension.
# design_gpu.md. Loaded only when a user has BOTH CUDA and KernelAbstractions.
# Core `src/` has zero GPU hooks; this ext adds the GPU factor path to PureSparse's generic entry
# points. The engine is backend-generic (ext/gpu_shared.jl, design_gpu_multibackend.md §B1) and
# shared with PureSparseAMDGPUExt; this file only supplies the CUDA device (`_default_backend`)
# plus the CUDA-only reference arms (cuSOLVER/cuBLAS left-looking + vendor solve).
module PureSparseCUDAExt

using PureSparse: PureSparse
using CUDA
using KernelAbstractions

_default_backend() = CUDABackend()          # the shim's device for this extension
_vendor_available() = true                   # cuSOLVER/cuBLAS present → the :vendor reference arm runs

include("gpu_shared.jl")                     # backend-generic engine (kernels + symbolic + numeric)
include("gpu_leftlooking_reference.jl")      # CUDA-only §4 reference arms (cuSOLVER potrf!/cuBLAS trsm!)
include("gpu_vendor_solve.jl")               # CUDA-only §8-gate reference solve (cuBLAS) — never shipped

end # module
