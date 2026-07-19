# PureSparse GPU offload — AMD ROCm/AMDGPU.jl weak-dep extension (design_gpu_multibackend.md, M8).
# Loaded only when a user has BOTH AMDGPU and KernelAbstractions. Shares the entire backend-generic
# engine (ext/gpu_shared.jl) with PureSparseCUDAExt — the pure KA kernels (proven machine-precision
# on gfx1151/gfx1152, benchmark/gpu/amd_kernel_test.jl) + the multifrontal driver, which reaches the
# device only through the §B1 shim. This file only supplies the ROCm device.
#
# Scope: end-to-end Cholesky + LDLᵀ on ROCm (correct; FP64 unoptimized — the Instinct-class MFMA
# tuning is the deferred M8 optimization half). The CUDA-only reference/vendor arms are NOT included
# here (they call cuSOLVER/cuBLAS); the shipped :auto path uses only the pure KA fronts.
module PureSparseAMDGPUExt

using PureSparse: PureSparse
using AMDGPU
using KernelAbstractions

_default_backend() = ROCBackend()           # the shim's device for this extension

include("gpu_shared.jl")                     # backend-generic engine (kernels + symbolic + numeric)

end # module
