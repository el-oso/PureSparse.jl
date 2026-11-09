# PureSparse GPU offload — Apple Metal/Metal.jl weak-dep extension. Loaded only when a user has
# BOTH Metal and KernelAbstractions. Shares the backend-generic engine (ext/gpu_shared.jl) with
# the CUDA and AMDGPU extensions; this file only supplies the Metal device.
#
# Apple GPUs have no FP64: Metal.jl refuses a Float64 array at allocation ("Metal does not
# support Float64 values"), so this backend factors and solves Float32 matrices only.
module PureSparseMetalExt

using PureSparse: PureSparse
using Metal
using KernelAbstractions

_default_backend() = MetalBackend()         # the shim's device for this extension
_vendor_available() = false                  # no cuSOLVER/cuBLAS on Metal → :vendor arm errors clearly

include("gpu_shared.jl")                     # backend-generic engine (kernels + symbolic + numeric)

end # module
