# Isolate the AMD compile segfault: is it the Atomix @atomic (and its Pair return), or the
# non-fused kernels? Tests, in order of suspicion.
using AMDGPU, KernelAbstractions, LinearAlgebra, Random, Printf
using KernelAbstractions: @kernel, @index, @localmem, @private, @synchronize, get_backend, @atomic
const BK = ROCBackend()
roc(A) = ROCArray(A)

# (a) minimal atomic — just the op, discard result
@kernel function _atom_a!(cnt)
    i = @index(Global)
    if i == 1
        @atomic cnt[1] += one(Int64)
    end
end
# (b) minimal atomic — capture old via .first (exactly what the fused kernels do)
@kernel function _atom_b!(cnt, out)
    i = @index(Global)
    if i == 1
        old = (@atomic cnt[1] += one(Int64)).first
        out[1] = old
    end
end
# (c) minimal atomic — Atomix returns Pair; try capturing new instead (no .first getfield)
@kernel function _atom_c!(cnt, out)
    i = @index(Global)
    if i == 1
        p = @atomic cnt[1] += one(Int64)
        out[1] = last(p)
    end
end

function try_kernel(name, launch)
    print("  ", rpad(name, 22))
    try
        launch(); KernelAbstractions.synchronize(BK)
        println("OK")
        return true
    catch e
        println("FAIL — ", sprint(showerror, e)[1:min(end, 120)])
        return false
    end
end

println("AMDGPU device: ", AMDGPU.device())
c = KernelAbstractions.zeros(BK, Int64, 1); o = KernelAbstractions.zeros(BK, Int64, 1)
try_kernel("(a) @atomic bare",   () -> _atom_a!(BK, 64)(c; ndrange = 64))
try_kernel("(b) @atomic .first",  () -> _atom_b!(BK, 64)(c, o; ndrange = 64))
try_kernel("(c) @atomic last()",  () -> _atom_c!(BK, 64)(c, o; ndrange = 64))
println("counter after runs: ", Array(c)[1])
