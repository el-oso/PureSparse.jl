# AD-support extension (P2, CLAUDE.md req 3): the frontal path is generic over real
# isbits `T`, but `QRStats.dropped_norm` is a Float64 diagnostic (a rank certificate,
# NOT on the differentiable path). `Float64(::ForwardDiff.Dual)` is deliberately
# undefined — dropping a derivative silently would be a bug — so `PureSparse._stat_f64`
# (base method `_stat_f64(::Real) = Float64(x)`) can't handle Duals from `src` alone.
# This more-specific method returns the Dual's primal (recursing, since a nested Dual's
# value is itself a Real/Dual), keeping `src` free of any ForwardDiff dependency.
module PureSparseForwardDiffExt

using PureSparse: PureSparse
using ForwardDiff: Dual, value

PureSparse._stat_f64(x::Dual) = PureSparse._stat_f64(value(x))

end
