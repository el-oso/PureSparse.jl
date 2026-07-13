# Always included (unlike dropin.jl itself, design.md §10 M4) so `activate!`/
# `deactivate!` exist regardless of the current Preference state — a user who hasn't
# activated the drop-in yet still needs to be able to call `activate!()` to turn it on
# for next time. See tuning.jl's `DROPIN_ACTIVE` derivation for why this can't be a
# same-session runtime toggle the way PureBLAS's `activate()`/`deactivate()` are.

"""
    activate!()

Opt in to the drop-in (design.md §10 M4): sets the `dropin_active` Preference so that,
from the NEXT Julia session onward, loading `PureSparse` also extends
`LinearAlgebra.cholesky`/`ldlt` for `SparseMatrixCSC`/`Symmetric`/`Hermitian` inputs to
run PureSparse's own factorizations instead of CHOLMOD. Requires a restart — Julia's
method table has no runtime "activate this override" primitive without `eval`
(forbidden, CLAUDE.md requirement 4); see `src/tuning.jl`'s `DROPIN_ACTIVE` comment.
"""
function activate!()
    Preferences.set_preferences!(@__MODULE__, "dropin_active" => true; force = true)
    @info "PureSparse.activate!(): dropin_active=true set. Restart Julia for LinearAlgebra.cholesky/ldlt on sparse matrices to route through PureSparse."
    return nothing
end

"""
    deactivate!()

Opt back out of the drop-in: sets `dropin_active = false`. Requires a restart (see
[`activate!`](@ref)).
"""
function deactivate!()
    Preferences.set_preferences!(@__MODULE__, "dropin_active" => false; force = true)
    @info "PureSparse.deactivate!(): dropin_active=false set. Restart Julia for LinearAlgebra.cholesky/ldlt on sparse matrices to route through CHOLMOD again."
    return nothing
end
