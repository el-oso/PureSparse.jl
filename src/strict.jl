# Runtime pre/postcondition checks layer (design.md §9.1 D6 / CLAUDE.md req 6) — separate
# from contracts.jl's compile-time TypeContracts (see that file's header comment: this is
# the "separate StrictMode.jl layer" it names). `StrictMode.checks_enabled()` is a
# compile-time-baked constant (default `false`), so every check below folds away entirely
# when disabled — zero runtime cost, matching CLAUDE.md req 5 (zero-alloc after symbolic)
# and req 4 (no runtime-detected branching on hot paths).
#
# M5a task 10 found this layer didn't exist anywhere in the codebase yet — not even for
# `cholesky!`/`ldlt!` (M1/M2) despite CLAUDE.md req 6 requiring it project-wide — so this
# is the first instance, wired into all three `*!` refactor entry points at once
# (`cholesky!`, `ldlt!`, `qr!`) rather than adding it QR-only and leaving the rest
# inconsistent.
#
# Each check is split into a `StrictMode.checks_enabled()`-gated public entry point (used
# at real call sites) and an ungated `_impl` (the actual comparison/throw logic) — tests
# call the `_impl` directly so the check LOGIC itself is verified without needing
# StrictMode's Preferences-based enable/disable toggle (which requires a process restart
# to take effect, too heavy for a per-PR test run).

using StrictMode

# Single choke point for reporting a violation, honoring StrictMode's own `fail_mode`
# preference (`:error` throws, `:warn` warns and continues) — mirrors StrictMode's own
# internal `_fail` (not exported; this is our own copy of the same convention using the
# public `StrictViolation`/`fail_mode` API).
function _strict_fail(kind::Symbol, target, details::AbstractString)
    v = StrictMode.StrictViolation(kind, target, String(details))
    if StrictMode.fail_mode() === :warn
        @warn sprint(showerror, v)
        return nothing
    end
    throw(v)
end

"""
    check_refactor_shape(A::SparseMatrixCSC, expected_m::Integer, expected_n::Integer, label::String)

Precondition for every `*!` refactor entry point (`cholesky!`/`ldlt!`/`qr!`): `A`'s
dimensions must match the factor's symbolic analysis. Cheap (`O(1)`) and catches the
most common refactor misuse (passing the wrong matrix). Only runs when
`StrictMode.checks_enabled()`.
"""
function check_refactor_shape(A::SparseMatrixCSC, expected_m::Integer, expected_n::Integer, label::String)
    StrictMode.checks_enabled() || return nothing
    _check_refactor_shape_impl(A, expected_m, expected_n, label)
end

function _check_refactor_shape_impl(A::SparseMatrixCSC, expected_m::Integer, expected_n::Integer, label::String)
    m, n = size(A)
    (m == expected_m && n == expected_n) || _strict_fail(
        :refactor_shape, label,
        "A is $(m)×$(n), expected $(expected_m)×$(expected_n) (the shape the factor's " *
        "symbolic analysis was built for) — `$label` refactors in place and requires an " *
        "identically-shaped, identically-patterned matrix",
    )
    return nothing
end

"""
    check_refactor_nnz(A::SparseMatrixCSC, expected_nnz::Integer, label::String)

Precondition for `cholesky!`/`ldlt!`: `A` must share the sparsity pattern the factor's
symbolic analysis was built from. A full pattern comparison isn't available from
currently-stored data (`Symbolic` keeps no copy of the original `colptr`/`rowval`, only
the derived `amap`), so this checks the structural invariant that *is* cheaply
available — `nnz(A)` must equal what `amap` was built for — which catches the
overwhelming majority of real misuse (wrong matrix, regenerated pattern) without adding
new stored state. Does NOT catch a same-`nnz`, different-layout pattern; that would
require storing the original `colptr`/`rowval`, out of scope here. Only runs when
`StrictMode.checks_enabled()`.
"""
function check_refactor_nnz(A::SparseMatrixCSC, expected_nnz::Integer, label::String)
    StrictMode.checks_enabled() || return nothing
    _check_refactor_nnz_impl(A, expected_nnz, label)
end

function _check_refactor_nnz_impl(A::SparseMatrixCSC, expected_nnz::Integer, label::String)
    nnz(A) == expected_nnz || _strict_fail(
        :refactor_pattern, label,
        "A has nnz=$(nnz(A)), expected $(expected_nnz) (the pattern the factor's symbolic " *
        "analysis was built for) — `$label` requires the same sparsity pattern as the " *
        "original; build a fresh factor instead if the pattern changed",
    )
    return nothing
end

"""
    check_finite(v::AbstractVector, label::String)

Postcondition: no `NaN`/`Inf` leaked into a factor's stored numeric values. An `O(nnz)`
scan, too expensive to pay unconditionally on every factorization — only runs when
`StrictMode.checks_enabled()`.
"""
function check_finite(v::AbstractVector, label::String)
    StrictMode.checks_enabled() || return nothing
    _check_finite_impl(v, label)
end

function _check_finite_impl(v::AbstractVector, label::String)
    @inbounds for val in v
        isfinite(val) || _strict_fail(:finite_factor, label, "non-finite value ($val) in $label's stored factor values")
    end
    return nothing
end
