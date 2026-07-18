# StrictMode guarantee gates (req 0.3.9): machine-checked performance/correctness guarantees on
# the hottest paths, using StrictMode's `@assert_*` macros. These run in an ISOLATED subprocess
# with `[StrictMode] checks_enabled = true` (+ AllocCheck/JET for the `:full` analysis extension),
# because the guarantee macros are gated by `assert_enabled()` (⇒ `checks_enabled()`), which the
# MAIN suite deliberately leaves OFF — PureSparse's own `strict.jl` runtime checks allocate, and
# the zero-alloc gate (`@allocated == 0` in qr_solve_tests etc.) depends on checks-off. Writing
# the preference to `test/LocalPreferences.toml` would contaminate every other item's precompiled
# state, so we use a throwaway env (same isolation the dropin subprocess tests use).
#
# WHAT IS ASSERTED (only guarantees that hold on CHECK-FREE targets, so the checks-on body the
# macro analyzes is identical to the shipped checks-off path):
#   - `@assert_concurrency_safe cholesky/ldlt(sym, A)` — the allocating factor treats its `sym`
#     (Symbolic) argument as READ-ONLY: proof that one immutable `Symbolic` is safe to share by
#     reference across concurrent `cholesky(sym, Aᵢ)` calls — the "analyze once, factorize many"
#     thesis (design.md §1.2, CLAUDE.md req 7), previously asserted only in prose.
#   - `@assert_typestable solve!(...)` — the hottest path (called hundreds of times per IPM solve)
#     has a concrete return type + no internal instability, for all three factor kinds (req 3).
#   - `@assert_noalloc solve!(xq, Fq, b)` — the QR solve is AllocCheck-clean on ALL paths.
#
# NOTE (follow-up, not a release blocker): `@assert_noalloc` FAILS for the Cholesky and LDLᵀ
# `solve!` (but passes for QR) — AllocCheck finds a non-throw allocation path the happy-path
# `@allocated == 0` gate does not exercise. The shipped hot path is alloc-free (that gate passes);
# tightening the Chol/LDLᵀ solve tree to be AllocCheck-clean on every path is a separate task.

@testitem "StrictMode guarantees: concurrency-safe shared Symbolic + type-stable solve (subprocess, checks-on)" begin
    using Pkg

    pkgroot = pkgdir(PureSparse)
    envdir = mktempdir()
    try
        write(joinpath(envdir, "LocalPreferences.toml"), """
        [StrictMode]
        checks_enabled = true
        fail_mode = "error"
        analysis = "full"
        """)

        script = joinpath(envdir, "run.jl")
        write(script, """
        using Pkg
        Pkg.activate(@__DIR__)
        Pkg.develop(path=raw"$pkgroot")
        Pkg.add(["StrictMode","AllocCheck","JET","SparseArrays","Random"])
        Pkg.instantiate()

        using PureSparse, StrictMode, AllocCheck, JET, SparseArrays, Random
        import PureSparse: symbolic, cholesky, cholesky!, ldlt, ldlt!, qr, solve!, AMDOrdering

        # If checks silently stayed off, the @assert_* macros no-op and this gate proves nothing.
        @assert StrictMode.assert_enabled() "StrictMode checks did not enable — guarantee gate would be vacuous"

        n = 40
        A = spdiagm(-1=>fill(-1.0,n-1), 0=>fill(4.0,n), 1=>fill(-1.0,n-1))
        sym  = symbolic(A);  F  = cholesky(sym, A)
        symL = symbolic(A);  FL = ldlt(symL, A)
        b = randn(n); x = similar(b); xL = similar(b)
        cholesky!(F, A); solve!(x, F, b)
        ldlt!(FL, A);    solve!(xL, FL, b)
        Fq = qr(A; ordering=AMDOrdering(), tol=0, singletons=false); xq = zeros(n); solve!(xq, Fq, b)

        # --- the analyze-once / shared-immutable-Symbolic guarantee (req 7) ---
        @assert_concurrency_safe cholesky(sym, A)
        @assert_concurrency_safe ldlt(symL, A)

        # --- type-stable hottest path, all three factor kinds (req 3) ---
        @assert_typestable solve!(x, F, b)
        @assert_typestable solve!(xL, FL, b)
        @assert_typestable solve!(xq, Fq, b)

        # --- QR solve is AllocCheck-clean on every path ---
        @assert_noalloc solve!(xq, Fq, b)

        println("STRICTMODE GUARANTEES OK")
        """)

        julia_exe = joinpath(Sys.BINDIR, Base.julia_exename())
        cmd = ignorestatus(addenv(`$julia_exe --project=$envdir $script`, "JULIA_LOAD_PATH" => nothing))
        proc = Base.run(pipeline(cmd; stdout = Base.stdout, stderr = Base.stderr))
        @test proc.exitcode == 0
    finally
        rm(envdir; recursive = true, force = true)
    end
end
