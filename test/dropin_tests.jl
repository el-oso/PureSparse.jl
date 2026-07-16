# M4 drop-in (design.md §10). `DROPIN_ACTIVE` is a compile-time Preference-backed
# const (tuning.jl) — it can't be flipped inside the already-running test process the
# way a runtime flag could (that's the whole point: the override genuinely doesn't
# exist until opted in, no `eval`, CLAUDE.md requirement 4). So this test item spawns
# an ISOLATED subprocess with its own temp `--project` (own `LocalPreferences.toml`
# setting `dropin_active = true`, entirely separate from `test/`'s own environment —
# writing to `test/LocalPreferences.toml` directly would contaminate every OTHER test
# item's precompiled state, since Preferences are project-scoped) and checks that the
# subprocess script — which exercises the real stdlib entry points, not PureSparse's
# own API — exits 0.

@testitem "M4 drop-in: activate!'d LinearAlgebra.cholesky matches stdlib surface (subprocess)" begin
    using Pkg

    pkgroot = pkgdir(PureSparse)
    envdir = mktempdir()
    try
        write(joinpath(envdir, "LocalPreferences.toml"), """
        [PureSparse]
        dropin_active = true
        """)

        script = joinpath(envdir, "run.jl")
        write(script, """
        using Pkg
        Pkg.activate(@__DIR__)
        Pkg.develop(path=raw"$pkgroot")
        Pkg.add(["LinearAlgebra", "SparseArrays", "Random"])
        Pkg.instantiate()

        using PureSparse, LinearAlgebra, SparseArrays, Random

        @assert PureSparse.DROPIN_ACTIVE "dropin_active Preference did not take effect"

        rng = MersenneTwister(11)
        n = 25
        Ir = Int[]; Jc = Int[]; V = Float64[]
        rowsum = zeros(n)
        for j in 1:n, i in (j + 1):n
            rand(rng) < 0.2 || continue
            v = randn(rng)
            push!(Ir, i); push!(Jc, j); push!(V, v)
            rowsum[i] += abs(v); rowsum[j] += abs(v)
        end
        for j in 1:n
            push!(Ir, j); push!(Jc, j); push!(V, rowsum[j] + 1.0)
        end
        A = sparse(Ir, Jc, V, n, n)
        Afull = sparse(Symmetric(A, :L))

        # Bare SparseMatrixCSC entry point (not PureSparse.cholesky — the STDLIB name,
        # which only resolves to PureSparse's algorithm because dropin_active flipped
        # the method table; this is the actual M4 gate, not a PureSparse-API check).
        F = LinearAlgebra.cholesky(Afull)
        @assert typeof(F) <: PureSparse.SupernodalFactor
        @assert PureSparse.issuccess(F)

        b = randn(rng, n)
        x = F \\ b
        Ad = Matrix(Symmetric(A, :L))
        @assert norm(Ad * x - b) / (norm(Ad) * norm(x) + eps()) < 1e-8

        L = Matrix(F.L)
        p = F.p
        PAP = Ad[p, p]
        @assert norm(L * L' - PAP) / norm(PAP) < 1e-9

        @assert isapprox(logdet(F), logdet(LinearAlgebra.cholesky(PAP)); rtol=1e-10)
        @assert isapprox(det(F), det(Ad); rtol=1e-6)

        # shift
        Fs = LinearAlgebra.cholesky(Afull; shift=3.0)
        xs = Fs \\ b
        @assert norm((Ad + 3.0I) * xs - b) / (norm(Ad) * norm(xs) + eps()) < 1e-8

        # perm: still a correct factorization for whatever F.p ends up being
        myperm = collect(n:-1:1)
        Fp = LinearAlgebra.cholesky(Afull; perm=myperm)
        pp = Fp.p
        @assert sort(pp) == collect(1:n)
        Lp = Matrix(Fp.L)
        @assert norm(Lp * Lp' - Ad[pp, pp]) / norm(Ad[pp, pp]) < 1e-9

        # check=true throws, check=false doesn't
        Abad = sparse(Diagonal(fill(-1.0, 4)))
        threw = try
            LinearAlgebra.cholesky(Abad)
            false
        catch e
            e isa PosDefException
        end
        @assert threw
        Fbad = LinearAlgebra.cholesky(Abad; check = false)
        @assert !PureSparse.issuccess(Fbad)

        # Int32 indices
        A32 = SparseMatrixCSC{Float64,Int32}(Afull)
        F32 = LinearAlgebra.cholesky(A32)
        @assert PureSparse.issuccess(F32)

        # --- ldlt drop-in (SQD/indefinite) ---
        npos, nneg = 15, 10
        nl = npos + nneg
        Il = Int[]; Jl = Int[]; Vl = Float64[]
        rowsuml = zeros(nl)
        addsym!(i, j, v) = (push!(Il, i); push!(Jl, j); push!(Vl, v);
            i != j && (push!(Il, j); push!(Jl, i); push!(Vl, v));
            rowsuml[i] += abs(v); i != j && (rowsuml[j] += abs(v)))
        for j in 1:npos, i in (j + 1):npos
            rand(rng) < 0.2 && addsym!(i, j, randn(rng))
        end
        for j in 1:nneg, i in (j + 1):nneg
            rand(rng) < 0.2 && addsym!(npos + i, npos + j, randn(rng))
        end
        for j in 1:npos, i in 1:nneg
            rand(rng) < 0.2 && addsym!(npos + i, j, randn(rng))
        end
        for j in 1:nl
            v = (rowsuml[j] + 1.0) * (j <= npos ? 1.0 : -1.0)
            push!(Il, j); push!(Jl, j); push!(Vl, v)
        end
        K = sparse(Il, Jl, Vl, nl, nl)
        Kfull = sparse(Symmetric(K, :L))

        FL = LinearAlgebra.ldlt(Kfull)   # bare stdlib entry point, free signs (no n_pos/n_neg kwarg exists)
        @assert typeof(FL) <: PureSparse.LDLFactor
        @assert PureSparse.issuccess(FL)

        bl = randn(rng, nl)
        xl = FL \\ bl
        Kd = Matrix(Symmetric(K, :L))
        @assert norm(Kd * xl - bl) / (norm(Kd) * norm(xl) + eps()) < 1e-8

        Ll = Matrix(FL.L)
        pl = FL.p
        PKPl = Kd[pl, pl]
        Dl = Diagonal(FL.d)
        @assert norm(Ll * Dl * Ll' - PKPl) / norm(PKPl) < 1e-9
        @assert isapprox(det(FL), abs(det(Kd)); rtol = 1e-4)   # CHOLMOD's own abs-value convention (verified separately, not assumed)
        @assert isapprox(logdet(FL), log(abs(det(Kd))); rtol = 1e-6)

        # negative-determinant case: logdet must stay REAL (CHOLMOD's observed
        # behavior — det(F)==30, not -30, for diag(2,-3,5) — matched exactly, not the
        # mathematically-cleaner signed product, which would need a complex log here)
        Kneg = sparse(Diagonal([2.0, -3.0, 5.0]))
        Fneg = LinearAlgebra.ldlt(Kneg)
        @assert isapprox(det(Fneg), 30.0; rtol = 1e-10)
        @assert logdet(Fneg) isa Real

        # shift, perm for ldlt too
        Fls = LinearAlgebra.ldlt(Kfull; shift = 2.0)
        xls = Fls \\ bl
        @assert norm((Kd + 2.0I) * xls - bl) / (norm(Kd) * norm(xls) + eps()) < 1e-8

        mypermL = collect(nl:-1:1)
        Flp = LinearAlgebra.ldlt(Kfull; perm = mypermL)
        plp = Flp.p
        Llp = Matrix(Flp.L)
        Dlp = Diagonal(Flp.d)
        @assert norm(Llp * Dlp * Llp' - Kd[plp, plp]) / norm(Kd[plp, plp]) < 1e-9

        # --- F.U extraction (convention verified against real stdlib/CHOLMOD output,
        # see dropin.jl's getproperty docstring): U = Lᵀ, materialized SparseMatrixCSC ---
        U = F.U
        @assert U isa SparseMatrixCSC
        @assert Matrix(U) == L'
        @assert norm(Matrix(U)' * Matrix(U) - PAP) / norm(PAP) < 1e-9
        Ul = Matrix(FL.U)
        @assert Ul == Ll'
        @assert all(Ul[i, i] == 1 for i in 1:nl)   # unit-upper for LDLᵀ, like CHOLMOD's own .U
        # stdlib-name issuccess (LinearAlgebra.issuccess is a DIFFERENT function from
        # PureSparse's exported issuccess — the drop-in must extend the stdlib one too)
        @assert LinearAlgebra.issuccess(F) && LinearAlgebra.issuccess(FL)

        # --- SimplicialLDLFactor property parity: .p/.L/.U reflect the LIVE state,
        # i.e. AFTER updowndate! mutates the factor in place ---
        G = PureSparse.simplicial(FL; grow = Float64(nl))
        @assert G.p == FL.p
        Lg = Matrix(G.L)
        @assert norm(Lg * Diagonal(G.d) * Lg' - PKPl) / norm(PKPl) < 1e-9
        w = zeros(nl); w[1] = 0.5; w[4] = -0.25   # H-block (positive) support: keeps SQD
        @assert PureSparse.updowndate!(G, w, +1) === :ok
        Kmod = Kd + w * w'
        Lg2 = Matrix(G.L)                          # re-extracted AFTER the update
        pg = G.p
        @assert norm(Lg2 * Diagonal(G.d) * Lg2' - Kmod[pg, pg]) / norm(Kmod) < 1e-9
        @assert istril(G.L) && all(==(1.0), diag(Matrix(G.L)))
        @assert Matrix(G.U) == Lg2'
        @assert LinearAlgebra.issuccess(G)

        # activate!/deactivate! exist and don't throw (restart-required to actually take
        # effect — see tuning.jl's DROPIN_ACTIVE comment; not re-checked live here)
        PureSparse.deactivate!()
        PureSparse.activate!()

        println("DROPIN_SUBPROCESS_OK")
        """)

        julia_exe = joinpath(Sys.BINDIR, Base.julia_exename())
        # `JULIA_LOAD_PATH`, if set in the PARENT process's environment (ReTestItems'
        # own worker isolation sets it to just `@`, observed on CI — a plain local
        # `julia --project=test` run doesn't have it set at all), is inherited by
        # `Base.run` by default and excludes `@stdlib` — breaking this subprocess's own
        # `using Pkg` with `ArgumentError: Package Pkg not found in current path`,
        # confirmed by reproducing that exact error via `JULIA_LOAD_PATH=@`. Unsetting
        # it lets the subprocess construct Julia's normal default LOAD_PATH from
        # `--project` alone, same as any standalone `julia --project=... script.jl`
        # invocation.
        cmd = ignorestatus(addenv(`$julia_exe --project=$envdir $script`, "JULIA_LOAD_PATH" => nothing))
        proc = Base.run(pipeline(cmd; stdout = Base.stdout, stderr = Base.stderr))
        @test proc.exitcode == 0
    finally
        rm(envdir; recursive = true, force = true)
    end
end

@testitem "M4 gate: downstream-consumer smoke suite passes unmodified with dropin active (subprocess)" begin
    # design.md §10 M4 gate, first clause: "with dropin active, a downstream
    # SparseArrays-dependent smoke test suite passes unmodified." The suite is
    # test/downstream_smoke.jl — stdlib names only, zero PureSparse knowledge in its
    # body (see its header; it also passes verbatim against plain CHOLMOD, which is
    # what makes it a genuine downstream stand-in). Same isolated-subprocess-with-own-
    # Preferences pattern as the testitem above; PureSparse appears ONLY in this
    # runner's setup preamble (loading the package is what installs the override),
    # never in the smoke file itself.
    using Pkg

    pkgroot = pkgdir(PureSparse)
    smoke = joinpath(pkgroot, "test", "downstream_smoke.jl")
    envdir = mktempdir()
    try
        write(joinpath(envdir, "LocalPreferences.toml"), """
        [PureSparse]
        dropin_active = true
        """)

        script = joinpath(envdir, "run.jl")
        write(script, """
        using Pkg
        Pkg.activate(@__DIR__)
        Pkg.develop(path=raw"$pkgroot")
        Pkg.add(["LinearAlgebra", "SparseArrays", "Random", "Test"])
        Pkg.instantiate()

        # Setup preamble: load PureSparse (installs the drop-in) and PROVE the smoke
        # below will exercise PureSparse, not CHOLMOD — without this guard a silently
        # inactive drop-in would make the whole testitem vacuous. `import`, not
        # `using`: a downstream module never has PureSparse's exports in its own
        # namespace, and a bare `using` here would make the smoke's unqualified
        # `cholesky`/`ldlt` ambiguous (caught by running this, not assumed).
        import PureSparse
        using LinearAlgebra, SparseArrays
        @assert PureSparse.DROPIN_ACTIVE "dropin_active Preference did not take effect"
        @assert LinearAlgebra.cholesky(sparse(1.0I, 2, 2)) isa PureSparse.SupernodalFactor

        include(raw"$smoke")   # unmodified; its body never mentions PureSparse
        println("DOWNSTREAM_SMOKE_OK")
        """)

        julia_exe = joinpath(Sys.BINDIR, Base.julia_exename())
        # `JULIA_LOAD_PATH`, if set in the PARENT process's environment (ReTestItems'
        # own worker isolation sets it to just `@`, observed on CI — a plain local
        # `julia --project=test` run doesn't have it set at all), is inherited by
        # `Base.run` by default and excludes `@stdlib` — breaking this subprocess's own
        # `using Pkg` with `ArgumentError: Package Pkg not found in current path`,
        # confirmed by reproducing that exact error via `JULIA_LOAD_PATH=@`. Unsetting
        # it lets the subprocess construct Julia's normal default LOAD_PATH from
        # `--project` alone, same as any standalone `julia --project=... script.jl`
        # invocation.
        cmd = ignorestatus(addenv(`$julia_exe --project=$envdir $script`, "JULIA_LOAD_PATH" => nothing))
        proc = Base.run(pipeline(cmd; stdout = Base.stdout, stderr = Base.stderr))
        @test proc.exitcode == 0
    finally
        rm(envdir; recursive = true, force = true)
    end
end

@testitem "M4 drop-in: inactive by default in the normal test environment" begin
    using LinearAlgebra, SparseArrays
    # test/'s own environment never sets dropin_active, so this file's presence alone
    # must not affect PureSparse's default (inactive) state or LinearAlgebra's own
    # cholesky method table.
    @test PureSparse.DROPIN_ACTIVE == false
    @test !isdefined(PureSparse, :sparse_L)
    @test length(methods(LinearAlgebra.cholesky, (SparseMatrixCSC{Float64,Int},))) == 1
end
