#!/usr/bin/env julia
# Build the puresparse_smoke executable via juliac --trim (experimental) — the
# "juliac --trim smoke build succeeds" gate (section 10 of docs/design.md). Run: `julia juliac/build.jl`,
# then execute `juliac/build/puresparse_smoke` (exit 0 + printed residuals ⇒ pass).
#
# The driver is the registered JuliaC.jl package, resolved in `juliac/toolenv`; Julia itself
# ships no juliac entry point. The driver runs in its own environment, so the package being
# built is named explicitly with `--project=$ROOT`.

const ROOT = normpath(joinpath(@__DIR__, ".."))
const TOOLENV = joinpath(@__DIR__, "toolenv")
const OUTDIR = joinpath(@__DIR__, "build")
mkpath(OUTDIR)

const NAME = "puresparse_smoke"
const OUT = joinpath(OUTDIR, NAME * (Sys.iswindows() ? ".exe" : ""))
const ENTRY = joinpath(@__DIR__, "entry.jl")

run(`$(Base.julia_cmd()) --startup-file=no --project=$TOOLENV -e "using Pkg; Pkg.instantiate()"`)

cmd = `$(Base.julia_cmd()) --startup-file=no --project=$TOOLENV
       -e "using JuliaC; JuliaC.main(ARGS)" --
       --project=$ROOT --output-exe $NAME --experimental --trim=safe --verbose $ENTRY`

# `--output-exe` takes a bare name and writes into the working directory.
@info "PureSparse: building trimmed smoke executable" OUT
run(Cmd(cmd; dir = OUTDIR))
@info "PureSparse: built" OUT filesize_bytes = (isfile(OUT) ? filesize(OUT) : 0)
