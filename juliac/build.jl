#!/usr/bin/env julia
# Build the puresparse_smoke executable via juliac --trim (Julia ≥ 1.12, experimental) —
# the M1 "juliac --trim smoke build succeeds" gate (design.md §10). Mirrors
# PureBLAS.jl/juliac/build.jl. Run: `julia juliac/build.jl`, then execute
# `juliac/build/puresparse_smoke` (exit 0 + printed residuals ⇒ pass).

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUTDIR = joinpath(@__DIR__, "build")
mkpath(OUTDIR)

const JULIAC = normpath(joinpath(Sys.BINDIR, "..", "share", "julia", "juliac", "juliac.jl"))
isfile(JULIAC) || error("juliac.jl not found at $JULIAC — needs Julia ≥ 1.12")

const OUT = joinpath(OUTDIR, "puresparse_smoke" * (Sys.iswindows() ? ".exe" : ""))
const ENTRY = joinpath(@__DIR__, "entry.jl")

cmd = `$(Base.julia_cmd()) --startup-file=no --project=$ROOT $JULIAC
       --output-exe $OUT --experimental --trim=safe --verbose $ENTRY`

@info "PureSparse: building trimmed smoke executable" OUT
run(cmd)
@info "PureSparse: built" OUT filesize_bytes = (isfile(OUT) ? filesize(OUT) : 0)
