# Benchmarking

## Methodology

`benchmark/gate.jl` runs the M1 wall-time gate: [Chairmarks.jl](https://github.com/LilithHafner/Chairmarks.jl)
medians (not min), single-thread pinned (`BLAS.set_num_threads(1)`), `evals=1` (each
sample is a fresh, independent timed call — appropriate since factorization refactor is
cheap enough per-call that batching isn't needed at the matrix sizes tested), a sample
budget capped at 30 samples / 1.5s per measurement. `benchmark/benchmarks.jl` is a
separate [PkgBenchmark.jl](https://github.com/JuliaCI/PkgBenchmark.jl) suite for
commit-to-commit self-regression (`judge(PureSparse, "HEAD", "base")`) — it answers "did
my change make my code slower?", which the gate does not.

For a methodologically-valid (clock-locked) run, lock CPU frequency first — this repo
doesn't duplicate a locking script; PureBLAS.jl's `bench/fleet_freqlock.sh lock` covers
the same machine. An unlocked run still produces real measured numbers, just noisier.

## Configurations

Three of the design's four configurations (`docs/design.md` §9.3) are measured — the
fourth, CHOLMOD+PureBLAS, is **N/A**, blocked on PureBLAS's documented
`BLAS.lbt_forward`-from-a-live-Julia-process limitation (see PureBLAS.jl's docs):

1. **PureSparse + PureBLAS** (primary) — the actual shipped stack.
2. **PureSparse + OpenBLAS** (kernel-attribution arm) — `benchmark/openblas_backend.jl`
   re-`include`s `src/numeric/llt.jl` verbatim under a different kernel binding (OpenBLAS
   via `LinearAlgebra.LAPACK`/`BLAS` instead of PureBLAS), so this isolates
   PureBLAS-vs-OpenBLAS kernel efficiency from the sparse scheduling layer — no
   algorithm duplication, same source file, different `using`.
3. **CHOLMOD (SparseArrays) + OpenBLAS** (baseline).

Both **own-ordering** (each stack's own AMD) and **same-permutation** (each stack fed the
*other's* chosen permutation via `GivenOrdering`/`perm=`) arms run — the latter isolates
factorization throughput from ordering quality and is part of the gate, not supplementary.

## Current result

As of 2026-07-13 on `neuromancer` (NOT clock-locked — see caveat above), **6/14**
matrix-arm combinations beat CHOLMOD+OpenBLAS on warm numeric refactor. This does **not**
yet meet M1's gate ("strictly faster on at least half the set"). The root cause has been
diagnosed (not guessed — measured): it is not an ordering-quality gap (PureSparse's AMD
fill matches or beats CHOLMOD's on every failing case) but a relaxed-amalgamation
limitation that under-merges supernodes on bushy elimination trees (2D grid Laplacians
being the clearest failing case). See `ROADMAP.md`'s "CURRENT FOCUS" section for the full
table and diagnosis — that file is the living source of truth for gate status; this page
won't be kept in perfect sync with every run.

## Reproducing

```bash
julia --project=benchmark benchmark/gate.jl            # measure + save + print gate verdict
julia --project=benchmark benchmark/gate.jl report      # print verdict from the last saved JSON only
```

Results are written to `benchmark/results/gate_<hostname>.json` (gitignored — per-host
measurement caches aren't committed, matching PureBLAS.jl's convention for its own
per-host benchmark data).
