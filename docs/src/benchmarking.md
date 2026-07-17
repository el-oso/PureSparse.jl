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

## Sparse QR (M5)

### Methodology

`benchmark/qr_gate.jl` runs the M5 wall-time gate with the same discipline as M1's:
Chairmarks medians, single-thread pinned, `evals=1`, 20 samples / 1.5s per
measurement. Like M1/M2/M4, the gate compares PureSparse's **warm `qr!` refactor**
(the zero-allocation, StrictMode-verified path — the primary "analyze once, factorize
many" API) against SuiteSparseQR's [spqr2011](@cite) factorization. Since stdlib
SuiteSparseQR exposes no analyze-once/refactorize split, its **cold** factorization is
its best case, so the gate is **PureSparse warm `qr!` vs SuiteSparseQR cold** (recorded
as design_qr.md **D13**). This is deterministic on the PureSparse side: the warm path
allocates nothing, so its per-call timing is near-constant (no GC-pause variance).
Both **own-ordering** (PureSparse's COLAMD vs SPQR's default) and **same-permutation**
arms run, as for Cholesky. PureSparse's number is the **best of `:column`/`:frontal`
warm** per matrix-arm — the same choice `qr(A; method = :auto)` makes for a real
caller (`:column` wins the singleton-dominated stratum, `:frontal` the rest).

The gate set is stratified into three regimes (design_qr.md §9.3), and the M5
closeout gate requires **every stratum to pass, both arms** — not just a majority:

- `i_singleton` — LP-shaped, singleton-dominated matrices (`lp_slack`, `staircase`).
- `ii_sparse_R` — genuinely sparse R, little dense work (`banded_ls`, `grid_ls`).
- `iii_flop_rich` — dense-panel-heavy problems where BLAS-3 fronts pay
  (`dense_arrow`, `random_tall`).

Two context arms are measured alongside but are **not** part of the pass/fail
verdict: PureSparse's own `cholesky(AᵀA)` normal equations (the §1.2 "when not to
use QR" alternative) and [faer](@cite)'s sparse QR via a `ccall` shim (its
ordering/threshold choices differ, so gating on it would conflate ordering quality
with kernel throughput).

### Current result

As of 2026-07-16, the M5 closeout gate **PASSES 16/16** — every stratum, both arms —
confirmed on **both** clock-locked hosts (`neuromancer` and `galen`, `performance`
governor). Two-host clock-locked agreement is this project's bar for a gate verdict.
Per stratum:

| Stratum | Passing | Where it stands |
|---|---|---|
| `i_singleton` | **6/6** | warm singleton refactor reuses the pattern-fixed peel set, so the singleton-dominated matrices refactor almost for free on `:column` (e.g. `lp_slack_n800x150` 0.018 ms vs SPQR cold 0.18 ms) |
| `ii_sparse_R` | **6/6** | the multifrontal `:frontal` path wins every sparse-R case (`banded_ls` 0.34 ms vs SPQR 0.93–1.65 ms; `grid_ls_70x50` 5.0 ms vs SPQR 7.6–10.1 ms) |
| `iii_flop_rich` | **4/4** | clean sweep — `:frontal`'s BLAS-3 fronts win every flop-rich case (`random_tall` 8.7 ms vs SPQR 14.4–15.8 ms) |

![M5 QR gate per-stratum comparison](assets/qr_gate_strata.png)

Two methodology corrections landed with this result and are worth stating plainly.
First, the gate now compares PureSparse's **warm `qr!`** against SPQR cold (D13, above)
rather than cold-vs-cold — matching how M1/M2/M4 already gate, and eliminating the
cold-path GC-pause variance that made single-sample verdicts unreliable. Second, an
earlier round of M5 gate numbers had been timing a **broken** blocked multifrontal
path that silently dropped ~2/3 of columns (so it clocked a fraction of the real
work); that bug is fixed, and the numbers above are the first measured on the corrected
factorization. `ROADMAP.md` is the living source of truth for the full diagnosis trail;
this page won't be kept in perfect sync with every run.

### The flagship dense-panel case (7000×4000)

Where the multifrontal path's BLAS-3 architecture is actually exercised — a
7000×4000 random matrix at 1% and 10% density
(`benchmark/faer_vs_puresparse_7000x4000.jl`) — PureSparse's `:frontal` path is
**tied with [faer](@cite)** and **~30% faster than SuiteSparseQR**, at identical
COLAMD fill (neuromancer, clock-locked, cold factorize-only medians of 10 samples):

| density | PureSparse `:frontal` | faer | SuiteSparseQR | vs faer | vs SPQR |
|---|---|---|---|---|---|
| 1% | **6.47 s** | 6.62 s | 8.25 s | 1.02× | 1.28× |
| 10% | **6.69 s** | 6.93 s | 8.98 s | 1.04× | 1.34× |

![7000×4000 sparse QR comparison](assets/qr_faer_comparison.png)

!!! warning "Prior flagship numbers withdrawn"
    Earlier versions of this page reported PureSparse `:frontal` at 1.72 s / 0.89 s,
    "2.3–6.5× faster than faer/SPQR". Those numbers were an **artifact of the
    correctness bug** — the broken blocked path dropped columns and timed only a
    fraction of the real work. They are withdrawn; do not cite them. The numbers above
    are the corrected measurement.

Both PureSparse and faer sit near **~13 GFlop/s single-threaded** here, and a
panel-width (NB) sweep confirms that rate is **shape-limited** — the skinny-K trailing
updates of a sparse QR can't reach square-`gemm` peak, which is exactly why all three
implementations cluster near it. Ordering is not the gap either: PureSparse's COLAMD
fill matches SuiteSparse's to within 0.1% (nnz(R) ratio 1.001 at 1%, 1.000 at 10%),
and faer also uses COLAMD. The remaining lever is the PureBLAS dense-`gemm` microkernel
itself, pursued separately.

(The `:column` path takes ~100 s here — this is exactly the regime `method = :auto`
exists to route away from.)

### Reproducing

```bash
julia --project=benchmark benchmark/qr_gate.jl          # measure + save + print gate verdict
julia --project=benchmark benchmark/qr_gate.jl report    # verdict from the last saved JSON only
julia --project=benchmark benchmark/plot_qr_comparison.jl  # regenerate the two plots above
                                                           # from the SAVED JSON (never re-measures)
```

The plots regenerate from `benchmark/results/qr_gate_neuromancer.json` and
`benchmark/results/faer_vs_puresparse_7000x4000_neuromancer.json` — saved measurement
snapshots; re-running a benchmark to make a plot is against this repo's benchmarking
rules (results→JSON first, plots from saved JSON only).

## GPU multifrontal (M6)

The GPU backend (a weak-dependency CUDA extension) factors on the device with a
**multifrontal** supernodal engine: an upward-closed etree frontier splits the small
fronts onto the CPU from the large "crown" fronts on the GPU, and the whole dense
inner loop runs through **pure-Julia KernelAbstractions.jl kernels** — no
cuSOLVER/cuBLAS on the shipped path (they remain only as reference arms). The dense
kernels beat cuBLAS FP64 on the shapes that matter because Julia is IEEE-strict and the
kernels use `muladd`; the whole factor stays device-resident, and solves run on-device
too (only the right-hand side and solution vectors cross the bus).

!!! note "Measurement status"
    These are galen (RTX 4070, clock-locked 1920 MHz) medians of the **warm,
    device-resident refactor** — the zero-allocation path the API is built around, so
    per-sample time is near-deterministic and the honest viz is a median line/bar (a
    violin would be a flat line — the same reasoning as the QR gate figure above). The
    formal pinned SPD+SQD stratum gate (`docs/design_gpu.md` §8), run on two
    clock-locked hosts against CHOLMOD/OpenBLAS, is a separate, later artifact.

### Symmetric quasi-definite LDLᵀ (the interior-point / KKT case)

On symmetric quasi-definite KKT systems `[H Aᵀ; A −D]` (the interior-point workload
PureSparse targets, §"Analyze once, factorize many"), the pure GPU LDLᵀ **matches and
slightly exceeds** the cuSOLVER/cuBLAS reference, reaching **5.08× over single-thread
CPU `ldlt!`** at H = 44³ (n ≈ 87 k, nnz(L) ≈ 46 M) — and the speedup grows with size:

![GPU LDLᵀ speedup vs CPU](assets/gpu_ldlt_speedup.png)

The dotted line is an earlier pure path that used a *standalone* triangular solve on the
crown fronts; it stalled at ~3.4× because the LDLᵀ diagonal was still factored on the
host. The shipped path (solid blue) instead runs a **fused signed-LDL front** — one
kernel that factors the diagonal block (fixed-pivot signed regularization + on-device
inertia) *and* solves the tall panel — which is what closes the gap to the vendor path.
Per crown-front shape, that fused front removes the host round-trip whose cost scales as
`nscol³`, so it wins big on the flop-heavy fronts (up to **7.15×** at the near-root
supernodes), for a **flop-weighted 4.42×** over the vendor front; it only loses on the
smallest fronts, where a CPU block is genuinely cheap:

![Fused signed-LDL front vs vendor front](assets/gpu_front_kernel.png)

### SPD Cholesky

The SPD Cholesky path uses the analogous fused Cholesky front and sits at **vendor
parity** (e.g. 2.91× over CPU `cholesky!` at a 44³ grid Laplacian — GPU speedups are
lower here than on the KKT case because the pure grid's fronts are sparser and less
GPU-favorable than the KKT coupling fill).

### Memory: the bounded arena

The multifrontal update matrices live in a **bounded stack-with-compaction arena** (each
front builds its Schur complement in a work slot, then compacts it onto a stack over the
space its children freed), rather than a monotonic per-front allocation. That is the
difference between OOM and fit on the large KKTs — **5.9× smaller at 44³**, and the ratio
grows with problem size:

![Bounded vs monotonic arena](assets/gpu_arena.png)

### Reproducing

```bash
julia --project=benchmark benchmark/plot_gpu_comparison.jl  # regenerate the three plots
                                                            # above from the SAVED JSON
```

The figures regenerate from `benchmark/results/gpu_multifrontal_galen.json` (a saved
measurement snapshot), never from a live benchmark — same rule as the rest of this page.
The correctness oracles that back these numbers (device factor matches CPU factor at
machine precision, exact inertia) live in `benchmark/gpu/`.
