# PureSparse.jl — Roadmap

PureSparse.jl is a pure-Julia sparse direct solver: fill-reducing ordering, supernodal and
multifrontal Cholesky/LDLᵀ, sparse QR, and GPU backends, with dense per-front work routed
through [PureBLAS.jl](https://github.com/el-oso/PureBLAS.jl). This file lists the milestones,
their status, the gate that defines "done" for each, and the work that remains. The
architecture is described in [`docs/design.md`](docs/design.md); the rules every change must
follow are in [`CONTRIBUTING.md`](CONTRIBUTING.md). Version 0.1.0 is tagged.

Milestone numbers are stable identifiers and are not reused. M3 (the original CUDA plan) was
folded into M6, so there is no M3 section.

Status values: **done**, **in progress**, **planned**, **shelved**, **research**.

## Summary

| Milestone | Scope | Status |
|---|---|---|
| M1 | AMD ordering, symbolic analysis, supernodal LLᵀ, solve | done |
| M2 | LDLᵀ for quasi-definite systems, update/downdate, refinement | done |
| M4 | Drop-in `cholesky`/`ldlt` for `SparseMatrixCSC` | done |
| M5 | Sparse QR | done |
| M6 | GPU (CUDA) multifrontal Cholesky and LDLᵀ | done |
| M7 | GPU sparse QR | shelved |
| M8 | Multi-backend GPU (AMD ROCm, then Intel oneAPI) | in progress |
| M9 | MLIR investigation | research |

## M1 — Ordering, symbolic analysis, supernodal LLᵀ

**Status: done.**

Scope: approximate minimum degree (AMD) ordering, elimination tree and column counts,
fundamental supernodes with relaxed amalgamation, left-looking supernodal LLᵀ for symmetric
positive definite input, forward/backward solves, and the "analyze once, factorize many times"
API (`symbolic`, `cholesky!`, `solve!`).

Gate (all met):

- Correctness on a matrix zoo against a dense `BigFloat` oracle, cross-checked against
  CHOLMOD's output through `SparseArrays`.
- `cholesky!` on an existing factor allocates zero bytes.
- Wall time: `median_seconds(PureSparse+PureBLAS) < median_seconds(CHOLMOD+OpenBLAS)` on the
  KKT/FEM benchmark set, under both own-ordering and same-permutation arms, strictly faster on
  at least half the set. A size sweep from n = 2 to 2048 also passes.
- AMD fill is at most 1.15 times the fill of the stdlib ordering on the zoo.
- A `juliac --trim` build of a factor-and-solve entry point succeeds.

Open work: none for the milestone. See "Cross-cutting work" for the `solve!` allocation item.

## M2 — LDLᵀ, update/downdate, refinement

**Status: done.**

Scope: LDLᵀ for symmetric quasi-definite (SQD) systems such as interior-point KKT matrices,
with a fixed pivot order and signed regularization (no dynamic pivoting), inertia tracking
`(n_pos, n_neg, n_zero)` in `FactorStats`, supernodal and simplicial storage, rank-1 and
rank-k update/downdate (Davis–Hager), split solves, iterative refinement (`refine!`), and an
interior-point usage guide.

Gate (all met):

- Synthetic interior-point iterate sequences factor without perturbation on the
  well-conditioned set; forced-regularization paths are covered by separate tests.
- Computed inertia matches the inertia by construction.
- Update/downdate round trip within 100·eps·n (measured about 1.4e-16 relative).
- `ldlt!` on an existing factor allocates zero bytes.
- Wall-time gate against CHOLMOD's sparse `ldlt` on SQD matrices.

Open work: the batched multiple-rank update variant of Davis–Hager; rank-k currently applies
successive rank-1 updates.

## M4 — Drop-in replacement for `LinearAlgebra.cholesky` / `ldlt`

**Status: done.**

Scope: `activate!`/`deactivate!` (Preferences-gated) route `cholesky` and `ldlt` on real
`SparseMatrixCSC` input to PureSparse, with stdlib surface parity (`logdet`, `det`, `diag`,
`issuccess`, `check=`, `shift=`, `perm=`, `Symmetric` wrappers, `Int32` indices, and
`SparseMatrixCSC` extraction of `L`/`U`/`p`).

Gate (met): with the drop-in active, the stdlib expectations captured in `dropin_tests.jl`
pass against PureSparse factors, and the M1 wall-time gate still holds through the drop-in
entry point.

Open work: none.

## M5 — Sparse QR

**Status: done.** Design: [`docs/design_qr.md`](docs/design_qr.md),
[`docs/design_qr_m5b.md`](docs/design_qr_m5b.md).

Scope: COLAMD column ordering, column-singleton peeling, a left-looking column Householder
path and a multifrontal path, rank detection with dead-column dropping, least squares for
tall matrices, and basic and minimum-norm solutions for wide matrices. `qr!` on an existing
factor allocates zero bytes.

Gate (met, 16 of 16 strata): warm PureSparse `qr!` is faster than SuiteSparseQR on the
benchmark set under both the own-ordering and same-permutation arms. The 7000×4000 case is
at parity with the Rust `faer` library, which is reported as context rather than gated.

Open work:

- Complex element types in the multifrontal path (conjugate Householder).
- `BigFloat` and other non-`isbits` element types in the multifrontal path.

The multifrontal path is generic over real `isbits` element types.

## M6 — GPU (CUDA) multifrontal Cholesky and LDLᵀ

**Status: done.** Design: [`docs/design_gpu.md`](docs/design_gpu.md). Supersedes the original
M3 plan.

Scope: a multifrontal engine written in KernelAbstractions with device-resident factors, a
symbolic-time etree frontier that splits small subtrees to the CPU and large supernodes to
the GPU, a bounded stack-with-compaction front arena, fused pure-Julia front kernels (SPD
`potrf`+`trsm`, signed-LDL diagonal plus panel solve), and a level-scheduled batched device
solve. Update/downdate stays on the CPU.

Gate (met), measured on an RTX 4070 (FP64) on a 3D-grid SPD and KKT stratum, factor plus
solve, medians:

- Pure-Julia GPU path at least 1.0 times a path built from the CUDA vendor libraries
  (cuSOLVER, cuBLAS) at every stratum size; measured 1.9 to 4.4 times.
- Faster than CHOLMOD+OpenBLAS at every stratum size.

The original target of 5 times a single-thread CPU PureSparse was not met on the
factor-plus-solve median (best 4.5 times) and was replaced by the vendor-parity clause above.
Results are in [`docs/src/benchmarking.md`](docs/src/benchmarking.md).

Open work: a second NVIDIA GPU for a cross-host measurement; pinning the frontier
auto-policy.

## M7 — GPU sparse QR

**Status: shelved.** Designs and reviews: `docs/design_qr_gpu.md`, `docs/design_qr_gpu_v3.md`,
and `docs/reviews/design_qr_gpu_*`.

Three probes on an RTX 4070 (FP64, warm medians) measured a pure-Julia GPU QR front against
cuSOLVER `geqrf` and cuBLAS before any production code was written: the single-workgroup
Householder panel was 3 to 10 times slower than `geqrf`; the projected front-level time was
1.5 to 2.7 times slower; and the batched trailing-update gemm ran at 0.77 of cuBLAS (faster
on the `nn` shape, 0.58 on the `tn` shape). The milestone is shelved because these results
leave no stage where a pure-Julia kernel is expected to beat the vendor library. CPU sparse
QR (M5) is unaffected. The probe scripts are under `benchmark/gpu/qr_*_phase0.jl`.

Reopen if a probe shows the `tn` batched gemm and the panel reaching vendor parity.

## M8 — Multi-backend GPU (AMD ROCm, Intel oneAPI)

**Status: in progress.** Design: [`docs/design_gpu_multibackend.md`](docs/design_gpu_multibackend.md).

Scope: run the M6 engine on non-NVIDIA GPUs in FP64. The engine is backend-generic
(`ext/gpu_shared.jl`); each backend is a thin extension that supplies its KernelAbstractions
backend. AMD ROCm comes first, then Intel oneAPI. Apple Metal is out of scope (no FP64).

Status by half:

- Functional: done on ROCm. Cholesky and LDLᵀ run end to end and match the CPU factor to
  about 1e-16 at every frontier cutoff (tested on a gfx1152 integrated GPU, which has weak
  FP64 throughput). The path is correct but not tuned.
- Performance: waiting on a measurement. AMD CDNA and Intel Max parts have FP64 matrix
  cores that the vendor BLAS libraries use and that a scalar KernelAbstractions kernel cannot
  emit. A GEMM probe (`benchmark/gpu/dgemm_vendor_probe.jl`) on a datacenter AMD GPU decides
  whether the pure kernels stay primary there or the dense fronts dispatch to rocBLAS and
  rocSOLVER.

Gate: end-to-end correctness on each backend; on the performance half, faster than the
multithreaded CPU path, with the ratio to the vendor libraries recorded per backend.

Open work: the CDNA probe; Intel oneAPI extension; per-backend dense-front dispatch if the
probe shows vendor kernels win.

## M9 — MLIR investigation

**Status: research, uncommitted.**

Question: can a dense front kernel written once in MLIR (`linalg.matmul`, `vector.contract`)
lower to each vendor's FP64 matrix instructions (AMD MFMA, Intel XMX, NVIDIA tensor cores)
and so reach vendor GEMM speed without vendor libraries? A prototype would be compared with
(a) calling rocBLAS through AMDGPU.jl, (b) hand-written `llvmcall` to MFMA intrinsics, and
(c) MLIR through Reactant.jl or MLIR.jl. MLIR's sparse-tensor dialect targets sparse tensor
algebra, not direct factorization, so it would only apply to the dense fronts.

Promotion criteria: the MLIR route matches the vendor FP64 GEMM on the crown-front shapes
and has a maintainable Julia integration. Otherwise M8 uses vendor kernels on matrix-core
hardware.

## Cross-cutting work

- **All-paths zero allocation:** `@allocated == 0` holds on every refactor and solve, but
  AllocCheck's all-paths proof fails for `cholesky!`, `ldlt!`, the multifrontal `qr!` and
  their `solve!`; the column-QR `qr!`/`solve!` pass. Most reported sites are in PureBLAS's
  level-3 kernels (per-thread workspace creation and growth, the threaded `gemm` path) and
  need fixing there; PureSparse's own sites are in `_panelview` and `_solve_Lt_generic!`.
  Once clean, `test/strictmode_guarantees_tests.jl` adds `@test_noalloc` for these entry
  points.
- **Re-measure on Julia 1.13:** the size sweep has been rerun on Julia 1.13.3 and PureBLAS
  0.1.2 (Zen 3, fixed clock; faster than CHOLMOD at every size). The named-matrix gates
  (`gate.jl`, `gate_ldlt.jl`, `qr_gate.jl`, `gpu_gate.jl`) and the Zen 4 size sweep still
  report results from earlier toolchains.
- **Analysis time:** on 2D and 3D Laplacians, symbolic analysis is slower than faer's
  and CHOLMOD's. AMD is the largest single stage (roughly 40% of analysis);
  its remaining cost is spread across the main elimination
  loop. The preliminary column-count pass used only for sibling ordering is another
  candidate.
- **Tree-level parallelism:** the numeric factorization runs supernodes sequentially, so
  extra threads only help inside large dense kernels. Matrices with many small
  supernodes, such as 2D and 3D grids, do not speed up with threads. Independent subtrees
  of the elimination tree could be factored in parallel; this needs a design.
- **Threaded kernels slow the factorization down:** on a 30×30×30 Laplacian,
  `cholesky!` with 4 PureBLAS threads takes about 1.4 times as long as with 1, although
  the dominant kernels run alone at the same shapes get faster with 4 threads (except
  `syrk!` at 900×300). The cause inside the factorization is not yet identified.
- **Registration:** registration in the General registry depends on PureBLAS being
  registered first.
- **LU:** unsymmetric sparse LU through a collaboration with PureKLU.jl. Not started.

## Scope: not supported

- Complex element types in Cholesky/LDLᵀ and in the QR multifrontal path. This is a v1.0
  non-goal ([`docs/design.md`](docs/design.md) §1.1); complex QR is listed under M5.
- General indefinite systems with 2×2 pivoting. LDLᵀ is fixed-pivot by design (M2).
- Distributed-memory factorization.
- GPU sparse QR (M7) and GPU update/downdate.
