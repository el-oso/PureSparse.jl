# Interior-Point Guide

PureSparse's LDLᵀ path targets the workload interior-point optimizers actually have:
factoring a symmetric quasi-definite (SQD) KKT system `[H Aᵀ; A −C]` (H, C positive
definite — the Hessian/barrier and slack blocks) many times per solve, with the
sparsity **pattern fixed** across iterations and only the numeric values changing as
the barrier parameter shrinks.

## Per-iteration refactor (the common case)

Analyze once, refactor every iteration:

```julia
using PureSparse, SparseArrays

K = ...                              # [H A'; A -C], lower triangle stored
S = PureSparse.symbolic(K)
F = PureSparse.ldlt(S, K; n_pos = size(H, 1), n_neg = size(C, 1))

for iter in 1:max_iters
    # ... update H/C's diagonal entries in K.nzval for the new barrier parameter ...
    PureSparse.ldlt!(F, K)           # same pattern, refactor in place
    F.stats.n_perturbed > 0 && @warn "regularization engaged" F.stats.n_perturbed
    dz = F \ rhs
    # ... IPM step using dz ...
end
```

`n_pos`/`n_neg` build the expected pivot-sign pattern (`n_pos` leading `+1`s, `n_neg`
trailing `−1`s, **in `K`'s original column order** — composes correctly with AMD
reordering, see [`ldlt`](@ref)) from Vanderbei's 1995 result that an SQD matrix is
strongly factorizable (LDLᵀ with 1×1 pivots exists under any symmetric permutation).
`ldlt!` never throws on an ill-conditioned or near-singular KKT system: signed
regularization (QDLDL/Clarabel-style — design.md §5.1) forces every pivot to its
expected sign and above a magnitude floor, so the factorization always completes.

## Reading regularization and inertia

`F.stats` after every `ldlt!` call:

- `(n_pos, n_neg, n_zero)` — the OBSERVED inertia before any forcing. An SQD matrix
  should read `(n_pos, n_neg, 0)` exactly matching construction; if it doesn't (e.g.
  `n_zero > 0` on paper-SQD structure), that is diagnostic of numerical trouble in the
  KKT system upstream, not something PureSparse decides for you.
- `n_perturbed`/`max_perturbation` — how many pivots were forced and by how much. A
  downstream IPOPT/MA57-style consumer that wants exact-inertia semantics can run its
  own `δI`/`−δI` regularization loop *on top of* PureSparse by reading these fields and
  refactoring (`ldlt_delta = 0` via `Preferences.set_preferences!` disables PureSparse's
  own forcing as a safety net, or leave it on as a fallback) — PureSparse reports and
  continues, it does not run that loop itself (design.md §5.2).

## Iterative refinement

Whenever `n_perturbed > 0`, the factor exactly solves a *perturbed* system, not the
true `K` — [`refine!`](@ref) recovers accuracy against the real matrix:

```julia
x = similar(rhs)
PureSparse.refine!(x, F, K, rhs; iters = 2)
```

## Update/downdate (structural changes between iterations)

For the less common case where the KKT system's *structure* changes between iterations
— an active-set method adding/removing a constraint row, a cutting-plane method
appending a row — rather than just its values, `simplicial`/`updowndate!` apply a
rank-1 change in `O(changed nnz)` without a full refactor:

```julia
G = PureSparse.simplicial(F)                        # one-time conversion, allocates
status = PureSparse.updowndate!(G, w, +1)            # A + w*w', O(changed nnz)
status === :ok || PureSparse.issuccess(G) || error("rebuild via simplicial(ldlt(...))")
x = G \ rhs                                          # simplicial split solves, no refactor
```

`updowndate!` returns `:refactor_required` if an update's fill exceeds a column's
slack (design.md §7 — no silent reallocation, ever) or `:not_definite` if a downdate
would change the factor's inertia; either way `issuccess(G)` becomes `false` and `G`
must be rebuilt via a fresh `simplicial(ldlt(...))`. This is the tool for occasional
structural changes, not the per-iteration diagonal update above — `ldlt!` is cheaper
and simpler when only values change.
