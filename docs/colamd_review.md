# Adversarial review — `src/ordering/colamd.jl` (COLAMD column ordering)

Reviewer: independent Opus pass, branch `colamd-impl` @ `b5fc607`. Scope: the M5a task-3
COLAMD implementation, its tests, and provenance. Sources cross-checked directly:
Davis–Gilbert–Larimore–Ng 2004 (TOMS, "[P]") and Larimore 1998 MS thesis ("[T]"),
both read via rendered PDF pages (the thesis text layer is garbled). CHOLMOD/COLAMD C
source was **not** read, per CLAUDE.md requirement 1.

**Verdict: BLOCKER 0 · DEFECT 1 · NIT 4.** The core algorithm is a faithful, carefully
reasoned transcription of thesis Algorithm 1 (pp. 25–26) with the paper's `l_k = 0`
discard grafted on. Permutation validity is solid (reproduced across 400+ random cases
plus my own checks). The one substantive finding is the `l_k = 0` synthesis (DEFECT-1),
which is own-engineering that neither source specifies and whose quality-equivalence to a
paper-faithful `l_k` is asserted-not-proven. The wide-matrix test-peel decision
(the implementer's top-flagged risk) **checks out** — I reproduced the diagnosis
independently and it does not hide a real ordering bug.

---

## What was independently verified as CORRECT (not just plausible)

These are load-bearing checks I actually performed, not assumptions.

- **Tests pass as reported.** `runtests(PureSparse; name=r"COLAMD")` → 4/4 items,
  **2539/2539** assertions, 0 failures (reproduced; the summary's "2539/2539" is exact).

- **Core algorithm = thesis Algorithm 1, line-for-line** (verified against rendered
  thesis pp. 25–26):
  - initial score `d_j = Σ_{i∈C_j}(|R_i|−1)` — colamd.jl:184 `s += rdeg[i] - 1`. ✔
    matches thesis p.25 init loop.
  - phase 1 set-differences: `w(i)=|R_i|; w(i) -= |j|; if w(i)==0 absorb` —
    colamd.jl:323–332 (`w[i]=rdeg[i]+t`, `w[i]-=thick[j]`, `w[i]==t ⇒ rowlive=false`).
    ✔ matches thesis p.25 exactly, `|j|`=super-column thickness.
  - phase 2 sum + further mass elimination: `d_j=Σw(i); if d_j==0 order j` —
    colamd.jl:341–380 (`s += w[i]-t`, `s==0 ⇒ mass-eliminate`). ✔ thesis p.26.
  - super-column detection after phase 2, pairwise within hash buckets — colamd.jl:406–447.
    ✔ thesis p.26 + §4.2.6 hashing.
  - final degree `d_j = d_j + |R_r| − |j|` computed AFTER super-column detection —
    colamd.jl:460 `sc = psc[j] + rdeg[r] - thick[j]`. ✔ thesis p.26 final loop. The
    `−thick[j]` (which is absent from [P] Algorithm 3) is genuinely from the thesis, not
    an unfounded deviation.

- **§4.8 recommended variant — all five choices present** (task item 4):
  COLMMD initial metric (colamd.jl:184, `Σ(|R_i|−1)`, not the AMD bound) ✔;
  AMD-style external-row-degree during elimination (phase 1/2 set-differences) ✔;
  no initial aggressive absorption (init scoring, colamd.jl:168–194, has no w-array
  set-difference pass) ✔; aggressive row absorption ON during elimination
  (colamd.jl:327–332, fires even for i∉C_c) ✔; super-columns ON (colamd.jl:406–447) ✔.

- **Pivot-row id-reuse aliasing invariant (flagged item 3) is CORRECT.** The merged pivot
  row reuses the id `r` of the first live row of the pivot column ([T] §4.2.4). Claim:
  every stale `C_j` reference to the OLD row under id `r` is pruned in phase 1. Proof
  holds: `r ∈ C_j ⇔ j ∈ R_r-old`; and `R_r-old \ {c} ⊆ R_r-new` (since `r ∈ C_c`, so
  old-`r`'s pattern is one of the sets unioned into the new pivot row). Phase 1 iterates
  **every** `j ∈ R_r-new` and drops `i==r` (colamd.jl:320), so all stale references are
  removed before phase 2 ever sums them. `c` itself is `_COLAMD_ORDERED` and never
  re-scanned. Aggressive-absorption-then-later-column interactions are caught by phase 2's
  `rowlive[i]` re-prune (colamd.jl:348). The invariant is sound. It is own-engineering
  correctly reasoned; the thesis §4.2 uses a different (parent-tree) layout, so the
  argument genuinely is the implementer's.

- **Provenance is clean.** Variable names (`iw`, `cstart`, `clen`, `rstart`, `rlen`,
  `rdeg`, `dhead/dnext/dprev`, `hhead/hnext`) mirror the project's own `amd.jl` and
  generic integer-workspace conventions, not COLAMD C identifiers. No struct field names
  or constants traceable to CHOLMOD source. (One numeric coincidence — see NIT-3.)

- **Bounds safety of the score-tightening.** `score = min(score, nactive−1)`
  (colamd.jl:207) and `min(sc, nactive_rem − thick[j])` (colamd.jl:461) keep every score
  `≤ n−1`, so `dhead[d+1]` (dhead sized `n`) never over-indexes and the pivot-selection
  scan (colamd.jl:232–234) terminates in-bounds. `mindeg` is only ever lowered, so it
  stays a valid lower bound for the upward scan. ✔

---

## DEFECT-1 — the `l_k = 0` discard is a source-ungrounded synthesis whose quality-equivalence is unproven (colamd.jl:291–294, 369–370, 383–399)

**Task item 3 asked whether D9 is implemented "VERBATIM from Algorithm 2/3." It cannot
be, and is not — and the gap is real, not pedantic.**

Neither source gives the algorithm the code actually needs. [P] Algorithm 2/3 carry the
`l_k = 0` discard but **explicitly assume no super-columns** (p.364: "To simplify the
presentation, we assume there are no super-columns"). [T] Algorithm 1 carries
super-columns + mass elimination but has **no `l_k` and no discard branch at all** (it
unconditionally does `C_j = C_j ∪ {r}` in final scoring, pp. 25–26). The code must
combine both, so the `l_k = 0` handling is inherently the implementer's own synthesis.
That is legitimate and unavoidable; the problem is the specific synthesis:

1. **`nrep[r]` (the code's `l_k`) is mutated during phase-2 mass elimination**
   (colamd.jl:370, `nrep[r] = max(nrep[r] - thick[j], 0)`). This has **no grounding in
   either source**: [P] computes `l_k` once, *before* the symbolic update, and never
   decrements it; [T] has no such quantity. Semantically `l_k` counts represented
   *rows*, but line 370 decrements it by an eliminated *column's* thickness.

2. **The discard is checked *after* phase 2** (colamd.jl:387), whereas [P] Algorithm 3
   checks `l_k = 0` *between* phase 1 and phase 2 (so phase-2 degree work is skipped
   entirely when `l_k = 0`). Combined with (1), the code can therefore fire the discard
   in cases where a paper-faithful pre-computed `l_k` would be `> 0` — driven to zero by
   mass elimination rather than being zero to begin with.

**Why it is not a BLOCKER:** the *core* D9 property is preserved — when the discard
fires, `continue` (colamd.jl:398) skips the final-scoring loop, so `{r}` is never appended
to any `C_j` (no phantom reference, the exact bug D9 exists to prevent). Survivors are
re-inserted with their prior scores (colamd.jl:391), matching "keep old `d_j`."
Permutations stay valid: survivors remain LIVE and in the degree list, so the
`while kout < nactive` loop still orders every column exactly once (`@assert kout ==
nactive` at colamd.jl:471 held across all tests and my traces). I hand-traced the test's
minimal trigger `A = [1 1]` (1×2, two single-entry columns): the discard fires with
`rlen[r]=0` after col 2 is correctly further-mass-eliminated → `perm = [1,2]`. Correct.

**Why it is still a DEFECT:** design_qr.md §2.2 pt 2 states "implement from Algorithm 2/3
verbatim … treat the bullet as a summary, not the spec," and CLAUDE.md req 1 demands every
line survive "where did this come from?". Line 370 does not, and the eager-discard timing
is a behavioral choice, not a transcription. I could not *construct* a witness where the
final permutation differs from a paper-faithful `l_k` (the not-strong-Hall cases I built
all discarded with `rlen[r]=0`, where timing is irrelevant), and there is a plausible
argument the divergence is quality-neutral (a row representing 0 candidate rows is a
dead-end that contributes no meaningful fill, so dropping `{r}` is harmless). But
"plausibly harmless and I couldn't break it" is not the standard this project sets for a
correctness-adjacent branch.

**Ask the implementer to do one of:** (a) move the discard to match [P] Algorithm 3
(compute `l_k` before the symbolic update, check before phase 2) and drop line 370; or
(b) keep the synthesis but add a header derivation proving output-equivalence, plus a
targeted test that exercises `nrep[r] → 0 via mass elimination with rlen[r] > 0` (the one
regime I could not witness) and pins the resulting order. Right now that regime is
untested and unproven.

---

## NIT-1 — the "2×-vs-greedy-mindeg" gate barely discriminates (test/colamd_tests.jl:99–125)

The comment says it "catches a badly-broken implementation." I measured what actually
trips it. Over 3000 random tiny instances (`m∈2:10, n∈2:8`), **neither COLAMD nor natural
nor reverse order ever exceeds 2× greedy** (COLAMD worst 1.20×, natural/reverse 0 failures).
On the three medium sizes in the same testitem: COLAMD 1.00–1.02×, but **natural** 1.10–1.45×
and **reverse** 1.12–1.45× — both still pass 2×. So an implementation that degenerated to
natural order would pass this gate. It guards only against *catastrophic* (>2×) breakage,
not the "badly-broken" quality the comment implies, and it never asserts COLAMD's real
signal (it sits right at greedy, ~1.0×). Not vacuous (it is a real computation and would
catch a total blow-up), but weak. Suggest asserting a tighter ratio (e.g. ≤ 1.3× greedy)
or `elimination_fill(colamd) ≤ elimination_fill(natural)` so the gate has teeth.

## NIT-2 — the wide-matrix test peels away the very property it should test (test/colamd_tests.jl:147–169)

The implementer's diagnosis is **correct and I reproduced it** (task item 5a/5c). Running
COLAMD's raw permutation vs `nnz(qr(A).R)` (SPQR default) on wide `m<n` sparse matrices,
raw ratios hit 4.75× / 1.78× / 1.45× — matching their reported "worst 5.55×." But on a
*fair* same-basis comparison (all no-peel, all `ORDERING_FIXED`), COLAMD is consistently
the **best** ordering:

```
shape     colamd_raw  natural  amd_on_AᵀA   spqr_default(peels)
80x120         2275     4482       2518          479
60x140         2968     6249       3558         2052
110x160        4019     8986       4239         3450
```

So the raw 4.75× is entirely SPQR's internal singleton pre-elimination ([P]/SPQR §2.1,
a separate pipeline stage — design §2.3, M5a task 9), not a COLAMD defect; composing a
matching peel in the test (giving worst 1.002) **is** a legitimate apples-to-apples proxy
for what the real pipeline will do (5b: yes). **However**, after peeling, the residual `B`
is squarish, so the test exercises COLAMD on a near-square residual, **not** on a wide
matrix — a genuinely wide-specific ordering bug would be invisible here. It is invisible
today only because no such bug exists (my no-peel table proves it), which the *test* does
not establish. Cheap fix: add one raw assertion, e.g.
`nnzR_fixed(A, colamd) ≤ nnzR_fixed(A, natural)` on the wide block, so the wide input is
actually ordered and scored.

## NIT-3 — dense-threshold constants coincide numerically with COLAMD's C defaults (tuning.jl:97–99)

`COLAMD_DENSE_FLOOR = 16`, `mult = 10.0`, threshold `max(16, 10·√dim)`
(colamd.jl:140–141). This is *numerically identical* to COLAMD's actual C-library default
(`knob = 10`, `max(16, 10·√n)`) — exactly the "just happens to match" pattern CLAUDE.md
req 1 flags. **Checked: provenance is salvaged.** design_qr.md D1 traces the
`max(16, mult·√dim)` shape and `mult=10` to the **AMD** package User Guide (`AMD_DENSE=10`,
design.md §2.2 pt 6), a permitted source, and the code deliberately rejects [P]'s own
stated 50%-density default. Same-author (Davis) convergence on the same convention is
expected. Already adjudicated in design review; noting only so the coincidence is on the
record for this code, not just the design.

## NIT-4 — `−thick[c]` reconciliation of eq. (2) and the defensive `resize!` are own-engineering (colamd.jl:291–294, 250–254)

`lr = max(lsum − thick[c], 0)` (colamd.jl:294) generalizes [P] eq. (2)'s `−1` to
super-columns of thickness `|c|`. Reasonable and flagged in the header, but note it is a
reconciliation the implementer invented (the thesis, the implementation reference, has no
`l_k`), so its correctness rests on the same synthesis as DEFECT-1. The `resize!` branch
(colamd.jl:250–254) is documented-unreachable defensive code that would grow `iw` beyond
the `O(|A|)` storage invariant if ever hit; acceptable as a guard, but it is the one place
the "storage never exceeds 2·nnz" contract is not structurally enforced.

---

## Answer to the specific ask on the wide-matrix test-peel

**It does not hide a real bug.** The implementer's root-cause (SPQR's always-on singleton
peel inflates the baseline) is demonstrably true — I reproduced the 4.75× raw gap and
confirmed it collapses to parity under a matching peel, while COLAMD on the same no-peel
basis beats both natural order and AMD-on-AᵀA. The peel is a fair proxy for the eventual
pipeline. The only residual issue is a *test-coverage* gap (NIT-2): the peel makes the
scored input squarish, so wide-specific behavior is asserted by proxy rather than directly.
The ordering algorithm itself is sound on wide inputs.
