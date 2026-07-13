# Provenance & Licensing

PureSparse.jl is MIT licensed. It reimplements the algorithm family SuiteSparse's CHOLMOD
implements (supernodal sparse Cholesky/LDLᵀ), but CHOLMOD's Supernodal and Modify modules
are GPL — so PureSparse is a **strict clean-room reimplementation**: design and code
derive only from published academic papers, official user-guide documentation, and
independent reasoning. **CHOLMOD/SuiteSparse source code is never read, in any form** —
not on GitHub, not via search-result snippets, not from language-model recall of source
text, not via a third-party port derived from that source. Black-box comparison against
CHOLMOD's *output* (via Julia's `SparseArrays` stdlib, as a correctness oracle and
benchmark baseline) is fine and used throughout PureSparse's test suite — only the source
is off-limits.

Every algorithm name, struct field name, and numeric constant in the codebase must
survive the question "where did this come from?" with an answer that is a paper citation,
a user-guide citation, or an in-repository independent derivation — never "it happens to
match CHOLMOD's default." This was tested twice during design review and caught real
violations (a field-naming coincidence and a threshold-value coincidence, both renamed/
re-derived before implementation began — see `docs/design.md` §0).

## Provenance table

| Component | Source (papers/books/guides only) |
|---|---|
| AMD ordering | Amestoy, Davis, Duff, *An Approximate Minimum Degree Ordering Algorithm*, SIAM J. Matrix Anal. Appl. 17(4), 1996; AMD package *User Guide* (dense-row default) |
| Elimination tree, postorder | Liu 1986; Davis, *Direct Methods for Sparse Linear Systems* (SIAM, 2006), ch. 4 |
| Column counts | Gilbert–Ng–Peyton 1994; Davis's book, ch. 4 |
| Fundamental supernodes | Liu, Ng, Peyton 1993 |
| Relaxed amalgamation | Ashcraft–Grimes 1989; Ng–Peyton 1993 (concept only — the numeric thresholds are PureSparse's own free tunables, no external provenance) |
| Left-looking supernodal LLᵀ | Ng–Peyton 1993; Rothberg–Gupta 1993 |
| SQD / LDLᵀ regularization (M2) | Vanderbei 1995; Stellato et al. (OSQP/QDLDL); the Clarabel paper |
| Update/downdate (M2) | Davis–Hager, *Modifying a Sparse Cholesky Factorization*, SIAM J. Matrix Anal. Appl., 1999, and the 2001 multiple-rank follow-up |
| GPU hybrid design (M3) | Rennich et al. 2016 (concept only) |
| Workspace-size bounds | independent derivation from the left-looking update schedule |

See `docs/design.md` §11 (in the repository) for the full policy and the design review
that enforced it.
