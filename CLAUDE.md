# PureSparse.jl — notes for coding agents

- The rules for contributions (numbered requirements, test and benchmark methodology) are in
  [`CONTRIBUTING.md`](CONTRIBUTING.md). Follow them exactly; requirement 1 (clean-room
  provenance: never read CHOLMOD or SuiteSparse source) is the one most easily violated by
  accident.
- Architecture and the approved design: [`docs/design.md`](docs/design.md) and the other
  `docs/design*.md` files. Milestone status and open work: [`ROADMAP.md`](ROADMAP.md).
- Treat the design documents as a contract. If you are tempted to skip, substitute or
  reinterpret a requirement, stop and ask.
- For hard published algorithms (AMD approximate degree, left-looking scheduling,
  Davis–Hager update/downdate), work from the paper's exact form or delegate to a stronger
  model; do not reimplement from memory.
- Measure before asserting a performance claim; verify facts by reading or running code.
- Comments and commit messages describe the code as it is, for a reader who has only the
  repository.
- No Python anywhere in the repository's tooling.
