---
title: Current state
kind: state
summary: CPU contract implementation and explicit evidence limits for the current Cybersand control programme.
owns: [state.summary]
verified: 2026-10-19
---

# Current state

## T-100 — Accepted research baseline

The T-100 research programme landed in PR #21 at
`95d8788686980974d4f849e892c33f531d992fe7`.
## Implemented CPU foundation

T-101 adds the installable CPU package,
strict gameplay/observation/action contracts, backward-compatible ACT003 mappings,
a versioned extension registry and immutable reproducible run records.

| Area | Evidence |
|---|---|
| CPU package and command line | Editable installation, environment report and recorded example exercised locally |
| Contracts, registry and run store | 27 focused tests; corruption, concurrency, lifecycle and privilege negative controls |
| Command-line conformance | Three additional tests for immutable records, optional-import isolation and invalid input |
| Linux/Windows package workflow | Added; its final-head hosted result is recorded in the PR |
| Physical host adapter | Separate T-102 increment; not part of T-101 acceptance |
| Authored gameplay controllers | Separate T-103 increment |
| Trained weights or scored learning runs | No accepted training result in T-101 |
| Runtime export, model service and device qualification | No accepted result in T-101 |

The [contract evidence](evidence/2026-10-19-t101-contracts.md) records source digests,
observed checks and corrected failures. The [package guide](PACKAGE.md) contains
commands; [progress checkpoints](PROGRESS.md) record ongoing implementation.

## Host authority

The refreshed Cybersand source at
`f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497` includes ACT-004 articulated actors
merged through PR #181. Its owner API has configure, reset, observation, submit,
cancel, step and snapshot operations. T-102 owns qualification of that boundary.
No Cybersand source mutation is part of T-101.

Package tests and a recorded protocol example establish no gameplay success,
learned-policy value or owner feel acceptance. GPU qualification is unmeasured.
