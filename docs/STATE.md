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
## Accepted CPU foundation

T-101 merged through [PR #22](https://github.com/techrote/neural-cyber-automata/pull/22)
at `9e6da91ff520e27bc78d11dd8196844a9fa37ff3`, adding the installable CPU package,
strict gameplay/observation/action contracts, backward-compatible ACT003 mappings,
a versioned extension registry and immutable reproducible run records.

| Area | Evidence |
|---|---|
| CPU package and command line | Editable installation, environment report and recorded example exercised locally |
| Contracts, registry and run store | 27 focused tests; corruption, concurrency, lifecycle and privilege negative controls |
| Command-line conformance | Three additional tests for immutable records, optional-import isolation and invalid input |
| Linux/Windows package workflow | Both operating systems and the documentation check passed on the exact final T-101 head |
| Physical host adapter | T-102 implementation and actual Linux conformance described below |
| Authored gameplay controllers | Separate T-103 increment |
| Trained weights or scored learning runs | No accepted training result in T-101 |
| Runtime export, model service and device qualification | No accepted result in T-101 |

The [contract evidence](evidence/2026-10-19-t101-contracts.md) records source digests,
observed checks and corrected failures. The [package guide](PACKAGE.md) contains
commands; [progress checkpoints](PROGRESS.md) record ongoing implementation.

## Persistent actual host

The [host adapter](HOST_ADAPTER.md) implements owned reset/step over one persistent
Godot process, preserving the accepted native material/Rapier causal schedule.
Recorded playback, actual ACT003 sampled wire/metal and actual ACT004 physical
lanes have distinct identities. Six-tick decisions, twelve-tick helper leases,
exclusive observation expiry, cancellation and terminal durations are tested.

The [T-102 evidence](evidence/2026-10-19-t102.md) reports 23 focused passing cases,
including fifteen real integrations, and bounded 1,200-tick pilots for Monowheel
and Ripper. Realtime mode advances while a client is delayed and returns the
completed terminal state once. Linux host CI verifies pinned source and runtime
before executing the integration tests; final-head results belong to the PR.

These tests qualify the adapter on the selected host and CPU runtime. The actual
host has no separate passive-wrench decomposition, arbitrary body replacement,
support-removal or assist-mutation API. Those capabilities remain unavailable.
Windows physical runtime, desktop pacing and owner feel remain unmeasured.

## Host authority

The refreshed Cybersand source at
`f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497` includes ACT-004 articulated actors
merged through PR #181. Its owner API has configure, reset, observation, submit,
cancel, step and snapshot operations. T-102 owns qualification of that boundary.
No Cybersand source mutation is part of T-101.

Package tests and a recorded protocol example establish no gameplay success,
learned-policy value or owner feel acceptance. GPU qualification is unmeasured.
