---
title: Architecture
kind: architecture
summary: How the project is put together and where new features plug in. Nothing is built; only the boundary set by D-001 is accepted, and the detailed design waits on proposals D-002 to D-008.
owns: [architecture.modules, architecture.seams, architecture.invariants]
verified: 2026-10-13
---

# Architecture

Untagged lines would be `[CURRENT]` (built); nothing is built yet. Accepted but unbuilt items are tagged `[TARGET D-###]`. Designs that are only proposed are **not** described here; they live in [DECISIONS.md](DECISIONS.md) as `[PROPOSED]` and move here once accepted.

## Shape

- [TARGET D-001] This repository owns everything needed to *make* NCA behaviours: training code, test worlds, evaluation, trained weights' records and the documents that describe how a behaviour is to be used.
- [TARGET D-001] The cybersand repository owns the engine. Anything this repo hands to cybersand crosses one documented boundary; the form of that boundary is decided by the proposals below.

Proposed overall flow, for orientation only (every box depends on an unaccepted decision):

```text
signal fields (D-005) ──► nanite network ──► intents (D-002) ──► world rules resolve ──► cells
                              ▲    │
                hidden plane (D-003) ◄┘
trained in: toy world (D-007) with gradient / evolution methods (D-008)
on: CPU / modern GPU / K80 lanes (D-006) · shipped as: integer runtime (D-004)
```

## Modules

| Module | Owns | Talks to | Status |
|---|---|---|---|
| Documentation and run records | `docs/`, `tools/check_docs.py` | everything | current (docs tooling only, see [STATE.md](STATE.md)) |
| NCA tooling (training, worlds, evaluation) | this repo's code, once written | cybersand only through the boundary | [TARGET D-001] |

## Extension points

None are defined yet. Proposed seams (model registry, world registry, training-method registry, behaviour registry) are part of D-006 to D-008 and are added here when those are accepted.

## Invariants

- [TARGET D-001] Nothing in this repository edits cybersand. Integration changes to cybersand happen in cybersand, through its own process.

## Hard to change later

- The nanite output format (D-002) and hidden-state storage (D-003): every trained behaviour depends on them. They are flagged for the owner before any training beyond the M1 reproduction.
