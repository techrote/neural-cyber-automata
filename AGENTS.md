---
title: Agent instructions — neural-cyber-automata
kind: navigation
summary: Task entry, current charter, authority and evidence requirements.
owns: [agents.entry]
verified: 2026-10-18
---

# Agent instructions

This repository serves training and testing intelligent behaviours for Cybersand. Its scope includes ACT-004 physical actors, movement and usable skills, human and model control, coordination and useful NCA experiments.

## Read first

1. [Documentation map](docs/README.md).
2. [Current state](docs/STATE.md).
3. [Constraints](docs/CONSTRAINTS.md) and [decisions](docs/DECISIONS.md).
4. [Scope migration](docs/SCOPE_AND_MIGRATION.md) and the [implementation programme](docs/ROADMAP.md).
5. The task issue, all comments, its read-set and dependencies.
6. Live branches, overlapping PRs and relevant host revision.

The current user instruction takes precedence over older project instructions. Record a material change of scope in the decisions register; do not turn obsolete instructions into a permission loop. D-010 records the 18 October 2026 recharter. It authorizes research and design here, not a claim that a proposed control algorithm is accepted or implemented.

## Authority and status

Code on the default branch determines what exists. Latest owner instructions determine authorized scope. Accepted decisions specify requirements; research recommendations and implementation issue prompts do not prove runtime behaviour.

Use these distinctions:

- Current or partial: exists on the cited revision.
- Accepted design: authorized but possibly unbuilt.
- Proposed: a design candidate or experiment to assess.
- Deferred: preserved but off the immediate path.
- Evidence: reviewed, built, tested, ran on target, measured, owner accepted.

A paper result never becomes a Cybersand result. An untrained network, a mock service, a metadata flag, a green documentation check and a trained/evaluated policy are different evidence.

## Working rules

- One issue, one branch, one PR. Preserve unrelated work. Required checks must pass on the final head before a merge, then verify the merge.
- A research prepass edits documentation and task definitions. Do not launch training, acquire compute, call a paid model service or modify a host runtime as a side effect.
- Cybersand remains read-only for this prepass. Host integration changes need a corresponding Cybersand task and that project's acceptance process.
- Keep the gameplay plant in Cybersand: Rapier integrates bodies/contacts, authored assists and compliant connections shape their behaviour, and the native cellular owner controls material changes. Controllers submit bounded requests. D-011 permits deliberate flex/play and gyro assistance without momentum accounting.
- Preserve ACT003 wire/metal, perception, decision placement, coordination and training-privilege distinctions. A new version cannot silently reinterpret existing packets or presets.
- Implement a registered extension with a declared contract before a framework-wide abstraction. Keep future choices replaceable.
- Every PR states its documentation impact and exact evidence limits.
- Run `python tools/check_docs.py --strict` for documentation changes. Do not weaken the checker to admit false status claims.

## Experiments and publication

Training and benchmarks record the host and learner revisions, build/environment, seed sets, resolved configuration, observation/action schema, cadence, dataset and checkpoint hashes, budget, failures and evaluation suite. Use [RUNS.md](docs/RUNS.md) and immutable per-run manifests. Missing hardware is unmeasured, not a passing result.

Compare matched physical capability and information budgets. Keep evaluator-only state out of policy input. Distinguish requested actions, admitted actions, actual actuation and passive/contact motion. Log interventions and resets.

Public repository hygiene: no personal context, machine nicknames, local paths, credentials or private traces. Copyright line only; record third-party code provenance and licence before copying or porting. Large checkpoints, datasets and videos stay outside Git with hashes. Windows instructions should be complete and low friction.
