---
title: Constraints
kind: constraints
summary: The rules of the project. What must always hold, what must not be done (and until when), and the default preferences.
owns: [constraints.register]
verified: 2026-10-18
---

# Constraints

Every entry is tagged. `FORBIDDEN` always states its lift condition (`until: …`) or `permanent`; only the owner lifts it.

## Required

### C-001 — Licence

[REQUIRED] Copyright line only (`© 2026 techrote. All rights reserved.`), no LICENSE file. Owner's standing rule. If third-party code is ever copied or ported, its licence is recorded in [research/SOURCES.md](research/SOURCES.md) first; copyleft code would make this repo inherit that licence and needs the owner's approval.

### C-002 — Evidence before claims

[REQUIRED] Training, benchmark and quality claims name the device, environment, commit, config and seed, and link a row in [RUNS.md](RUNS.md). "Not measured" is written as such, never as zero. Owner's standing rule.

### C-003 — Public repository hygiene

[REQUIRED] This repository is public. No personal context, local paths, machine nicknames, credentials or tokens in files, issues, pull requests or logs. Owner's standing rule.

### C-005 — Large artifacts stay out of Git

[REQUIRED] Checkpoints, videos and datasets go to GitHub release assets (or another store the owner approves), with their SHA-256 recorded in [RUNS.md](RUNS.md). Only tiny test fixtures live in Git. Owner accepted 2026-10-13 (I-007).

## Current scope and fidelity

### C-004 — Historical engine-agnostic bridge restriction

[SUPERSEDED by D-010] The old M4 restriction was accepted on 13 October. The 18 October owner request authorizes Cybersand-specific research and rechartering this repository. The current prepass leaves Cybersand code and docs unchanged; later host edits belong to a corresponding Cybersand task and acceptance process.

### C-006 — Plausible, stable authored mechanics

[REQUIRED] D-011 governs fidelity. Bounded gyro assistance, approximate joints and controlled flex/play are allowed. No finite gyro momentum ledger or mechanically perfect connection is required. Train and evaluate the same declared gameplay mechanics, with equal assistance and limits in matched comparisons.

### C-007 — Keep evidence and execution scope distinct

[REQUIRED] This prepass produces research/design and task definitions. It does not run training or call model services. A later implementation task must record its budget and exact runtime. Unknown or unavailable readings are not zero; a learned metadata flag is not evidence of training. A proposed method becomes an executed experiment only when its task is commissioned.

## Preferred

### C-020 — Modular and extensible first

[PREFERRED] Choose the approach that keeps later sweeping changes cheap: plug-in registries for worlds, models, trainers and behaviours, rather than code paths edited in place. Owner's standing priority.

### C-021 — Performance and effectiveness

[PREFERRED] Measure before and after changes to hot paths (training step, world step, runtime inference) and record numbers with their conditions. Owner's standing priority.

### C-022 — Windows-first, low-friction owner steps

[PREFERRED] Anything the owner runs works on Windows where possible, takes one bundled command block or a direct download link, and states prerequisites before the commands. Owner's standing preference.
