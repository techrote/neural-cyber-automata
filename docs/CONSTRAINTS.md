---
title: Constraints
kind: constraints
summary: The rules of the project. What must always hold, what must not be done (and until when), and the default preferences.
owns: [constraints.register]
verified: 2026-10-13
---

# Constraints

Every entry is tagged. `FORBIDDEN` always states its lift condition (`until: …`) or `permanent`; only the owner lifts it. Rules an agent has only suggested live in [IDEAS.md](IDEAS.md) as `[PROPOSED]` until the owner accepts them.

## Required

### C-001 — Licence

[REQUIRED] Copyright line only (`© 2026 techrote. All rights reserved.`), no LICENSE file. Owner's standing rule for their work. If third-party code is ever copied or ported, its licence is recorded in [research/SOURCES.md](research/SOURCES.md) first; copyleft code would make this repo inherit that licence and needs the owner's approval.

### C-002 — Evidence before claims

[REQUIRED] Training, benchmark and quality claims name the device, environment, commit, config and seed, and link a row in [RUNS.md](RUNS.md). "Not measured" is written as such, never as zero. Owner's standing rule for long-horizon projects.

### C-003 — Public repository hygiene

[REQUIRED] This repository is public. No personal context, local paths, machine nicknames, credentials or tokens in files, issues, pull requests or logs. Owner's standing rule.

## Preferred

### C-020 — Modular and extensible first

[PREFERRED] Choose the approach that keeps later sweeping changes cheap: plug-in registries for worlds, models, training methods and behaviours, rather than code paths edited in place. Owner's standing priority.

### C-021 — Performance and effectiveness

[PREFERRED] Measure before and after changes to hot paths (training step, world step, runtime inference) and record numbers with their conditions. Owner's standing priority.

### C-022 — Windows-first, low-friction owner steps

[PREFERRED] Anything the owner runs works on Windows where possible, takes one bundled command block or a direct download link, and states prerequisites before the commands. Owner's standing preference.
