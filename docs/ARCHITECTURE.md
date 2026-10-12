---
title: Architecture
kind: architecture
summary: How the project is put together, what each part owns, and where new features plug in. Nothing is built; everything below is accepted design marked TARGET.
owns: [architecture.modules, architecture.seams, architecture.invariants]
verified: 2026-10-13
---

# Architecture

Untagged lines would be `[CURRENT]` (built); nothing is built yet. Accepted but unbuilt items are tagged `[TARGET D-###]`. The design is engine-agnostic (D-009): it talks about a grid world, cells, signals and intents, never a specific game engine.

## Shape

```text
                 ┌──────────── signal fields (D-005) ◄── world events / controllers
                 ▼
 world grid ──► perception ──► cell network ──► intents (D-002) ──► world rules resolve ──► world grid
                     ▲                │
                     └─ hidden-state plane (D-003) ◄┘

 trainers (D-008: gradient, evolution) drive many batched worlds (D-007) on CPU / modern GPU / K80 lanes (D-006)
 trained network ──► quantiser ──► integer runtime library (D-004) ──► any host program
```

- [TARGET D-006] Python package `nca` holds everything except the integer runtime.
- [TARGET D-004] The integer runtime is a separate dependency-free C++ library in `runtime/`, with byte-for-byte parity against a Python integer reference.

## Modules

| Module | Owns | Talks to | Status |
|---|---|---|---|
| `nca.registry` | Name → factory tables for every plug-in kind | everything | [TARGET D-006] |
| `nca.models` | Cell networks and their perception | worlds (through the step interface), trainers | [TARGET D-002] |
| `nca.worlds` | World state, rules, intent resolution, signal fields; toy world `nanosand` | models, trainers, eval | [TARGET D-007] |
| `nca.train` | Training loops: gradient, evolution strategies | models, worlds, runs | [TARGET D-008] |
| `nca.eval` | Versioned behaviour suites and scoring | worlds, models | [TARGET D-007] |
| `nca.runs` | Run folders, config hashing, seeding, RUNS.md rows | every runnable command | [TARGET D-006] |
| `nca.view` | Videos and interactive viewers | runs, worlds | [TARGET D-006] |
| `nca.quant` | Integer conversion and Python integer reference | models, runtime | [TARGET D-004] |
| `runtime/` (C++) | Integer inference only; no world rules | hosts | [TARGET D-004] |

## Extension points

### Plug-in registries

- [TARGET D-006] Models, worlds, trainers, behaviours (task + reward/loss + suite) and signal types each register under a name. New experiments add files and register; they do not edit existing modules (C-020).

### World step interface

- [TARGET D-002] A world exposes: observe (grid + hidden plane + signals → per-cell inputs), apply intents (resolve conflicts, conserve mass), and inject signals. Any world that implements it, toy or host-engine adapter, works with every model and trainer.

### Signal sources

- [TARGET D-005] Event generators (explosion, breach, attractor, scripted controller) write into named signal channels. A model declares which channels it reads.

## Invariants

- [TARGET D-002] Total mass per material is unchanged by any step except explicit, budgeted conversions. Checked by tests on every world.
- [TARGET D-006] Same config + seed on the same device gives identical results; every run writes a RUNS.md row (C-002).
- [TARGET D-004] Python integer reference and C++ runtime agree byte-for-byte on the behaviour suite.
- [TARGET D-009] No module imports or mirrors a specific engine's types, material IDs or cell layout before M4.

## Hard to change later

- The intent format (D-002) and hidden-state width (D-003): every trained behaviour depends on them. Width and intent set are configuration values, not constants, so experiments can vary them.
