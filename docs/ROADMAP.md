---
title: Roadmap
kind: roadmap
summary: The order of upcoming work, as milestones with exit tests and tasks with IDs. Nothing here is built; finished work moves to the state page.
owns: [roadmap.order, roadmap.tasks]
verified: 2026-10-13
---

# Roadmap

This page owns **order and scope**. Issue numbers and creation dates never imply order. Finished tasks leave this page: what they built is described in [STATE.md](STATE.md), and their evidence is in [evidence/](evidence/README.md) and [RUNS.md](RUNS.md).

Gate key: ⚙️ an agent can verify it · 👁 needs the owner's eyes or hands (watching a result, or running a command on a GPU machine).

**Design basis:** all tasks follow the accepted decisions D-002 to D-009. The project is in an engine-agnostic experiment phase (D-009): no task names a specific engine's materials, cell layout or scheduler before M4.

## M0 — Foundations: package, environments, run records

**M0 is done when:** `nca doctor` prints a device and software report in CI (CPU) and on at least one GPU on an owner machine, and a smoke run adds its own row to [RUNS.md](RUNS.md).

| Task | Status | What | Depends on | Gate | Write-set | Issue |
|---|---|---|---|---|---|---|
| T-001 | [PLANNED T-001] | Python package skeleton with plug-in registries; CPU test workflow | — | ⚙️ | `pyproject.toml`, `src/nca/` (package root, `registry.py`), `tests/`, `.github/workflows/tests.yml` | [#2](https://github.com/techrote/neural-cyber-automata/issues/2) |
| T-002 | [PLANNED T-002] | Pinned `modern` and `legacy-k80` environments, Windows setup scripts, `nca doctor` | T-001 | 👁 | `env/`, `scripts/`, `src/nca/doctor.py` | [#3](https://github.com/techrote/neural-cyber-automata/issues/3) |
| T-003 | [PLANNED T-003] | Run records: config hashing, seeding, run folders, automatic RUNS.md row | T-001 | ⚙️ | `src/nca/runs/`, `docs/RUNS.md` | [#4](https://github.com/techrote/neural-cyber-automata/issues/4) |
| T-004 | [PLANNED T-004] | M0 debt sweep and docs audit | T-001, T-002, T-003 | ⚙️ | refactors within M0 write-sets, `docs/` | [#5](https://github.com/techrote/neural-cyber-automata/issues/5) |

- **T-001 checks:** ⚙️ `pip install -e .` and `pytest` pass in CI; ⚙️ a registry test shows a dummy model and dummy world can be registered and looked up by name without editing existing files (C-020).
- **T-002 checks:** ⚙️ `nca doctor` runs on CPU in CI and reports Python, PyTorch, CUDA, device name and compute capability, or "no GPU" (never blank); 👁 the owner runs one bundled command block per environment on a GPU machine and pastes the report; the report is saved as an evidence record. The legacy environment must report `sm_37` among PyTorch's supported architectures before any K80 claim.
- **T-003 checks:** ⚙️ two runs with the same config and seed produce the same config hash and identical CPU outputs; ⚙️ a run without a RUNS.md row fails the test suite.

## M1 — Baseline: reproduce Growing NCA

**M1 is done when:** an NCA trained on a small image grows that image from a single seed cell and regrows it after damage; the owner has watched the video; and steps-per-second for training is measured on every available device lane, each linked from [RUNS.md](RUNS.md).

| Task | Status | What | Depends on | Gate | Write-set | Issue |
|---|---|---|---|---|---|---|
| T-005 | [PLANNED T-005] | Growing NCA model and gradient trainer (perception filters, alive masking, stochastic updates, sample pool, damage) | T-001, T-003 | ⚙️ | `src/nca/models/growing.py`, `src/nca/train/gradient.py`, `tests/` | [#6](https://github.com/techrote/neural-cyber-automata/issues/6) |
| T-006 | [PLANNED T-006] | Rollout viewer: frames to video/GIF, plus an interactive "click to damage" viewer that runs on Windows | T-005 | 👁 | `src/nca/view/` | [#7](https://github.com/techrote/neural-cyber-automata/issues/7) |
| T-007 | [PLANNED T-007] | Device lane benchmark: training steps/s and memory on CPU, K80 (one die), and any modern GPU | T-002, T-005 | 👁 | `src/nca/bench/`, `docs/HARDWARE.md`, `docs/evidence/` | [#8](https://github.com/techrote/neural-cyber-automata/issues/8) |
| T-008 | [PLANNED T-008] | M1 debt sweep and docs audit | T-005, T-006, T-007 | ⚙️ | refactors within M1 write-sets, `docs/` | [#9](https://github.com/techrote/neural-cyber-automata/issues/9) |

- **T-005 checks:** ⚙️ a tiny configuration (16×16 target, few hundred steps) trains on CPU in CI and its loss falls below a recorded threshold; ⚙️ the implementation is written from the published description, not copied (C-001); sources listed in [research/SOURCES.md](research/SOURCES.md).
- **T-006 checks:** 👁 the owner opens a video and the interactive viewer from a direct link or one command, and sees growth, damage and regrowth. Playback speed and zoom are settings, not fixed values.
- **T-007 checks:** 👁 the owner runs one command per machine; ⚙️ results are written as an evidence record with device, driver, environment and settings; anything not run is listed as "not measured".

## M2 — Nanosand: first nanite behaviour

**M2 is done when:** in the toy world, nanites move towards an alarm source, regroup after part of the swarm is destroyed, react to at least two kinds of world event, and still do so with random update order and noise; mass error is exactly zero throughout; a versioned behaviour score and a video exist; the owner judges the behaviour from the video.

| Task | Status | What | Depends on | Gate | Write-set | Issue |
|---|---|---|---|---|---|---|
| T-009 | [PLANNED T-009] | Toy world "nanosand": a few materials, gravity, walls, nanite material, intent resolution, mass checks, many copies batched on GPU | T-001, T-003 | ⚙️ | `src/nca/worlds/nanosand/`, `tests/` | — |
| T-010 | [PLANNED T-010] | Signal fields (alarm source) and a nanite network with a hidden state plane | T-009 | ⚙️ | `src/nca/models/nanite.py`, `src/nca/worlds/nanosand/signals.py` | — |
| T-011 | [PLANNED T-011] | Behaviour evaluation suite v1: fixed scenarios and seeds, versioned, with scores | T-009 | ⚙️ | `src/nca/eval/`, `evals/v1/` | — |
| T-012 | [PLANNED T-012] | Evolution-strategies trainer with robustness training; first behaviour "swarm to alarm, regroup after damage" | T-010, T-011 | 👁 | `src/nca/train/es.py`, `configs/behaviours/swarm_alarm.*` | — |
| T-018 | [PLANNED T-018] | Event-driven signal sources: explosion, breach and attractor events write signal fields; the swarm reacts (alarm, repair, retreat) | T-010, T-012 | 👁 | `src/nca/worlds/nanosand/events.py`, `configs/behaviours/` | — |
| T-013 | [PLANNED T-013] | M2 debt sweep and docs audit | T-009 … T-012, T-018 | ⚙️ | refactors within M2 write-sets, `docs/` | — |

- **T-009 checks:** ⚙️ total mass per material is identical before and after 10,000 steps across 64 random worlds; ⚙️ same seed gives byte-identical worlds on CPU; ⚙️ GPU and CPU results agree for the same seed, or the difference is recorded as a gotcha with its cause.
- **T-018 checks:** ⚙️ each event type is a registered plug-in that writes only through the signal-field interface (D-005); ⚙️ behaviour suite v1 gains event scenarios under a new suite version; 👁 the owner watches the swarm respond to two different events in one video.
- **T-011 checks:** ⚙️ the suite has a version number; scores from different versions are never compared (rule recorded in the suite's README).
- **T-012 checks:** ⚙️ the trained behaviour beats an untrained network on suite v1 by a margin recorded before training starts; 👁 the owner watches the video and says whether it reads as "a swarm". Swarm speed and cohesion are exposed as training settings so alternatives can be shown side by side.

## M3 — Integer runtime

**M3 is done when:** a dependency-free C++ library runs the M2 nanite network in integer arithmetic, matches a Python integer reference byte-for-byte on every suite v1 scenario, builds in CI on Windows (MSVC) and Linux, and its cost per nanite update is measured on a desktop CPU.

| Task | Status | What | Depends on | Gate | Write-set | Issue |
|---|---|---|---|---|---|---|
| T-014 | [PLANNED T-014] | Quantisation to integer weights and a Python integer reference; score loss measured on suite v1 | T-012 | ⚙️ | `src/nca/quant/` | — |
| T-015 | [PLANNED T-015] | C++ integer runtime library with byte-for-byte parity tests; Windows and Linux CI builds | T-014 | ⚙️ | `runtime/`, `.github/workflows/runtime.yml` | — |
| T-016 | [PLANNED T-016] | CPU cost benchmark: nanite updates per second per core, single and multi-thread | T-015 | ⚙️ | `runtime/bench/`, `docs/evidence/` | — |
| T-017 | [PLANNED T-017] | M3 debt sweep and docs audit | T-014, T-015, T-016 | ⚙️ | refactors within M3 write-sets, `docs/` | — |

## M4 — Host-engine bridge (needs the owner's go-ahead)

**M4 is done when:** a written interface proposal for a chosen host engine (cybersand is the motivating candidate: hidden-state storage, intent resolution inside the host's scheduling rules, signal-field writers, CPU budget per tick) has been accepted by the owner, and a first active material runs in a host test scene behind a default-off setting.

Tasks are added when M3 is complete and the owner approves starting M4 (C-004).
