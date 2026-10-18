---
title: Decisions
kind: decisions
summary: Current owner scope and gameplay fidelity, research proposals, and preserved historical decisions.
owns: [decisions.register]
verified: 2026-10-18
---

# Decisions

Newest first. IDs are never reused. A replaced decision keeps its entry and is tagged `[SUPERSEDED by D-###]`. Accepted designs are described in [ARCHITECTURE.md](ARCHITECTURE.md) as `[TARGET D-###]` until built.

D-002 to D-008 record the original 13 October acceptance, qualified at that time by D-009. The current scope is D-010 and fidelity target D-011. Historical text below explains the earlier rationale; superseded clauses are not current gates. D-002/D-003/D-005 retain their meaning only for a cellular experiment, not for every actor controller.

## Current entries

### D-010 — This repository serves Cybersand behaviour training and testing

[ACCEPTED] 2026-10-18 by owner; recorded by the ACT-004 research prepass, [T-100 / #10](https://github.com/techrote/neural-cyber-automata/issues/10).

The repository now exists in service of training and testing machine learning and intelligent behaviours for Cybersand and its actors. Include neural and cybernetic movement control, skills/usables, human and model command sources, coordination and relevant NCA experiments. Reorganize the research plan accordingly.

This supersedes the mandatory engine-agnostic/NCA-first milestone order, mandatory toy-world-first and no-RL-first choices. Prior methods remain candidates in their appropriate tracks. Modular design, truthful evidence, public-repository hygiene and copyright requirements remain. Cybersand-specific research is authorized immediately; this prepass does not modify its runtime or launch training.

### D-011 — Plausible gameplay through emulation, with deliberate flex and play

[ACCEPTED] 2026-10-18 by owner, clarifying the same prepass.

The target is plausible, expressive, stable actor behaviour. Emulate the useful effects of physical mechanisms instead of simulating every physical detail. Gyros do not require finite angular-momentum storage; bounded assist torque is an allowed design. Joints and attachments may use approximate compliant constraints, small controlled play and damping.

Do not turn conservation accounting, perfect prismatic geometry or zero positional error into research acceptance gates. Assess the authored mechanics for responsiveness, personality, intelligible response to contacts/recoil, recovery and absence of distracting twitch or glitches. Keep the same configured mechanics for human and learned controllers in a matched comparison.

This is not authorization to erase Rapier bodies, ignore intended collisions, bypass native material ownership or disguise numerical explosions as skill. Those implementation responsibilities still follow ACT-004 and its latest owner direction.

### D-012 — Layer intent, skills and local feedback over shared game actuation

[PROPOSED] Research recommendation, 2026-10-18.

Extend ACT003 rather than create a parallel command world. Different sources may provide mission intent, skill parameters, wire input or supported direct actuator commands. The host applies the same declared actuator/assist profile and validates lifetime/observation identity. Fast local feedback continues while a slow model reasons.

Start with exclusive authority and explicit takeover. Shared assistance, graph coordination and longer-lived mission contracts are separate comparisons. See [architecture](ARCHITECTURE.md) and [command design](design/COMMAND_AND_SKILL_CONTRACT.md).

### D-013 — Start with matched baselines and small learned policies

[PROPOSED] Research recommendation, 2026-10-18.

Use the actual authored actor mechanics to compare a tuned scripted/feedback baseline, imitation, and small recurrent or residual policies. PPO is an initial candidate; SAC, CPG parameter control, modular graphs and world models need a discriminating reason. Freeze observation/action semantics and benchmark provenance before ranking them. See [training and evaluation](design/TRAINING_AND_EVALUATION.md).

CPU conformance is the initial requirement; optional accelerators are qualified independently. Benchmark game stepping and inference before prescribing hardware or quantisation.

## Historical entries

### D-009 — Experiment first; keep the design engine-agnostic

[SUPERSEDED by D-010] Historical acceptance: 2026-10-13 by owner: "keep things flexible/don't make it cybersand specific yet let's just experiment first".

- **Decision:** until the owner approves a host-engine bridge (M4), everything here is a general NCA experiment platform. Names, interfaces and data formats describe a generic grid world, cells, signals and intents, not cybersand's materials, cell layout or scheduler. cybersand stays the motivating example and a read-only reference.
- **Options considered:** engine-agnostic experiments first (chosen) · design around cybersand's cell format from the start (tighter fit later; locks choices before anything is learned).
- **Why:** keeps the most options open while it is unknown which behaviours work at all (C-020).
- **Consequences:** D-003 and D-005 are written in generic terms; host-specific questions move to M4; C-004 forbids editing cybersand until then.
- **Revisit if:** M2 results show a behaviour worth fitting to a specific engine.

### D-008 — Training method ladder: gradients first, evolution strategies for world behaviours

[SUPERSEDED by D-010] Historical acceptance: 2026-10-13 by owner ("all as guessed").

- **Decision:** gradient training (backpropagation through many update steps) wherever the goal is differentiable, such as a target shape or pattern. Evolution strategies (score many slightly varied copies, move towards the better ones) for behaviours scored in a non-differentiable world, such as "swarm reaches the alarm source". No reinforcement learning at first.
- **Options considered:** gradients + evolution strategies (chosen) · reinforcement learning from the start (more general; far more samples and tuning) · evolution strategies only (simplest; slow for shape tasks).
- **How hard to change later:** easy; methods are separate trainer plug-ins (D-006).

### D-007 — Train in a standalone toy world first ("nanosand")

[SUPERSEDED by D-010] Historical acceptance: 2026-10-13 by owner ("all as guessed").

- **Decision:** a small grid world written in PyTorch inside this repo: a handful of materials, gravity, walls, an active "nanite" material and exact conservation of mass, with thousands of copies batched on a GPU. It is a test bed, never treated as any engine's truth.
- **Options considered:** toy world first (chosen: fast, GPU-friendly, no engine coupling) · train directly against a real engine (exact physics; CPU-bound and slow) · toy world only, never validated in an engine.
- **How hard to change later:** medium; behaviours may need fine-tuning when moved to a real engine (R-006).

### D-006 — One device-agnostic PyTorch trainer, two pinned environments

[SUPERSEDED by D-010] Historical acceptance: 2026-10-13 by owner ("all as guessed").

- **Decision:** one Python package that runs unchanged on CPU, on current NVIDIA GPUs with current PyTorch, and on the Tesla K80 (Kepler, `sm_37`) with the last PyTorch builds that support it. Two pinned environments (`modern`, `legacy-k80`), a device report command, and no features the legacy environment lacks in shared code (no half precision, no `torch.compile`).
- **Options considered:** one codebase, two environments (chosen) · K80 only · modern GPUs only.
- **Why:** keeps the K80 useful for long unattended runs and sweeps without making it a requirement; NCA models are small and train in FP32.
- **How hard to change later:** easy; dropping a lane removes one environment.

### D-005 — Signal fields are the only channel from the world to the cells

[ACCEPTED] 2026-10-13 by owner ("all as guessed").

- **Decision:** the world (or any controlling program) writes coarse regional signal layers, such as *alarm*, *attract*, *repair*, *threat direction*. Active cells read them as extra inputs. World events influence cells only through these fields.
- **Options considered:** signal fields (chosen: one small, testable interface) · cells query world objects directly (flexible; couples networks to engine internals) · no external input (cannot react to events).
- **Consequences:** reacting to large world events (T-018) is built on this.
- **How hard to change later:** medium; new signal types are easy, changing the interface means retraining.

### D-004 — Integer inference at runtime, bit-identical across platforms

[SUPERSEDED by D-010] Historical acceptance: 2026-10-13 by owner ("all as guessed").

- **Decision:** trained networks are converted to 8-bit integer weights with fixed-point arithmetic for the runtime library. A Python integer reference and the C++ runtime produce identical bytes for identical inputs.
- **Options considered:** integer (chosen: deterministic on every platform, small, fast with SIMD) · 32-bit float (results can differ between compilers and platforms) · float with strict maths (closer, still fragile).
- **How hard to change later:** medium; training includes quantisation-aware steps.

### D-003 — Cell memory lives in a separate hidden-state plane

[ACCEPTED] 2026-10-13 by owner ("all as guessed"). Retained only as the separate hidden-state-plane principle for cellular experiments. The historical engine-agnostic/M4 storage restriction in the rationale below is superseded by D-010.

- **Decision:** each active cell carries a small private state (about 8–16 numbers) in a hidden-state plane alongside the visible world grid, owned by the NCA module and moved together with its cell. How a host engine stores this plane is decided in M4.
- **Options considered:** separate plane (chosen) · squeeze state into a host's existing per-cell bytes (too little memory) · enlarge every cell (costs memory for every material).
- **How hard to change later:** easy inside this repo; hard once a host engine depends on it.

### D-002 — Cells output intents; the world's rules decide what happens

[ACCEPTED] 2026-10-13 by owner ("all as guessed").

- **Decision:** an active-cell network outputs requests (stay, move in a direction, attach, convert a material, emit a signal) plus its new hidden state. The world's own rules resolve competing requests. Mass is conserved: cells never create or destroy matter except through explicit, budgeted conversions.
- **Options considered:** intents (chosen: physics stays authoritative; balance rules enforceable) · the network writes cells directly (most expressive; can duplicate or delete matter and break determinism).
- **How hard to change later:** hard; every trained behaviour depends on this output format.

### D-001 — Build NCA behaviours in this separate repository

[SUPERSEDED by D-010] Historical acceptance: 2026-10-13 by owner: "seems ideal for developing some VERY interesting cybersand behaviours like nanotech swarms and distributed pseudo-intelligences", then "here's a repo to use to get started on the planning and implementation roadmap".

- **Decision:** neural cellular automata are developed here, in `techrote/neural-cyber-automata`, starting with planning and an implementation roadmap. Motivating behaviours: nanotech swarms and distributed pseudo-intelligences, with cybersand as the motivating host (scope narrowed to engine-agnostic experiments by D-009).
- **Options considered:** this separate repo (chosen: the owner created it) · building inside cybersand · not pursuing NCAs.
- **Consequences:** this repo owns training, test worlds and evaluation; host integration is a later, separately approved step (C-004).
- **Revisit if:** integration needs a tighter coupling than a library and a documented interface.
