---
title: Decisions
kind: decisions
summary: Every significant choice, who made it, when, what else was considered, and why. All current entries are accepted; the project is in an engine-agnostic experiment phase (D-009).
owns: [decisions.register]
verified: 2026-10-13
---

# Decisions

Newest first. IDs are never reused. A replaced decision keeps its entry and is tagged `[SUPERSEDED by D-###]`. Accepted designs are described in [ARCHITECTURE.md](ARCHITECTURE.md) as `[TARGET D-###]` until built.

D-002 to D-008 were proposed by an agent on 2026-10-13 and accepted by the owner the same day ("all as guessed"), with the qualification recorded as D-009.

## Entries

### D-009 — Experiment first; keep the design engine-agnostic

[ACCEPTED] 2026-10-13 by owner: "keep things flexible/don't make it cybersand specific yet let's just experiment first".

- **Decision:** until the owner approves a host-engine bridge (M4), everything here is a general NCA experiment platform. Names, interfaces and data formats describe a generic grid world, cells, signals and intents, not cybersand's materials, cell layout or scheduler. cybersand stays the motivating example and a read-only reference.
- **Options considered:** engine-agnostic experiments first (chosen) · design around cybersand's cell format from the start (tighter fit later; locks choices before anything is learned).
- **Why:** keeps the most options open while it is unknown which behaviours work at all (C-020).
- **Consequences:** D-003 and D-005 are written in generic terms; host-specific questions move to M4; C-004 forbids editing cybersand until then.
- **Revisit if:** M2 results show a behaviour worth fitting to a specific engine.

### D-008 — Training method ladder: gradients first, evolution strategies for world behaviours

[ACCEPTED] 2026-10-13 by owner ("all as guessed").

- **Decision:** gradient training (backpropagation through many update steps) wherever the goal is differentiable, such as a target shape or pattern. Evolution strategies (score many slightly varied copies, move towards the better ones) for behaviours scored in a non-differentiable world, such as "swarm reaches the alarm source". No reinforcement learning at first.
- **Options considered:** gradients + evolution strategies (chosen) · reinforcement learning from the start (more general; far more samples and tuning) · evolution strategies only (simplest; slow for shape tasks).
- **How hard to change later:** easy; methods are separate trainer plug-ins (D-006).

### D-007 — Train in a standalone toy world first ("nanosand")

[ACCEPTED] 2026-10-13 by owner ("all as guessed").

- **Decision:** a small grid world written in PyTorch inside this repo: a handful of materials, gravity, walls, an active "nanite" material and exact conservation of mass, with thousands of copies batched on a GPU. It is a test bed, never treated as any engine's truth.
- **Options considered:** toy world first (chosen: fast, GPU-friendly, no engine coupling) · train directly against a real engine (exact physics; CPU-bound and slow) · toy world only, never validated in an engine.
- **How hard to change later:** medium; behaviours may need fine-tuning when moved to a real engine (R-006).

### D-006 — One device-agnostic PyTorch trainer, two pinned environments

[ACCEPTED] 2026-10-13 by owner ("all as guessed").

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

[ACCEPTED] 2026-10-13 by owner ("all as guessed").

- **Decision:** trained networks are converted to 8-bit integer weights with fixed-point arithmetic for the runtime library. A Python integer reference and the C++ runtime produce identical bytes for identical inputs.
- **Options considered:** integer (chosen: deterministic on every platform, small, fast with SIMD) · 32-bit float (results can differ between compilers and platforms) · float with strict maths (closer, still fragile).
- **How hard to change later:** medium; training includes quantisation-aware steps.

### D-003 — Cell memory lives in a separate hidden-state plane

[ACCEPTED] 2026-10-13 by owner ("all as guessed"), engine-agnostic per D-009.

- **Decision:** each active cell carries a small private state (about 8–16 numbers) in a hidden-state plane alongside the visible world grid, owned by the NCA module and moved together with its cell. How a host engine stores this plane is decided in M4.
- **Options considered:** separate plane (chosen) · squeeze state into a host's existing per-cell bytes (too little memory) · enlarge every cell (costs memory for every material).
- **How hard to change later:** easy inside this repo; hard once a host engine depends on it.

### D-002 — Cells output intents; the world's rules decide what happens

[ACCEPTED] 2026-10-13 by owner ("all as guessed").

- **Decision:** an active-cell network outputs requests (stay, move in a direction, attach, convert a material, emit a signal) plus its new hidden state. The world's own rules resolve competing requests. Mass is conserved: cells never create or destroy matter except through explicit, budgeted conversions.
- **Options considered:** intents (chosen: physics stays authoritative; balance rules enforceable) · the network writes cells directly (most expressive; can duplicate or delete matter and break determinism).
- **How hard to change later:** hard; every trained behaviour depends on this output format.

### D-001 — Build NCA behaviours in this separate repository

[ACCEPTED] 2026-10-13 by owner: "seems ideal for developing some VERY interesting cybersand behaviours like nanotech swarms and distributed pseudo-intelligences", then "here's a repo to use to get started on the planning and implementation roadmap".

- **Decision:** neural cellular automata are developed here, in `techrote/neural-cyber-automata`, starting with planning and an implementation roadmap. Motivating behaviours: nanotech swarms and distributed pseudo-intelligences, with cybersand as the motivating host (scope narrowed to engine-agnostic experiments by D-009).
- **Options considered:** this separate repo (chosen: the owner created it) · building inside cybersand · not pursuing NCAs.
- **Consequences:** this repo owns training, test worlds and evaluation; host integration is a later, separately approved step (C-004).
- **Revisit if:** integration needs a tighter coupling than a library and a documented interface.
