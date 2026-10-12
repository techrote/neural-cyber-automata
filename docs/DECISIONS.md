---
title: Decisions
kind: decisions
summary: Every significant choice, who made it, when, what else was considered, and why. Only D-001 is decided; D-002 to D-008 are proposals waiting for the owner.
owns: [decisions.register]
verified: 2026-10-13
---

# Decisions

Newest first within each group. IDs are never reused. A replaced decision keeps its entry and is tagged `[SUPERSEDED by D-###]`. When a proposal is accepted, its tag changes here and its design moves into [ARCHITECTURE.md](ARCHITECTURE.md) as `[TARGET D-###]`.

## Accepted

### D-001 — Build NCA behaviours for cybersand in this separate repository

[ACCEPTED] 2026-10-13 by owner: "seems ideal for developing some VERY interesting cybersand behaviours like nanotech swarms and distributed pseudo-intelligences", then "here's a repo to use to get started on the planning and implementation roadmap".

- **Decision:** neural cellular automata are developed here, in `techrote/neural-cyber-automata`, starting with planning and an implementation roadmap. The motivating behaviours are nanotech swarms and distributed pseudo-intelligences inside cybersand.
- **Options considered:** this separate repo (chosen: the owner created it) · building inside cybersand (couples experimental ML tooling to the engine's build and docs) · not pursuing NCAs.
- **Why:** NCAs are local per-cell rules, which matches cybersand's cellular model; training is a separate Python/GPU workload with its own environments.
- **Consequences:** this repo owns training, test worlds and evaluation. Anything that touches cybersand is a later, separately approved step (I-001).
- **Revisit if:** integration needs a tighter coupling than a library and a documented interface.

## Proposed (not decided)

Each entry below was suggested by an agent on 2026-10-13 and waits for the owner. Nothing may be built on a proposed decision.

### D-008 — Training method ladder: gradients first, evolution strategies for world behaviours

[PROPOSED] Raised by agent 2026-10-13.

- **Proposal:** use gradient training (backpropagation through many update steps, as in the Growing NCA work) wherever the goal is a differentiable target such as a shape or pattern. Use evolution strategies (score many slightly varied copies, move towards the better ones) for behaviours scored in a non-differentiable world, such as "swarm reaches the alarm source". Reinforcement learning is not used at first.
- **Options:** A gradients + evolution strategies (proposed) · B reinforcement learning from the start (more general; far more samples and tuning) · C evolution strategies only (simplest; slow for shape tasks).
- **How hard to change later:** easy. Methods are separate trainer plug-ins (D-006).

### D-007 — Train in a standalone toy world first ("nanosand"), cybersand later

[PROPOSED] Raised by agent 2026-10-13.

- **Proposal:** a small sand world written in PyTorch inside this repo: a handful of materials, gravity, walls, a nanite material and conservation of mass. It runs thousands of copies at once on a GPU. Its rules imitate cybersand's style but are deliberately simpler; it is never treated as cybersand truth.
- **Options:** A toy world first (proposed: fast, GPU-friendly, no engine coupling) · B train directly against cybersand (exact physics; CPU-bound and slow, per the owner's note that physics is cybersand's main cost) · C toy world only, never validated in cybersand.
- **Why:** behaviour learning needs very many simulated steps; the toy world provides them. Differences from cybersand are measured in M4, not assumed away.
- **How hard to change later:** medium. Trained behaviours may need retraining or fine-tuning when moved to the real engine.

### D-006 — One device-agnostic PyTorch trainer, two pinned environments

[PROPOSED] Raised by agent 2026-10-13.

- **Proposal:** one Python package that runs unchanged on CPU, on current NVIDIA GPUs with current PyTorch, and on the Tesla K80 (Kepler, `sm_37`) with the last PyTorch builds that still support it. Two pinned environment files (`modern`, `legacy-k80`), a device report command, and no features the legacy environment lacks (no half precision, no `torch.compile`).
- **Options:** A one codebase, two environments (proposed) · B K80 only (cheapest hardware use; ties the project to old software) · C modern GPUs only (simplest; leaves the K80s idle).
- **Why:** keeps the K80 useful for long unattended runs and parameter sweeps without making it a requirement. NCA models are tiny and train in plain FP32, which Kepler handles.
- **How hard to change later:** easy. Dropping a lane removes one environment file.

### D-005 — Signal fields are the only channel from the world to nanites

[PROPOSED] Raised by agent 2026-10-13.

- **Proposal:** the game writes coarse regional signal layers (for example *alarm*, *threat direction*, *repair needed*, *attract here*). Nanite cells read them as extra inputs. Explosions, the player, damage or a strategist character (I-003) influence nanites only through these fields.
- **Options:** A signal fields (proposed: one small, testable interface) · B nanites query world objects directly (flexible; couples the network to engine internals) · C no external input (pure local rules; cannot react to world events).
- **How hard to change later:** medium. Adding a new signal type is easy; changing the interface itself means retraining.

### D-004 — Integer inference at runtime, bit-identical across platforms

[PROPOSED] Raised by agent 2026-10-13.

- **Proposal:** trained networks are converted to 8-bit integer weights with fixed-point arithmetic for the runtime library. A Python integer reference and the C++ runtime must produce identical bytes for identical inputs.
- **Options:** A integer (proposed: deterministic on desktop and Web, small, fast with SIMD) · B 32-bit float (simpler; results can differ between compilers and platforms, which breaks replays) · C float with forced strict maths (closer, still fragile across Web/desktop).
- **Why:** cybersand defines deterministic ordering and hashing for its simulation; a learned material must not be the part that breaks it.
- **How hard to change later:** medium. Training must include quantisation-aware steps.

### D-003 — Nanite memory lives in a separate state plane beside the cell grid

[PROPOSED] Raised by agent 2026-10-13.

- **Proposal:** each nanite needs a small private state (about 8–16 hidden numbers). cybersand's cell has four bytes, of which two are generic material state, so the hidden state is kept in a separate per-cell plane owned by the nanite module and moved together with the cell.
- **Options:** A separate plane (proposed) · B squeeze state into the two existing bytes (no engine change; too little memory for useful behaviour) · C enlarge every cell (simple to read; costs memory for every material).
- **How hard to change later:** hard once integrated. This is the main cybersand-facing design question and is only settled in M4.

### D-002 — Nanites output intents; the engine decides what happens

[PROPOSED] Raised by agent 2026-10-13.

- **Proposal:** a nanite network outputs requests (stay, move in a direction, attach, convert a material, emit a signal) plus its new hidden state. The world's normal rules resolve competing requests. Mass is conserved: nanites never create or destroy matter except through explicit, budgeted conversions.
- **Options:** A intents (proposed: physics stays authoritative; balance rules enforceable) · B the network writes cells directly (most expressive; can duplicate or delete matter and break determinism).
- **How hard to change later:** hard. Every trained behaviour depends on this output format.
