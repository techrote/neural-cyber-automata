---
title: Decisions
kind: decisions
summary: Current owner scope and gameplay fidelity, research proposals, and preserved historical decisions.
owns: [decisions.register]
verified: 2026-10-22
---

# Decisions

Newest first. IDs are never reused. A replaced decision keeps its entry and is tagged `[SUPERSEDED by D-###]`. Accepted designs are described in [ARCHITECTURE.md](ARCHITECTURE.md) as `[TARGET D-###]` until built.

D-002 to D-008 record the original 13 October acceptance, qualified at that time by D-009. The current scope is D-010 and fidelity target D-011. Historical text below explains the earlier rationale; superseded clauses are not current gates. D-002/D-003/D-005 retain their meaning only for a cellular experiment, not for every actor controller.

## Current entries
### D-029 — Prepare matched IK trials; no training; [ACCEPTED SCOPE] 2026-10-23; [endpoint-loss hypothesis, serial/parallel packets and excluded finals](decisions/2026-10-23-ik-matched-preparation.md).
### D-028 — Polar regression investigation; [ACCEPTED SCOPE] 2026-10-23; [diagnosis and proposal only; no training, criterion relaxation, final opening or accounting reset](design/IK_POLAR_REGRESSION.md).
### D-027 — Bounded polar-feature IK follow-up; [ACCEPTED] 2026-10-23; [new conditional TITAN allocation, fresh final and unchanged gates](decisions/2026-10-23-ik-polar-followup.md).
### D-026 — Reviewed recovery of the same frozen IK evaluation
[ACCEPTED] 2026-10-23: [preserved markers, accounted costs and no retraining](decisions/2026-10-23-ik-evaluation-recovery.md).

### D-025 — Preserve the original IK allocation through an admitted source amendment

[ACCEPTED] 2026-10-23: [original allocation, fresh admissions and necessary source amendment](decisions/2026-10-23-ik-continuation.md).

### D-024 — Narrow Windows retained-label compatibility and one gated TITAN batch

[ACCEPTED] 2026-10-23 by owner. See the [scope and unchanged admission gates](decisions/2026-10-23-ik-windows-labels.md).

### D-023 — Risk-managed LEARN-JOLT-003 resource admission
[ACCEPTED] 2026-10-23: [8 GiB watchdog and two independent reviews](decisions/2026-10-23-jolt-resource-risk.md).
### D-022 — CPU identity advisory for Jolt-003
[ACCEPTED] 2026-10-23 by owner: [portable CPU contract and ballpark timings](decisions/2026-10-23-jolt-cpu-portability.md).

### D-021 — Explicitly gated long-running Jolt CI entry

[ACCEPTED] 2026-10-22 by owner. See the
[full decision and unchanged admission gates](decisions/2026-10-22-jolt-ci.md).

### D-020 — Remove operational wall-clock limits and rerun CI

[ACCEPTED] 2026-10-22 by owner, after PR #73: remove all time limits, rerun CI, then stop until the owner reports its completion. This supersedes D-019's operational wall-clock ceilings and time-fit admission requirements. It does not launch the conditional physical comparison or authorize merging this change.

The separately versioned [configuration v2](../configs/learning/jolt-learn-003-v2.json) uses explicit null wall, phase-reserve and owner-transport limits. Execution v3 propagates that policy through capture, startup, requests, audit, publication, cleanup and independent observation. Elapsed time remains finite telemetry; phase ordering, all 39 counters, storage/memory integrity, source/runtime/review admission and qualification thresholds retain their requirements. Simulated task ticks, recovery deadlines and inference quality thresholds remain unchanged. Historical finite configurations, receipts and published evidence stay immutable.

Remove the repository's CI job timeout overrides and synthetic diagnostic wall guards. [GitHub-hosted runner limits](https://docs.github.com/en/actions/reference/limits) still impose a six-hour job maximum. Queue fresh CI for the changed source, then stop; do not wait for its result or consume the experimental allocation.

### D-019 — Implement and conditionally execute LEARN-JOLT-003

[PARTIALLY SUPERSEDED by D-020: operational time limits and time-fit admission]

[ACCEPTED] 2026-10-22 by owner: complete the prepared
[LEARN-JOLT-003 task](design/LEARN_JOLT_003_PROMPT.md). The
[actual commission](https://github.com/techrote/neural-cyber-automata/issues/62#issuecomment-6347420584)
records fresh allocation `learn-jolt-003-20261022-a1` for implementation,
independent review and exactly one 7,200-second CPU comparison conditional on
all prospective gates. D-018 remains the historical preparation boundary.

Implement the separately versioned safe-retirement path, unchanged learning
workload, exact 39 ceilings, frozen case/controller order, bounded evidence and
complete independent audit. Preserve PR #70 and subsequent unrelated IK work.
The [implementation contracts](design/LEARN_JOLT_003.md) identify new versions
without changing policy/action semantics or repairing old failed evidence.
Admission requires exact clean executable source, two independent reviews,
passing required checks and defensible complete time/memory/storage feasibility.
If feasibility remains unresolved, publish admission failure before spending
the allocation. No retry, resume, extension, tuning, reseeding or host change is
included. Learned, owner-feel and morphology gates remain separate.

The [completion record](evidence/2026-10-22-learn-jolt-003-completion.md) accepts
the reviewed offline software and publishes the permitted admission refusal.
Complete time and memory feasibility remain unresolved. The requested larger
audit allocation was assessed without silently enlarging the retained whole
ceiling; no executable budget change or physical invocation occurred.

### D-018 — Calculate and review a separately versioned Jolt follow-up

[ACCEPTED] 2026-10-22 by owner: calculate and review a new allocation and create
the next project-step prompt following the consumed #62 experiment and PR #70.

This commission covers allocation arithmetic, independent planning review and a
concrete [LEARN-JOLT-003 task prompt](design/LEARN_JOLT_003_PROMPT.md). The
[allocation proposal](design/LEARN_JOLT_003_ALLOCATION.md) recommends one fresh
7,200-second CPU invocation, unchanged fixed training/comparison quantities,
explicit safe-retirement failure semantics and prospective resource/admission
gates. These are proposal inputs for a later separately commissioned task.

No experiment, benchmark, checkpoint execution, host modification or extension
of LEARN-JOLT-002 is commissioned by this preparation. Final executable-source,
runtime, serialization, memory/storage and budget admission remain required
before any later invocation. Planning reviews explicitly exclude execution
admission. Historical failed results and learned/owner-feel/morphology gates
remain unchanged; #62 remains open.

### D-017 — Complete the separately commissioned fourth Jolt recovery diagnostic

[ACCEPTED] 2026-10-20 by owner: complete [#64 at its JOLT-RECOVERY-004 recommendation](https://github.com/techrote/neural-cyber-automata/issues/64#issuecomment-6300675643).

This fresh commission covers an offline controller/evidence correction, exact
source and declaration review, and one prospectively published matched
diagnostic. The earlier recommendation was proposal-only; the subsequent owner
request commissions this bounded work. Consumed arrival/v1/v2/v3 allocations
and their original failed qualifications remain immutable and separate.

Keep one host, 32 opened cases, three controller families and 96 fresh episodes.
Global maxima are 4,000 decisions, 24,000 requested ticks, 4,500 owner calls and
900 whole-process seconds. Each episode retains 40 decisions/240 ticks, giving
the tighter effective totals of 3,840 decisions/23,040 ticks. Reserve 120 of the
900 seconds for shutdown, durable sealing and independent final verification;
stop admitting new episodes before that reserve is endangered. No retries,
extensions, resumed episodes, outcome-driven candidate search or learning calls
are included. Source/host/profile/task identities, all paired cases and the
original teacher acceptance must pass independent admission before the host starts.

Fix the extra compressed-journal staging/inventory lifecycle with synthetic
durability and strict-inventory checks, and use retained observations for a
versioned ordinary settling/landing correction. Preserve fresh pulse authority,
the two-edge cap and minimal supervisor. An original final-inventory failure
cannot be repaired into teacher qualification by later forensic subchecks.
Only a complete passing teacher result releases #62's teacher dependency;
collection/learning and morphology retain their separate declarations and gates.
The [v4 evidence page](evidence/2026-10-20-jolt-recovery-v4.md) records the offline
source and historical costs; the execution commit/tree and admission are fixed
prospectively before spending this single allocation.

### D-016 — Build a CUDA-trained inverse kinematics layer

[ACCEPTED] 2026-10-20 by owner: open the CUDA IK issue chain, then begin execution.

The [IK programme / #49](https://github.com/techrote/neural-cyber-automata/issues/49)
commissions a fixed articulated-chain point task, optional CUDA training,
independent numerical references, branch-aware datasets, a small learned inverse
model, CPU export and physical qualification of the exact delivered artifact.
The first bounded implementation is the Runner right-sole contract in #50.
Its two rotary joints control point x/y; the observed chassis is a reference
frame, not a solver actuator. Authored play and assistance remain governed by D-011.

Freeze source-bound geometry and numeric acceptance before learning. Numerical
IK, raw learned predictions, bounded correction and full reference fallback
retain separate identities and contribution measures. A numerical or analytic
reference can be the useful result; learned metadata or GPU availability cannot
establish trained value or physical success.

This fixed-chain task is independent of the failed locomotion/morphology gate in
#16. Journal/recovery tasks #43/#48 remain related evidence work, not blanket
entry blockers. CUDA/device qualification, complete run provenance and finite
prospective budgets remain required for actual experiments. No newly purchased
compute or paid service is included. Missing joint-target capabilities require
the separately tracked Cybersand task and host acceptance in #55; this increment
changes no host source or existing ACT003/ACT004 packet semantics.

### D-015 — Recover Jolt and qualify evidence before the next learned skill

[ACCEPTED] 2026-10-20 by owner: start JOLT-RECOVERY-001 and EVIDENCE-001
in parallel with independent source ownership and regular repository checkpoints.

Add a versioned contact/progress recovery controller while preserving both
existing authored references, all historical failures and the configured host.
Keep ordinary movement, posture, launch timing and correction in the controller.
A minimal physical recovery supervisor handles invalid state, lost authority,
expired commands and explicitly declared emergency recovery; it must not become
an implicit authored movement policy for a future learner.

Complete durable diagnostic recording and native-close verification. A newly
declared bounded diagnostic keeps a distinct source, allocation and evidence
identity; it cannot repair or replace the earlier truncated owner journal.
Measure actual impulses, airborne time, recovery, response and chatter together
with arrivals. Use the accepted findings to design LEARN-JOLT-002 with three
independent seeds, fresh held-out geometry/starts, a prospective compute budget
and meaningful learned actuation. Preserve the 75% mean / 50% every-seed
reliability, character, runtime and integrity gates. Morphology transfer stays
blocked until learned qualification and its separate continuation requirements.

### D-014 — Execute the complete current implementation programme

[ACCEPTED] 2026-10-19 by owner: complete all current issues and record progress in the repository every 30 minutes.

This commissions T-101 through T-110, including their bounded CPU implementation,
testing and experiments. The earlier prepass-only wording describes T-100 and
does not prevent this work. Each issue retains its dependency and evidence gates,
one branch and one PR. Conditional experiments use a named behaviour, topology
and budget recorded before execution. No paid model service or newly acquired
compute is included. The existing Cybersand public API can be used from an
adapter owned here; Cybersand source changes retain that project's task process.

Progress checkpoints record source identities, tested results, failures and remaining gates. A negative experimental result is preserved; it is not replaced with an unsupported success claim or an enlarged unrecorded search.

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
