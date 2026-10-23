---
title: Cybersand behaviour implementation programme
kind: roadmap
summary: Bounded tasks, dependencies and acceptance gates following the ACT-004 research prepass.
owns: [roadmap.order, roadmap.tasks]
verified: 2026-10-22
---

# Implementation programme

Prepared by T-100. D-014 commissions this issue-based execution programme. Assignment authorizes bounded work subject to actual dependencies; [current state](STATE.md), [run records](RUNS.md) and each PR establish what has been implemented and measured.

D-010 defines the Cybersand mission. D-011 sets plausible gameplay emulation with deliberate flex/play. D-012/D-013 contain research recommendations whose value is established through the experiments below.

## First useful milestone

A compatible game profile, a persistent adapter, a convincing authored baseline and one small learned skill can be compared on unseen scenarios with recorded feel, response, success and cost.

T-101 can start from this repository immediately after the prepass lands. T-102 can prepare recorded/sampled transport before ACT-004 is accepted, but its physical-game acceptance needs the actual host. Do not wait for all 29 rigs or a new NCA engine to test one available rig.

## Active task definitions

Nine of ten programme tasks have accepted results. Morphology transfer in
[#16](https://github.com/techrote/neural-cyber-automata/issues/16) is the remaining
programme task; its fixed reliability prerequisite and continuation gates still apply.

| Task | Status | Scope | Depends on | Issue |
|---|---|---|---|---|
| T-101 | Merged [PR #22](https://github.com/techrote/neural-cyber-automata/pull/22) | Package, gameplay-profile contracts and reproducible run records | T-100 | [#11](https://github.com/techrote/neural-cyber-automata/issues/11) |
| T-102 | Merged [PR #23](https://github.com/techrote/neural-cyber-automata/pull/23) | Persistent Cybersand host adapter and causal stepping conformance | T-101 | [#12](https://github.com/techrote/neural-cyber-automata/issues/12) |
| T-103 | Merged [PR #24](https://github.com/techrote/neural-cyber-automata/pull/24) | Authored feedback baselines, compliant response and actor feel fixtures | T-102 | [#13](https://github.com/techrote/neural-cyber-automata/issues/13) |
| T-104 | Merged [PR #25](https://github.com/techrote/neural-cyber-automata/pull/25) | Demonstration capture, consumed-action replay and intervention datasets | T-101, T-102 | [#14](https://github.com/techrote/neural-cyber-automata/issues/14) |
| T-105 | Merged [PR #26](https://github.com/techrote/neural-cyber-automata/pull/26) | First learned actor skill: imitation and recurrent residual control | T-102, T-103, T-104 | [#15](https://github.com/techrote/neural-cyber-automata/issues/15) |
| T-106 | Preparation merged in [PR #30](https://github.com/techrote/neural-cyber-automata/pull/30), primary provenance/recovery in [PR #36](https://github.com/techrote/neural-cyber-automata/pull/36), and guarded preparation in [PR #38](https://github.com/techrote/neural-cyber-automata/pull/38); transfer blocked | Morphology sharing, articulation and recovery generalization | T-105 | [#16](https://github.com/techrote/neural-cyber-automata/issues/16) |
| T-107 | Merged [PR #27](https://github.com/techrote/neural-cyber-automata/pull/27) | Usable skills and feedback-driven task composition | T-103, T-104 | [#17](https://github.com/techrote/neural-cyber-automata/issues/17) |
| T-108 | Merged [PR #31](https://github.com/techrote/neural-cyber-automata/pull/31); native-lease conformance corrected in [PR #40](https://github.com/techrote/neural-cyber-automata/pull/40) | Human takeover and asynchronous decision-model or LLM adapter | T-102, T-107 | [#18](https://github.com/techrote/neural-cyber-automata/issues/18) |
| T-109 | Export implementation/no-go in [PR #29](https://github.com/techrote/neural-cyber-automata/pull/29); exact Linux Torch CPU bundle qualified by separate composite evidence in [PR #41](https://github.com/techrote/neural-cyber-automata/pull/41); original no-go runs preserved | Runtime export, end-to-end throughput and deployment qualification | T-105 | [#19](https://github.com/techrote/neural-cyber-automata/issues/19) |
| T-110 | Bounded no-go merged in [PR #28](https://github.com/techrote/neural-cyber-automata/pull/28); cancellation and repeated-run accounting corrected in [PR #35](https://github.com/techrote/neural-cyber-automata/pull/35) | Conditional distributed NCA and coordination experiments for Cybersand | T-101, T-102, T-104 | [#20](https://github.com/techrote/neural-cyber-automata/issues/20) |

Each issue contains an execution prompt, read-set, write-set, checks, limits and budget. No task may claim trained/physical/live-model evidence from a schema, mock or metadata flag.

The current prerequisite path is contact-aware [Jolt recovery #48](https://github.com/techrote/neural-cyber-automata/issues/48)
and [complete evidence #43](https://github.com/techrote/neural-cyber-automata/issues/43),
then [LEARN-JOLT-002 / #62](design/LEARN_JOLT_002.md), then T-106. Evidence software
is accepted in PR #58. The [v1 incomplete attempt](evidence/2026-10-20-jolt-recovery-v1.md),
[v2 negative teacher result](evidence/2026-10-20-jolt-recovery-v2.md) and
[v3 failed qualification](evidence/2026-10-20-jolt-recovery-v3.md) remain preserved.

The separately commissioned [v4 result](evidence/2026-10-20-jolt-recovery-v4.md),
published through [PR #66](https://github.com/techrote/neural-cyber-automata/pull/66)
for [#64](https://github.com/techrote/neural-cyber-automata/issues/64), satisfies
all thirteen teacher checks with complete evidence: 30/32 arrivals versus 27/32,
three historical rescues, zero success regressions and 32/32 purposeful-airborne
cases. This releases #62's teacher dependency. Two leftward failures remain,
including worse final distance for `reference-floor-left-63`; consistent useful
second-coast landing progress and owner feel are not established. All 32 cases
remain opened development evidence. The teacher result supplies no learned
qualification.

The [learning implementation](LEARN_JOLT_002_RUNBOOK.md) and its source reviews
admitted one declared allocation at clean source
`a68e687ced52bef778375a2753b1362dea3704fa`, with the 182-file closure retained in
the [failed-allocation evidence](evidence/2026-10-21-jolt-learn-002.md).
That allocation was consumed: it retained five of six final checkpoints and
stopped before any open or sealed evaluation. The original journal contains
6,720 records against the worker's claimed 24,195; complete physical evidence
and terminal execution receipts are missing. Paired learned reliability remains
unmeasured. #62 stays open and blocked, and T-106 remains blocked.

The [implementation checklist](design/LEARN_JOLT_002_IMPLEMENTATION.md) preserves
the historical admission requirements; its boxes are not a current execution
status. Publication and independent readback preserve the failed result and do
not reopen its allocation. The learned 75% mean / 50% every-seed reliability gate
and all separate morphology gates remain unchanged. Any further physical
experiment requires a separately commissioned prospective allocation; no retry,
resume, retuning or extension of this one is admitted.

The [offline execution v2 repair](evidence/2026-10-22-jolt-learning-reliability.md)
adds checked recording, independent cleanup and exact lifecycle/output binding.
Its source reviews explicitly exclude prospective experiment admission. The
48-tick emergency stop remains allocation-aborting, and complete runtime
feasibility remains unmeasured. This increment supplies no paired learned result.

The separately commissioned [LEARN-JOLT-003 offline completion](evidence/2026-10-22-learn-jolt-003-completion.md)
closes reviewed v3 inventory and terminal-learning defects. Its full pipeline
assessment records the historical finite-time refusal. The later
[D-020 revision](DECISIONS.md#d-020--remove-operational-wall-clock-limits-and-rerun-ci)
removes operational time limits and requests CI followed by a stop. Full-worker/native
memory and final-source admission remain unresolved; the allocation remains
unspent. It supplies no paired learned result, qualified family or morphology
release; #62 and T-106 retain the existing requirements.

## Dependency gates

**T-101:** Can start after the prepass merge; no ACT-004 implementation dependency for the package and recorded fixtures.

**T-102:** The compatible ACT-004 host was accepted through Cybersand PR #181 at the pinned revision in [current state](STATE.md). Qualification uses that actual runtime.

**T-103:** Needs one accepted ACT-004 profile and actual game adapter; host mechanics changes use a separate Cybersand task.

**T-104:** Recorded/sampled preparation can run earlier; any ACT-004 dataset claim requires the physical game lane.

**T-105:** Requires actual game adapter, stable baseline/profile, datasets and frozen evaluation criteria. Run only when this implementation/experiment task is commissioned.

**T-106:** Software preparation and read-only admission are implemented. Physical teacher qualification, learning and transfer comparison require the designated per-rig prerequisite to pass the fixed **75% mean / 50% every-seed success** thresholds together with the original gain, character and runtime gates, and require accepted additional actor profiles. The current designated primary fails. A passing separately commissioned prerequisite, a distinct continuation declaration and a qualified semantic waypoint reference are required before morphology learning.

**T-107:** Requires accepted ACT-004 mount/usable capabilities. An authored skill implementation can precede learned skill replacement.

**T-108:** Deterministic mock providers first. A live provider requires an explicit chosen service and bounded call/spend budget in the commissioned task.

**T-109:** The exact evaluated Linux Torch CPU bundle has [composite qualification](evidence/2026-10-20-t109-torch-cpu.md), with its interrupted training lineage and original no-go runs retained. Additional platforms, accelerators, multiple learned actors and continuous realtime pacing remain separately qualified lanes.

**T-110:** Off the first actor-control critical path. Commission one named behaviour and topology before implementation; no open-ended cellular engine programme.


The host adapter uses the accepted Cybersand mechanics, including deliberate assistance and compliant connections. The host remains separately responsible for actor bodies, native materials and usables. Missing host facilities become bounded Cybersand follow-ups, not a silent change to the research environment.

## What can run in parallel

After T-101, transport/conformance and dataset-format preparation can proceed independently, provided full T-104 completion still uses an accepted T-102 adapter. After T-102, baseline tuning (T-103) and capture/replay (T-104) have separate primary write-sets.

After the baseline and dataset, one learned skill (T-105) and authored usable composition (T-107) can proceed separately. T-108 does not require the low-level policy to be neural: the same skill interface can use authored control. Morphology software preparation and read-only admission (T-106) can proceed independently of qualification of a separately evaluated export candidate (T-109). Physical morphology work retains the fixed 75% mean / 50% every-seed prerequisite and its separate continuation gates.

T-110 is a conditional side track with one named behaviour/topology at a time. It does not block the actor-control milestone.

## Experiment and acceptance gates

Before a learning run, freeze the profile, observation/action semantics, scenario split, reward, comparison metrics and budget. Begin with the [training specification](design/TRAINING_AND_EVALUATION.md) ceilings: three seeds and two learned configurations, at most 200,000 decisions or two hours per seed initially. These are caps, not a prediction of sufficient learning.

Routine CI checks contracts, reset, sample handling and small fixtures. Long training and live model calls use explicit experiment tasks. Preserve failed and capped runs. A no-go or inconclusive result can satisfy a research issue if it answers the scoped question honestly and leaves reproducible evidence.

Promotion requires matched capabilities and declared information/timing, no critical lifecycle failures, useful task performance, measured runtime cost and separately recorded owner feel judgement. Do not demand perfect joints, a gyro momentum ledger or exact energy conservation.

## CUDA IK programme

D-016 commissions [IK-CUDA-000 / #49](https://github.com/techrote/neural-cyber-automata/issues/49).
The first pilot targets Runner's right sole using two rotary coordinates and an
observed root frame. Its [contracts](IK_LAYER.md), [prospective acceptance](IK_ACCEPTANCE.md)
and [separate host requirements](IK_HOST_HANDOFF.md) distinguish nominal geometry,
actual CUDA learning, exact CPU export and physical tracking evidence.

| Issue | Deliverable | Dependency | Evidence state |
|---|---|---|---|
| [#50](https://github.com/techrote/neural-cyber-automata/issues/50) | Registered rig/effector/target/result contracts | Entry | Nominal profile and software fixtures |
| [#51](https://github.com/techrote/neural-cyber-automata/issues/51) | Optional CUDA runtime and resumable run support | #50 | [Software conformance](evidence/2026-10-21-ik-runtime-software.md) and [Windows V100 CUDA/CPU qualification](evidence/2026-10-22-ik-cuda-qualification.md); 32 steps per lane with step-8 resume; Linux CUDA remains unmeasured |
| [#52](https://github.com/techrote/neural-cyber-automata/issues/52) | Batched FK and independent numerical IK | #50 | [Independent geometry/reference implementation and conformance](evidence/2026-10-21-ik-kinematics.md); actual CUDA parity unmeasured |
| [#53](https://github.com/techrote/neural-cyber-automata/issues/53) | Branch-aware data/curriculum and sealed family splits | #52 | [Dataset implementation](IK_DATASETS.md) and [complete 10,220-task CPU corpus](evidence/2026-10-22-ik-datasets.md); final candidate scoring remains sealed |
| [#54](https://github.com/techrote/neural-cyber-automata/issues/54) | Three-seed compact CUDA IK training | #51, #53 | [Preparation and harness](IK_TRAINING.md), with [one independently admitted/verified TITAN batch](evidence/2026-10-23-ik-titan-first-batch.md); full training and three-seed comparison remain unexecuted |
| [#55](https://github.com/techrote/neural-cyber-automata/issues/55) | Host joint-target handoff and NCA adapter | #50, accepted separate Cybersand capability | Specification only |
| [#56](https://github.com/techrote/neural-cyber-automata/issues/56) | Exact CPU runtime export | #54 | Unimplemented |
| [#57](https://github.com/techrote/neural-cyber-automata/issues/57) | Physical qualification and runbook for the exported bundle | #55, #56 | Unmeasured |

After #50, CUDA infrastructure, reference geometry and host-interface preparation
can proceed independently. Export precedes final physics scoring so the scored
artifact is the delivered artifact. A host-capability gap does not prevent
nominal learning preparation; it prevents claims of actual tracking.

The independent fixed-chain task does not relax or inherit #16's failed
locomotion/morphology prerequisite. Journal #43 and recovery #48 are related
work, not unconditional blockers. Each issue retains a distinct branch/PR,
independent review, exact-head checks and bounded experiment declaration.

## Preserved legacy programme

All following tasks are deferred historical scopes. Their original IDs are retained and their original descriptions remain at [bootstrap 20d0634](https://github.com/techrote/neural-cyber-automata/blob/20d063423fda1e037f328fb32c40c00c73b121a5/docs/ROADMAP.md). Replacements do not mark the old work completed.

| Task | Status | Original scope | Current destination | Original issue |
|---|---|---|---|---|
| T-001 | [DEFERRED] | Package skeleton | T-101 | [#2](https://github.com/techrote/neural-cyber-automata/issues/2) |
| T-002 | [DEFERRED] | Modern/legacy environments | T-101 / T-109 | [#3](https://github.com/techrote/neural-cyber-automata/issues/3) |
| T-003 | [DEFERRED] | Run records | T-101 / T-104 | [#4](https://github.com/techrote/neural-cyber-automata/issues/4) |
| T-004 | [DEFERRED] | M0 audit | Per-task evidence/review | [#5](https://github.com/techrote/neural-cyber-automata/issues/5) |
| T-005 | [DEFERRED] | Growing NCA baseline | T-110 | [#6](https://github.com/techrote/neural-cyber-automata/issues/6) |
| T-006 | [DEFERRED] | NCA damage viewer | T-110 | [#7](https://github.com/techrote/neural-cyber-automata/issues/7) |
| T-007 | [DEFERRED] | Device benchmark | T-109 | [#8](https://github.com/techrote/neural-cyber-automata/issues/8) |
| T-008 | [DEFERRED] | M1 audit | Per-task evidence/review | [#9](https://github.com/techrote/neural-cyber-automata/issues/9) |
| T-009 | [DEFERRED] | Nanosand toy world | T-110 | — |
| T-010 | [DEFERRED] | Signals and nanite network | T-110 | — |
| T-011 | [DEFERRED] | Behaviour suite | T-110 | — |
| T-012 | [DEFERRED] | Evolution-strategies swarm | T-110 | — |
| T-013 | [DEFERRED] | M2 audit | Per-task evidence/review | — |
| T-014 | [DEFERRED] | Quantisation | T-109 / T-110 | — |
| T-015 | [DEFERRED] | Integer runtime | T-109 / T-110 | — |
| T-016 | [DEFERRED] | Cellular CPU benchmark | T-109 / T-110 | — |
| T-017 | [DEFERRED] | M3 audit | Per-task evidence/review | — |
| T-018 | [DEFERRED] | World-event signal response | T-110 | — |

See [scope migration](SCOPE_AND_MIGRATION.md) for the rationale. The original NCA methods remain available; their old milestone order no longer controls actor research.
