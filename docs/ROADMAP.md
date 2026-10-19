---
title: Cybersand behaviour implementation programme
kind: roadmap
summary: Bounded tasks, dependencies and acceptance gates following the ACT-004 research prepass.
owns: [roadmap.order, roadmap.tasks]
verified: 2026-10-19
---

# Implementation programme

Prepared by T-100. D-014 commissions this issue-based execution programme. Assignment authorizes bounded work subject to actual dependencies; [current state](STATE.md), [run records](RUNS.md) and each PR establish what has been implemented and measured.

D-010 defines the Cybersand mission. D-011 sets plausible gameplay emulation with deliberate flex/play. D-012/D-013 contain research recommendations whose value is established through the experiments below.

## First useful milestone

A compatible game profile, a persistent adapter, a convincing authored baseline and one small learned skill can be compared on unseen scenarios with recorded feel, response, success and cost.

T-101 can start from this repository immediately after the prepass lands. T-102 can prepare recorded/sampled transport before ACT-004 is accepted, but its physical-game acceptance needs the actual host. Do not wait for all 29 rigs or a new NCA engine to test one available rig.

## Active task definitions

| Task | Status | Scope | Depends on | Issue |
|---|---|---|---|---|
| T-101 | Merged [PR #22](https://github.com/techrote/neural-cyber-automata/pull/22) | Package, gameplay-profile contracts and reproducible run records | T-100 | [#11](https://github.com/techrote/neural-cyber-automata/issues/11) |
| T-102 | Merged [PR #23](https://github.com/techrote/neural-cyber-automata/pull/23) | Persistent Cybersand host adapter and causal stepping conformance | T-101 | [#12](https://github.com/techrote/neural-cyber-automata/issues/12) |
| T-103 | Merged [PR #24](https://github.com/techrote/neural-cyber-automata/pull/24) | Authored feedback baselines, compliant response and actor feel fixtures | T-102 | [#13](https://github.com/techrote/neural-cyber-automata/issues/13) |
| T-104 | Merged [PR #25](https://github.com/techrote/neural-cyber-automata/pull/25) | Demonstration capture, consumed-action replay and intervention datasets | T-101, T-102 | [#14](https://github.com/techrote/neural-cyber-automata/issues/14) |
| T-105 | Merged [PR #26](https://github.com/techrote/neural-cyber-automata/pull/26) | First learned actor skill: imitation and recurrent residual control | T-102, T-103, T-104 | [#15](https://github.com/techrote/neural-cyber-automata/issues/15) |
| T-106 | [PLANNED T-106] | Morphology sharing, articulation and recovery generalization | T-105 | [#16](https://github.com/techrote/neural-cyber-automata/issues/16) |
| T-107 | Merged [PR #27](https://github.com/techrote/neural-cyber-automata/pull/27) | Usable skills and feedback-driven task composition | T-103, T-104 | [#17](https://github.com/techrote/neural-cyber-automata/issues/17) |
| T-108 | Merged [PR #31](https://github.com/techrote/neural-cyber-automata/pull/31) | Human takeover and asynchronous decision-model or LLM adapter | T-102, T-107 | [#18](https://github.com/techrote/neural-cyber-automata/issues/18) |
| T-109 | [PLANNED T-109] | Runtime export, end-to-end throughput and deployment qualification | T-105 | [#19](https://github.com/techrote/neural-cyber-automata/issues/19) |
| T-110 | [IN-PROGRESS T-110] | Conditional distributed NCA and coordination experiments for Cybersand | T-101, T-102, T-104 | [#20](https://github.com/techrote/neural-cyber-automata/issues/20) |

Each issue contains an execution prompt, read-set, write-set, checks, limits and budget. No task may claim trained/physical/live-model evidence from a schema, mock or metadata flag.

## Dependency gates

**T-101:** Can start after the prepass merge; no ACT-004 implementation dependency for the package and recorded fixtures.

**T-102:** The compatible ACT-004 host was accepted through Cybersand PR #181 at the pinned revision in [current state](STATE.md). Qualification uses that actual runtime.

**T-103:** Needs one accepted ACT-004 profile and actual game adapter; host mechanics changes use a separate Cybersand task.

**T-104:** Recorded/sampled preparation can run earlier; any ACT-004 dataset claim requires the physical game lane.

**T-105:** Requires actual game adapter, stable baseline/profile, datasets and frozen evaluation criteria. Run only when this implementation/experiment task is commissioned.

**T-106:** Starts only after one useful per-rig learned skill and accepted additional actor profiles exist.

**T-107:** Requires accepted ACT-004 mount/usable capabilities. An authored skill implementation can precede learned skill replacement.

**T-108:** Deterministic mock providers first. A live provider requires an explicit chosen service and bounded call/spend budget in the commissioned task.

**T-109:** Requires a trained/evaluated candidate; additional platforms and accelerators are separately qualified lanes.

**T-110:** Off the first actor-control critical path. Commission one named behaviour and topology before implementation; no open-ended cellular engine programme.


The host adapter uses the accepted Cybersand mechanics, including deliberate assistance and compliant connections. The host remains separately responsible for actor bodies, native materials and usables. Missing host facilities become bounded Cybersand follow-ups, not a silent change to the research environment.

## What can run in parallel

After T-101, transport/conformance and dataset-format preparation can proceed independently, provided full T-104 completion still uses an accepted T-102 adapter. After T-102, baseline tuning (T-103) and capture/replay (T-104) have separate primary write-sets.

After the baseline and dataset, one learned skill (T-105) and authored usable composition (T-107) can proceed separately. T-108 does not require the low-level policy to be neural: the same skill interface can use authored control. After a useful learned candidate, morphology work (T-106) and export/performance (T-109) can be independent.

T-110 is a conditional side track with one named behaviour/topology at a time. It does not block the actor-control milestone.

## Experiment and acceptance gates

Before a learning run, freeze the profile, observation/action semantics, scenario split, reward, comparison metrics and budget. Begin with the [training specification](design/TRAINING_AND_EVALUATION.md) ceilings: three seeds and two learned configurations, at most 200,000 decisions or two hours per seed initially. These are caps, not a prediction of sufficient learning.

Routine CI checks contracts, reset, sample handling and small fixtures. Long training and live model calls use explicit experiment tasks. Preserve failed and capped runs. A no-go or inconclusive result can satisfy a research issue if it answers the scoped question honestly and leaves reproducible evidence.

Promotion requires matched capabilities and declared information/timing, no critical lifecycle failures, useful task performance, measured runtime cost and separately recorded owner feel judgement. Do not demand perfect joints, a gyro momentum ledger or exact energy conservation.

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
