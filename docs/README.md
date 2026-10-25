---
title: Documentation map
kind: navigation
summary: Where to find current state, the ACT-004 research design and its execution programme.
owns: [docs.map]
verified: 2026-10-22
---

# Documentation map

Flight-learning #74: [Trainer preparation](FLIGHT_LEARNING.md), [prospective preregistration / final custody](FLIGHT_PREREGISTRATION.md), and [experiment-admission package](evidence/2026-10-25-4ly004-admission-package.md).

| Question | Owning page |
|---|---|
| What exists now? | [STATE.md](STATE.md) |
| How do we preflight and dispatch the *actual* Jolt-003 training run? | [Manual admitted long-run workflow and single-use claim](LEARN_JOLT_003_RUNBOOK.md#github-actions-long-run-dispatch) |
| What changed in the mission and old tasks? | [SCOPE_AND_MIGRATION.md](SCOPE_AND_MIGRATION.md) |
| What is the main recommendation? | [Research prepass](research/PREPASS.md) |
| Which physical and neural controllers should we compare? | [Control methods](research/CONTROL_METHODS.md) |
| How much flex, play and assistance should the mechanics allow? | [Gameplay control profile](design/GAMEPLAY_CONTROL_PROFILE.md) |
| How are modules and host boundaries divided? | [ARCHITECTURE.md](ARCHITECTURE.md) |
| How do humans, decision models and LLMs share control? | [Command and skill contract](design/COMMAND_AND_SKILL_CONTRACT.md) |
| How are observations, stepping, learning and evaluations defined? | [Training and evaluation](design/TRAINING_AND_EVALUATION.md) |
| What can be implemented next and with what gates? | [ROADMAP.md](ROADMAP.md) |
| What is accepted, proposed or superseded? | [DECISIONS.md](DECISIONS.md) |
| What must be preserved? | [CONSTRAINTS.md](CONSTRAINTS.md) |
| What exact source was inspected? | [Source prepass](evidence/2026-10-18-source-prepass.md) |
| What questions distinguish the candidates? | [Research questions](research/QUESTIONS.md) |
| Which primary sources support the design? | [Sources](research/SOURCES.md) |
| What remains optional? | [IDEAS.md](IDEAS.md) |
| What compute is supported or measured? | [HARDWARE.md](HARDWARE.md) |
| How is the first CUDA IK task defined and qualified? | [IK contracts](IK_LAYER.md), [acceptance](IK_ACCEPTANCE.md) and [host handoff](IK_HOST_HANDOFF.md) |
| How do independent FK, Jacobians and bounded inverse references work? | [IK kinematics](IK_KINEMATICS.md) and [reference evidence](evidence/2026-10-21-ik-kinematics.md) |
| How are geometric IK data, guarded splits and final-suite access handled? | [IK datasets](IK_DATASETS.md) and [configuration recipes](../configs/ik/datasets/README.md); [declared corpus and evidence](evidence/2026-10-22-ik-datasets.md) |
| How do I install, declare and resume the optional IK device workload? | [IK runtime operations](IK_RUNTIME.md), [software evidence](evidence/2026-10-21-ik-runtime-software.md) and [Windows V100 qualification with matched CPU costs](evidence/2026-10-22-ik-cuda-qualification.md) |
| Which runs occurred? | [RUNS.md](RUNS.md) |
| How do I install and run the CPU package? | [Package operations](PACKAGE.md) |
| How do I launch the pinned physical or sampled game host? | [Host adapter](HOST_ADAPTER.md) |
| How do I use authored skills and repeat matched response fixtures? | [Controllers](CONTROLLERS.md) |
| What is the separate arrival-aware Jolt teacher and how is it measured? | [Jolt arrival reference](evidence/2026-10-20-jolt-arrival-reference.md) |
| What happened in the contact-recovery diagnostics? | [V4 teacher qualification and remaining limits](evidence/2026-10-20-jolt-recovery-v4.md), [v3 improvement with failed acceptance](evidence/2026-10-20-jolt-recovery-v3.md), [complete v2 negative result](evidence/2026-10-20-jolt-recovery-v2.md), [preserved v1 failure](evidence/2026-10-20-jolt-recovery-v1.md) |
| What must the next learned Jolt experiment prove? | [LEARN-JOLT-002 design](design/LEARN_JOLT_002.md), [implementation checklist](design/LEARN_JOLT_002_IMPLEMENTATION.md) |
| How are the separate learned Jolt contracts, static split and single bounded invocation operated? | [LEARN-JOLT-002 runbook](LEARN_JOLT_002_RUNBOOK.md) |
| What happened to the single learned Jolt allocation? | [Consumed failure, preserved evidence and nonqualification](evidence/2026-10-21-jolt-learn-002.md) |
| What recording and execution repairs followed that failure? | [Versioned offline reliability repair and remaining experiment gate](evidence/2026-10-22-jolt-learning-reliability.md) |
| What allocation and task are proposed for the next learned Jolt step? | [LEARN-JOLT-003 budget and review](design/LEARN_JOLT_003_ALLOCATION.md), [ready-to-issue task prompt](design/LEARN_JOLT_003_PROMPT.md) |
| How does the separately commissioned Jolt follow-up preserve safe failures and admit one bounded comparison? | [LEARN-JOLT-003 contracts](design/LEARN_JOLT_003.md) and [operations](LEARN_JOLT_003_RUNBOOK.md) |
| What completed LEARN-JOLT-003 software review, and why is its experiment not admitted? | [Offline completion, independent reviews and complete pipeline refusal](evidence/2026-10-22-learn-jolt-003-completion.md); [original handover](evidence/2026-10-22-learn-jolt-003-handover.md) |
| What measured Jolt-003 memory, and which two independent admission reviews remain unresolved? | [Native/synthetic memory qualification prepass and unsigned source-bound review packets](evidence/2026-10-22-jolt003-memory-admission-prepass.md) |
| Why can Jolt-003 use different CPUs across GitHub runners, and how are timings compared? | [D-022 CPU hardware portability](decisions/2026-10-23-jolt-cpu-portability.md) and [measured Intel/EPYC calibration comparison](evidence/2026-10-23-jolt-cpu-portability-checkpoint.md) |
| What stops an over-memory Jolt experiment and who admits its resources? | [D-023 risk policy](decisions/2026-10-23-jolt-resource-risk.md), [single-use runbook](LEARN_JOLT_003_RUNBOOK.md#process-tree-rss-watchdog-and-v2-resource-review-d-023) and [resource assessment](evidence/2026-10-23-jolt003-resource-assessment-review.md) |
| What supersedes the Jolt wall-clock limits, and what is the CI handoff? | [D-020](DECISIONS.md#d-020--remove-operational-wall-clock-limits-and-rerun-ci); [timing revision and CI handoff](progress/2026-10-22-learning-time-limits.md) |
| How do I capture demonstrations and replay consumed input? | [Dataset guide](../configs/datasets/README.md) |
| How do I compose usable actions and cancel ownership? | [Usable skills](USABLE_SKILLS.md) |
| How do I run the bounded recurrent actor-learning screen? | [Learning operations](LEARNING.md) |
| What is prepared for morphology comparison and why is training gated? | [Morphology evidence](evidence/2026-10-19-t106.md) |
| What did the bounded Jolt direction-frame prerequisite find? | [Direction-frame result](evidence/2026-10-19-t106-direction-frame.md) |
| What conditional morphology code exists behind the failed prerequisite? | [Guarded comparison preparation](evidence/2026-10-19-t106-learning.md) |
| How do asynchronous intent and human takeover keep local control running? | [Asynchronous control](ASYNC_CONTROL.md) |
| How do I convert and qualify an evaluated CPU policy? | [Runtime export](RUNTIME_EXPORT.md) |
| Which exact Linux Torch CPU bundle passed qualification, and with what limits? | [Torch CPU result](evidence/2026-10-20-t109-torch-cpu.md) |
| What did the bounded per-part Jolt graph experiment find? | [Cellular experiment](evidence/2026-10-19-t110-screen.md) |
| What happened during programme execution? | [Progress checkpoints](PROGRESS.md) |
| What traps matter? | [GOTCHAS.md](GOTCHAS.md) |
| What do the terms mean? | [GLOSSARY.md](GLOSSARY.md) |

Research/design pages describe candidates. The source audit is pinned evidence. Implementation tasks are not completed merely because their prompts exist.
