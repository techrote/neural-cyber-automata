---
title: Scope and programme migration
kind: history
summary: How the NCA-first plan becomes a Cybersand behaviour programme, including the owner's gameplay-emulation clarification.
owns: [scope.migration]
verified: 2026-10-18
---

# Scope and programme migration

## Owner direction

The owner requested a research/design prepass for [ACT-004 / Cybersand #178](https://github.com/techrote/cybersand/issues/178), with human, decision-model and frontier-LLM command sources, and authorized completely reorienting this repository toward training and testing intelligent behaviours for Cybersand.

The owner then clarified the fidelity target: plausible game behaviour through emulation of physical reality. Finite gyro momentum accounting and perfect mechanical joint representations are not requirements. Deliberate flex, play and imprecision are welcome when they reduce twitchiness and glitches. D-010 and D-011 record those two instructions.

This is a research and design commission. It does not itself launch training or implement ACT-004.

## Original work retained

The original programme remains at [bootstrap revision 20d0634](https://github.com/techrote/neural-cyber-automata/tree/20d063423fda1e037f328fb32c40c00c73b121a5), including its [roadmap](https://github.com/techrote/neural-cyber-automata/blob/20d063423fda1e037f328fb32c40c00c73b121a5/docs/ROADMAP.md) and [decisions](https://github.com/techrote/neural-cyber-automata/blob/20d063423fda1e037f328fb32c40c00c73b121a5/docs/DECISIONS.md). The replacement documentation branch descends from it. Neither its code tree nor the audited default branch contains a trainer, checkpoint or learning runtime.

Several old issues, including #3 and #9, describe prerequisites as “merged” although those referenced tasks were open and no implementation was present. Historical wording is not evidence of completed work.

## Migration choices

| Former scope | Current treatment | Reason |
|---|---|---|
| Generic NCA first; host bridge only at M4 | Cybersand-specific contract and training research immediately | Current owner direction |
| T-001 package/registries | T-101 | Include actor, sensor, actuator, policy, skill, trainer and evaluator adapters |
| T-002 environments/K80 | T-101 and T-109; legacy lane optional | A usable actor-control loop need not wait for old GPU qualification |
| T-003 run records | T-101 and T-104 | Preserve lineage; immutable records avoid concurrent ledger writes |
| T-004/T-008/T-013/T-017 debt sweeps | Review gates in every new task | Do not create stages that imply unbuilt work is complete |
| T-005 Growing NCA / T-006 viewer | Deferred comparator in T-110 | Pattern regrowth is background, not an actor-control prerequisite |
| T-007 device benchmark | Pipeline measurements in T-109 | Game stepping, transport and inference can dominate a small network |
| T-009–T-012 nanosand/swarm/ES | Conditional cellular track T-110 | Require a named useful Cybersand behaviour and report transfer limits |
| T-014–T-016 integer runtime | Cellular option; actor export assessed in T-109 | Integer network parity does not establish whole-world determinism |
| T-018 event fields | Retained in T-110 | Preserve distributed pseudo-intelligence ideas |

Old IDs stay defined as deferred historical scope. New tasks have new IDs. No old experiment is marked completed merely because its replacement was designed.

## Host relationship

Cybersand owns the gameplay plant: Rapier bodies/contact integration, native materials, effects, usables and authored actuator/constraint assists. This repository owns training orchestration, comparisons, datasets, evaluation and policy artifacts. Its adapters consume a documented, versioned host interface.

“Train in the actual environment” means the configured game mechanics, including deliberate emulation, compliance and assistance. It does not require a mechanically exact replica of a real machine.

Cybersand remains read-only during this prepass. A missing host interface gets a bounded host-side follow-up; a toy or sampled environment cannot be relabelled as the ACT-004 game implementation.

## NCA remains a research option

Preserve three distinct mechanisms: recurrent modules per rigid-body part; actor-graph updates through permitted peer messages; and spatial cellular updates over a separate coarse field or active material. They have different memory, topology, communication cost and objectives. Each is eligible when it helps a specific Cybersand behaviour.
