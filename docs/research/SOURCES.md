---
title: Sources
kind: reference
summary: External articles, papers and engine documents this project draws on, what is taken from each, and licence notes.
owns: [research.sources]
verified: 2026-10-13
---

# Sources

Ideas and published descriptions may be used freely. Code may only be copied or ported after its licence is recorded here and checked against C-001; so far nothing has been copied.

## Neural cellular automata

| Source | What this project takes from it | Licence notes |
|---|---|---|
| Mordvintsev et al., "Growing Neural Cellular Automata", Distill, 2020 — <https://distill.pub/2020/growing-ca/> | The baseline model and training recipe reproduced in M1: perception filters, alive masking, stochastic updates, sample pool, damage during training | Article text CC-BY; any accompanying code must be checked before reuse |
| Randazzo et al., "Self-classifying MNIST Digits", Distill, 2020 — <https://distill.pub/2020/selforg/mnist/> | Evidence that cells can agree on a shared, global answer through local messages only (relevant to I-002) | As above |
| Niklasson et al., "Self-Organising Textures", Distill, 2021 — <https://distill.pub/selforg/2021/textures/> | Training NCAs towards patterns and textures rather than fixed shapes (relevant to I-008) | As above |
| Randazzo et al., "Adversarial Reprogramming of Neural Cellular Automata", Distill, 2021 — <https://distill.pub/selforg/2021/adversarial/> | How injected signals or rogue cells can steer a whole NCA (relevant to D-005 and to swarm robustness) | As above |

## Training methods

| Source | What this project takes from it | Licence notes |
|---|---|---|
| Salimans et al., "Evolution Strategies as a Scalable Alternative to Reinforcement Learning", 2017 — <https://arxiv.org/abs/1703.03864> | The evolution-strategies method proposed in D-008 for behaviours scored in a non-differentiable world | Paper; no code taken |

## cybersand (read-only reference)

| Document | What this project takes from it |
|---|---|
| [Materials and rule kernels](https://github.com/techrote/cybersand/blob/main/docs/systems/materials-and-rule-kernels.md) | The four-byte cell (material, two generic state bytes, update epoch) that motivates D-003 |
| [Determinism and boundary transfers](https://github.com/techrote/cybersand/blob/main/docs/architecture/determinism-and-boundary-transfers.md) | Phased, coordinate-seeded deterministic updates that motivate D-004 and R-003 |
| [Simulation tick and threading](https://github.com/techrote/cybersand/blob/main/docs/architecture/simulation-tick-and-threading.md) | Fixed 60 Hz tick on desktop; single cellular worker on compatibility Web (I-006) |

These describe cybersand as of 2026-10-13; re-read the live documents before relying on them in M4.
