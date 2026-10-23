---
title: Frozen TITAN IK evaluation recovery and no-go result
kind: evidence
summary: Same-candidate evaluation completed inside original allocation after paid recovery and fresh admission; accuracy and utility fail.
owns: [evidence.ik_frozen_evaluation_recovery]
verified: 2026-10-23
---

# Frozen TITAN IK evaluation recovery

The owner-authorized recovery completed evaluation of all three original frozen
candidates. No training, candidate reselection, normalization change, budget
reset or new allocation occurred. The measured result is **no-go for accuracy
and practical utility**; learned contribution passes. No export or physical
qualification follows. PR #78 remains draft and issue #54 remains open.

## Source and admission

Executable commit `d3dbbe2d1c559f3c4ce636383d01d702d9956013` has 149-file closure
SHA-256 `4a774d8ef247246ecf6c7d647e0032ca2f563410d9c956629667f8f486c12587`.
All six hosted checks passed on that exact head. Local IK discovery passed
406 tests with two existing skips before the final byte-copy regression;
the final rebind, overhead and recovery fixtures passed separately. Strict
documentation checks passed. A full faithful-copy rebind/load rehearsal verified
that all real files stayed unchanged before actual preparation.

The archive-layout fix selects the authenticated top-level corpus prefix.
The later CLI fix preserves canonical proof type identity and the earliest
process timestamp. [Historical failures](../progress/2026-10-23-ik-recovery-refusal.md),
opening records and all original paid receipts remain immutable. Original
corpus archive SHA-256 remains
`97e3c547c7056e6ef727dc07437f469fde54b545bb091503a791aed85a4f99ce`.
The same frozen set remains
`e9e1ea0e1bb4831b4323a78fef14a9a1012c97c12a61eae8833f5933c0161b85`.

The [actual rebound declaration](https://github.com/techrote/neural-cyber-automata/pull/78#issuecomment-6361987031)
has file SHA-256 `245c4ba3325345e2b5efcbd58b4347262bfaa700eee8e76f764199c5d7c45960`
and semantic SHA-256 `3a3ec1fa6b02e2ac4aed656111954d047ea5acef7ff5e8cb96e849820a78108d`.
Two fresh non-author agents independently reviewed source, complete histories,
real paid ledgers, partition isolation, scoring and resource feasibility:
[admission A](https://github.com/techrote/neural-cyber-automata/pull/78#issuecomment-6362066059)
and [admission B](https://github.com/techrote/neural-cyber-automata/pull/78#issuecomment-6362039084).
Their aggregate semantic SHA-256 is
`062b6a48c8ddbbadcc75072fb616513f5ba7ae57d2b0b818984b7d7c49332ff6`.
Implementation contributors did not count as these fresh admissions.

## Accounting and runtime

Append-only records retain the original historical failed-process charges.
Each seed additionally pays ten seconds for the observed 5.962-second launcher
refusal, measured recording work and a sixty-second publication reserve:
72.984 / 72.984 / 72.985 seconds. No worker attempt slot was consumed by that
pre-supervisor refusal. Later evaluation attempts inherit this wall debit and
parent chain. Original failed markers are preserved and only their exact,
paid historical identities are reconciled. Unknown future work is not excused.

Evaluation used the same isolated NVIDIA TITAN RTX at `cuda:0`, capability 7.5,
72 SMs and 25,769,476,096 bytes reported capacity, Windows 11 AMD64,
Python 3.12.10, Torch 2.8.0+cu126, NumPy 2.3.5 and driver 581.80.
The V100 was not initialized. Scoring and matched batch-one timing run on CPU
with one Torch thread; CUDA restores the admitted continuation context.
No optimizer update occurs during evaluation. Peak allocator reserve was
2,097,152 bytes for each restored CUDA context, below the 2 GiB ceiling;
this excludes driver/context/external memory.

The supervised worker returned zero after 441.407 complete process seconds.
Publication was verified inside the conservative per-seed charges below.
All ledgers are completed, with no pending reservations or new unknown-work
holds. Remaining time is not authorization to rerun this completed experiment.

| Seed | Attempts total | Evaluation wall charge, s | Cumulative charged wall, s | Remaining original wall, s |
|---|---:|---:|---:|---:|
| 41011 | 4 | 479.485 | 1454.891 | 5745.109 |
| 42013 | 3 | 486.672 | 1447.875 | 5752.125 |
| 43019 | 3 | 493.719 | 1529.314 | 5670.686 |

Each seed retains 1,720 optimizer steps, 210,400 presentations and 80,800 tuning
calls. Recovery adds exactly 65,200 evaluation/timing charges and one checkpoint
per seed, giving 43 / 42 / 42 total checkpoint charges. All ceilings remain
original: four attempts, 45 checkpoints and 7,200 seconds per seed.

## Complete comparison and fixed verdict

Every seed reports five comparators over all 2,940 scheduled final tasks:
2,800 nominal tasks, including 280 withheld-region tasks, and 140 stress tasks.
All scheduled rows and refusals remain in their denominators. Timing retains
100 warmups and 10,000 measured calls per comparator, all 150,000 raw measured
latencies across three seeds, with no rerun or selected faster repeat.

| Seed | Learned nominal p95, cells | Nominal feasible success | Withheld feasible success | Learned CPU p95, ms | Numerical CPU p95, ms |
|---|---:|---:|---:|---:|---:|
| 41011 | 0.216149 | 15.071% | 0.357% | 0.8438 | 1.3149 |
| 42013 | 0.176660 | 52.250% | 32.143% | 0.6647 | 0.9028 |
| 43019 | 0.209830 | 7.286% | 0.357% | 0.8318 | 1.2787 |

Pooled learned nominal p50/p95/worst errors are 0.095182 / 0.201796 /
0.461551 cells, with 2,089/8,400 feasible successes (24.869%). Both independent
analytic and numerical references achieve 8,400/8,400 nominal feasible successes.
The learned-disabled current-reference comparator achieves 2,247/8,400 and the
matched untrained initialization 39/8,400. Contribution passes relative to the
untrained initialization; it is not sufficient to promote the learned model.

All 760 scheduled eligible continuity pairs per seed are complete, with zero
admitted branch switches or changes above 0.25 radians. Nominal outputs contain
no nonfinite, malformed, command-bound or hard-bound violations. Direct mode
uses zero numerical corrections and zero reference fallbacks.

The learned CPU p95 is below numerical p95, but the unchanged practical-utility
gate also requires accepted quality. That prerequisite fails. Every seed and
the aggregate therefore report accuracy no-go, contribution pass and utility
no-go. No timing speed ratio overrides the geometric acceptance failure.

## Terminal artifacts and evidence limits

Final continuation checkpoint SHA-256 identities are:

- 41011: `e80ba839e5e755aec02a663a6c84ae6c6c093df274cca6ede68e105015cf825b`.
- 42013: `85b533336223adf8356f38e41aebef79e7b4c801129d188e83013cc332e3bc7b`.
- 43019: `6f5f2e64748472f0329ded25171f53344ac2c65ab75981579d37ec15cd258503`.

[Independent terminal audit A](https://github.com/techrote/neural-cyber-automata/pull/78#issuecomment-6362338573),
[audit B](https://github.com/techrote/neural-cyber-automata/pull/78#issuecomment-6362319542)
and the [result index](ik-frozen-recovery-20261023.json)
bind original declarations, source, same selected candidates, retained failures,
ledger chains, checkpoint tensors, all comparator denominators and timing
samples. Large comparison reports, checkpoint bytes, declarations and audits
remain outside Git in maintainer-held bundle `ik-rebound-execution-d3dbbe2`,
retrievable by the index hashes. The final suite remains opened/development-only;
no further training or tuning on these results is authorized. This is a complete
nominal software evaluation with a negative result, not CPU export or physical
host qualification.


Result index SHA-256: `b0a4ba45db8babab83bf3ef6790234d618d4fee42b5e61b8ebb320f068aa098a`.

© 2026 techrote. All rights reserved.
