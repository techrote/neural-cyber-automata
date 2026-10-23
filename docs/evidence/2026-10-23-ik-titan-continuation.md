---
title: TITAN three-seed training and failed final evaluation
kind: evidence
summary: Three independently verified forty-epoch training runs; final archive-layout failure opened the suite and blocked reuse before scoring.
owns: [evidence.ik_titan_continuation]
verified: 2026-10-23
---

# TITAN three-seed training and failed final evaluation

[MEASURED, PARTIAL] #54 / PR #78 completed the declared training for seeds
41011, 42013 and 43019 on the isolated TITAN RTX. Independent actual readback
verified every training ledger, checkpoint and update receipt. Final evaluation
**failed before its first predictor call** because the reader omitted the
archive's top-level directory. The durable final opening already existed.
The suite is opened/development-only; failed supervision and overrun receipts
block reuse. No final scores, CPU timing, learned quality or export exist.

The [machine-readable evidence index](ik-titan-continuation-20261023.json)
retains exact source, declaration, checkpoint, allocation, cost and audit hashes.
Its SHA-256 is `2e93278282e6a5d0fbed163de09d0c87035c18fc687d72bb14b3324daee3bbbf`.
Large original artifacts remain outside Git and are retrievable by those hashes.
The [original single batch](2026-10-23-ik-titan-first-batch.md) remains unchanged
historical evidence in the same allocation.

## Authority and exact sources

D-025 records the owner's explicit continuation and necessary-amendment scope.
The original accepted corpus, seeds, schedule, model, loss, optimizer,
normalization, acceptance thresholds and allocation ceilings did not change.
The V100 remained prohibited and supplied no performance baseline.

Training continuation executed clean commit
`a9c1c837c17d7f042117db22d036b11b8735c236`, source closure
`4fb2e4b541f65bda18de9489503dc8ab095c97ed3c73b6c1fbff2c437f82954c`.
Its [published amendment](https://github.com/techrote/neural-cyber-automata/pull/78#issuecomment-6355802213)
preserved the original declaration, first checkpoint, admission and all charges.
Six exact-head CI checks passed and
[two actual prospective reviews](https://github.com/techrote/neural-cyber-automata/pull/78#issuecomment-6355881709)
admitted that training continuation.

After training, a CPU readback exposed a 1 MiB parser bound on real 2.1 MB
tuning reports. The narrow repair uses an explicit 16 MiB bound only for
authenticated tuning reports; small receipts retain 1 MiB. Both independent
reviewers reproduced the repaired full 120-report proof without Torch or
accepted final payload access. An evaluation-only child binds all 11,205
completed-run files, old training provenance, remaining budgets and actual
tuning-selected candidates. It cannot authorize a training attempt.

Final evaluation executed clean commit
`fb392b023d522caeb99bddb1ae709d57be051861`, source closure
`7f140d712e42a23b4ecc7bf1129ae2890a7d23c92d3fdd32bb58cd13a2bd09cc`.
The [evaluation-only declaration](https://github.com/techrote/neural-cyber-automata/pull/78#issuecomment-6360303058)
has file SHA `ac058a6eac60bfbe5c0e87a707e28ab7fdebe1e6e930927f0b2e5125c1ea44de`.
Six exact-head CI checks and
[two fresh evaluation admissions](https://github.com/techrote/neural-cyber-automata/pull/78#issuecomment-6360362915)
preceded execution. Earlier suspended evaluation admissions were not reused.
Those prospective checks missed the original archive's actual member prefix;
CI and review did not prove successful final execution.

## Verified training

Every seed completed 40 epochs, 1,720 CUDA Adam updates, 210,400 presentations
and 80,800 tuning decisions, including all 40 complete 2,020-row reports.
CPU-only checkpoint audits verified 5,250 changed learned parameter elements,
six finite Adam states at step 1,720 and unchanged normalization/bounds buffers.
All original baseline files and cumulative ledger chains authenticated.

| Seed | Cumulative checkpoints | Cumulative CUDA training seconds | Cumulative training ledger charge, seconds | Tuning-selected step |
|---|---:|---:|---:|---:|
| 41011 | 42 | 49.325223 | 707.344 | 774 |
| 42013 | 41 | 50.827894 | 693.172 | 1720 |
| 43019 | 41 | 56.387335 | 767.532 | 645 |

Seed 41011 includes its original first batch and charges; nothing was reset.
Continuation process/charge pairs were 628.594/679.313 seconds for 41011,
643.265/693.172 for 42013 and 717.360/767.532 for 43019. Charged publication
reserves stay inside each original 7,200-second allocation. Every training
supervisor returned zero and verified its whole process within charged wall.

Actual target receipts identify NVIDIA TITAN RTX, driver 581.80, capability
7.5, 72 SMs and compiled `sm_75`. UUID fingerprint:
`e5df8b0db3a22718306c0f68e7ec85b69f1555759649550d74fc44f563c841d9`.
Each seed peaked at 69,206,016 reserved allocator bytes within the unchanged
2 GiB ceiling. Allocator measurements exclude driver/context/external memory.

Independent actual training audits:
[41011](https://github.com/techrote/neural-cyber-automata/pull/78#issuecomment-6359756834),
[42013](https://github.com/techrote/neural-cyber-automata/pull/78#issuecomment-6359943685),
[43019](https://github.com/techrote/neural-cyber-automata/pull/78#issuecomment-6360167393).
Tuning-only frozen candidate SHA:
`e9e1ea0e1bb4831b4323a78fef14a9a1012c97c12a61eae8833f5933c0161b85`.
The manifest records both complete-training and selected checkpoint hashes.

## Failed final opening and cost boundary

The candidate-bound opening was durably published at `2026-10-23T08:05:03Z`.
Its semantic SHA is
`769b01e29d075ecd1ed331a9849b9d81d4f041e0931c2358113079cb0f6890ea`.
It binds the three actual tuning-selected hashes and original registry/suite.

The reader then requested `corpus/final-00000.jsonl`. The authenticated original
archive instead stores
`runner-ik-corpus-v1-20261022/corpus/final-00000.jsonl`.
`TarFile.getmember` raised `KeyError` before the first `extractfile`, evaluation
attempt, device configuration or predictor call. Independent header-only audits
authenticated the unchanged archive and its 180 members/46 final shard headers
with payload extraction disabled. The mixed pending journal was not extracted.

No evaluation ledger event, checkpoint, comparison or CPU timing was produced;
all three training states and all 11,205 baseline files remain unchanged.
Source/trace audits place this phase's failure before CUDA initialization;
this is control-flow evidence, not new GPU telemetry. Training device evidence
remains the separately verified actual CUDA receipts above.

The failed worker returned 1. Supervision records 121.734 process seconds and
130.984 seconds through publication. Because failure preceded ledger attempt
creation, no additional wall was charged inside the three ledgers. The
supervisor conservatively holds all remaining wall: 6,492.656 / 6,506.828 /
6,432.468 seconds, retains unknown-work/overrun markers, and forbids reuse.
This is an uncharged-work protection; it is not a claim that the physical
process exceeded a 7,200-second allocation. Zero evaluation-call charges do
not make the invocation free or replayable.

Independent failed-outcome audits:
[gate review](https://github.com/techrote/neural-cyber-automata/pull/78#issuecomment-6360491048)
and [compatibility review](https://github.com/techrote/neural-cyber-automata/pull/78#issuecomment-6360498436).
Both actual outcome guards refuse further work. Reviewers acknowledged that
their prospective fixture review missed the real archive layout.

The failure, opening, holds and original artifacts remain immutable. No retry,
marker reset, replacement allocation, GPU substitution, retuning or additional
training followed. The final archive-reader defect and failed supervision are
unresolved; the gated evaluation is incomplete and no quality claim is supported.

© 2026 techrote. All rights reserved.
