---
title: First admitted TITAN compact IK batch
kind: evidence
summary: One independently admitted and verified seed-41011 CUDA update on the original corpus, paused with all remaining gates intact.
owns: [evidence.ik_titan_first_batch]
verified: 2026-10-23
---

# First admitted TITAN compact IK batch

[Issue #54](https://github.com/techrote/neural-cyber-automata/issues/54)
and draft [PR #78](https://github.com/techrote/neural-cyber-automata/pull/78)
now record exactly one supervised seed-41011 batch: 128 presentations and one
finite CUDA forward/backward/Adam update, followed by a verified pause. Training
stopped there. No complete seed, three-seed comparison or learned quality is claimed.

## Source, original corpus and prospective admission

Execution used clean commit `b9d1ccda294b3111dbb227704d2b8cb867a84a75`, integrating
main `9747ef0`. Its 144-file executable closure has SHA-256
`2e08895a16ca82199feedf9aea0c490e84d44b6c1d3ef623595d3a718e2cb1ae`.
All six exact-source CI jobs passed: Windows/Linux CPU, optional CPU learning,
pinned host, strict docs and the inherited memory-envelope check. Local focused
CPU conformance passed 88 tests; strict docs checked 103 files without errors/warnings.

The original 7,340,716-byte archive freshly matched SHA-256
`97e3c547c7056e6ef727dc07437f469fde54b545bb091503a791aed85a4f99ce`.
CPU-only preparation authenticated 5,260 training and 2,020 tuning rows and
exactly recomputed all 14 original means/scales from 5,140 training-only inputs.
Two source-bound inventories admit only 534 exact retained/Windows label pairs
and 17 exact whole-task replay pairs after original-byte authentication. Original
tasks, labels, inputs, configuration and normalization are retained. No numerical
tolerance, reordering, missing-row, fixture or sealed-final exception was added.

The [complete prospective declaration](https://github.com/techrote/neural-cyber-automata/pull/78#issuecomment-6355144196)
was published before CUDA initialization. Its file SHA-256 is
`c4162e81f2cfbbd9bddfb86fbafc2d4b9c4b62f141197bfbb5e15a6db18b214e`;
semantic SHA-256 is `37183126562dae4d3d917b46bca2197a8e58ad41bf0365eb8ecc7558ef600ca3`.
Two independent Codex agents separately authenticated and reviewed source/CI,
corpus/splits/normalization, allocation/resume, scoring/selection/final protection
and TITAN environment/resource feasibility. Their substantive
[first admission](https://github.com/techrote/neural-cyber-automata/pull/78#issuecomment-6355177315)
and [second admission](https://github.com/techrote/neural-cyber-automata/pull/78#issuecomment-6355183093)
preceded the [aggregate admission and invocation](https://github.com/techrote/neural-cyber-automata/pull/78#issuecomment-6355192400).
The aggregate semantic hash is
`f34cf4f39829bc4584c1803660b65d3e3bf4619c26615bbfd629126e905b3a4c`.

Every original seed retains its fixed 1,720-step / 210,400-presentation /
80,800-tuning / 65,200-evaluation allocation, 45 checkpoints, four attempts,
7,200 aggregate seconds and 2 GiB allocator ceiling. This batch consumed seed
41011's existing allocation; it was not a free calibration run.

## Actual TITAN result

Only the TITAN UUID was visible, mapped to `cuda:0`; private UUID fingerprint:
`e5df8b0db3a22718306c0f68e7ec85b69f1555759649550d74fc44f563c841d9`.
Runtime identified NVIDIA TITAN RTX, capability 7.5, 72 SMs, compiled `sm_75`
route, driver 581.80 and 25,769,476,096 measured capacity bytes. Windows 11 AMD64 /
Python 3.12.10 / Torch 2.8.0+cu126 / NumPy 2.3.5 / CUDA build 12.6 were bound.
The allocator fraction was derived from that actual TITAN capacity. The V100
was never initialized or used; its historical results supplied no admission or baseline.

| Quantity | Observed result |
|---|---|
| Seed / batches / optimizer attempts and commits | 41011 / 1 / 1 attempted and committed |
| Presentations / checkpoint charges | 128 / 2, including retained initialization |
| Tuning / evaluation calls | 0 / 0 |
| Loss / pre-clip gradient norm | 0.18387065827846527 / 0.56528240442276; finite |
| CUDA placement | All model, batch, FK, loss, gradients and Adam tensors on selected `cuda:0` |
| Independent FK guard | 128 rows; maximum 1.1278636006563628e-6 cells against 2e-4 ceiling |
| Learned parameters changed | All 5,250 elements; six finite Adam states at step 1 |
| Peak allocator allocated / reserved | 67,304,960 / 69,206,016 bytes (64.187 / 66 MiB) |
| Synchronized host / CUDA-event training time | 0.12965439999243245 / 0.12965251159667968 seconds |
| Supervisor complete process / charged wall | 23.672 / 28.031 seconds; verified, no timeout/interruption |
| Terminal ledger | Paused, one reconciled batch, no unresolved work |

The targeted metadata observer retained 78 samples, no query errors, 30 C maximum
temperature and a sampled whole-card memory maximum of 2,263 MiB versus a
1,949 MiB baseline. These whole-card readings include unrelated/external
allocations; they are distinct from the allocator peak and are not a workload benchmark.

## Independent readback and preserved boundary

Both [checkpoint/ledger audit A](https://github.com/techrote/neural-cyber-automata/pull/78#issuecomment-6355223663)
and [device/receipt/checkpoint audit B](https://github.com/techrote/neural-cyber-automata/pull/78#issuecomment-6355229580)
passed. They authenticated checkpoint bytes/markers, reconstructed the original
first-batch sample identities, verified the event chain and reservations, and
decoded checkpoints on CPU with all GPUs hidden. No second optimizer/model
execution occurred. Normalization and command/measurement bound buffers stayed equal.

Initial checkpoint SHA-256:
`3cd37bd14dc4c19abe340d0ffa030bb1bd36774065c28e11d29075987fde9d7d`.
Post-update checkpoint SHA-256:
`808aa87c8b9e8d4a32fdcbff8bb185d49efe79da144fbae36a368dc90e82faf5`
(89,027 bytes). The complete [immutable result index](ik-titan-first-batch-20261023.json)
binds 27 artifact hashes and the source/declaration/allocation/runtime identities.
Its SHA-256 is `8b5a36232fff16bf9ac9e52086ab5c2f90f5ffa3c7a84b22c45adb4b47b03b9c`.
Original datasets, checkpoints, ledgers, metadata samples and CPU audits remain
in the maintainer-held external bundle identified by that index; no private
paths, GPU UUID or large artifacts enter Git.

Seeds 42013 and 43019 have zero attempts or charges. The final suite remains
unopened, with no tuning selection, frozen final candidates, CPU export or
physical result. Any later owner-authorized continuation must check out the
exact declared execution source and use the same run and remaining allocation.
This evidence publication is not an additional training authorization.

© 2026 techrote. All rights reserved.
