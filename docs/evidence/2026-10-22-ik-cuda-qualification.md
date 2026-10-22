---
title: Windows V100 IK runtime qualification and matched CPU costs
kind: evidence
summary: Prospectively declared real CUDA placement, finite Adam updates and checkpoint resume, with complete matched CPU costs and scoped device acceptance.
owns: [evidence.ik_cuda_qualification]
verified: 2026-10-22
---

# Windows V100 IK runtime qualification

[Issue #51](https://github.com/techrote/neural-cyber-automata/issues/51) already
had its software implementation through [PR #68](https://github.com/techrote/neural-cyber-automata/pull/68).
This follow-up executes its two remaining actual-device gates using that
runtime. Executable source, accepted CPU environments, configuration and
earlier evidence are unchanged. Open PR #76's unrelated Jolt work is preserved.

## Prospective admission and identities

The [prospective issue publication](https://github.com/techrote/neural-cyber-automata/issues/51#issuecomment-6351007674)
and [evidence release](https://github.com/techrote/neural-cyber-automata/releases/tag/ik-cuda-qualification-20261022)
precede every optimizer operation. The release was published at
**20:49:31 UTC on 22 October 2026**, before the first attempt. Its two exact
declaration assets retain their original bytes. Independent admission review
found no source blocker and performed no device workload.

| Identity | Value |
|---|---|
| Clean executable revision | `95873260c2411da320cac40d28644df2a45e6738` |
| Exact raw source closure | 131 files; `fb3ebcb0a77e3db377d95e8f2e82eac463320c3f1b708dddb85f316ef1df6434` |
| Configuration | `runtime-smoke-v1.json`; semantic SHA-256 `31600ef59d3491f48c22ff7bcb086de5c8df78f1588cfbbb67064a8e075e5350` |
| Dataset/generator/split | `f8121642a24ec1515c6187cbc7e4b747e39d45f08c188e0ab583fc854479c4f6`; fixed 256-point lattice, training-only smoke family; no quality holdout |
| Runner profile | `6cb15832e240f82f4bcd582d2c26015ef986e44f335da683abaf9746e31503c4` |
| Descriptor host revision | `f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497`; host not executed |
| CUDA declaration raw SHA-256 | `3ae256c6c6a60769d54c8b9aa4fc14acfb1b697e9bf5cf4f25eafc14fb84b65c` |
| CPU declaration raw SHA-256 | `e428fbd68e95073eaff4f7e3e2904d3a45d9890ad9ea561ced262d9921a95bc0` |

Both lanes use Windows 11 AMD64, CPython 3.12.10, Torch **2.8.0+cu126**, CUDA
build 12.6, NumPy **2.3.5**, cuDNN 91002 and one Torch thread. The existing
PowerShell setup and metadata-only doctor were exercised. The complete
declarations record all package versions. Torch Git revision is
`a1cb3cc05d46d198467bebbb6e8fba50a325d4e7`; build-config SHA-256 is
`39fc996f9a7576f7d777f02b1eb2b460b7cbad05855c11c6d040f1f94003c69d`.

The selected **Tesla V100-SXM2-16GB**, capability **7.0**, has 80 multiprocessors,
17,031,102,464 reported bytes and installed driver **581.80**. Preflight finds
compiled architectures `sm_61, sm_70, sm_75, sm_80, sm_86, sm_90` and selects
`sm_70`. The selected GPU alone is exposed as `cuda:0`; hardware UUIDs are not
published. Free memory at initial preflight is 16,697,720,832 bytes. The CPU
match uses the same CUDA wheel on an AMD Ryzen 5 5600X, six cores / twelve
logical processors. CPU and CUDA environment identities are respectively
`f04a600bc35406e3b9c096115b97e895978f52628d63e78da78a3d6b06dd76eb` and
`b6e5f083d9e1e35d66691f9fb60c98c83aceddd7e74694cec926d6206657ffc5`.

FP32, highest matmul precision, TF32 disabled, deterministic algorithms enforced,
deterministic cuDNN with benchmarking disabled and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` apply to both lanes. Autocast, compilation,
fused/foreach Adam and custom kernels are disabled.

## Complete bounded execution

The unchanged 4→32→2 network, dropout 0.1 and Adam learning rate 0.001 use seed
5101, batch 32 and 32 requested steps. Each lane first executes
`smoke --stop-after 8`, then `resume` under the same raw declaration hash and
original run directory. CUDA executes first; CPU follows. Every attempt completes
without failure, cancellation, OOM or environment continuation. The first
attempt is an intentional boundary pause, not final qualification.

Each lane's ceiling remains **256 attempted steps, 8,192 samples and 120 aggregate
charged seconds**, including startup, both attempts and finalization/readback.
The CUDA allocator fraction is 0.5; each checkpoint is capped at 32 MiB. The
successful terminal allocation refuses further training even though unused
capacity remains. No sweep, extra CUDA test, batch change or selected repeat ran.

| Lane | Run ID | Steps / samples | Attempt charges (s) | Total charged (s) | Measured outer invocations (s) |
|---|---|---|---|---|---|
| CUDA | `ik-da321827bee543cd9ddb1b040b6bffcd` | 8+24 / 1,024 | 13.906 + 10.515 | 24.421 | 9.2085467 + 5.9394411 = 15.1479878 |
| CPU | `ik-944a47df0d5546fda0970bb6cd9957a2` | 8+24 / 1,024 | 9.391 + 9.891 | 19.282 | 4.6950267 + 5.3176554 = 10.0126821 |

Charges conservatively include the existing five-second finalization reserve
per attempt. They are distinct from worker stage timings and external
PowerShell invocation stopwatches. Both lanes together consumed 64 optimizer
steps and 2,048 samples; charged totals are **43.703 seconds**. Every original
attempt and its consumed costs remain retained; there is no replacement run.

All 64 optimizer-boundary receipts show finite loss and gradients, changed model
weights and finite FP32 placement evidence. Each receipt checks **26 tensors**:
two data, one input, four model, one output, one geometry, one loss, four
gradient and twelve optimizer tensors. CUDA receipts identify `cuda:0`; CPU
receipts identify `cpu`. The CUDA result is `cuda_smoke_passed`; CPU is
`cpu_smoke_passed`. Metadata-only preflight remains `qualification=unmeasured`;
qualification comes from actual completed work.

Resume binds the step-eight checkpoint hash to the next attempt request and
restores model, Adam, sampler/permutation/cursor, normalization and all saved
Python/NumPy/Torch CPU/CUDA RNG components. Counters proceed 8→9→32 and cursor
256→288→1,024. The CPU sampler produces identical sample sequences across the
matched lanes. No cross-device equality of weights or random dropout is claimed.

## Timing and memory

All durations below sum both attempts. Synchronized host timing brackets the
checked optimization workload; CUDA event intervals also include host-induced
stream gaps from placement/finite checks. They are not kernel-only timings.
No warmup or outlier removal is applied.

| Measurement | CPU | CUDA |
|---|---|---|
| Synchronized training (s) | 0.075995100 | 0.491718200 |
| CUDA event interval (s) | null: inapplicable | 0.491313801 |
| Samples / synchronized training second | 13,474.553 | 2,082.494 |
| Preflight including validation/import (s) | 4.140 | 8.312 |
| Loading/model construction/restore (s) | 1.922 | 2.063 |
| Generation and transfer (s) | 0.031 | 0.000 at recorded clock resolution |
| Checkpoint publication (s) | 0.157 | 0.215 |
| Peak allocated bytes, maximum per attempt | null: CUDA counters do not measure CPU memory | 68,178,432 (65.0200 MiB) |
| Peak reserved bytes, maximum per attempt | null: CUDA counters do not measure CPU memory | 69,206,016 (66 MiB) |

The small checked workload is slower on CUDA. These observations establish
compatible execution and measured cost; they predict neither large-batch
throughput nor the separately commissioned three-seed trainer's performance.
Stage accounting uses the runtime's Windows monotonic clock; generation's
recorded zero is below its resolution, not absent measurement. Training timing
uses the finer performance counter and synchronized CUDA events. Stages do not
cover all parent/startup/shutdown costs; use charged and outer totals separately.
Allocator peaks exclude driver/context/external allocations. CPU RSS and full
host/device process peaks remain unmeasured.

## Retention, review and limits

The [evidence release](https://github.com/techrote/neural-cyber-automata/releases/tag/ik-cuda-qualification-20261022)
retains original declarations, every request/reservation/boundary/checkpoint,
preflight, progress log, worker result, immutable manifest and seal. The
publication index and independent readback record accompany the final archive.
The [run ledger](../RUNS.md) preserves both run identities and cumulative cost.

The [publication index](ik-cuda-20261022/index.json) identifies the **310-member,
1,579,701-byte archive**, SHA-256
`1735d1d51bb08bb619e461d4eb7153abd59aad8bf3802838d3f4b9ee04f3eb30`.
Its [external ZIP](https://github.com/techrote/neural-cyber-automata/releases/download/ik-cuda-qualification-20261022/ik-cuda-qualification-20261022.zip)
contains every original attempt and all 66 checkpoint payloads, plus the
per-member SHA-256/byte inventory, comparison and reproducible independent
readback script. Check the archive hash before extraction, then check each
member against `archive-inventory.json`. Checkpoints stay outside Git.
The committed [summary](ik-cuda-20261022/summary.json) is a small cost index;
its hash and the independent-review hash are bound by the publication index.

The [independent offline audit](ik-cuda-20261022/independent-review.json) passes
**7,418 checks** without findings: 131 raw source files, four immutable attempts,
282 artifact references, 64 reservation/step pairs, 66 checkpoint payloads and
markers, 264 model / 768 Adam / 165 RNG tensor snapshots and all four invocation
receipts. CPU-only deserialization verifies step-eight request binding,
Adam's 8→9 progression, matching initial model hashes and four complete sampler
epochs per lane. It inspects CUDA RNG bytes without replaying them. This
separate offline evidence processing takes 2.241623200 seconds and adds no
optimizer work; it is distinct from original runtime charges. Uninterrupted
CUDA bitwise equivalence was not rerun.

On the publication branch, the bare `python -S` IK suite passes **237 cases:
200 passed / 37 explicit optional skips**. The pinned optional environment,
with GPU visibility disabled, passes **237 cases: 235 passed / two explicit
manual-CUDA skips**. Those unrequested tests remain skipped; the actual CUDA
evidence is the prospectively declared production CLI execution above. No
long GPU run enters routine PR CI. Final-head hosted checks and merge
verification are recorded by the owning PR and issue after publication.

This accepts only the identified **Windows / Torch 2.8.0+cu126 / V100 / fixed
FP32 IK smoke** combination and its same-device resume. Linux CUDA, other GPU
models, actual GPU OOM, cross-device continuation, independent batched-FK CUDA
parity, useful learned IK, exact CPU export and physical tracking remain
unmeasured. The previously accepted software failure controls retain their
original evidence; this successful allocation does not physically exercise them.
No Jolt, morphology or owner-feel gate changes.

© 2026 techrote. All rights reserved.
