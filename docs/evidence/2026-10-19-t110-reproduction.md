---
title: T-110 independent reproduction declaration
kind: evidence
summary: One newly commissioned reproduction of the original frozen Jolt screen, with unchanged implementation and criteria, separately bounded execution and complete artifact publication.
owns: []
verified: 2026-10-19
---

# Independent reproduction of the frozen Jolt screen

The renewed instruction to complete the programme commissions one new faithful
reproduction after the [archive review](2026-10-19-t110-review.md) found that the
original runtime artifact set was unavailable. The
[new declaration](../../configs/cellular/t110-reproduction-20261019.json)
precedes all reproduction rollouts. This is a new experiment with its own
evidence identity; it does not recover or replace the original trajectories,
checkpoints, timestamps or runtime measurements.

## Frozen execution and budget

| Field | Declared value |
|---|---|
| Reproduction identity | `t110-jolt-three-part-reproduction-20261019` |
| Learner revision | `626099f98bf37f961c45796914b31926dc05f29d` |
| Learner tree | `5d46e52b32122bbb2b4127f150541a420b93e66e` |
| Complete source-file aggregate | `ab0783530cecb26250605ac3be9cf22b0a7e62d1f39b1fc627a5ae1b526076f5` |
| Original experiment declaration | `e5a66f5a830b9e097a5bbb9647ade4bea42c82e26fb5718273b5bae2b992de5d` |
| Actual host | Cybersand `f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497`, pinned Godot/native/Rapier bytes |
| Runtime | Linux x86-64, Python 3.12.14, NumPy 2.3.5, Torch 2.8.0+cpu, one Torch thread |
| Training seeds | `17011`, `17017`, `17021` |
| Planned task matrix | Eight demonstrations, nine tuning episodes, ninety paired heldout/stress episodes, twelve saved untrained controls |
| Selection | Thirty fixed BC epochs; fixed final checkpoint for all six candidates |
| Limits | One attempt, at most 4,000 game decisions and 900 seconds; no retry, tuning or extension |

A clean detached checkout of the original learner executes the unchanged
runner, feature mapping, model, training procedure and evaluator. The source
manifest must match the [recovered original manifest](t110-frozen-source-manifest.json).
No code or data from later roadmap work enters that source checkout.

This new screen starts with an empty prior-attempt ledger because it is a
separate commission. The historical screen retains its original 3,315-decision /
227.1865-second aggregate and every recorded failure. If the new attempt fails
or caps, that result and its surviving files remain the completed attempt under
this declaration; it does not receive another run.

## Publication and acceptance

The declaration must be publicly retrievable before execution. A separate launch
receipt records its published commit and file hash before any reproduction
rollout. All raw episode files, communication traces, source and run manifests,
initial/final checkpoints, checkpoint freeze, training histories, summary and
failure logs will be retained for a read-only integrity audit and complete
archive publication.

The old runner's internal artifact names remain unchanged. The new outer
reproduction identity, launch receipt and archive index distinguish this
execution from the historical screen. No new file is claimed to be a recovered
historical runtime artifact.

The old no-go remains preserved whatever the new result. The reproduction will
report its own unchanged utility-gate outcome, per-seed results, failures,
timing and evidence limits. It is not permission to select a more favorable
checkpoint or change the quiet-arrival criterion after seeing outcomes.
Owner feel, physical morphology mutation and broader graph usefulness remain
separate unmeasured claims.

## Combined conformance

The learning workflow now prepares the pinned physical host alongside Torch.
Its test runner requires the cellular actual-host case to be present and rejects
a skipped result. The full discovered suite runs once, so combined cellular
conformance no longer depends on two separate jobs that each skip the required
combination. Source/workflow changes establish this required gate; a hosted
passing result remains tied to the eventual published final head.

The local combined check subsequently passed all **38 focused tests with zero
skips in 6.500 seconds**, using the pinned CPU and actual host runtime and the
unchanged cellular implementation reviewed at
`9e76b979c8b39b6daa3dde108cfbb742ec2b32a1`. Its first invocation had 37 passes
and one socket-permission failure before host startup; the permitted local
transport invocation then passed the complete suite. This is source/host
conformance, not another learning screen or a change to either experiment's
training budget.

## Preserved import failure and declared launch correction

The first CLI invocation started at `2026-10-19T19:15:54.420510Z` and exited
before importing the learner, constructing a host, creating the screen output
directory or consuming any game work. Its outer wrapper resolved the virtual
environment's Python symlink to the base interpreter. That interpreter raised
`ModuleNotFoundError: No module named 'torch'`. The
[byte-exact completion receipt](t110-reproduction-import-failure-receipt.json)
records exit code 1 and **3.311061632000019 seconds**. The launch receipt,
empty stdout and original traceback remain retained under the new reproduction
identity. There were zero episodes, game decisions, ticks, training updates or
checkpoints; this failure produces no physical or learning result.

The [pre-runtime correction declaration](../../configs/cellular/t110-launch-correction-20261019.json)
explicitly authorizes one corrected CLI invocation. The
[corrected launcher](../../tools/cellular_reproduction_launch.py) preserves the
virtual-environment entry point instead of dereferencing its symlink. A
read-only import check through that entry point confirmed Python 3.12.14,
Torch 2.8.0+cpu and NumPy 2.3.5 with no CUDA; it started no host or training.

This amendment preserves the failed invocation and passes its exact elapsed
time through the original runner's
[prior-budget mechanism](../../configs/cellular/t110-reproduction-prior-budget-20261019.json).
The aggregate ceiling remains 4,000 game decisions and 900 seconds, leaving
896.688938368 seconds for the corrected screen. Both CLI invocations are
explicitly accounted. No sample, source, runtime, checkpoint-selection or
evaluation change accompanies the wrapper correction, and there is no further
retry. The correction, receipt, prior budget and launcher must be publicly
retrievable before that invocation starts.

## Corrected reproduction result

The single permitted corrected invocation was published at `be3ea792695116caf886cf475a0479fcae778d77` and
ran as GitHub Actions run `39323844679`. It launched at
`2026-10-19T20:59:54.021436Z` and completed at
`2026-10-19T21:02:14.025106Z` with exit code 0. The frozen learner checkout
remained clean. The outer process used 139.980692305 seconds; the screen itself
recorded 138.697095888 seconds. Together with the preserved 3.311061632000019
second import failure, the screen ledger reports 142.00815242300004 aggregate
seconds, still inside the unchanged 900-second ceiling.

The completed screen retained all **119** declared rows, six frozen checkpoints
and **3,297 game decisions**. It made zero direct-cell-write requests and
recorded no execution failure. The exact compact result receipt is
[t110-reproduction-result-20261019.json](t110-reproduction-result-20261019.json).

The unchanged utility gate returned **`no-go-at-this-screen`**. Graph and
no-message each succeeded in **1/6** normal holdout trials; the authored
reference succeeded in **0/6**. Their paired normal success gain was exactly
zero (mean and 95% interval `0.0`). The graph's capped-completion gain was
slightly negative (mean **-0.03333333333333336 s**, 95% interval
**[-0.0783333333333334, 0.0]**). Angle, inference, integrity, reference-success
and stress checks passed; only `normal_benefit` failed. This confirms the
frozen screen's no-go rather than providing evidence to promote graph
communication.

The complete 387-file Actions artifact is
`t110-corrected-reproduction-evidence` (artifact `12220438092`, ZIP
**16,054,974 bytes**, SHA-256
`9e8c20698c842e7cc649d96aba468987e38b46a1dad9244dbf22a5a0e503b607`).
Its exact `screen/screen-summary.json` SHA-256 is
`cea07abb3b8fea267077dbf54c12055dbc53a05e17149cba3863c5a29f98cd88`.
No retry, retuning or checkpoint reselection followed. Owner feel and physical
topology mutation remain unmeasured.

The historical no-go remains preserved separately; this reproduction confirms
the same bounded conclusion with a new evidence identity rather than rewriting
the historical run.

© 2026 techrote. All rights reserved.
