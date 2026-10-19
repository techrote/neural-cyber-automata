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

The independent screen has not been executed at the declaration stage.

© 2026 techrote. All rights reserved.
