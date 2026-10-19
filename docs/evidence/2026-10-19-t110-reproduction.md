---
title: T-110 independent reproduction and audited unfinished prefix
kind: evidence
summary: A separately declared faithful reproduction retains 112 verified episodes and seven missing controls after executor loss; its final receipts and termination remain unconfirmed, and the historical no-go is preserved.
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
| Limits | One commissioned screen; 4,000 game decisions and 900 aggregate seconds; the separately published pre-runtime correction below accounts for both CLI invocations without resetting this ceiling |

A clean detached checkout of the original learner executes the unchanged
runner, feature mapping, model, training procedure and evaluator. The source
manifest must match the [recovered original manifest](t110-frozen-source-manifest.json).
No code or data from later roadmap work enters that source checkout.

The original declaration started this new commission with an empty
prior-attempt ledger. The published launcher correction below subsequently
charged the import failure to that ledger. The historical screen retains its
original 3,315-decision / 227.1865-second aggregate and every recorded failure.
An interrupted or capped attempt retains its surviving files and missing
receipts; it receives no additional run.

## Publication and acceptance

The declaration must be publicly retrievable before execution. A separate launch
receipt records its published commit and file hash before any reproduction
rollout. All produced raw episode files, communication traces, source and run
manifests, initial/final checkpoints, checkpoint freeze, training histories and
failure logs are retained for read-only audit and archive publication. Missing
final receipts are recorded as missing.

The old runner's internal artifact names remain unchanged. The new outer
reproduction identity, launch receipt and archive index distinguish this
execution from the historical screen. No new file is claimed to be a recovered
historical runtime artifact.

The old no-go remains preserved whatever the new result. A completed
reproduction would report its own unchanged utility-gate outcome, per-seed
results, failures, timing and evidence limits. The unfinished prefix below
computes no aggregate gate. The commission does not permit selecting a more
favorable checkpoint or changing the quiet-arrival criterion after outcomes.
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

At the correction's declaration stage the independent physical screen had not
executed. The original declaration was subsequently verified at public commit
`9facc2f50ecc759ce1170587d325a5b215e0edc3`, and the correction and launcher at
`1aede29234e3132aaae8f625d7957ef700f31797`, before the corrected launch.

## Observed execution: audited unfinished prefix

The original corrected invocation began at `2026-10-19T19:26:41.023947Z` using
the clean original `626099f9…` source. Its executor transport disconnected and
the original session later became inaccessible. The last retained raw artifact
was written at `2026-10-19T19:38:17.238651Z`. After the declared deadline, a safe
interrupt request through the original session returned that the session was
unknown. That response establishes neither cancellation nor process termination.
This execution thread did not launch or replay it after the loss. A later
concurrent GitHub execution is recorded separately below.

The [public-safe evidence index](t110-reproduction-prefix-index.json) binds the
published declarations, exact file hashes, missing episode identities and
separately delivered archives. A read-only audit verified:

| Retained evidence | Verified coverage |
|---|---|
| Episode records | 112: eight demonstrations, nine tuning, ninety heldout/stress and five saved untrained controls |
| Actual consumed work | 3,096 decisions and 18,576 native ticks |
| Training lineage | 180 eligible authored training rows; six fixed final candidates and six initial checkpoints; 30 epochs and 120 optimizer updates per fitted candidate |
| Source and runtime | All 38 frozen source hashes, exact experiment declaration and pinned host/runtime identities |
| Record and artifact integrity | 113 run manifests, 227 unique artifact references, and 356 raw screen files totaling 21,946,293 bytes |
| Locality | 1,981 local graph frames; reconstructed feature difference exactly zero; node ownership, one-hop message bounds, fault schedules and private-memory clears verified |
| Request attribution | 3,096 admissions and the consumed source envelope at every recorded native tick; maximum JSON action-echo difference `5.551115123125783e-16` |
| Outcome computation | All 112 retained per-episode metric records independently recomputed; no aggregate utility gate computed |
| Reset lineage | 111 following episode resets have distinct recorded run identities |

Seven final untrained-control episodes are absent. The original runner's
`screen-summary.json`, final screen run manifest and corrected process
completion receipt are also absent. **This is an unfinished reproduction.**
The confirmed decision/tick counts cover the retained prefix only. The final
aggregate budget counters, wall time, exit status and work after the last
retained episode are unknown. A passed prefix audit does not prove completion,
final budget compliance or zero unrecorded work.

The earlier immutable prefix is separately delivered as
`t110-reproduction-prefix-20261019T194247Z.zip`, SHA-256
`48dd01b9c084dfe435ffa69f655aebaf9c3bca3b4c5f50d6166278acabede178`.
The audited archive is
`t110-reproduction-audited-prefix-20261019.zip`, SHA-256
`38905e57031da728558f10cd30569a3886f123a53330432510b445337d9c8986`.
It includes the later cancellation response, failure helper, audit programs,
derived reports and a per-file index. All 356 original raw screen files are
byte-identical across these two ZIPs. The earlier ZIP is unchanged. Owner
delivery of the audited ZIP and its index was confirmed successful at
`2026-10-19 20:08 UTC`; both archives now have verified external delivery.
No public download URL is asserted. Original diagnostic bytes remain in the
archives, with machine paths omitted from this public record.

Numerical T110 outcomes remain withheld from the separate unconfirmed T105
training process. No new qualification is inferred from this prefix. The
historical no-go and unavailable historical raw archive remain unchanged.

## Prospective quiet-arrival cancellation

The audit identified a lifecycle limit in original `626099f9…`: a quiet-arrival
episode can finalize its record without an explicit owner-cancellation receipt.
The following serialized episode resets into a distinct run; native deadline
episodes record their terminal revocation. These facts do not establish the
final unrecorded execution boundary.

The current runner now requests owner cancellation when quiet arrival precedes
the native terminal, records that zero-tick revocation before finalizing the
episode, and clears controller memory and queued messages when the invocation
closes. The original source checkout and historical/new reproduction records
remain byte-identical; this prospective fix establishes no new task result.

A focused actual-host regression uses a stationary at-goal policy fixture with
no fitted weights. It verifies a changed control generation and observation
identity at the same completed tick, unchanged body motion and physical part/
joint state, a persisted final cancellation event, and clearing of nonzero
private memory and delayed messages. Joint control targets are permitted to
clear separately from physical state. The focused cellular suite passed all
**39 tests with zero skips in 14.194 seconds** on the pinned CPU/host runtime.
Its first invocation had 38 passes and one overbroad test assertion that treated
the cleared joint control target as physical state; the corrected assertion
passed the full suite. This conformance work is separate from both learning
screens and changes neither screen's budget or outcomes.

## Concurrent GitHub repeat and budget deviation

A concurrent publication at `be3ea792695116caf886cf475a0479fcae778d77`
added a push-triggered workflow and repeated the original frozen learner as
[Actions run 39323844679](https://github.com/techrote/neural-cyber-automata/actions/runs/39323844679).
It launched at `2026-10-19T20:59:54.021436Z` and completed at
`2026-10-19T21:02:14.025106Z` with exit code zero. This was an additional
execution, not the original 19:26 invocation moving to GitHub or resuming.
The workflow was removed in the subsequent result publication to prevent
further automatic runs.

The completed GitHub screen reported 119 rows, 3,297 decisions, six final
checkpoints and 138.697095888 screen seconds; its outer process reported
139.980692305 seconds. The compact [result receipt](t110-reproduction-result-20261019.json)
retains the published summary and artifact hashes. Its reported
142.008152423-second aggregate includes only the original 3.311061632-second
import failure and this GitHub screen. It omits the interrupted local execution.

**The two corrected executions have at least 6,393 confirmed decisions**, above
the original aggregate 4,000-decision ceiling. Complete aggregate wall time and
any unrecorded local work remain unknown. No further allocation was declared
by this execution thread before the concurrent repeat. The repeated result
therefore cannot be represented as the sole permitted corrected invocation or
as evidence that the original aggregate budget was respected.

The GitHub receipt reports the same bounded `no-go-at-this-screen` decision:
graph and no-message each succeeded in 1/6 normal trials, authored in 0/6,
and paired graph-minus-no-message success gain was zero. These observations
are retained under their distinct execution identity; they do not promote
communication or change the original historical no-go. No numerical outcome
from this repeat was supplied to the separate T105 training or selection work.

The published 387-file Actions artifact is `t110-corrected-reproduction-evidence`,
ID `12220438092`, 16,054,974 ZIP bytes, SHA-256
`9e8c20698c842e7cc649d96aba468987e38b46a1dad9244dbf22a5a0e503b607`.
The screen-summary SHA-256 is
`cea07abb3b8fea267077dbf54c12055dbc53a05e17149cba3863c5a29f98cd88`.
The separately saved unfinished local archive and all historical source and
result hashes remain unchanged. Owner feel and physical topology mutation
remain unmeasured.

© 2026 techrote. All rights reserved.
