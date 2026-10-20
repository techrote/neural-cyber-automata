---
title: Current state
kind: state
summary: CPU contract implementation and explicit evidence limits for the current Cybersand control programme.
owns: [state.summary]
verified: 2026-10-19
---

# Current state

## T-100 — Accepted research baseline

The T-100 research programme landed in PR #21 at
`95d8788686980974d4f849e892c33f531d992fe7`.
## Accepted CPU foundation

T-101 merged through [PR #22](https://github.com/techrote/neural-cyber-automata/pull/22)
at `9e6da91ff520e27bc78d11dd8196844a9fa37ff3`, adding the installable CPU package,
strict gameplay/observation/action contracts, backward-compatible ACT003 mappings,
a versioned extension registry and immutable reproducible run records.

| Area | Evidence |
|---|---|
| CPU package and command line | Editable installation, environment report and recorded example exercised locally |
| Contracts, registry and run store | 27 focused tests; corruption, concurrency, lifecycle and privilege negative controls |
| Command-line conformance | Three additional tests for immutable records, optional-import isolation and invalid input |
| Linux/Windows package workflow | Both operating systems and the documentation check passed on the exact final T-101 head |
| Physical host adapter | T-102 implementation and actual Linux conformance described below |
| Authored gameplay controllers | Jolt/Vector feedback skills and matched actual-host screen below |
| Trained weights or scored learning runs | No accepted training result in T-101 |
| Runtime export, model service and device qualification | No accepted result in T-101 |

The [contract evidence](evidence/2026-10-19-t101-contracts.md) records source digests,
observed checks and corrected failures. The [package guide](PACKAGE.md) contains
commands; [progress checkpoints](PROGRESS.md) record ongoing implementation.

## Persistent actual host

The [host adapter](HOST_ADAPTER.md) implements owned reset/step over one persistent
Godot process, preserving the accepted native material/Rapier causal schedule.
Recorded playback, actual ACT003 sampled wire/metal and actual ACT004 physical
lanes have distinct identities. Six-tick decisions, twelve-tick helper leases,
exclusive observation expiry, cancellation and terminal durations are tested.

The [T-102 evidence](evidence/2026-10-19-t102.md) reports 23 focused passing cases,
including fifteen real integrations, and bounded 1,200-tick pilots for Monowheel
and Ripper. Realtime mode advances while a client is delayed and returns the
completed terminal state once. Linux host CI verifies pinned source and runtime
before executing the integration tests; final-head results belong to the PR.

These tests qualify the adapter on the selected host and CPU runtime. The actual
host has no separate passive-wrench decomposition, arbitrary body replacement,
support-removal or assist-mutation API. Those capabilities remain unavailable.
Windows physical runtime, desktop pacing and owner feel remain unmeasured.

## Authored skill reference

The [authored controllers](CONTROLLERS.md) implement Jolt directed hops, Vector
burst traversal and their declared recovery mappings. The accepted-adapter
screen ran 18 matched episodes across firm/balanced/compliant tuning and
step/pulse/reversal fixtures. All six step variants reached their target;
compliant Jolt missed the reversal deadline, and Vector drifted after completing
arrival. These limitations and all exact source/artifact hashes are in the
[T-103 evidence](evidence/2026-10-19-t103.md).

Thirty-three focused controller/measurement checks pass. Forced perturbation
fixtures are published but explicitly refuse unsupported host operations;
arbitrary disturbance, support removal, payload replacement and joint override
remain physically unmeasured. Owner feel approval is not recorded.

## Demonstration and replay boundary

The [data API](../configs/datasets/README.md) captures every consumed base tick,
the original source observation, dispatch-time view, source/policy identity,
requested/admitted action, actual impulse/motion, intervention and lifecycle
boundary. Unknown, refused, mixed or corrected consumption cannot acquire an
ordinary on-policy likelihood. Per-episode files publish immutably and family
splits reject contamination.

The [T-104 evidence](evidence/2026-10-19-t104.md) contains three actual capture
pipelines with 296 base receipts, compatible-host replay and separately recorded
skill outcomes. The +48-cell Jolt invocation succeeded by its local estimate;
the +20-cell invocation remained unsuccessful. No optimization occurred. The
recorded script transfers are not human or live-model demonstrations, and replay
qualifies copied local trajectories on the same build, not whole-world identity.

## Authored usable skills and composition

The [usable runtime](USABLE_SKILLS.md) provides typed approach, aim/use, recovery,
grip, carry and release over the accepted Vector launcher and Hammerhead cutter.
Pouncer charge-hold is the separate disarming example. Twenty-nine focused tests
pass, including thirteen actual-host integrations. The five-case
[T-107 pilot](evidence/2026-10-19-t107.md) exercised native clearance refusal with
recoil, declared recovery, physical pickup/carry/use/release, a retained failed
late pickup, persistent applied effects and charge cancellation without launch.

Owner control and observed attachment generations are independently fenced.
Local skill predicates remain separate from evaluator success; the scripted
human/model source labels do not claim an actual input session or live service.
Independent arbitrary object replacement and owner feel remain unmeasured.

## First learned Jolt skill

The optional [CPU learning lane](LEARNING.md) implements behavioural cloning and
recurrent residual PPO with original latent-action likelihoods, ordered sequence
updates, valid duration/continuation masks and immutable checkpoints. Its
[T-105 screen](evidence/2026-10-19-t105.md) used three policy seeds, six frozen
final checkpoints and eight unseen paired scenarios. It completed 11,755 game
decisions in 685.6 seconds under the predeclared 13,000-decision / 30-minute cap.

Mean success was 45.83% for cloning and 50.00% for residual PPO, versus 25.00%
for the authored reference. Both passed the declared improvement, integrity,
character and inference gates. Every failed scenario and weaker seed remains
recorded. The complete artifact audit found no held-out data in optimization.
An original deterministic-evaluation likelihood label defect is preserved in
historical records and corrected in the final source; it did not alter trained
weights or wire outcomes. Owner feel remains unjudged.

These are bounded exploratory improvements. Neither family met the independently
predeclared 75% mean / 50% minimum-seed morphology entry gate. Transfer learning
remains blocked by that result. The fixed BC seed 1103 candidate is eligible for
a separate CPU export qualification, with its directional failures retained.

## Gated morphology preparation

The [morphology preparation](evidence/2026-10-19-t106.md) pins accepted Jolt,
Vector, Monowheel and Ripper contracts with eight part slots, seven connection
slots and separate presence, measurement and activity masks. Optional padded
recurrent and two-round shared graph candidates have 13,048 and 13,192 parameters.
Forty-one focused checks and independent review cover locality, padding,
source-age limits, recurrence and confirmed owner-revocation boundaries.

The immutable admission runner retains the failed predeclared T105 reliability
prerequisite and emits zero training, candidate-game and optimizer work. Its
fixed conditional comparison budget is preparatory; no transfer result exists.
A separate [32-request / 192-tick scripted profile diagnostic](evidence/2026-10-19-t106-conformance.md)
passed on all four profiles at public source `557799ce`. It retains the first
zero-request startup failure; both attempts total 54.754654 seconds within the
original 60-second allocation. This establishes interface conformance; learned
transfer, adaptation, damage recovery and owner feel remain unmeasured.
The research question remains blocked by the stated prerequisite.

A separately commissioned [Jolt direction-frame BC follow-up](evidence/2026-10-19-t106-direction-frame.md)
completed its designated primary local invocation at public `1222d2e8`: 4,693 requests / 28,158 ticks,
11,760 BC optimizer steps, all 136 episodes and all twelve initial/final checkpoints, in 943.606 seconds.
Direction-frame and world-frame mean success were 0.25 and 0.291667, with minimum seed success 0.125
and 0.0. Both failed the unchanged 75% mean / 50% minimum-seed gate; other character/runtime failures
remain in the [immutable result](evidence/t106-direction-frame-result.json). The separate concurrently
started GitHub invocation is accounted for outside these primary totals and was not used for selection.
The [combined provenance record](evidence/t106-direction-frame-result-20261019.json) retains both:
9,353 confirmed requests / 56,118 ticks / 23,520 optimizer steps exceeded the original 6,000-request
allocation. The concurrent execution is an aggregate budget deviation.
No morphology qualification, training, transfer or adaptation follows this no-go.

The [conditional capture and learning source](evidence/2026-10-19-t106-learning.md) retains the original
comparison allocations and complete source/checkpoint admission. The guarded orchestrator, semantic
waypoint reference and full comparison summary pass 99 focused software checks without a physical
morphology experiment. An immutable primary designation prevents a concurrent repeat from replacing
the failed prerequisite. A passing separately commissioned prerequisite and a qualified semantic
waypoint reference would still be required before any transfer claim.

## Asynchronous source ownership

The [asynchronous adapter](ASYNC_CONTROL.md) connects bounded mock/local-worker
providers and typed human/script intent to the same usable skill runtime.
Original observation/plan age, actual actor control generation and attachment
changes are fenced independently. Local feedback continues while retired model
work remains busy; an unacknowledged call cannot accumulate replacement calls.

The [T-108 evidence](evidence/2026-10-19-t108.md) distinguishes proposal admission,
local skill feedback and actual owner receipts. Continuously advancing tests
exercise delay, jitter, duplicate/reordered replies, outage, takeover, reset and
available equipment changes. Model smart plans and their motor leases retain
original 120-tick expiry; longer explicit typed local missions are separate.
No live service or owner feel verdict is claimed.
The frozen actual pilot passed ten cases, recording 338 owner operations and
790 completed native ticks in 19.169 seconds with no skips or source drift.

## Evaluated CPU export and deployment no-go

The optional [runtime export lane](RUNTIME_EXPORT.md) converts the fixed trained
BC seed 1103 policy into a hash-bound NumPy GRU with declared preprocessing,
recurrence, wire mapping and exact runtime context. It fences original source
observations, ownership changes, deadlines and unsupported profiles. Conversion
verifies the accepted T105 training/evaluation lineage before admitting weights.

The [single fixed-candidate screen](evidence/2026-10-19-t109-deployment-no-go.md)
completed with an immutable no-go on published source `eb953615`. Both physical
trajectory pairs exceeded the unchanged tolerances. The fixed-count throughput
phase measured 79.033537896 seconds; the requested full ten-minute observation
remains outstanding. A proven realtime reason-filter defect has a prospective
checker correction and a separate same-receipt audit. Eleven focused checker
tests and four actual evaluated-bundle integration tests passed, the latter
using one host and 36 ticks. These engineering results do not qualify the
deployment. The original interrupted training termination remains unconfirmed,
and the complete original failure and distinct follow-up evidence are preserved.

## Conditional per-part graph result

The [fixed Jolt graph experiment](evidence/2026-10-19-t110-screen.md) completed
119 episodes with six preselected trained checkpoints, normal and delayed/lost/
partitioned/reset communication, and saved untrained controls. The three-part
star has private recurrence, one neighbor round and root-only action readout.
The equal-size no-message candidate shares the same local features and action
limits. All original declarations, failed attempts and audit corrections remain
in the [evidence index](evidence/2026-10-19-t110.md).

The predeclared result is no-go: graph and no-message each succeeded in one of
six normal holdout trials, with zero paired success gain; the graph was slightly
slower under the capped-completion metric. Each untrained method succeeded in
three of six. The trained successes did not require a purposeful hop, a retained
limitation of the frozen quiet-arrival task. Other declared gates passed, but
these observations provide no reason to promote communication in this screen.
Aggregate work stayed within 4,000 decisions / 900 seconds: 3,315 decisions and
227.1865 seconds through final ledger publication. Owner feel remains unjudged.

The independent local reproduction stopped producing records after 112 episodes,
3,096 confirmed decisions and 18,576 ticks; its completion and termination are
unconfirmed. Its audited archive is saved. A concurrently launched GitHub repeat
later completed 119 rows / 3,297 decisions and reported the same no-go. These are
separate executions. Their combined confirmed decisions are at least **6,393**,
above the original aggregate **4,000** ceiling; the GitHub receipt's reported
142.008-second total omits the interrupted local work. This deviation is retained
in [the reconciliation and evidence](evidence/2026-10-19-t110-reproduction.md),
with both immutable artifact identities. It does not qualify graph utility.

The current runner also records actual owner cancellation on early quiet arrival
and clears private memory/messages before closing. A dedicated actual-host
regression passed; frozen historical and reproduction source/records are unchanged.
Owner feel and unsupported physical topology changes remain unmeasured.

## Host authority

The refreshed Cybersand source at
`f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497` includes ACT-004 articulated actors
merged through PR #181. Its owner API has configure, reset, observation, submit,
cancel, step and snapshot operations. T-102 owns qualification of that boundary.
No Cybersand source mutation is part of T-101.

Package tests and a recorded protocol example establish no gameplay success,
learned-policy value or owner feel acceptance. GPU qualification is unmeasured.
