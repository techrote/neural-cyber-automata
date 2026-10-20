---
title: Jolt advisory v2 — complete diagnostic, teacher gate failed
kind: evidence
summary: Complete durable 96-episode comparison restores airborne character but loses arrival reliability; learning and morphology remain blocked.
owns: []
verified: 2026-10-20
---

# Jolt advisory v2 diagnostic

The independently declared [#60](https://github.com/techrote/neural-cyber-automata/issues/60)
invocation completed all 96 episodes, including durable sealing and independent
post-exit verification, in **444.46566076900126 seconds**. The recovery teacher
arrived in **25/32 cases**, compared with **27/32** for the arrival reference.
It restored purposeful airborne movement in **32/32**, but fails the unchanged
teacher acceptance rule. No dataset collection, learning or morphology is admitted.

[PR #61](https://github.com/techrote/neural-cyber-automata/pull/61) contains the
advisory lifecycle, refused-decision evidence correction and this result. The
[original truncated arrival diagnostic](2026-10-20-jolt-arrival-reference.md)
and [failed recovery v1 attempt](2026-10-20-jolt-recovery-v1.md) remain unchanged.
A complete new result neither repairs their bytes nor changes their status.
The original [result](jolt-recovery-v2-result-20261020.json),
[outer receipt](jolt-recovery-v2-supervisor-20261020.json),
[teacher audit](jolt-recovery-v2-review-20261020.json),
[journal seal](jolt-recovery-v2-seal-20261020.json) and labelled
[character aggregation](jolt-recovery-v2-character-summary-20261020.json)
are retained with their byte identities.

## Prospective source and scope

The public freeze preceded execution in PR #61 and issue comments on #60, #48
and #43. The physical run used a clean detached checkout of that exact commit.

| Identity | Frozen value |
| --- | --- |
| Source revision | `8da294ae4a263dd95c34b181abb795b906060c2d` |
| Git tree | `c701a9d37edb5567fab9c7db0fb595fafec2ece5` |
| Canonical declaration SHA-256 | `d7cc13335b453a755e9b7d950ac52d982aff553c025cbe0fbd47b81b5b0729eb` |
| Candidate specification SHA-256 | `a3f4da6ba34f09884937fdd522290043d9f64909481b3fafab8bcd13d3202037` |
| Source-file inventory SHA-256 | `9f8cbc56e8c8fa6a4753f228460f06d52cc42f19ea695fec32890f37563da92a` |
| Read-only host revision | `f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497` |

The [declaration](../../configs/baselines/jolt-recovery-diagnostic-v2.json)
preserves v1 controller/supervisor bytes, both earlier references, every kinetic
constant, the task, host mechanics and physical profile. Decisions consume six
base ticks at 60 Hz. The goal remains a 240-tick deadline with supported arrival,
four-cell tolerance, speed at most six cells/s, angle at most 0.32 radians and
twelve continuous quiet physical ticks. Maximum helper lease remains twelve ticks.

The sole behavioral change is advisory local settling. A quiet sampled estimate
requests a neutral hold with valid existing observation/deadline authority.
Only the caller validates the task outcome and cancels local/native intent.
Fresh disturbance resumes ordinary control. The separate recorder schema retains
refused decisions without inventing physical frames or granting completion.

This new allocation permits one invocation, 96 episodes, 3,840 decisions,
23,040 requested ticks, 4,226 owner calls and 900 whole-process seconds including
startup, closure and independent verification. It permits zero optimizer updates,
learned-policy calls or checkpoint execution. No previous rows were reused.

## Matched outcomes and character

All 32 cases were already opened. Each family ran all of them; the eight historical
evaluation-origin cases are reported separately and are not sealed evaluation.
Means weight cases equally. The immutable result contains every row and failure.

| Measurement | Original teacher | Arrival reference | Recovery v2 |
| --- | ---: | ---: | ---: |
| Arrivals / 32 | 17 (53.125%) | 27 (84.375%) | 25 (78.125%) |
| Training-origin arrivals / 24 | 13 | 22 | 19 |
| Opened evaluation-origin arrivals / 8 | 4 | 5 | 6 |
| Purposeful airborne cases / 32 | 32 | 25 | 32 |
| Confirmed airborne primary pulses | 63 | 25 | 42 |
| Mean airborne seconds | 1.685938 | 0.578646 | 1.052604 |
| Mean peak toward-speed, cells/s | 69.790112 | 53.974276 | 39.269321 |
| Mean overshoot, cells | 19.007839 | 0.606258 | 0 |
| Mean capped completion, seconds | 2.720313 | 2.085417 | 2.801042 |
| Mean first toward-motion response, seconds | 0.177604 | 0.227604 | 0.496354 |
| Mean ordinary trim reversals/s | 0.105127 | 0.490190 | 0.410481 |
| Mean absolute angle, radians | 0.046669 | 0.026368 | 0.027598 |
| Worst case inference p95, ms | 1.333349 | 2.680419 | 0.957702 |

Recovery requests, native admission, consumed primary edges, confirmed actuator
pulses and airborne confirmation all agree at **42**. Ten are second pulses.
The original requests 66 primary edges but confirms 63 airborne pulses; the
arrival reference confirms 32 actuator pulses but only 25 airborne pulses.
Requests and actual airborne motion therefore remain separate measurements.

Recovery has 2,021 airborne ticks, 24 ordinary recovery events and five declared
emergency-recovery events. Twenty-eight of the 29 events have at least one cell
of subsequent observed progress. That association does not establish which
action caused progress or turn each event into a rescued arrival.

Nineteen supervisor intervention decisions occur in five cases. Arrival is
22/27 among cases without emergency intervention and 3/5 among cases with one.
Both strata remain in the 25/32 total; assisted outcomes are not relabelled as
autonomous. Unknown-support and unknown-impulse tick counts are zero.

Mean summed aggregate linear-impulse magnitude is 569.868285 / 476.165192 /
587.488989 native units per case for original / arrival / recovery. Mean summed
absolute angular impulse is 134.160094 / 79.401531 / 120.329240. These are native
actuator receipts, including assistance; passive/contact wrench decomposition is
unavailable. The first aggregate actuator receipt follows an active request by
one tick for all cases, but assistance prevents attributing this solely to policy
intent. Mean per-case owner-roundtrip p95 is 64.246 / 69.226 / 65.348 ms, including
durable transport overhead. It is distinct from policy inference and motion response.

## Unchanged teacher gate

Independent frozen-source replay reproduces every one of the candidate's 900
decisions. Pulse readiness and fresh physical recovery reasons match all 42
requests, including the ten second pulses. The acceptance specification changes
only the candidate/version identity from v1; every numeric threshold is unchanged.

| Requirement | Result | Decision |
| --- | --- | --- |
| Strictly improve contemporaneous arrival | 25 versus 27 | Fail |
| Rescue at least three named historical failures | Two | Fail |
| Rescue required unsupported/drop case | Still fails | Fail |
| Rescue a supported historical failure | `demo-floor-left-60-2` | Pass |
| At most one regression | Four | Fail |
| Restore at least one missing-airborne case | All seven restored | Pass |
| Retain overall purposeful-airborne count | 32 versus 25 | Pass |
| Ordinary trim increase at most 0.5/s | -0.079709/s | Pass |
| Mean absolute angle increase at most 0.1 rad | +0.001230 rad | Pass |
| Mean peak toward-speed ratio 0.7–1.5 | 0.7275562273201698 | Pass |
| Every case inference p95 at most 5 ms | Maximum 0.957702 ms | Pass |
| At most two primary edges, fresh second-hop authority | Maximum two; all replayed | Pass |

The two rescued arrival-reference failures are `demo-floor-left-60-2` and
`reference-drop28-left-57`. The four regressions are `demo-floor-right-48-0`
and `demo-floor-right-60-0`, `-1`, `-2`. The other remaining failures are
`demo-floor-left-60-1`, `reference-floor-left-63` and
`reference-drop12-right-45`. The required unsupported/drop rescue remains absent.

All seven failures reach the unchanged task deadline at tick 240, with two
confirmed pulses, final raw support, restored energy/cooldown and no failed
admission. The three right-60 regressions finish 3.037 cells from the target but
never accumulate twelve quiet physical ticks; their longest quiet span is three.
The right-48 regression lands its second hop at tick 212 and ends 4.508 cells
away. The left-60 failure relands at 182, has a fresh supported no-progress
window by 204 and the native blocked heuristic by 228, ending 6.082 cells away.
Both the two-pulse cap and late pulse bounds prevent another launch. The
left-63 and drop-12-right failures reland at 178/175, resume supported movement,
and end 7.428/7.302 cells away. These receipts identify deadline and remaining
ground-progress weaknesses; they do not expose a passive contact-force cause.

In all fourteen right-floor cases, the retained sequence begins supported launch
anticipation at local tick 12, loses raw support by native tick 14 / sampled tick
18, cancels that pending launch and waits for a new unsupported-edge stall window.
Recovery anticipation starts at 36 and its pulse is sourced at 42 / applied at 43.
First toward motion is 0.716667 seconds in each of those cases. This observation
ties the delay to the contact/anticipation sequence, while leaving its physical
force cause unmeasured. It provides a focused correction target beyond inference.
The [independent behavior audit](jolt-recovery-v2-independent-behavior-audit-20261020.json)
retains every failed case's native support, launch, landing, progress and quiet
intervals, alongside exact replay and the unchanged acceptance verdict.

The seven-pair v1 prefix speed ratio, 0.58924, is preserved as a partial finding.
The complete v2 ratio passes the prospectively fixed vigor gate. It does not
waive failed arrival/rescue/regression checks. Lower motion response speed and
longer completion remain useful diagnostic tradeoffs despite passing that gate.

## Repaired lifecycle case

At the failed v1 boundary in `demo-floor-right-36-1`, quiet samples at ticks
222/228/234 hid a speed violation at native tick 223. V1 retired its lease at
234 with only eleven continuous quiet ticks and no subsequent physical receipt.

The new run requests its advisory hold at 234 with a valid six-tick remaining
lease. The unchanged task observes success at **tick 235**; the admitted packet
finishes at tick 240. All forty local frames now bind to forty physical steps.
This new receipt demonstrates the correction in this invocation. It does not
claim tick 235 existed in the original failed attempt.

## Durable evidence and cost

The full raw journal has **85,438,519 bytes**, 5,616 linked records and SHA-256
`fd7873ada1f280f4c35595b0105d98ef6f9051853c0f90dae58ac908a6bdb1f7`.
Its 8,684,383-byte compressed copy has SHA-256
`4c7adfb5da0de90d80bf7760886e9c361864c2b38977e6e4a5b4d6232f523dd6`.
The complete seal binds the raw/compressed bytes, journal chain, all native
exchanges and episode receipts. The post-exit verifier confirms all 196 indexed
artifacts, all 14,694 physical ticks, every episode and final native close.

| Actual accounting | Preserved v1 | New v2 | Both increments |
| --- | ---: | ---: | ---: |
| Host starts / terminations | 1 / 1 | 1 / 1 | 2 / 2 |
| Episode rows | 24 (one incomplete) | 96 complete | 120 |
| Dispatched six-tick decisions | 479 | 2,449 | 2,928 |
| Requested / confirmed ticks | 2,874 / 2,874 | 14,694 / 14,694 | 17,568 / 17,568 |
| Reserved / answered owner calls | 549 / 549 | 2,711 / 2,711 | 3,260 / 3,260 |
| Whole-process seconds | 61.76433001999976 | 444.46566076900126 | 506.229990789001 |
| Optimizer / checkpoint / learned-policy executions | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |

V2 confirms no pending requests, uncertain tail, failed admission, controller
refusal or administrative cutoff. The worker closes the native host at
206.3609206380006 internal seconds and finishes sealing at 327.0383383589997.
The reported 444.46566076900126 total also includes startup and independent
post-exit verification. Worker and owned process group are confirmed terminated;
the 900-second limit was neither extended nor reused.

## Archive delivery verification

The [artifact index](jolt-recovery-v2-index-20261020.json) binds all 206 archived
files: 197 original output files, five closed process/verification receipts and
four derived reviews. The complete archive has **27,852,949 bytes**, SHA-256
`7325c1674a9007e7b2c8becb5abc2e37e2d5183c5c899eda4b4be37e88ce2a53`.
A fresh persisted readback matches that identity and every indexed member.
All original source-artifact bytes remain unchanged.

Two earlier direct transfers reported success but left shorter archives of
17,213,270 and 19,216,860 bytes. They ended after 203 and 204 complete members,
respectively, with compressed-stream EOF errors. Both failures, hashes and
retained versions are recorded in the separate
[delivery audit](jolt-recovery-v2-delivery-audit-20261020.json).
Building and verifying a temporary archive before atomic installation, followed
by guarded replacement, produced the verified complete readback. The mutating
layer remains undetermined; this observation does not explain the older journal
truncation. Delivery repair consumed no physical or training allocation.

## Review, publication and next work

The integrated offline Jolt suite passed 127 tests, including the unchanged
process-group cleanup assertion with consistent process visibility. Independent
source/declaration review preceded execution. All five hosted checks passed on
the frozen physical source. Integration with accepted main `a265ca79` preserves
the independently implemented IK contracts; all 163 combined Jolt and IK tests
pass in 20.153 seconds. Final publication uses its own final-head checks and
merge verification, with the corresponding receipts recorded in PR #61.

The evidence result and the teacher result have separate acceptance. A complete
durable comparison can satisfy EVIDENCE-001 while failed behavior leaves
JOLT-RECOVERY-001 open. The original truncation cause remains undetermined.
No authored result meets the learned reliability prerequisite by itself.

The [independent evidence audit](jolt-recovery-v2-independent-evidence-audit-20261020.json)
accepts the new diagnostic's EVIDENCE-001 qualification. It binds all 93 frozen
source inputs, every native/episode source and receipt, 892 twelve-tick and eight
six-tick candidate leases, 69 native cancellations, 28 terminal episodes and
close serial 2711. Cancellation and terminal counts can overlap. The closed writer
records 5,616 per-record fsyncs plus final synchronization and a directory fsync.
Two report-only assumptions were corrected after evidence verification: parsing
the final literal `completed` as JSON, and equating cancellation with the actor's
`suspended` flag. The audit records both; the native revocation proof and all
original evidence remain intact. The full verifier was not repeated for reporting.

[LEARN-JOLT-002](../design/LEARN_JOLT_002.md) and its
[implementation checklist](../design/LEARN_JOLT_002_IMPLEMENTATION.md) remain
conditional on an accepted teacher. The next controller correction is tracked in
[JOLT-RECOVERY-003 / #64](https://github.com/techrote/neural-cyber-automata/issues/64). It must resolve
the arrival regressions and unsupported/drop failure while retaining the now
measured airborne, vigor and latency bounds. This increment performs no automatic
tuning loop, new architecture search, training or morphology transfer.
