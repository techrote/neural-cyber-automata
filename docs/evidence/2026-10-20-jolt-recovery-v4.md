---
title: Jolt recovery v4 — accepted teacher and complete single-run evidence
kind: evidence
summary: The sole commissioned diagnostic passes all thirteen unchanged teacher checks with 30/32 arrivals, complete original evidence and two retained leftward failures.
owns: [evidence.jolt-recovery-v4]
verified: 2026-10-21
---

# Jolt recovery v4

The single JOLT-RECOVERY-004 diagnostic **passes the unchanged teacher gate**:
30/32 arrivals versus 27/32 for the contemporaneous arrival reference, three
historical rescues, zero arrival regressions and 32/32 purposeful-airborne cases.
The original final verifier, supervisor and external executor completed
successfully, with native closure and **544.9560992200022 whole-process seconds**.
The [original independent review](jolt-recovery-v4-review-20261020.json)
reproduces all 2,264 controller decisions and 96 native task/character outcomes.

This satisfies the authored-teacher dependency for
[LEARN-JOLT-002 / #62](https://github.com/techrote/neural-cyber-automata/issues/62).
It does not establish learned skill, sealed evaluation, morphology qualification
or owner acceptance of feel. Collection/training still require the separate
versioned implementation, independent review, fresh split and prospective
execution declaration in the [learning plan](../design/LEARN_JOLT_002.md) and
[implementation checklist](../design/LEARN_JOLT_002_IMPLEMENTATION.md).
The diagnostic allocation is consumed; no tuning, retry or extension followed.

## Unchanged acceptance

The [original result](jolt-recovery-v4-result-20261020.json) and independent
review retain the original arrival predicate: raw support, Euclidean distance
at most four cells, speed at most six cells/s and absolute angle at most 0.32
radians for twelve consecutive native ticks. The 240-tick deadline and at most
two candidate primary edges remain unchanged. All thirteen behavioral checks
pass, together with complete original evidence and zero authority, lifecycle,
integrity or admission defects.

| Criterion | Arrival reference | Recovery v4 | Unchanged gate |
| --- | ---: | ---: | --- |
| Arrivals | 27/32 | **30/32** | Strict improvement |
| Historical rescues | — | **3/5** | At least three; required drop and supported case |
| Arrival regressions | — | **0** | At most one |
| Purposeful-airborne cases | 25/32 | **32/32** | Retain overall count and restore a named missing-air case |
| Mean ordinary trim reversals/s | 0.490190 | **0.258764** | Increase at most 0.5/s |
| Mean absolute angle, radians | 0.026368 | **0.026197** | Increase at most 0.1 |
| Mean peak toward speed, cells/s | 53.974276 | **42.979247** | Ratio **0.796291**, within 0.7–1.5 |
| Worst case inference p95, ms | 2.924240 | **3.029225** | Every case at most 5 |
| Maximum requested primary edges | 1 | **2** | Fresh physical justification and admission for every edge |

The three historical rescues are `demo-floor-left-60-2`,
`reference-drop12-right-45` and `reference-drop28-left-57`. The required unsupported
drop and a supported failure are included. All seven named missing-airborne
reference cases are restored; v2/v3's 32/32 airborne count is retained.
The original controller remains 17/32; v4 has 23/24 training-origin and 7/8
evaluation-origin arrivals. All 32 cases were already open development cases.

Zero arrival regressions is a success-status comparison, not a claim that every
trajectory improved. Both left60-1 and reference-left63 still fail, with the
latter substantially farther from the target. Mean capped completion time is
2.218229 seconds versus 2.085417 for arrival; higher arrival count does not imply
faster mean completion under this metric. The 30/32 aspiration did not replace
any acceptance requirement.

## Frozen source and allocation

The owner commissioned [the fourth-run recommendation](https://github.com/techrote/neural-cyber-automata/issues/64#issuecomment-6300675643);
[D-017](../DECISIONS.md#d-017--complete-the-separately-commissioned-fourth-jolt-recovery-diagnostic)
records that fresh scope. Accepted main at entry was
`d7d4abe49c537426ef6c6aed0be60de4cdd8b70e`. Earlier allocations stayed closed.

[The prospective publication](https://github.com/techrote/neural-cyber-automata/issues/64#issuecomment-6301588932)
contains the complete declaration, exact source/tree, five passing source checks,
historical ledger and complete independent admission **before host start**.
The [original admission](jolt-recovery-v4-admission-20261020.json) verifies the
clean detached checkout, 115 source/test/configuration inputs and pinned host.
Later documentation/evidence commits do not become the physical run's source.

| Frozen identity | Value |
| --- | --- |
| Source commit | 46a22f98dce78be3c4ba534d955e58d5691c7d96 |
| Source tree | 90e22caf259228083308c1ce23e047c3c27df63e |
| Declaration SHA-256 | bc3cf6843d670a9305fa2f70bfa7531c865fa05363f3d55c88f8c554e520d30c |
| Controller SHA-256 | d7903561e37bc0845b3993d3b608ff83200021084e4d41e6f8ab19dff38edeec |
| Candidate specification SHA-256 | 2a04924885e3147881fef5ed3421c7c0e43457faa83e5c13b8e3c351c0384038 |
| External executor SHA-256 | c8a4f0699f45a67d3cc287c53aa6cbcbd4c0ed553dfde4c3b53e7f66c750ef85 |
| Admission SHA-256 | 015e750b0bfbeeca35b3b364d1c77e75f284aef7cf7c580aca853b3aab57c2b4 |
| Unchanged host commit | f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497 |
| Unchanged host tree | 445a185e9f9824b16953fec27bf0157dfeedca77 |

The [manifest](jolt-recovery-v4-manifest-20261020.json) retains exact case IDs,
seeds, profiles, observation/action schemas, cadence, environment and byte hashes
for Godot, native/Rapier artifacts, eleven host scripts and 37 native inputs.
No host, profile, task, gyro, original deadline or teacher threshold changed.

| Resource | Fixed ceiling | Actual |
| --- | ---: | ---: |
| Attempts / persistent host starts | 1 / 1 | 1 / 1 |
| Fresh episodes, 32 cases × three families | 96 | 96 |
| Decisions / requested ticks | 4,000 / 24,000 | 2,264 / 13,584 |
| Effective per-episode-derived total | 3,840 / 23,040 | Within both limits |
| Per-episode decisions / ticks | 40 / 240 | At most 40 / 240 |
| Owner calls, including lifecycle/cleanup | 4,500 | 2,531 |
| Whole-process seconds | 900 | 544.9560992200022 |
| Shutdown/sealing/verification reserve | 120 within 900 | Preserved |
| Optimizer updates / checkpoints / learned calls | 0 / 0 / 0 | 0 / 0 / 0 |
| Automatic retries / extensions | 0 / 0 | 0 / 0 |

One original monotonic start propagates through executor, supervisor and worker.
Normal calls reserve fifteen seconds for the complete exchange; latest admission
is elapsed 765 seconds and normal work ends by 780. Worker termination must occur
by 840, leaving at least sixty seconds, including six for finalization, before
the original ceiling. Actual last normal response was elapsed 257.324383 seconds.
The external measurement starts before supervisor spawn and ends after its actual
exit and confirmation that the owned process tree is gone. Subsequent executor
receipt publication and later offline reviews are separate accounting.

## Implementation and measured motion

The separately registered `authored-jolt-recovery@4` preserves v3's initial
contact-transition and pulse-vector methods and all earlier controller versions.
Its ordinary controller adds bounded interior settling: current raw support
inside caller tolerance requests slow inward traction, with counter-traction
for excess speed and hysteresis against small boundary crossings. Interior
neutral requires distance at most 0.75 times tolerance, speed at most two cells/s
and the unchanged posture bound. Support uncertainty, gaps and authority or
emergency boundaries invalidate capture. The caller still owns actual completion.

Exactly six ticks after a second pulse, a consecutive raw-supported observation
with cooldown acknowledgment and upward speed at least six cells/s can request
one six-tick traction coast. The witness and lease expire together. This adds no
primary edge or supervisor movement. The [controller reference](../CONTROLLERS.md#interior-settling-in-recovery-v4)
defines the contract and preserves the tick-223 aliasing regression.

The [retained prepass](jolt-recovery-v4-retained-prepass-20261020.json) reproduced
754 original v3 decisions before execution; v4's probes changed 123 requests.
Retained observations after a changed action were never treated as a physical
counterfactual. Actual v4 replay verifies 715 candidate decisions, including
70 capture requests, five post-pulse coasts and three ground-correction decisions
across two cases. No interior neutral quiet-hold decision occurred, so that
branch has synthetic coverage rather than physical validation in this run.

| Target case | V3 longest quiet span | Actual v4 outcome | Final distance, v3 → v4 |
| --- | ---: | --- | ---: |
| demo-floor-right-36-1 | 9 ticks | Quiet 151–162; arrival at 162 (2.700 s) | 3.765681 → 3.069391 |
| reference-drop28-left-57 | 10 ticks | Quiet 153–168; arrival at 164 (2.733 s) | 3.898402 → 3.529405 |
| demo-floor-left-60-1 | 0 ticks | No quiet tick; deadline at 240 | 5.055274 → 4.939332 |
| reference-floor-left-63 | 0 ticks | No quiet tick; deadline at 240 | 8.131480 → 12.503336 |

The [four-case native comparison](jolt-recovery-v4-case-comparison-20261020.json)
retains every continuous span and final speed. Drop28-left57 stays quiet through
the four remaining native ticks of its already-issued six-tick action.
All 32 first-pulse source ticks and first-toward-motion latencies match v3.
Fourteen initial right-floor transitions retain the 0.316667-second response.

All 37 candidate primary pulses have confirmed upward native actuation, including
five freshly justified second edges. The left60-1 second pulse lands at 168,
0.717163 cells horizontally farther from the target; later supported progress
is 1.259933 cells, producing only a 0.115943-cell final improvement. Reference-left63
lands at 179 instead of 163, after 22 rather than six airborne ticks; its landing
is 3.805695 cells farther away, followed by only 1.469040 cells of supported
progress. These traces do not establish consistent useful progress from the coast.

Emergency assistance comprises 19 supervisor intervention decisions in five
cases: three successful and the two remaining failed cases. Per-case native
impulses, airborne time, latency, ordinary recovery and chatter remain in the
result and independent review. Aggregate actuator impulses and observed motion
do not identify passive/contact-force causes. No causal coast benefit or
unmeasured perturbation robustness is claimed.

## Original evidence and independent review

New exclusive final-path publication synchronizes, closes and reads back JSON
and gzip bytes before sealing. Strict inventory rejects missing/surplus files,
links, special entries and changed bytes. Partial publication stays failed;
neither software nor review repairs an original verdict. The durable journal
and minimal physical supervisor remain preserved; v4 adds explicit transport,
cleanup and outer execution accounting.

The original [supervisor](jolt-recovery-v4-supervisor-20261020.json),
[finalization marker](jolt-recovery-v4-finalization-20261020.json) and
[executor](jolt-recovery-v4-executor-20261020.json) record `completed=true`.
The supervisor and executor record return code zero and no timeout.
The verifier accepts 196 indexed artifacts; including the summary, the exact
inner inventory is **197 files**. Nine original outer receipt/log files remain.
The 5,256-record owner chain binds 2,531 reservations to 2,531 responses:
2,264 steps, 96 resets, 96 profiles, 74 cancels and one close.
All 13,584 requested ticks are confirmed, with no pending tail.

The [original evidence/cost audit](jolt-recovery-v4-evidence-review-20261020.json)
checks raw/gzip equality, complete gzip EOF, record chains, source, original
hashes, native closure and nested wall intervals. Worker sealing completed at
elapsed 429.917187 seconds; original verification completed by the supervisor's
prepublication measure of 544.893780. Durable primary receipt publication measured
544.898435 and external observed exit 544.956099 seconds, all within 900.

The separate full behavioral review initially stopped on an auditor schema error:
it indexed optional `mount_generation` as mandatory after replaying all 96 rows.
The [correction record](jolt-recovery-v4-auditor-correction-20261020.json) preserves
the failed attempt and frozen script hash. A separately hashed derivative changes
only that optional-field access and records provenance. Required identities,
physical evidence, source, metrics, gates and the passing original in-window
verifier remain unchanged. Its complete read-only replay passes.

The [v3 failure](2026-10-20-jolt-recovery-v3.md) remains failed: its original
`completed=false` receipt and undeclared duplicate pending gzip are preserved,
with the historical file's origin still unestablished. V4 does not retroactively
qualify v3, v2, v1 or the truncated original-arrival journal.

## Costs and retained artifacts

| Invocation | Hosts | Rows | Decisions | Episode-confirmed ticks | Owner calls | Whole-process seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original arrival | 1 | 64 | 1,549 | 9,294 | 1,722 reported | 221.630556 |
| Recovery v1 | 1 | 24 | 479 | 2,874 | 549 | 61.764330 |
| Recovery v2 | 1 | 96 | 2,449 | 14,694 | 2,711 | 444.465661 |
| Recovery v3 | 1 | 96 | 2,303 | 13,818 | 2,568 | 294.951525 |
| Recovery v4 | 1 | 96 | 2,264 | 13,584 | 2,531 verified | 544.956099 |
| Cumulative | **5** | **376** | **9,044** | **54,264** | **10,081 reported** | **1,567.768171** |

The [historical ledger](jolt-recovery-v4-historical-costs-20261020.json) preserves
each source, original result/receipt hash and failure. Original arrival calls
remain reported because its journal ends at unanswered request 843; cumulative
calls retain that qualification. V1's 24 rows include an incomplete episode.
Cumulative wall time uses original whole-process intervals, including v4's
actual exit, and excludes later offline review, publication and CI conformance.

The unchanged archive `jolt-recovery-v4-evidence-20261020.tgz` contains
**243 regular files / 26,268,108 compressed bytes**, with SHA-256
**27afcae51f20a02fce7888b385574f92d0932dd4ab8aedd4393df0f6e2454e44**.
Large raw evidence stays outside Git. The [original member manifest](jolt-recovery-v4-archive-members-20261020.json)
binds each file; its explicit self-hash is externally bound as
`56fd6563543a5aec63ffc5b59155e4255867422aedd6285267f7a2a398b49a66`.

The [progress checkpoint](https://github.com/techrote/neural-cyber-automata/pull/66#issuecomment-6322984047)
records the interrupted publication and subsequent removal of the unpublished
checkout by workspace maintenance. The retained archive was recovered unchanged
on 21 October. Original results and independent reviews are copied byte-for-byte
from it; the unpublished local completion commit is not treated as a public
revision or as the physical source.

The new [archive recovery verification](jolt-recovery-v4-archive-recovery-verification-20261021.json)
and [acceptance reconciliation](jolt-recovery-v4-acceptance-reconciliation-20261021.json)
are separately dated read-only reviews of those restored bytes. The
[publication index](jolt-recovery-v4-index-20261021.json) distinguishes original
archive members from later recovery/publication records. Recovery introduces
no host starts, owner calls, physical ticks, optimizer updates or changed evidence.

## Software validation and integration

The frozen offline suite ran 682 tests: 521 passed and 161 optional/physical cases
were expected skips without a game host. All 57 new tests passed: eleven controller,
thirteen publication, fifteen diagnostic, four supervisor and fourteen executor
cases. The archive preserves the original test log. These checks cover authority,
expiry, aliasing, strict inventory, partial writes, complete-exchange deadlines,
immutable failure receipts and process-tree cleanup.

The unexecuted first checkpoint `80ca7d265fa3dba9141b36ea201ae579136e1dc2` failed
Windows CPU because checkout changed previously unpinned LF bytes and synthetic
supervisor tests implicitly selected Windows. Narrow LF rules and explicit POSIX
modeling corrected those issues without skipping tests or changing production
behavior, declaration or acceptance.

Executed source `46a22f9` passed all five checks before admission:
[docs](https://github.com/techrote/neural-cyber-automata/actions/runs/39497919318/job/118887674862),
[Linux CPU](https://github.com/techrote/neural-cyber-automata/actions/runs/39497919375/job/118887674931),
[Windows CPU](https://github.com/techrote/neural-cyber-automata/actions/runs/39497919375/job/118887675490),
[actual host](https://github.com/techrote/neural-cyber-automata/actions/runs/39497919327/job/118887675084)
and [learning CPU](https://github.com/techrote/neural-cyber-automata/actions/runs/39497919319/job/118887675269).
[PR #66](https://github.com/techrote/neural-cyber-automata/pull/66) requires the
same five checks on the final publication head before merge, then verification
of main. That later head does not replace the physical source. Routine CI
conformance is separate from this diagnostic's allocation and cumulative costs.
