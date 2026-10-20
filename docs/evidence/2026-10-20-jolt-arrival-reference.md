---
title: Versioned Jolt arrival reference diagnostic
kind: evidence
summary: A separately declared authored teacher improvement and paired development diagnostic; learned morphology entry remains blocked.
owns: []
verified: 2026-10-20
---

# Versioned Jolt arrival reference

[Issue #43](https://github.com/techrote/neural-cyber-automata/issues/43) implements
a separately named authored reference for the remaining useful-skill dependency.
The current owner instruction to advance the repository commissions this bounded
follow-up. It leaves the original teacher, learned task/reward, failed prerequisite
and 75% mean / 50% every-seed reliability requirement unchanged.

## Observed failure and candidate

Read-only replay of the preserved primary direction-frame evidence found that
the original teacher succeeded in 13 of 24 training episodes and four of eight
evaluation episodes. Some failed episodes launch again while near the target
and already moving toward it. The accepted host provides Jolt horizontal
traction while supported; opposite movement input cannot brake a free-air hop.

The new `authored-jolt-arrival@1` reference permits at most one supported hop per
goal. Its fixed analytic pulse estimate accounts for flight travel, supported
stopping room and one decision interval of reaction. It then uses a bounded
ground position servo. The local arrival estimate requires measured support,
speed no greater than six cells/s, wrapped angle no greater than 0.32 radians,
and a conservative 12-tick observed quiet span. That remains a local estimate,
separate from the unchanged evaluator's per-base-tick completion predicate.

A synthetic check also reproduced unsupported/tilted completion in the original
teacher. Replaying all 32 preserved original authored episodes found that this
particular mismatch did not cause their recorded failures. It is not presented
as the explanation for their failed learned prerequisite.

## Frozen diagnostic scope

The [declaration](../../configs/baselines/jolt-arrival-diagnostic-v1.json) binds
the complete candidate specification, original source hashes, accepted host and
all 32 already-open case/seed pairs. One invocation compares both teachers:
at most 64 episodes, 2,560 six-tick decisions, 15,360 requested base ticks,
2,818 total owner requests and 600 seconds including startup and finalization.
Every request is reserved before transmission; an uncertain response retains
its budget and raw journal entry. There is no automatic retry or optimization.

The [runner](../../tools/jolt_arrival.py) requires an explicitly published clean
source revision and verifies every source/configuration blob before starting the
physical host. It retains complete episode records, owner exchanges, individual
task metrics, cancellations, failures and source/runtime identities. The original
teacher and candidate receive matched host seeds, goals, sensors, cadence and
physical assistance.

All these cases have already been opened. Any measured difference is a
development diagnostic; it cannot qualify a learned policy or open morphology
training. Owner feel and continuous realtime pacing are separately assessed.

## Recorded outcome and audit status

The new teacher reached quiet arrival in **27/32 cases**, versus **17/32** for
its contemporaneous original baseline. The complete native episode traces
support these descriptive counts. **The declared complete owner-journal
requirement is not met**, so this run is not accepted as a complete diagnostic.
The original runner-reported `completed` status is preserved unchanged alongside
the separate incomplete audit; no physical invocation was repeated.

The single physical invocation used public revision
`5d9d6a13d2f273ee1b4fd3ce0b50a6482c807382`, tree
`57b8a4e86db08ea3713a2205eb4aa5269d29309d`. Its exact 86-file source closure,
original declaration, candidate specification, accepted host scripts and native
artifacts passed independent identity checks. The original task/reward and
Jolt physical profile are unchanged. Canonical declaration SHA-256 is
`269cef7a852fe8d690bb95ab9187576a3babe0f1fa5d1e8ee28fafc2730c474a`;
candidate specification SHA-256 is
`4faeca424f3a2c87ed358eaa16053cab93d5c439b71b8dccc7ed41a1a8980ac5`.

| Quantity | Recorded result | Evidence scope |
| --- | ---: | --- |
| Episodes | 64 | All 64 episode and metric files are present and decode |
| Decisions / consumed ticks | 1,549 / 9,294 | Complete episode receipts, also reported in summary |
| Total owner requests | 1,722 reported | Full raw owner journal is missing; only its prefix is independently available |
| Internal wall time | 219.47143778100144 s | Original result counter |
| Complete supervisor wall time | 221.63055612800235 s | One normal worker exit, within 600 s |
| Host starts / terminations | 1 / 1 reported | Source, receipt and current process absence support closure; journal tail is missing |
| Training / checkpoint / optimizer work | 0 | Authored controllers only |

## Journal discrepancy and preservation

The original `owner-journal.jsonl.gz` has 2,481,084 bytes and SHA-256
`dd5a1236d8f50048c20d0a2a2fb090d2803a96cc2f260169662c382229ec7172`,
matching its original summary index. It nevertheless fails gzip completion: the
compressed stream stops after a sync-flush boundary, with neither a complete
DEFLATE stream nor the final gzip trailer. This is a substantive record cutoff,
not merely a missing checksum footer. Its cause is unestablished.

The recoverable prefix contains 1,759 complete JSONL records: 37 episode starts,
36 completions, 843 request reservations and 842 responses. It accounts for
740 requested six-tick intervals / 4,440 requested ticks, but only 739 confirmed
intervals / 4,434 ticks. Its last entry reserves request 843 at elapsed
94.78852914000163 seconds; that response and all subsequent exchanges are absent.
There is no raw closing-host record. These prefix counters do not replace the
larger original summary counters, and the missing exchanges are not reconstructed.

The archive retains every original file, the unchanged original summary and
supervisor, a separate independently readable compressed prefix, and a derived
audit. The prefix is labelled incomplete and has its own identity. The complete
64 episode traces support case-level outcomes, but they cannot supply missing
configure/reset/cancel/close exchanges or complete aggregate owner accounting.

The current runner replays and validates its closed owner journal before allowing
completion. Fifteen offline regressions pass, including truncated gzip, a valid-gzip
prefix, missing responses, counter/row disagreement and incomplete native closure.
The unchanged original journal is rejected as `arrival_journal_stream` with zero
host, model or optimizer work. This independently reviewed software correction is
separate from the frozen physical invocation above. Original source, data,
reported status, failures and hashes remain unchanged. Issue #43 retains the
unmet complete-journal acceptance requirement.

## Paired development outcomes

| Previously open cases | Original teacher | Arrival reference | Change |
| --- | ---: | ---: | ---: |
| Training-origin cases | 13/24 (54.17%) | 22/24 (91.67%) | +9 successes |
| Evaluation-origin cases | 4/8 (50.00%) | 5/8 (62.50%) | +1 success |
| All cases | 17/32 (53.13%) | 27/32 (84.38%) | +10 successes |

These are newly recorded paired baseline episodes. Their counts happen to match
the earlier historical teacher result; the two executions keep separate source,
episode and receipt identities. There are 13 rescued cases, three regressions
and two failures shared by both teachers.

| Mean across 32 cases per teacher | Original | Arrival reference |
| --- | ---: | ---: |
| Overshoot, cells | 19.0078 | 0.6063 |
| Final distance, cells | 9.9811 | 3.8244 |
| Capped completion, seconds | 2.7203 | 2.0854 |
| Peak speed toward target, cells/s | 69.7901 | 53.9743 |
| Absolute angle, radians | 0.0467 | 0.0264 |
| Ordinary trim reversals, per second | 0.1051 | 0.4902 |
| Primary edges | 2.0625 | 1.0000 |
| Purposeful airborne confirmation | 32/32 | 25/32 |

The new reference reduces mean overshoot by about 96.8% and consumes exactly one
primary edge per case. All 27 new successes end actually supported, with final
speed no greater than 4.274 cells/s, absolute angle no greater than 0.129 radians,
and distance no greater than 3.424 cells. Increased trim reversals and reduced
airborne behavior remain relevant gameplay tradeoffs; owner feel is unjudged.

Seven short leftward successes consume a pulse but have no unsupported physics
tick: all six training-origin left-24/left-36 cases and the open-evaluation
left-27 case. They are not absent-pulse cases. Purposeful confirmation changes
from 24/24 to 18/24 on training and from 8/8 to 7/8 on open evaluation; the
aggregate loss alone does not prove a violation of an evaluation-only bound.

## Remaining failure classes

All five new-reference failures miss the unchanged task deadline, have zero
overshoot and contain one purposeful pulse.

| Case | Paired outcome | Final distance, cells | Observed pattern |
| --- | --- | ---: | --- |
| `demo-floor-left-60-1` | Regression | 5.449 | Supported, blocked approach stall |
| `demo-floor-left-60-2` | Regression | 4.364 | Supported, blocked approach stall |
| `reference-floor-left-63` | Shared failure | 7.828 | Supported, blocked approach stall |
| `reference-drop28-left-57` | Shared failure | 4.200 | Supported, blocked approach stall |
| `reference-drop12-right-45` | Regression | 35.008 | Prolonged unsupported configuration after the first pulse |

For the four supported stalls, the final eight decision observations report
both grounded and blocked. Continued leftward requests make only 0.04–0.24 cells
of net progress; this does not by itself identify a particular friction or
contact cause. The drop regression consumes its pulse at tick 30, then reports
unsupported at each decision from tick 42 through 234, with zero horizontal
request. It is not a repeated-pulse overshoot failure.

The next useful control investigation is separately declared contact/progress
recovery, retaining these two failure classes and the short-pulse tradeoff.
This version remains a development reference: open evaluation is only 5/8,
there are no learned policy seeds, the owner journal is incomplete, and the
fixed learned 75% mean / 50% every-seed prerequisite remains unmet. Morphology
learning/transfer, a new learned qualification and owner-feel acceptance remain
unadmitted.

## Software verification and evidence files

Before the physical invocation, independent review covered controller behavior,
source admission, native identity/terminal checks and entire-process cleanup.
The final pre-run software suite ran 449 tests in 33.998 seconds: 398 passed
and 51 expected host/artifact checks skipped. The initial Windows failure was traced exactly to LF-to-CRLF
conversion of the pinned original baseline. An explicit `.gitattributes` entry
preserves its original LF bytes without relaxing the strict source pin; all
five hosted jobs then passed at the frozen head. After the journal correction and
integration of the runtime/CI follow-ups, the software suite ran 482 tests in
46.827 seconds: 425 passed and 57 expected host/artifact checks skipped. Strict
documentation validation passed 58 files without errors or warnings.

The [original result](jolt-arrival-reference-result-20261020.json),
[original supervisor](jolt-arrival-reference-supervisor-20261020.json),
[independent audit](jolt-arrival-reference-audit-20261020.json) and
[archive index](jolt-arrival-reference-index-20261020.json) preserve their separate
claims and identities. The index identifies the owner-delivered 11,501,074-byte archive with 138 members
(134 original and four derived), SHA-256
`55d8be5e14ca7aeeeb42bdbf7d2e58d6e970806017e19d9fedd965c8e68ed4d4`.
Every archive member was verified against its source bytes; the original journal
remains explicitly incomplete inside that valid archive.
No public path or reconstructed journal is substituted for missing raw evidence.
