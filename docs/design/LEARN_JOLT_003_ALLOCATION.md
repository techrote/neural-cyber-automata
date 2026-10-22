---
title: LEARN-JOLT-003 allocation proposal and review
kind: guide
summary: Calculated one-attempt CPU budget and explicit admission conditions for a separately versioned learned Jolt comparison; no execution commission.
owns: [learning.jolt003-allocation-proposal]
verified: 2026-10-22
---

# LEARN-JOLT-003 allocation proposal and review

**Recommend one fresh 7,200-second whole-process CPU allocation**, with the
original fixed training and evaluation workload and 1,200 seconds reserved after
ordinary work stops. This is a reviewed planning proposal in response to the
owner's request to calculate an allocation and prepare the next project prompt.
It neither starts nor admits an experiment. The [next-step prompt](LEARN_JOLT_003_PROMPT.md)
requires a later owner commission and final executable-source/runtime admission.

The inspected base is PR #70 merge
`f23401eb793f9708040c4047a8146ab01ec3befc`, tree
`345549edb928a72c0478cd626a0a154098af85b5`. The original
[LEARN-JOLT-002 failure](../evidence/2026-10-21-jolt-learn-002.md), its consumed
declaration and all historical evidence remain unchanged. Issue
[#62](https://github.com/techrote/neural-cyber-automata/issues/62) remains open;
learned qualification, owner feel and morphology #16 remain unaccepted.

## Reproducible calculation

The standard-library [calculator](../evidence/jolt-learn-003-allocation-calculator-20261022.py)
reproduces the [complete proposal JSON](../evidence/jolt-learn-003-allocation-plan-20261022.json),
including all 39 operation ceilings, input hashes, formulas and timing limits.
It reads existing configuration and retained cost metadata without importing a
learning framework, loading a checkpoint, starting a host or invoking a model.

```sh
python docs/evidence/jolt-learn-003-allocation-calculator-20261022.py
```

The JSON has a proposal-only schema, `offline_scope_only: true` and
`prospective_experiment_admitted: false`. It is not an executable declaration.

## Fixed experiment and physical budget

Keep the 32-wide GRU, five continuous outputs and two buttons, policy seeds
190103/190207/190301, BC epoch 40, PPO eight updates of 256 decisions per seed,
four PPO epochs, fixed physical windows of length 16, six ticks per decision and
40 decisions/240 ticks per physical invocation. Use one CPU worker, one persistent
accepted host and one Torch thread. No GPU, purchased compute or paid service.

| Stage | Episode-start ceiling | Decision ceiling | Requested ticks |
| --- | ---: | ---: | ---: |
| Teacher: 64 episodes × 40 | 64 | 2,560 | 15,360 |
| PPO: 3 seeds × 8 updates × 256 | 6,144 | 6,144 | 36,864 |
| Open: 4 cases × 8 controllers × 40 | 32 | 1,280 | 7,680 |
| Sealed: 16 cases × 14 controllers × 40 | 224 | 8,960 | 53,760 |
| **Effective total** | **6,464** | **18,944** | **113,664** |

Open controllers are authored, idle and six learned finals. Sealed controllers
are authored, idle and all twelve learned initial/final models. Freeze all six
final hashes before **any** sealed outcome, including authored and idle.
The inherited 20,000-decision/120,000-tick aggregate backstops grant no extra work
above the stage caps. Refused or uncertain requests remain charged.

PPO's episode ceiling assumes at least one confirmed retained decision per
episode. Enforce at most 256 starts per update and 6,144 overall; abort on a
zero-decision capture. A safe failure consumes its own case-stream entry and
its actual decisions. It does not earn a replacement episode or reset a counter.

Owner operations are bounded by
`18,944 step + 6,464 reset + 6,464 profile + 6,465 cancel + 1 close = 38,338`.
There is one extra cancellation allowance for failed cleanup, with no
unbounded cancellation loop. Separate configure/observe operations remain zero.

## Time allocation and evidence for its size

All deadlines are offsets from the **same original outer-process clock**.

| Interval | Seconds | Absolute cutoff |
| --- | ---: | ---: |
| Startup, admission recheck, collection, fitting, checkpoint publication and all physical comparisons | 6,000 | T + 6,000 |
| Worker native closure, journal close, complete neural readback, final artifacts and worker cleanup | 900 | T + 6,900 |
| External inventory/cost verification, verifier cleanup and log synchronization | 240 | T + 7,140 |
| Final executor receipts, original stdout and independently observed exit | 60 | T + 7,200 |
| **Whole-process maximum** | **7,200** | **T + 7,200** |

The worker cleanup reserve is **1,200**, the executor verification reserve is
**300**, and the finalization reserve is **60** seconds. These reserves overlap
by definition: usable worker tail is `1,200 − 300 = 900`; verifier interval is
`300 − 60 = 240`. They do not add time to 7,200. Within the 900-second worker tail,
plan 180 seconds for native closure/journal sealing and pre-audit publication,
600 for the complete independent readback, and 120 for final publication and
worker exit. The final source must enforce the original absolute deadlines and
stop admitting work before an in-flight operation can consume a reserved phase.

The [retained costs](../evidence/jolt-learn-002-costs-20261021.json) report
963.081273 internal seconds for 7,509 decisions, BC and almost all PPO, with no
evaluation rows or neural audit. That is **not** a verified whole-process time.
Applying its mixed-work rate to 18,944 decisions gives 2,429.699 seconds.
Separately, recovery v4 measured 544.956099 whole-process seconds for 2,264
decisions; its different-workload rate gives 4,559.915 seconds for 18,944.
These are alternative sensitivity estimates, not additive measurements or
completion guarantees. The 6,000-second active interval is 2.469 times the first
proxy and 1,440.085 seconds above the second.

The [forensic review](../evidence/jolt-learn-002-forensic-review-20261021.json)
places the last original reservation at 915.835715 seconds and ledger closure
at 963.081273: a 47.245558-second tail even though independent readback rejected
the journal before any checkpoint load or audit forward. The complete new
journal, neural readback and external verification costs remain unmeasured.
The new ceiling is a conservative planning choice; final admission must assess
its feasibility for the pinned source, hardware and serialization shape.
A capped attempt remains capped, with no automatic extension or retry.

## Learning, inference and artifact operation bounds

| Quantity | Proposed maximum | Derivation |
| --- | ---: | --- |
| BC optimizer calls | 23,040 | 3 × 40 × 64 × ceil(40/16) |
| BC ordinary forwards / eligible samples, each | 307,200 | 3 × 40 × 64 × 40 |
| BC prefix forwards | 368,640 | 3 × 40 × 64 × (0 + 16 + 32) |
| PPO optimizer calls | 24,576 | 3 × 8 × 4 × 256, allowing one-row episodes |
| PPO ordinary forwards, each module | 24,576 | 3 × 8 × 4 × 256 |
| PPO prefix forwards, each module | 33,792 | 352 × 3 × 8 × 4 |
| PPO authentication forwards, each module | 6,144 | 3 × 8 × 256 |
| Runtime module forwards | 25,248 | 12,288 PPO + 1,440 open + 11,520 sealed |
| Bootstrap forwards | 12,288 | Conservative two-module allowance per PPO episode |
| Independent audit forwards | 37,888 | Retained backstop; 25,248 + 12,288 = 37,536 |
| Charged learned controller invocations | 21,144 | 14,784 retained-row ceiling + 6,360 retirement attempts |
| Checkpoint writes / loads | 36 / 48 | 36 publications; 12 evaluation-cache + 36 audit loads |

For an episode of length L, prefix cost per epoch is
`b(L) = sum(range(0, L, 16))`: zero for L≤16, 16 for L=17…32, 48 for L=33…40.
Integer dynamic programming over every partition of 256 rows into episodes of
length 1…40 gives maximum 352. A witness is seven 33-row episodes and one 25-row
episode: `7 × 48 + 16 = 352`. Masks and skipped windows only reduce this cost.
The old 958,464-per-module PPO prefix ceiling can therefore be tightened to
33,792 without changing training. Packing every rollout into sixteen windows
would instead undercount fragmented episodes and is not allowed.

The original 15,744 controller-invocation ceiling was not the exact learned-row
rectangle: `6,144 + 4 × 6 × 40 + 16 × 12 × 40 = 14,784`. A retirement can charge
one attempted invocation without a physical row or model forward. The new cap
adds at most `6,144 + 24 + 192 = 6,360` such attempts. The auditor must reconstruct
these attempts independently instead of inferring every charge from a row.

Total separate operation maxima are **47,616 optimizer calls** and **880,288
module forwards**, including prefix/authentication/bootstrap/audit work. These
maxima need not occur simultaneously. Existing control flow suggests tighter
bootstrap/audit caps of 48/25,296; retain the wider finite backstops until the
actual new source proves the administrative-cutoff/retirement precedence.

## Proposed safe-failure semantics

Only `emergency_recovery_deadline` is eligible. Independently replay the original
48-tick emergency limit and final observation, prove the same lifetime and
native cancellation generation/state/cleared authority, and durably retain the
complete episode before advancing. The next invocation has a new identity from
the frozen schedule; the failed invocation never receives renewed time or action.
Unknown reasons, lost authority, invalid/non-finite state, native refusal,
recording failure, cancellation failure or budget exhaustion stop the allocation.

Represent the retirement as a complete **unsuccessful** outcome with an explicit
reason and cancellation evidence. Do not put a verified expected outcome into
the integrity-error field, synthesize a physical transition or relabel it as
arrival. Keep `success=false`, full-timeout capped completion cost of four seconds,
all original character/latency inputs and the complete paired denominator.
Unavailable metrics cannot become zero or passing. Episode completion counters
mean receipt-complete captures, including PPO administrative cutoffs; they do
not mean task success. Reconcile started/completed/aborted counts without
manufacturing records after an integrity failure.

Teacher failures keep all supported ordinary labels, including abstention;
overrides remain masked and outcome-based filtering is forbidden. PPO keeps
actual confirmed decisions, unchanged rewards, actor/critic masks and GAE gaps.
Use explicit terminal failure with zero bootstrap and no new value observation
or forward. Terminal failure takes precedence if detected during cutoff peek;
do not claim both a terminal and an ordinary bootstrap cutoff. No invented
penalty propagates through the existing masked emergency gap. This change
permits honest completion of a comparison; it does not establish better learning.

## Admission findings and remaining gates

Three independent reviews cover [operations](../evidence/jolt-learn-003-operations-review-20261022.json),
[time and storage](../evidence/jolt-learn-003-feasibility-review-20261022.json) and
[acceptance](../evidence/jolt-learn-003-acceptance-review-20261022.json).
Their scope is this allocation/design proposal. They cannot substitute for the
two or more independent reviews of the final executable source required later.

The main findings incorporated into the proposal are bounded positive progress,
charged no-forward retirement attempts, exact prefix arithmetic, explicit
terminal/cutoff precedence, intact loss masks and paired failure denominators,
and nonoverlapping time intervals. Existing v1/v2 code still rejects these new
semantics and retains hardcoded 1,800/120-second limits in admission, execution
and cost verification. Changing JSON alone cannot admit a larger allocation.

**Serialization and memory feasibility remain a mandatory pre-execution gate.**
The sum of the 39 nonnegative counter ceilings bounds reservation records at
1,901,556. At the historical average 114.681705 bytes/record, each reservation
artifact projects to 218,073,684 bytes (207.971 MiB), versus its 512 MiB reader.
An average is not a maximum: the prospective source must prove or enforce its
serialized record and total bounds, including terminal metadata. A 256-byte
per-record proof would yield 464.247 MiB before the small envelope, but is not
yet established. Ordinary JSON/sample readers have 64 MiB limits; raw episodes
have 32 MiB limits. Parsed objects and copying can use much more RAM than files.

The current event inventory plus one new retirement note per episode and one
allocation-stop note bounds journal records at 115,110; the 1 MiB record limit
alone permits over 112 GiB, so it is not a useful available-disk guarantee.
The calculated successful inventory allows 19,468 worker files plus 16 executor
files, before separately retained original outer stdout/exit. Recompute those
counts for actual new metadata. Resolve aggregate bytes, peak memory, available
disk, bounded readers and tail-time feasibility before admission; use separately
versioned bounded streaming if necessary. Do not loosen readers without a bound.

Final admission also binds the owner commission, clean source commit and full
closure, host/profile/runtime identities, fresh sealed split and opened-case
inventory, unchanged acceptance, all counters and actual phase enforcement.
No physical or neural experiment, checkpoint load, benchmark or host change
was performed to prepare this allocation. The proposed next task supplies the
implementation and admission work before any separately commissioned execution.

© 2026 techrote. All rights reserved.
