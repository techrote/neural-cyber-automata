---
title: LEARN-JOLT-003 implementation contracts
kind: reference
summary: Separately versioned safe-retirement, learning, admission and evidence contracts for the conditional fresh Jolt comparison.
owns: [learning.jolt003-contracts]
verified: 2026-10-22
---

# LEARN-JOLT-003 implementation contracts

[CURRENT] The separately versioned implementation follows the owner-issued
[task](LEARN_JOLT_003_PROMPT.md) and [D-019](../DECISIONS.md#d-019--implement-and-conditionally-execute-learn-jolt-003).
Its [configuration](../../configs/learning/jolt-learn-003-v1.json) retains the
original learning workload and qualification gates. The [runbook](../LEARN_JOLT_003_RUNBOOK.md)
describes prospective admission and one-invocation operation. Software
conformance alone does not admit the experiment or establish a useful learner.

The [offline completion evidence](../evidence/2026-10-22-learn-jolt-003-completion.md)
records three fresh independent reviews and the final source-bound typed
pipeline assessment. Complete original-clock time and full-worker/native memory
remain unresolved, so prospective physical admission is refused and allocation
`learn-jolt-003-20261022-a1` remains unspent. The requested audit increase to at
least 3,900 seconds does not establish feasibility under the retained 7,200-second
whole ceiling. Executable phase budgets and all 39 counters remain unchanged;
the original storage and coupled CPU reports remain historical component evidence.

## Version boundaries and unchanged inputs

| Contract | Version |
| --- | --- |
| Experiment | `nca-jolt-learn-003/v1` |
| Execution declaration | `nca-learn-jolt-003-execution/v3` |
| Worker run | `nca-learn-jolt-003-run/v1` |
| Reservations and budget snapshot | `jolt-budget-reservations/v3`, `nca-jolt-learn-budget/v3` |
| Owner durability and bounded serialization | `nca-owner-journal-durability/v3`, `nca-jolt-bounded-serialization/v3` |
| Independent neural audit | `nca-learn-jolt-003-independent-readback/v3` |
| Closed-cost verification | `nca-jolt-closed-cost-verification/v3` |
| Qualification | `nca-jolt-learning-qualification/v3` |

The registry exposes trainer `learn-jolt@3` and explicitly selects execution v3.
Discovery does not import Torch; the old `learn-jolt@2` entry remains unchanged.

Execution v1/v2 paths retain their historical meaning and byte contracts.
Policy, action, feature and weight bundles retain their existing v2 contracts;
their action semantics do not change. Cybersand mechanics, source and native
artifacts remain unchanged. The teacher is still `authored-jolt-recovery@4`
at physical source `46a22f98dce78be3c4ba534d955e58d5691c7d96`, bound by the exact
[original prerequisite](../../configs/learning/jolt-prerequisite-v1.json).

The new configuration hashes the unchanged architecture, seeds, losses,
hyperparameters, reward, original supervisor and qualification sections.
It rejects changing those inputs while presenting the change as execution v3.
The consumed LEARN-JOLT-002 allocation, its five finals and its incomplete
journal/lifecycle evidence remain historical failures.

## Complete unsuccessful retirement

Only `emergency_recovery_deadline` is eligible. The capture and independent
auditor replay every supervision input and the final pre-cancel observation.
They retain quiet-exit precedence, the original invocation deadline, and exactly
48 consumed emergency ticks. A reason string without the replay is rejected.

Native cancellation must preserve the original actor incarnation and unchanged
physical state, increment control generation once, and clear action, targets,
pending edges and helper lease. Valid cancellation does not require suspension.
The episode retains final observation, path, reason, context, actual decision
count and cancellation proof. There is no additional physical action or neural
forward under the retired invocation.

The versioned episode outcome is complete and unsuccessful. Its integrity-error
field remains empty only when supervision, native cancellation and durable
recording all verify. Its paired success is false and capped-completion time is
four seconds. Actual character, latency, impulse and recovery measurements
remain unchanged; unavailable measurements cannot pass qualification. Authored,
idle, BC and residual comparators use the same rule. Every other failure remains
allocation-fatal and preserves an incomplete or capped result.

Ordinary learned attempts reserve their checkpoint invocation before shared
supervision. A retirement there can have one charged attempt without a new
proposal, value, receipt or transition. An administrative-cutoff peek is recorded
separately: a retirement found there adds no bootstrap forward and takes
precedence over administrative bootstrapping.

## Learning and fixed order

Teacher failures retain every supported ordinary label and pulse abstention.
Emergency override intervals remain masked, and there are no positive secondary
labels. BC refuses an empty eligible teacher set. Compact ordinary teacher
sequences remain in memory; the 64 complete raw captures do not.

PPO retains actual confirmed decisions, including masked emergency intervals.
An override without a model value receives no critic loss. Reward terms, actor
and critic masks, physical length-16 windows and GAE cuts are unchanged. Safe
terminal failure uses zero bootstrap at the versioned episode boundary without
rewriting raw per-tick task flags or making a new value observation.

At a verified administrative cutoff immediately following a masked emergency
row, the persisted next-observation peek and its real value/forward costs remain
unchanged. The v3 learning projection uses zero internal bootstrap for that
masked final row; the original mask gap already cuts GAE. This narrowly scoped
projection avoids a legacy validator rejection and does not alter legacy rollout
semantics, raw task flags, losses or the recorded peek.

**No terminal penalty is added.** Original mask gaps cut return propagation;
retaining the failed outcome does not supply a new negative learning signal or
establish that learning improves. Empty or all-masked windows perform no
invented optimizer work.

The configuration contains all 6,144 possible PPO case IDs before execution:
2,048 continuous `random.Random(seed + 300000).choice(...)` draws for each of
190103, 190207 and 190301. Each new episode consumes the next entry, including a
safe failure. There is no outcome-selected replacement. Each of 24 updates
contains exactly 256 physical decisions, at least one per capture and at most
256 captures per update. Zero progress aborts.

Open and sealed comparison order is also frozen, case first and controller
second: 32 open and 224 sealed rows. All six final hashes precede every sealed
outcome, including authored and idle references. The
[expanded opened inventory](../../configs/learning/jolt-opened-inventory-v2.json)
conservatively includes all original 64 teacher and four open candidates. The
[unopened proof](../evidence/jolt-learn-003-unopened-sealed-proof-20261022.json)
binds original source order, immutable inventory, five-final state and zero
evaluation stage counters. It permits reuse of the original declared but
unreached sealed candidates without repairing the failed original journal.

## Operation, clock and storage enforcement

All 39 ceilings equal the [reviewed proposal](../evidence/jolt-learn-003-allocation-plan-20261022.json).
Stage totals are 6,464 starts, 18,944 decisions and 113,664 ticks; aggregate
20,000/120,000 backstops grant no additional stage work. Reservation precedes
work and refused or uncertain work is never refunded. The streaming ledger
holds counters and one record in memory, with an immutable disk spool and
independently replayed published prefixes.

| Phase | Absolute end from original T |
| --- | ---: |
| Ordinary work | 6,000 s |
| Worker closure/pre-audit publication | 6,180 s |
| Complete independent neural audit | 6,780 s |
| Worker final publication | 6,888 s |
| Worker exit, cleanup and log sync | 6,900 s |
| External inventory verifier | 7,128 s |
| Verifier cleanup and log sync | 7,140 s |
| Final receipts, original stdout and observed CLI exit | 7,200 s |

The two twelve-second cleanup intervals come from the existing phase reserves.
They add no time. Near-zero monotonic origins are valid. Every phase uses T;
no import, child process, audit or publication restarts the clock. Cleanup is
attempted independently after a recording failure and never turns late work
into successful completion. The retained capture `RunBudget(40, 1800)` is a
per-episode wall backstop, further limited by the remaining active interval.
It is not a new allocation or a replacement for the absolute phase clock.

Canonical reservation records are capped at 256 bytes and their count at
1,901,556. Published reservation artifacts remain below the unchanged 512 MiB
reader. Raw and sample episode envelopes use a 64 KiB base plus 128 KiB for
each actual decision; the 40-decision maximum is 5.0625 MiB, below the raw
32 MiB and sample 64 MiB readers. Metrics, checkpoint bundles, learning JSON,
journal records and all subprocess streams have separate enforced limits.
Serialization streams hashes, compressed bytes and full EOF readback; a partial
final-path file remains a failure artifact rather than being silently replaced.

`storage_bounds_v3()` recomputes the source's full byte and file envelope,
including both reservation prefixes, the live spool, partial files, lifecycle
and independent observer receipts. Admission requires this exact object and
available space for all artifacts plus at least one GiB reserve. Bounded sizes
do not establish throughput or peak resident memory: those remain separate
resource gates, supported by source reasoning and bounded synthetic conformance.

## Admission and qualification authority

An executable declaration requires the actual commission, unused allocation
and output identity, exact clean source revision/closure, configuration,
teacher, opened inventory, split, complete order, host/build and CPU runtime.
It embeds the original text and file hashes of at least two distinct final
source reviews, covering terminal/learning/qualification and full
operation/serialization/deadline feasibility. Review binding excludes only the
reviews themselves and creation time, avoiding a self-referential hash.

A review or declaration can be published in a documentation commit after the
exact executable commit. Execution still uses the clean reviewed executable
checkout and verifies embedded review bytes. Planning reviews and execution v2
reviews never satisfy this gate. False, missing or unresolved resource gates
refuse before physical work. A genuinely unresolved feasibility assessment is
published as admission failure, leaving this allocation unspent.

Qualification retains all original denominators and thresholds: 36/48 whole
family arrivals, at least 8/16 per seed, final-minus-own-initial mean gain of
12.5 percentage points with strictly positive lower 95% bound from all 27
policy-seed resamples, unchanged authored comparison including high-baseline
capped completion, character tolerances and per-case inference p95 at most
5 ms. Zero integrity, authority or lifetime regressions is mandatory. Complete
negative evidence is distinct from incomplete evidence. A qualified BC family
has preference; residual must satisfy its own initial-gain test. Owner feel and
morphology #16 remain separate gates.
