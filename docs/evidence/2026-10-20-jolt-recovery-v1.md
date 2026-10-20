---
title: Jolt recovery v1 — preserved incomplete diagnostic
kind: evidence
summary: The first contact-recovery diagnostic stopped at a sampled-arrival lifecycle mismatch; complete raw failure evidence and next-version boundaries are retained.
owns: [evidence.jolt-recovery-v1]
verified: 2026-10-20
---

# Jolt recovery v1

**The teacher is not qualified.** The single declared invocation stopped after
24 of 96 episode rows. It exposed a real disagreement between a controller's
sampled arrival estimate and the unchanged task's per-tick success predicate.
The runner refused a zero lease. It did not manufacture an arrival, renew retired
authority or continue the remaining cases.

## Source and allocation

The reviewed public source is
`ceb19f5742a782299197557fa53e2208523eb7f7`, tree
`59cd25fe3e977c00945752f0bafde044723843cb`, published before execution in
[#48](https://github.com/techrote/neural-cyber-automata/issues/48#issuecomment-6297367286).
The [declaration](../../configs/baselines/jolt-recovery-diagnostic-v1.json) has
canonical SHA-256
`2e5d4e3c554ec2382d10a95d347c1f933252050f2a18dff320711c2cc3c7fff7`.
It fixes the original, arrival and recovery references on the same 32 already
opened case/seed pairs, the original task and the pinned host
`f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497`.

| Account | Declared maximum | Actual |
| --- | ---: | ---: |
| Host attempts / terminations | 1 / 1 | 1 / 1 |
| Episode rows | 96 | 24: 23 complete, 1 incomplete |
| Dispatched decisions | 3,840 | 479 |
| Requested / confirmed physical ticks | 23,040 | 2,874 / 2,874 |
| Owner calls | 4,226 | 549, all answered |
| Whole-process seconds | 900 | 61.76433001999976 |
| Learned policy calls / checkpoints / optimizer updates | 0 | 0 |

The process exited 2 without a timeout, pending request or uncertain physical
tail. The host and owned process group terminated. Its allocation is closed;
unused budget does not authorize a retry or a resumed dataset.

## Failure boundary

The last case was `demo-floor-right-36-1`, recovery v1. It consumed 39 decisions
and 234 ticks before the next proposed decision was refused.

| Observation | Controller's sampled interpretation | Unchanged native task evidence |
| --- | --- | --- |
| Tick 222 | Supported, inside tolerance, quiet | An ordinary observed sample |
| Tick 223 | No decision at this tick | Speed 7.5989 cells/s exceeds the 6 cells/s limit; quiet dwell resets |
| Tick 228 | Supported, inside tolerance, quiet | Quiet dwell has restarted |
| Tick 234 | Three quiet samples span 12 ticks; reports `succeeded` and lease 0 | Only 11 consecutive quiet physics ticks; task is unfinished |
| Tick 235 | No observation exists | Eventual arrival is unknown |

Native cancellation 548 revoked the final generation without advancing physical
time; close 549 then ended the host. The original row's closure flag remains
false, while the separate raw-receipt audit establishes that cleanup actually
occurred. This distinction does not change the failed task outcome.

The episode retained 40 controller frames for 39 dispatched steps. The extra
terminal frame is valuable evidence, but violates the character replay's
one-frame-per-physical-step contract. Consequently `character_frames` prevents
publication of the complete seal. The original summary remains `incomplete`.

## What the partial behavior shows

On the eight reached training-origin pairs, the original reference recorded five
arrivals, the arrival reference eight, and recovery seven complete arrivals plus
one incomplete invocation. None of the five targeted historical failures or the
eight evaluation-origin cases was reached. These are not 32-case qualification
rates and do not establish a reliable learned skill.

Recovery restored receipt-confirmed airborne launches on
`demo-floor-left-24-{0,1,2}`. Across only the seven complete matched pairs, its mean
peak toward-speed ratio against the arrival reference was 0.5892403131810864.
That is a concrete vigor concern for the next full comparison, not a verdict on
the declared full-case gate.

The separately labeled in-memory analysis of the failed episode's 39 consumed
frames found two requested, admitted, consumed and airborne-confirmed pulses,
at native ticks 43 and 157; 102 airborne ticks; two recovery entries followed by
observed progress; and zero supervisor interventions. Recovery-phase entry and
later progress are not proof of eventual target arrival or isolated causal
actuator benefit. The host reports aggregate actuator impulse, not a separate
passive/contact wrench.

## Durable failure preservation

The raw journal contains 1,148 linked records and 16,767,143 bytes, SHA-256
`1515be1e5e95c8e435523b95214a491e093e93c690c41a96048c7ba9f54d66d6`.
The complete retained `gzip.pending` transport decodes exactly to those bytes.
It is deliberately not renamed to a successfully sealed journal.

Independent offline review matched all 51 indexed artifacts, 549 native request
and response pairs, every source/issued observation, 479 step events and all
2,874 tick receipts. The 23 complete episode task/reward/character replays match.
The failed episode's consumed-prefix task and character analysis is explicitly
derived, with its original 40th frame and failure retained. No original byte was
replaced, and no new host, physics, checkpoint or optimizer work was performed.

The archive `nca-jolt-recovery-v1-diagnostic-20261020.tar.gz` contains all 57
original and separately derived evidence members, each byte-verified after
packing. Its size is 5,520,107 bytes and SHA-256 is
`6be1740a1a623084f431ba7ed08f39fceb8d3ee9fa416d519c48ff6b57ede439`.
Large raw evidence stays outside Git; the repository retains the
[index](jolt-recovery-v1-index-20261020.json),
[original result](jolt-recovery-v1-result-20261020.json),
[outer receipt](jolt-recovery-v1-supervisor-20261020.json),
[rejecting teacher audit](jolt-recovery-v1-review-20261020.json) and
[independent closed-prefix audit](jolt-recovery-v1-prefix-audit-20261020.json).

The earlier arrival-reference truncated journal remains a separate historical
failure with an undetermined cause. This new journal is retained through native
cleanup and fails qualification for a different, observed reason.

## Corrective increment and gates

The separately commissioned [#60](https://github.com/techrote/neural-cyber-automata/issues/60)
adds advisory arrival under a new controller identity and records undispatched
decisions separately. It preserves v1 source, dynamics, task thresholds and
evidence. A fresh source/declaration and independent full comparison are required;
no v1 row becomes a v2 result. Teacher, complete-diagnostic, LEARN-JOLT-002 and
morphology qualification remain blocked at this checkpoint.

The integrated pre-run offline suite passed 141 tests. Initial hosted checks
found an installed-package path assumption in the new audit's positive fixture;
the audit now supports a matching explicit checkout, environment root or current
checkout, with package and Git-blob checks retained. Twelve focused gate tests
pass after that correction. The final PR records its applicable hosted results.
