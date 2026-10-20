---
title: Implementation checkpoints — 20 October afternoon
kind: evidence
summary: Preserved Jolt recovery and evidence checkpoints from 17:19 through 19:00 UTC.
owns: []
verified: 2026-10-20
---

# 20 October afternoon checkpoints

## 2026-10-20 17:19 UTC — Jolt recovery and evidence work started in parallel

Fresh accepted main is `b923f39c1d00219cc789505414f02e61db60570a` (PR #44).
The owner commissioned [JOLT-RECOVERY-001 / #48](https://github.com/techrote/neural-cyber-automata/issues/48)
and EVIDENCE-001 continuation of [#43](https://github.com/techrote/neural-cyber-automata/issues/43).
D-015 records that scope. Controller/supervisor code and evidence transport/metrics
have independent branches, file ownership and offline tests. A third reviewer
audits observation semantics, acceptance and the next learning experiment.

The original 64-episode result and truncated owner journal remain unchanged.
The current writer has no file fsync per exchange, and its supervisor does not
independently bind the final worker evidence; these are concrete durability gaps,
not an established explanation of the historical truncation. The proposed
correction records durable raw exchanges before transport, seals a closed copy,
and independently validates completion. Recovery work covers supported-blocked
progress, edge/landing/support changes and fresh-observation second-hop admission.

The pinned host/runtime setup is verified without a diagnostic rollout.
The next learning design must change the versioned action mapping: the current
residual learner copies authored launch buttons and therefore cannot demonstrate
learned launch timing. No training, checkpoint selection or morphology execution
has started. A single new matched three-reference diagnostic will be declared
after source review; no historical case is treated as sealed evaluation.

## 2026-10-20 17:44 UTC — Recovery and durability reviewed; diagnostic frozen

The independent controller and evidence workstreams are implemented. The recovery
controller owns ordinary movement, launch/aim, contact progress and landing;
the separate physical supervisor owns protocol refusal and a declared bounded
attitude emergency. The original controller, arrival reference, task, bridge,
physical profile and cadence remain pinned.

[EVIDENCE-001 PR #58](https://github.com/techrote/neural-cyber-automata/pull/58)
publishes durable pre-dispatch reservations, a closed raw/compressed journal seal,
native observation/receipt binding, recomputed task and character metrics, and
independent post-exit verification. Its public head is
`04ae94319e4788881b6dd7b126618c4bba37801e`; required CI is in progress.
The original journal's truncation cause remains undetermined and its incomplete
record remains unchanged.

The integrated offline suite passed 141 tests in 2.646 seconds. Strict
documentation validation passed 60 files without errors or warnings. Independent
review found and closed source-observation and replay-code binding gaps before
any physical diagnostic. Tests of refusal, malformed evidence and interrupted
execution are synthetic controls, not additional game runs.

The separate [recovery declaration](../../configs/baselines/jolt-recovery-diagnostic-v1.json)
has canonical SHA-256
`2e5d4e3c554ec2382d10a95d347c1f933252050f2a18dff320711c2cc3c7fff7`.
It permits one host invocation, 96 episodes across the three authored families,
3,840 decisions, 23,040 requested ticks, 4,226 owner calls and 900 whole-process
seconds including durable closure and independent verification. No retries,
extensions, optimization or learned checkpoints are permitted. The published
teacher rule requires improved matched arrival, both historical failure classes,
restored airborne character and retained style/latency bounds.

[LEARN-JOLT-002](../design/LEARN_JOLT_002.md) is a conditional design with full learned
aim and launch authority, a learned BC residual base, three training seeds and
fresh geometry/start separation. Its exact execution declaration remains future
work after teacher/evidence acceptance. No new physics, learning or morphology
run has started at this checkpoint.

## 2026-10-20 18:02 UTC — Failed v1 preserved; separate corrective scope declared

The sole public `ceb19f57` recovery diagnostic ended incomplete in
61.76433001999976 seconds, with 24/96 rows, 479 dispatched decisions, 2,874
requested/confirmed ticks and 549 answered owner calls. No timeout, pending
request, new learned checkpoint or optimizer work occurred. At tick 234, sampled
settling retired authority while an intervening native speed violation left the
unchanged task only eleven quiet ticks. The runner correctly refused lease zero.

[The complete failure record](../evidence/2026-10-20-jolt-recovery-v1.md) retains
the original summary, raw journal, pending compressed copy and undispatched
terminal frame. Independent audit binds every physical receipt and all native
cleanup; the original incomplete seal remains rejected. The 57-member archive,
SHA-256 `6be1740a1a623084f431ba7ed08f39fceb8d3ee9fa416d519c48ff6b57ede439`,
is saved and byte-verified. This result does not complete #43 or qualify #48.

Initial PR #59 hosted checks exposed an installed-package checkout assumption
in the acceptance audit. Its correction retains complete source-byte binding;
all twelve focused tests pass. This fix does not change the frozen physical run.

The separately commissioned [JOLT-RECOVERY-002 / #60](https://github.com/techrote/neural-cyber-automata/issues/60)
adds a versioned advisory-arrival boundary and independently archives refused
decisions. No kinetic tuning is included. Its fresh 96-episode, 900-second
allocation needs its own public source/declaration freeze and independent review;
it cannot resume the failed v1 attempt or spend its unused budget. V1's lower
observed prefix vigor remains a risk. All learning and morphology gates remain
unchanged and blocked pending complete evidence and useful teacher behavior.

## 2026-10-20 18:22 UTC — Corrective v2 integrated and independently reviewed

PR #58 and [PR #59](https://github.com/techrote/neural-cyber-automata/pull/59)
passed all five final-head checks and are merged. Accepted main is
`7fee91b18dd0a6bba7089870fb34c515cf2e9f5a`; its tree matches the reviewed
implementation and preserved negative evidence. Teacher qualification remains
open. Neither merge changes the frozen v1 result or its failed acceptance.

The separately versioned controller keeps sampled arrival advisory and retains
the original observation lease until the caller validates its outcome and
cancels current local/native intent. V1 controller and supervisor source hashes
remain pinned; every one of the 189 previously consumed candidate actions replays
unchanged. All kinetic constants and physical/task contracts are preserved.

The [recorder correction](../evidence/2026-10-20-evidence-refused-decisions.md)
separately retains an undispatched decision, exact source, reservation uncertainty
and native error cleanup. Legacy extra-frame records remain rejected, and a
refusal cannot become a completed diagnostic. Independent controller, recorder,
acceptance and declaration reviews found no remaining source blocker.

The integrated Jolt suite passed 127 checks in 18.287 seconds with consistent
process visibility, including the unchanged process-group cleanup assertion.
No physical diagnostic ran during these tests. The prospective
[v2 declaration](../../configs/baselines/jolt-recovery-diagnostic-v2.json) has
canonical SHA-256
`d7cc13335b453a755e9b7d950ac52d982aff553c025cbe0fbd47b81b5b0729eb`.
It independently permits one fresh 96-episode invocation, at most 3,840 decisions,
23,040 ticks, 4,226 owner calls and 900 whole-process seconds. It carries the full
prior v1 cost ledger, reuses zero rows, resumes no attempt and changes no numeric
teacher gate. Public source freezing precedes this one new invocation. Learning
and morphology remain blocked until their respective acceptance requirements.

## 2026-10-20 18:37 UTC — IK contracts implemented and independently reviewed

The owner began the [CUDA IK chain / #49](https://github.com/techrote/neural-cyber-automata/issues/49).
The first bounded increment, #50, owns the registered `runner-right-sole@1`
descriptor, immutable observation/target/result records, nominal geometry and
the prospective numeric acceptance declaration. Work started from `14b85d08`
and reconciled with accepted Jolt PR #59 at
`7fee91b18dd0a6bba7089870fb34c515cf2e9f5a`; unrelated source and evidence remain intact.

Independent geometry review reconstructed both local anchors and rotary zeros
from pinned Cybersand `f2e34cf9`, selected the sole point `(0,1)` and established
the two-dimensional feasible target generator. Contract review exercised
issued-root conversion, all lifecycle fields, explicit policy/grant identity,
exclusive source age/lease, cancellation/replay and result/proof refusal.
Absolute geometric goals remain distinct from bounded deltas and host tracking.

The installed dependency-free wheel passed the local 580-case CPU suite:
376 passed and 204 optional learner/physical-host cases were explicitly skipped.
After the final declaration update, all 36 focused IK tests pass. Strict
documentation checks pass all 65 files without errors or warnings. These are
software fixture results; no inverse training, CUDA or physical IK run occurred.

Acceptance review closed a trivial-fixture gap before publication: static
starts must need a reach, sweeps need displacement and reversals, response
trials need attributed motion, and paired stationary-reference controls must
fail often enough to demonstrate useful tracking. The 90-tick static deadline
includes completion of the full 18-tick dwell. Numeric, learned, exported and
physical gates retain separate verdicts and denominators.

The [contract guide](../IK_LAYER.md), [acceptance](../IK_ACCEPTANCE.md) and
[host handoff](../IK_HOST_HANDOFF.md) describe the implemented and missing pieces.
Next is the scoped PR, required final-head checks and verified merge. Existing
morphology, Jolt recovery and evidence gates remain unchanged.

## 2026-10-20 18:40 UTC — Full v2 evidence; teacher qualification remains failed

The independently declared public `8da294ae` invocation completed all 96 rows in
444.46566076900126 whole-process seconds, including startup, sealing and independent
post-exit verification. It used 2,449 decisions, 14,694 requested/confirmed ticks,
2,711 answered owner calls and one terminated host, with no pending tail, failed
admission, controller refusal, learned call, checkpoint or optimizer execution.
The complete raw journal has 85,438,519 bytes and 5,616 linked records; all 196
indexed artifacts and final native close verify. No v1 row was reused.

[The full result](../evidence/2026-10-20-jolt-recovery-v2.md) has original/arrival/
recovery successes 17/32, 27/32 and 25/32. Recovery restores purposeful airborne
movement in 32/32 and passes the complete 0.727556 vigor ratio, posture, trim,
latency and two-pulse/fresh-observation gates. Independent replay reproduces all
900 candidate decisions and 42 pulses, including ten second pulses. It fails
strict arrival improvement, at least three historical rescues, the required
unsupported/drop rescue and the at-most-one-regression gate. Two failures are
rescued and four previous successes regress. The opened evaluation subset is
4/8, 5/8 and 6/8; none of these cases becomes sealed learning evaluation.

All remaining failures are ordinary 240-tick deadlines. Three regressions are
inside tolerance but lack continuous quiet settling; four others remain short
after the second landing. The old v1 lifecycle failure succeeds at new native
tick 235, with a valid advisory hold and no extension beyond the original deadline.
Original v1 bytes still stop at 234. Independent durability review and archival
publication are being finalized before final-head CI and merge verification.

The [learning design](../design/LEARN_JOLT_002.md),
[interface checklist](../design/LEARN_JOLT_002_IMPLEMENTATION.md) and
[conditional issue #62](https://github.com/techrote/neural-cyber-automata/issues/62)
now specify learned aim/buttons, a learned BC residual base, immutable invocation
time across recurrent segments, matched supervision/history masks, three seeds,
fresh geometry/start fingerprints and complete prospective compute counters.
Collection remains blocked by the teacher result. No architecture or training
experiment follows automatically.

A live state check found #43 and #48 closed after PR #58/#59 despite their stated
incomplete requirements. Both were reopened, with issue reconciliation comments;
PR wording now avoids a closing-keyword phrase inside a negative statement.
The evidence requirement and teacher requirement retain independent outcomes.

## 2026-10-20 19:00 UTC — Independent acceptance and archive readback complete

Independent evidence review accepts the complete v2 diagnostic: all 93 frozen
source inputs, 196 indexed output artifacts, 5,616 linked journal records and
2,711 native exchanges bind correctly. Raw/compressed journal identity, durable
writer synchronization, final native close and process termination are verified.
The [evidence audit](../evidence/jolt-recovery-v2-independent-evidence-audit-20261020.json)
and [behavior audit](../evidence/jolt-recovery-v2-independent-behavior-audit-20261020.json)
remain separate. The latter reproduces all 900 candidate decisions and confirms
four failed teacher requirements; evidence acceptance does not qualify behavior.

The 206-member archive is saved and a fresh readback verifies every member and
the full 27,852,949-byte SHA-256
`7325c1674a9007e7b2c8becb5abc2e37e2d5183c5c899eda4b4be37e88ce2a53`.
Two shorter direct-transfer states are preserved with their hashes in the
[delivery audit](../evidence/jolt-recovery-v2-delivery-audit-20261020.json).
Their cause remains unestablished. Atomic staging and guarded replacement
completed delivery without altering any original diagnostic file or rerunning
physics. The failed v1 archive is also rechecked at its original byte identity.

Accepted concurrent main `a265ca79` adds the independently commissioned IK
contracts. Integration preserves that work and passes all 163 combined Jolt/IK
offline tests in 20.153 seconds. No host source, frozen diagnostic source,
physical threshold, learning gate or morphology gate changed during integration.
Final publication, required final-head CI and merge verification follow in PR #61.

The next narrow teacher task is [#64](https://github.com/techrote/neural-cyber-automata/issues/64):
support loss during launch anticipation, useful progress after the second
landing and enough quiet settling before the existing deadline. No third
physical allocation is opened. Conditional [#62](https://github.com/techrote/neural-cyber-automata/issues/62)
retains three training seeds, fresh geometry/start evaluation, meaningful learned
aim/buttons and a learned residual base; its collection and execution remain
blocked until teacher acceptance. Morphology #16 remains blocked independently.
