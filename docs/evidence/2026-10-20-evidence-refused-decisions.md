---
title: Jolt evidence for refused decisions
kind: evidence
summary: Versioned recording separates refused local frames from physical step receipts while retaining failed outcomes and strict legacy reads.
owns: []
verified: 2026-10-20
---

# Jolt evidence for refused decisions

The first recovery diagnostic stopped with a local arrival estimate at tick 234.
Its failed episode retained forty controller frames and thirty-nine physical
step events. The last frame had no admitted lease and no corresponding physical
step. Counting it in the physical frame array prevented character replay and
the final evidence seal. The failed invocation and its original artifacts remain
unchanged; the [preservation checkpoint](https://github.com/techrote/neural-cyber-automata/pull/59)
records the independent prefix audit.

This correction belongs to the separately declared continuation in
[#60](https://github.com/techrote/neural-cyber-automata/issues/60). It changes
recording and reading of local decisions. It does not change the task, controller
kinetics, physical cadence, supervisor authority or reliability thresholds.

## Decision and physical receipt boundary

The [capture path](../../src/nca/evaluation/jolt_arrival.py) now adds a controller
frame to `controller_frames` only after the recorder returns a physical step
event. If a local frame cannot produce such an event, capture retains it in the
separate `controller_decision_failures` sequence. Capture still stops immediately
on that failure.

The additive [decision audit](../../src/nca/evaluation/jolt_decisions.py) declares
`nca-jolt-controller-decisions/v2` in episode evaluation and result rows.

| Field | Meaning |
| --- | --- |
| `controller_frames` | One action/phase/inference frame for each retained physical step event. |
| `controller_decision_failures` | At most one unrecorded local frame, its exact source observation, failure, possible packet and owner-reservation counters. |
| `controller_decision_accounting` | Separate controller-call, physical-step, pretransport-refusal and unrecorded-response counts. |
| `error_cleanup` | A separate attempted/confirmed native cancellation result for an already failed episode. |

Each failure uses `nca-jolt-controller-decision-failure/v1`. Its source body must
match the last retained physical view, including support and velocity; a matching
observation ID alone is insufficient. A durable `controller_decision_failed`
journal event preserves the frame and source before cleanup. Journal replay binds
that event to the preceding native view and to the stored episode audit.

No owner reservation is added for a locally refused lease. If a step reservation
has already advanced, capture uses `owner_response_unrecorded`. That label does
not claim the request reached the native owner. A reservation fsync can fail after
counters advance and before transport; the separate counters and pending journal
state retain that uncertainty. A poisoned writer continues to reject all later
owner dispatch through it, including cleanup attempts.

## Native cleanup and failed qualification

Successful error cleanup records a compact `nca-jolt-error-cleanup/v1` receipt.
It binds the native cancellation serial, before/after observation hashes, unchanged
tick and physical state, revoked joint targets and incremented control generation.
Replay checks those fields against the retained cancellation event and native
journal response. No neutral physical step is substituted for cancellation.

An error-cleanup receipt does not turn the failed episode into a normal completed
episode. `owner_closure_confirmed` retains its conservative failed-outcome value.
The original error and administrative cutoff remain recorded. Any refusal still
prevents a completed journal seal and complete valid diagnostic pairing, even
when every consumed physical tick and the subsequent cleanup are measurable.

## Read compatibility and the next declaration

Legacy episodes without the new decision schema retain their original array
interpretation. Readers neither discard an extra legacy frame nor silently move
it into a new sequence. The original forty-frame/thirty-nine-step failure therefore
continues to fail complete task/character replay in its unchanged form.

The original declaration SHA and default command remain unchanged.
`nca-jolt-recovery-diagnostic/v1` continues to require `jolt-recovery-v1`.
A separately pinned `nca-jolt-recovery-diagnostic/v2` requires
`jolt-recovery-v2`, alongside the same original and arrival references.
Schema/family substitutions are rejected. Both recovery versions must supply
their supervisor's observation-anchored, deadline-capped lease; refusal is never
bypassed to finish a diagnostic. The new declaration has its own source freeze
and allocation. It cannot resume or replace the failed first invocation.

## Offline verification

The [refusal tests](../../tests/test_jolt_decision_refusals.py) use synthetic
receipts with the real environment adapter, episode recorder, task replay and
journal validator. The main regression returns thirty-nine six-tick physical
steps, refuses a fortieth local frame, preserves all forty local frames across
their separate sequences, and verifies measurable 234-tick task/character output
with native cancellation and an intentionally refused completion seal.

Negative controls cover uncertain reserved steps, reservation fsync failure before
transport, same-ID source changes, rehashed refusal-journal changes, forged
cleanup claims and attempted relabelling as completed. Existing legacy records
remain readable when their original arrays are complete, and extra legacy frames
remain errors. An initial fsync-test expectation assumed capture would return
after the writer was poisoned. The final episode-completion journal note
correctly raised instead, after the failed episode artifact had been retained.
The test now checks that propagation and those retained files; production
persistence was not relaxed to make the test pass.

The combined focused sandbox run passed 57 of 58 checks and reproduced the
previously documented process/procfs visibility failure in the unchanged Linux
process-group cleanup test. That unchanged test passed separately with consistent
process visibility. Neither production cleanup nor its assertion was weakened.

This source work starts no physical host, checkpoint, optimizer or learned policy.
Physical results, final-head checks and acceptance of the separately declared
continuation are recorded by its owning issue and PR. The historical journal
truncation cause remains undetermined, and no authored diagnostic qualifies the
learned skill or admits morphology transfer.
