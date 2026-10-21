---
title: Jolt learning execution and recording reliability
kind: evidence
summary: Offline versioned failure recording, journal readback and executor completion controls; the consumed experiment remains failed and no replacement is admitted.
owns: [evidence.jolt_learning_reliability]
verified: 2026-10-22
---

# Jolt learning execution and recording reliability

The renewed [#62 completion request](https://github.com/techrote/neural-cyber-automata/issues/62)
starts from accepted main `1c19c5aa7e715575d867675babd19bb08ae58e25`.
[PR #67](https://github.com/techrote/neural-cyber-automata/pull/67) already published
the implementation and the consumed allocation's complete retained evidence.
This increment addresses software reliability using synthetic tests. It starts
no replacement physical or learned experiment and creates no execution declaration.

The [original failed result](2026-10-21-jolt-learn-002.md), its frozen source
`a68e687ced52bef778375a2753b1362dea3704fa`, and every original evidence artifact
remain unchanged. The original journal has 6,720 records against 24,195 reported;
five of six final checkpoints and zero open or sealed rows survive. The cause of
the lost suffix and complete original process closure remain unknown. No code
change here restores missing historical evidence or qualifies a learned family.

## Version and admission boundary

The original learned policy/action schemas already use v2. The new **execution**
declaration, run manifest and readback schemas use `/v2` separately. Legacy
`freeze`, `run`, `worker` and `inventory` retain their v1 execution path and
historical semantics. New modes are `freeze-v2`, `run-v2`, `worker-v2`,
`inventory-v2` and `lifecycle-v2`. A consumed v1 declaration cannot enter v2.

V2 requires a distinct bounded allocation identifier and an exact numeric issue
#62 commissioning-comment reference. These fields validate syntax only. They do
not establish the comment's meaning, authorization or historical uniqueness;
those require prospective review. At least two independent source-bound reviews
must explicitly set `offline_scope_only: false` and
`prospective_experiment_admitted: true` before a declaration is accepted.
The reviews published with this repair explicitly have the opposite scope and
cannot admit an experiment. No replacement commissioning comment exists here.

Every execution path retains one attempt, no automatic retry or extension,
the original 1,800-second total ceiling, the fixed configuration and all original
model, action, observation, seed, split, task, profile and acceptance contracts.
The current repair does not establish that the complete experiment fits that
ceiling. A future commission must assess that feasibility before admission.

## Failure evidence and independent cleanup

An opt-in structured `SupervisorRetirement` preserves a bounded reason such as
`emergency_recovery_deadline`. It is raised before another neural forward or
action. The original emergency limit remains 48 consumed ticks. Retirement
still fails the allocation; this increment does not reset and continue with a
replacement episode. Legacy callers retain their original exception behavior.

V2 capture records whether a PPO bootstrap was never attempted or an actual
cutoff attempt failed. Neither case invents a valid value observation or feeds a
failed sequence to optimization. Original capture failure and cancellation
failure have separate bounded records; exception text and local paths are not
published. Native cancellation is validated by its generation, unchanged physical
state and cleared authority, so a valid unsuspended cancellation is recognized.
Cancellation does not erase the original failed row or make the run complete.

Cleanup independently attempts owner close, termination reservation, forced
native close, exit observation, closure notes and journal close. A poisoned
journal or exhausted reservation cannot suppress the native close attempt.
Recording errors remain visible and make the worker fail. A clean journal close
alone never upgrades an already failed worker.

## Journal readback

`VerifiedDurableJournal` uses an exclusive regular file, descriptor/path identity,
size and position checks, and a single-link requirement. Before each append it
independently checks the preceding tail. After each append it flushes, fsyncs
and independently reopens and reads the new bytes before returning to the caller.
Uncertain writes poison the journal and reject further writes.

Close performs full-file chain, byte-count, hash and record-count verification,
with a separately identified durability receipt. The v2 auditor requires the
exact receipt shape, completed close verification, matching full-file identity
and directory synchronization on POSIX. Windows software checks permit the
explicit unsupported-directory-sync flags; they do not qualify physical execution.
A successful fsync followed by a failed readback is
retained as attempted durability work, not an accepted journal record.

This checks local readback and completed synchronization calls. It cannot prove
storage survival under every hardware failure. Prefix bytes outside the previous
tail are fully checked at close, so an earlier prefix alteration can be discovered
then rather than at the next dispatch. The historical writer remains unchanged;
the new writer does not diagnose why the old suffix disappeared.

## Executor completion and binding

The v2 standard-library supervisor starts its clock before worker imports,
retains one worker, and records immutable startup, process ownership, worker
exit, verification, cleanup, finalization and terminal receipts. The worker phase
reserves 30 seconds and the verification phase reserves five seconds within the
unchanged 1,800-second ceiling. Cleanup, stream synchronization and receipt
publication remain inside their phase deadlines.

The durable terminal records only a completion candidate. Completion additionally
requires the original post-publication stdout verdict and an independently
observed zero exit. Missing stdout, a killed parent, publication failure, surviving
owned processes, mismatched identities or elapsed phase limits cannot qualify.
Failures retain stage evidence wherever publication is still possible; a hard
kill can still leave an incomplete lifecycle and is never reconstructed as success.

The independent lifecycle reader checks all 16 successful lifecycle artifacts,
nine stage hashes, four log streams, declaration identity, original clock,
ordered times, child exits and empty owned-process sets. That reader alone
reports `executor_lifecycle_only` and does not accept scientific results.
The CLI additionally verifies the complete worker inventory and binds its exact
inventory hash and original monotonic origin to the executor's retained summary.
Changing a file while keeping counts and other summary fields identical is
rejected. The combined verifier requires a complete passing independent learned
audit under the unchanged qualification rules before any `qualified` claim.

For read-only verification of a separately admitted v2 output, preserve the
original stdout bytes and actual CLI exit code, then use:

```sh
python tools/learn_jolt.py lifecycle-v2 --output RUN_DIRECTORY --terminal-stdout ORIGINAL_STDOUT_FILE --observed-exit-code OBSERVED_EXIT_CODE
```

The command cannot infer or recover a missing original exit code. Do not pass a
chosen zero or regenerated stdout. The [runbook](../LEARN_JOLT_002_RUNBOOK.md)
continues to mark its original declaration consumed; the commands above do not
authorize a new invocation.

## Verification and review

The [source and test record](jolt-learning-reliability-validation-20261022.json)
binds the complete source closure, environment and exact local test results.
Independent reviews cover the [worker and acceptance boundary](jolt-learning-reliability-worker-review-20261022.json),
[executor integration](jolt-learning-reliability-executor-review-20261022.json)
and [journal and lifecycle interaction](jolt-learning-reliability-journal-review-20261022.json).
Each review identifies its authored components and limits its acceptance to
offline software behavior. None is prospective experiment admission.

Regression coverage includes truncation, path replacement, short/uncertain writes,
tail and closed-chain corruption, missing receipts/stdout, publication failures,
phase overruns, child cleanup, schema mismatches, consumed declarations and changed
worker-inventory bytes. A synthetic unstable capture reaches 48 ticks, revokes
authority without suspension, retains the retirement reason and performs no
neural forward after retirement. Independent readback verifies that failed row
without promoting it to complete or qualified.

The first full offline suite encountered one existing process-isolation failure:
the sandbox and `/proc` did not expose a consistent PID namespace. Its isolated
rerun passed with consistent process visibility. The final complete suite uses
that environment and explicitly unsets actual-host/export inputs. The original
failure is retained in the validation record. Unit fixtures include small CPU
models and optimizer operations; these are software tests, not a replay or resume
of the consumed experiment. Final-head hosted CI remains a separate merge gate
and its final outcome is recorded on the linked PR and issue.

The first hosted head `f7466f014552699069ccd96268174cfde120c5b6` exposed a test
fixture portability error in all four code-conformance jobs. Backdating the
timeout fixture by 1,769.8 seconds produced a negative origin on freshly booted
runners, which the production executor correctly refused. The correction
translates only the synthetic clock's origin while retaining real elapsed time,
actual child timeout/cleanup and all production deadline checks. The original
failed checks and source identity are retained in the validation record and
[PR #70](https://github.com/techrote/neural-cyber-automata/pull/70).

## Remaining learned objective

Issue #62 remains open. It needs all six frozen final checkpoints and the full
224-row sealed paired comparison, complete physical/evidence closure and the
unchanged learned gates: at least 75% mean success, at least 50% for every policy
seed, own-initial mean gain of at least 12.5 percentage points with strictly
positive lower 95% bound from all 27 seed resamples, original
authored improvement, character and per-case inference limits. No result here
establishes those quantities. Owner feel and morphology #16 remain separate.

A fresh allocation needs a separately recorded owner commission and prospective
source, budget and experiment review. Reclassifying emergency retirement as an
ordinary failed episode and continuing would change experiment semantics; it
requires an explicit versioned design decision. Repeating the current experiment
without that assessment is not established as a route to completion.

© 2026 techrote. All rights reserved.
