---
title: Durable Jolt diagnostic evidence
kind: evidence
summary: Additive owner-journal persistence, independent receipt replay and character measurement; the historical truncation remains unresolved.
owns: []
verified: 2026-10-20
---

# Durable Jolt diagnostic evidence

The owner commissioned EVIDENCE-001, the remaining qualification work in
[#43](https://github.com/techrote/neural-cyber-automata/issues/43), alongside
[JOLT-RECOVERY-001](https://github.com/techrote/neural-cyber-automata/issues/48).
This page describes the evidence implementation. A new physical result needs
its own published source/configuration freeze, allocation, invocation and review.

## Historical result and diagnosis

The [original arrival diagnostic](2026-10-20-jolt-arrival-reference.md) retains
its 27/32 versus 17/32 descriptive arrivals and its incomplete journal. The
file matched its own indexed hash while stopping at reservation 843; its complete
episode records do not supply the missing owner exchanges. Its original
`completed` flag, raw bytes, hash and separate incomplete audit remain unchanged.

Source inspection establishes two gaps in the previous writer: it used
`gzip.flush()` without synchronizing each reservation/response to storage, and
the supervisor trusted worker exit without independently binding the final
summary to the closed files. These are failure mechanisms worth addressing.
They do **not** establish what caused the historical cutoff or how its bytes
were transported. This implementation does not claim to reproduce that incident.

## Persistence and completion boundary

The [durable writer](../../src/nca/evaluation/durable_journal.py) creates one
exclusive plain `owner-journal.jsonl` for the invocation. Every bounded JSONL
record carries an increasing sequence, its predecessor hash and its own hash.
A request reservation is written, flushed and `fsync`ed before transport.
The response is persisted before confirmed counters advance. Any persistence
failure poisons the writer and prevents a subsequent owner dispatch through it.
A stopped worker therefore leaves an explicitly unfinished readable prefix;
that prefix cannot be resumed or replaced by the same invocation.

Only after native close and process termination does finalization compress the
closed raw journal. It rereads the entire gzip through its trailer, compares its
decoded bytes and chain to the raw original, and replays every owner exchange.
The replay checks native serials, chronological episodes, requested/confirmed
ticks, cancellation, terminal generations and final host closure.

For the new durable lane, replay also verifies each stored episode's compressed
hash and record identity. It compares every submitted command and every physical
tick field shared by the journal and the episode. For this synchronous runner,
both the controller's source view and its issued view must exactly match the
preceding native observation, including physical contents. Retaining an old
observation ID while changing support or velocity cannot justify an action.
It recomputes the unchanged
`HopTask` rewards/outcomes and the character metrics from the retained actions,
frames and physical receipts. Copying the same erroneous metric into the journal,
episode and summary does not pass this independent calculation.

An immutable seal binds raw and compressed hashes, sizes, the final record chain,
the persistence receipt, row/counter identities and the completed replay. Final
JSON publications use a synchronized temporary file and an atomic no-overwrite
link. POSIX directory entries are also synchronized. Windows file data is
synchronized; directory-fsync support is explicitly recorded as unavailable.
Neither method promises survival of broken storage hardware.

The [outer supervisor](../../tools/jolt_arrival.py) independently reopens the
summary, manifest, full artifact inventory, seal, journal and episodes after the
worker exits. This offline verifier has only the remaining original wall budget,
with six seconds reserved for cleanup. Timeout, changed bytes, absent output or
an unconfirmed owned process group prevent completion. Its receipt binds the
verified summary and manifest hashes. No second physical invocation is launched.

## Character measurement

The [character audit](../../src/nca/evaluation/jolt_character.py) preserves
separate requested, admitted and consumed primary edges. A confirmed
primary-associated pulse additionally requires the observed cooldown transition,
energy use and an upward aggregate actuator impulse. Airborne confirmation needs
an unsupported physical tick within twelve ticks after such a pulse. Falling
before any pulse is measured as airborne time, not a purposeful launch.

Additional measures include actual aggregate linear/angular impulses, airborne
seconds, landing/support-loss transitions, the host's supported/blocked heuristic,
request changes, move/trim reversals, inference and owner-roundtrip latency.
Initial actuator assistance is reported separately from the first aggregate
actuation observed after an active policy request. Unknown impulse/support data
remain unavailable; they are not replaced with zero.

Recovery events retain the observed trigger, source position and subsequent
physical response. The report distinguishes ordinary contact/support recovery
from explicit emergency-supervisor interventions, and records whether subsequent
receipts show at least one cell of progress. These are descriptive temporal
associations; a phase label does not prove a successful recovery or isolate its
causal impulse. The native `blocked` field is a host progress heuristic, not a
direct contact-normal measurement. Passive contact wrench and owner feel remain
unmeasured.

## Additive recovery diagnostic

The original arrival declaration and default command remain unchanged. A new
`nca-jolt-recovery-diagnostic/v1` declaration requires its explicitly supplied
published SHA-256, the original opened cases, task, baseline, shaping and host
bridge. It names original, arrival and recovery families with separately frozen
candidate specifications. Its ceilings are 96 episodes, 3,840 decisions,
23,040 requested ticks, 4,226 owner requests, one host and 900 whole-invocation
seconds. Every episode retains six-tick decisions, at most forty decisions and
the unchanged 240-tick task deadline. No optimization, checkpoint execution,
automatic retry or budget extension is admitted.

The recovery candidate's packet uses the supervisor's observation-anchored,
deadline-capped lease. The earlier references retain their original twelve-tick
helper leases. The actual consumed interval and physical assistance are matched.
All 32 cases were previously opened; this diagnostic cannot qualify a learned
policy or unblock morphology transfer.

## Offline verification and limits

The tests cover durable reservation before dispatch, write/fsync failure,
unclean worker exit, chain tampering, truncated raw/compressed transports,
changed artifact bytes, rehashed false task/character claims, native/episode
receipt disagreement and a timeout in the independent final verifier. Character
controls distinguish initial falling, an unconfirmed requested hop, actual
airborne pulse confirmation, unavailable impulses, standing assistance and
ordinary versus emergency recovery.

Development preserved an initial Linux process-state test failure caused by
sandbox PID/procfs namespace mismatch. The unchanged process-group test passed
with consistent process visibility; neither production cleanup nor its assertion
was weakened. A fixture initially attempted to deep-copy immutable mapping
proxies and was corrected to thaw their canonical JSON. These were offline test
failures, not repeated physical experiments.

No physical host, learned policy, checkpoint or optimizer was started by this
evidence implementation. Final-head CI, the separately declared physical
diagnostic and its independent acceptance are recorded by the owning PR and
subsequent run evidence. The original incomplete diagnostic remains incomplete.
