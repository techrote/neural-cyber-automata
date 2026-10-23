---
title: IK continuation source amendment
kind: decisions
summary: Owner-authorized source amendment preserves original allocation lineage and final gates.
owns: [decisions.ik_continuation]
verified: 2026-10-23
---

# IK continuation source amendment

[ACCEPTED D-025] The owner authorizes completing #54 / PR #78 on the TITAN RTX
after fresh continuation admissions, within the existing allocations. The
owner explicitly authorizes amendments needed to repair the CPU scoring and
final-reader compatibility refusals. This extends D-024's completed single
batch boundary; it does not grant another allocation, corpus or schedule.

The original declaration, admission, first checkpoint and verified one-batch
ledger remain immutable. A separate source-bound amendment records every
baseline file hash, consumed charge, original allocation, current executable
source and TITAN UUID fingerprint. Original ledger source metadata continues
to identify the original allocation lineage. Every new checkpoint, result and
supervisor receipt additionally identifies the actual execution source and
amendment. No runtime may claim an old execution source for amended code.

Only the registered continuation runner, authenticated scorer/final reader and
their lazy registry entries may differ in the executable source closure. The
model, loss, optimizer, sampler, seeds, forty-epoch schedule, normalization,
corpus bytes and thresholds stay fixed. CPU checks and CI must pass on the
committed head before two distinct reviewers admit the actual amendment.
Reviews are prospective evidence; this decision does not itself admit CUDA.

Authenticated scoring uses the original retained task coordinates and complete
ordered denominators. The final extension independently checks retained frame
geometry after original byte authentication. It requires all three completed
training histories and tuning-only checkpoint selections, writes and reads
back the durable final opening, and only then reads original final shards.
It never reads the mixed pending journal or uses final labels as supervision.

The final worker retains the authenticated training view from before opening.
It charges common process startup/opening/comparison/timing overhead to every
seed, reserves all 65,200 evaluation calls before any comparator invocation,
and writes a reconciled checkpoint without changing optimizer steps. The
five CPU comparators use one thread, batch one, 100 warmups and 10,000 measured
calls each. Failures and unknown work block reuse; final-driven retuning or
an automatic retry is not authorized. Pass, no-go and inconclusive outcomes
remain distinct from exported-policy or physical-host qualification.

The V100 remains prohibited. Only the single UUID-isolated TITAN RTX may be
initialized. Hardware or validation failures that cannot be repaired within
these gates stop the experiment and must be reported.

© 2026 techrote. All rights reserved.
