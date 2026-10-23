---
title: Matched IK qualification and admission operations
kind: guide
summary: Separate non-training resource qualification, source-bound paired preparation, independent admissions and later owner execution gate.
owns: [ik.matched.operations]
verified: 2026-10-23
---

# Matched IK qualification and admissions

The owner commissions blocker closure and admissions, **not training**. Both
final suites remain excluded; evaluation capacity stays zero. The original
[matched design](design/IK_ENDPOINT_MATCHED_TRIALS.md) and all historical
packets, refusals, charges and results are preserved. Only new external
directories may hold rebound declarations and the qualification probe.

The shared TITAN ownership file excludes probe and trial groups across packets.
Windows job containment kills inherited child processes when the supervisor
exits. No GPU fallback or automatic retry exists. Isolate the actual TITAN UUID
using the [runbook](IK_TRAINING_RUNBOOK.md); public evidence uses its fingerprint.

First freeze clean source and pass exact-head CI. CPU-only probe preparation:

```powershell
python -m nca.train.ik_memory_probe prepare --root NEW_PROBE_DIRECTORY --previous-probe RETAINED_FAILED_PROBE
```

Two independent probe-specific records must bind that declaration, source and
CI. Only then may `run --root NEW_PROBE_DIRECTORY --review REVIEW_A --review
REVIEW_B` perform the one allocation-only probe. It creates exactly three
contexts, warms up one MiB, then retains twenty approved 256 MiB zero-filled
buffers per worker, holding all three for thirty seconds. No corpus, model,
optimizer, gradients, predictions or training are involved. Limits: 5 GiB
allocator per worker, 20 GiB total board use, 8 GiB hard committed host memory
per process, 80 C, active work at most 240 seconds and total paid ceiling 300
seconds. Missing/stale telemetry, insufficient projected headroom, a child
failure or any overrun fails qualification and retains the ownership claim.
No lower-concurrency fallback is implicit. Passing evidence establishes only
concurrent contexts/touched allocations, not learner throughput or equivalence.

Authenticate the successful probe's terminal receipt and every indexed artifact.
Then prepare a new paired packet without modifying its predecessors:

```powershell
python -m nca.train.ik_matched_prepare prepare --corpus ORIGINAL_CORPUS --registry ORIGINAL_REGISTRY --output NEW_PACKET --qualification NEW_PROBE_DIRECTORY
python -m nca.train.ik_matched_prepare inspect --output NEW_PACKET
```

Both modes retain the fixed data, normalization, sampler and seed set. Each
seed's fresh allocation binds a conservative 7,195-second coordination debit
inside 7,200 seconds, covering group startup, worker work, monitoring and
publication; this is prepaid accounting, not measured worker time. The group
stops by 7,190 seconds. Startup refusals after group claim charge unstarted
seeds; ambiguous active work retains the failure hold and cannot resume.

Two new independent **trial** admissions must bind the new packet and exact
source, corpus/final exclusion, accounting, CI and measured resource evidence.
Probe admissions cannot authorize training. `check-admission` validates them
without execution. Actual execution additionally requires a separate exact
owner execution record, which is not created by qualification or preparation.

After any future admitted training, terminal verification authenticates all 41
checkpoints, 40 tuning reports and 1,720 ordered batch receipts per seed. It
compares typed model/optimizer/RNG/sampler fingerprints across modes. Any
divergence denies consistent-parallel qualification; no speedup or numerical
equivalence is claimed before those measurements exist. Both-mode completion
and all original tuning gates remain required. Neither mode opens final data.

© 2026 techrote. All rights reserved.
