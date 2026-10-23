---
title: D-023 — LEARN-JOLT-003 risk-managed memory admission
kind: decisions
summary: Source-bound 8 GiB soft process-tree RSS stop and two independent resource/terminal reviews, without claiming kernel OOM prevention.
owns: [learning.jolt003-memory-resource-risk]
verified: 2026-10-23
---

# Risk-managed memory admission — D-023

[ACCEPTED] 2026-10-23 by owner: proceed with implementing an 8 GiB
process-tree memory watchdog, source-bound v2 resource assessment and two
genuinely independent final-source reviews as separate steps. This extends
[D-022](2026-10-23-jolt-cpu-portability.md) and the conditional one-allocation
commission, but does not itself admit the physical experiment or let its
reviewers omit source/evidence qualification.

The source-backed baseline is a maximum observed **~0.64 GB combined**
synthetic-worker-plus-*idle-native* RSS. A declared fourfold planning
allowance is ~2.56 GB against GitHub's advertised 16 GB Ubuntu runner.
Neither this measurement nor that multiplier is a worst-case bound on the
full worker, native stepped physics, PPO retained objects, audit and final
publication. The owner accepts implementing a risk-limiting watchdog, **not**
an assertion that memory admission has already passed.

## Required implementation

- For prospective portable v2 resource admission, bind the exact
  \`nca-jolt-003-process-tree-memory-watchdog/v1\` contract:
  8,589,934,592 bytes total **sampled** original observer and owned
  descendants (8 GiB), a 0.25-second sampling interval and fail-closed
  reaction to sampled overage or unreadable live process memory.
- Keep the existing Linux subreaper and PID-safe original process-tree
  revocation/reaping; every memory failure is a failure of the **same**
  one-use allocation. Never automatically retry, reseed, continue training
  or create another experimental output.
- Retain an original, separately hashable memory-watchdog receipt
  showing the sampled peak, number of samples, threshold crossings,
  monitoring errors and original tree cleanup status. Upload the
  receipt alongside partial/original raw evidence even on failure where
  runner survival permits.
- This is a **soft supervisory threshold**, not a cgroup or kernel
  \`memory.max\` enforcement. It cannot prevent instantaneous/sudden
  allocations, memory pressure from unrelated runner processes or kernel
  OOM before the next sample. Never present it as a hard upper bound.
- Publish source/config/runtime/host-bound assessment **drafts** from
  independent diagnostics, with unresolved \`peak_resident_memory\`
  findings retained. If an independent risk review later accepts this
  limitation, that reviewer must explicitly own the decision and its
  evidence; synthetic samples alone cannot set a gate true.
- Two separately authored, truly independent source-bound review records
  are required: terminal-learning/qualification and operations/serialization/
  resource envelope. Each binds exact reviewed executable source SHA,
  closure, pinned host/CPU software contract, configuration, prospective
  binding, and disclosed unresolved findings. Merely generating two JSON
  forms does not count as independent review.

## Unchanged constraints

The physical learner remains the fixed CPU-only LEARN-JOLT-003 workload:
39 counter ceilings, seeds and case ordering unchanged, original 5 ms
inference p95 and learning gates unchanged, sealed barrier upheld.
CPU model is advisory under D-022, with only approximate diagnostic
speed normalization. The GitHub six-hour hard job lifetime remains
outside repository control. Disk must be rechecked on the actual
execution runner. Existing LEARN-JOLT-002 failed evidence stays immutable.
No experimental \`execute\` invocation is authorized by a successful
watchdog unit test, synthetic memory CI run or unsigned review packet.
