---
title: D-022 — Jolt CPU portability and diagnostic normalization
kind: decisions
summary: Record changing CPU hardware without using its identity as an execution-admission gate; normalize only approximate diagnostic time.
owns: [learning.jolt003-cpu-portability-policy]
verified: 2026-10-23
---

# D-022 — CPU identity is recorded, not enforced

[ACCEPTED] 2026-10-23 by owner: do not enforce exact CPU identity
between GitHub-hosted runners. Record the hardware and normalize timing
comparisons for ballpark planning instead. This changes neither the
LEARN-JOLT-003 learning protocol nor permission to consume the experiment.

## Versioned compatibility policy

The current unlimited-time Jolt configuration v2 is unchanged and historical
configuration bytes are preserved. A distinct resource-admission schema,
`nca-jolt-003-resource-admission/v2`, and source-admission review schema,
`nca-learn-jolt-003-source-admission-review/v2`, bind the portable
`nca-jolt-cpu-runtime-contract/v1`. Both are required to use portable
review semantics; old v1 review/resource records retain their exact-runtime
interpretation.

The pinned *software and execution contract* remains strict: Linux x86_64,
Python 3.12, Torch 2.8.0+cpu, NumPy 2.3.5, CPU-only, single Torch/interop
thread, and deterministic algorithms. The exact executable commit, complete
source-file hashes, config/hash, 39 counters, training seeds, policy, host
revision/tree/binary identities and independent review admission remain
unchanged.

The runtime declaration and execution receipts still record the original
CPU model, kernel release, libc version, Python patch level and available
logical CPU count, but the v2 resource and two v2 review bindings compare
only the pinned CPU *capability* SHA-256, not those volatile hardware
details. Reviewers also bind `prospective_binding()`; for portable v2
resource records, its runtime projection is the capability digest. A
different CPU or kernel on an ephemeral GitHub runner is no longer a
sufficient reason to refuse an otherwise identical software/host
experiment. Hardware omissions or malformed telemetry still fail.

## Approximate performance normalization

Preflight and nonexperimental memory diagnostics record the exact runtime
metadata plus a fixed SHA-256 64 KiB buffer throughput calibration:
1,024 digest operations per repetition, repeated three times. It is a
single-core speed proxy, not an actual Torch, physics or full-worker
performance benchmark.

Given measured elapsed seconds `t`, observed calibration rate `R_obs`
and explicitly chosen reference rate `R_ref`, the approximate normalized
time is `t_ref = t * R_obs / R_ref`. Publish raw values, both measured
rates, method and sample conditions. Do not silently normalize memory
bytes, trained-policy outcomes, observed native physics, decision/tick
counts or the **actual per-case inference p95 <= 5 ms acceptance gate**.
Neither the calibration factor nor historical component timing constitutes
full-pipeline resource feasibility.

## Admission boundaries

This is **not** an automatic memory-admission waiver. Existing required
storage/reader/disk/peak-memory gates, the one-shot permanent allocation
claim, two genuinely independent source reviews and final-source CI remain.
The prior native/synthetic memory report is historical planning evidence,
not a hard worker/native peak. An accepted resource record and two accepted
independent reviews must still be prospectively published and independently
read back before an experimental `execute` dispatch. Missing evidence is
not treated as zero and the original consumed LEARN-JOLT-002 attempt
cannot be resumed.
