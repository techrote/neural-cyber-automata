---
title: Prepare matched endpoint-loss IK trials without training
kind: decisions
summary: Owner-selected preparation-only serial/parallel comparison with fixed loss intervention, unchanged gates and both final suites excluded.
owns: [decision.ik_matched_preparation]
verified: 2026-10-23
---

# D-029 — Matched IK trial preparation only

[ACCEPTED SCOPE] 2026-10-23. The owner requests investigation of the polar-v2
regression using training/tuning evidence, a reviewed next experiment, and a
matched parallel run to record evidence for future concurrency. When asked
whether to execute after gates or prepare both, the owner selected:
**"Prepare both; do not train yet".**

Prepare the [endpoint-loss ablation and matched serial/parallel packets](../design/IK_ENDPOINT_MATCHED_TRIALS.md).
The endpoint coefficient 4.0 is a single untested hypothesis, not a proven
accuracy fix or accepted optimization. It is separate from D-028's batch-order
proposal; preserve the original schedule in both matched packets. Preserve
all historical allocations, failures, charges, models and terminal markers.
Do not use either final suite, saved final score or payload for this diagnosis,
training, tuning or candidate selection. Evaluation allowance is zero.

The sole future GPU target is the UUID-isolated TITAN RTX. Prepare three fixed
seeds serially and, separately, at most three simultaneous seed processes under
one aggregate 20 GiB resource envelope. This is a concurrency proposal subject
to actual resource admission, not evidence that three workers are safe or fast.

CPU preparation, software fixture checks and independent design reviews are
authorized. Training, benchmarks and CUDA initialization are not authorized.
Future execution needs a reviewed group supervisor/watchdog, exact-head CI,
two fresh independent source/declaration-bound TITAN execution admissions and
an owner instruction to execute. Preparation reviews cannot stand in for them.

© 2026 techrote. All rights reserved.
