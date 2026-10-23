---
title: Bounded polar-feature IK follow-up commission
kind: decisions
summary: Owner-authorized implementation and gated new TITAN allocation with fresh final evaluation and unchanged criteria.
owns: [decisions.ik_polar_followup]
verified: 2026-10-23
---

# Bounded polar-feature IK follow-up

[ACCEPTED D-027] The owner explicitly requests implementation of the
[target-feature follow-up](../design/IK_ACCURACY_FOLLOWUP.md), verified fresh
held-out isolation, exact-head CI, two genuine independent admissions and
TITAN RTX execution with the tuning stop gate. This is new prospective work,
not a reset or continuation of the completed v1 allocation.

Keep model width/depth, losses, optimizer, original training seeds and finite
training schedule fixed. Preserve every old corpus, normalization, opening,
checkpoint, refusal and charge. Version the transformed training-only normalizer
and final-only geometry. Do not use old final cases for optimization or tuning.
Acceptance thresholds remain unchanged. If any seed misses the original raw
quality gate on tuning, stop and keep fresh final sealed.

The initial proposed final lane fails existing guards. A deterministic revision
before any labels/scores uses ordinary lane (0.74,0.78), withheld strip
(0.02,0.06), and new family/trajectory namespace at seed 53023. Geometry replay
passes; label generation and sealing remain separate prerequisites. No guard
is relaxed. The boundary stress diagnostics are not nominal patch separation.

Fresh exact-source reviews, immutable source/environment/normalizer/suite
identity and cumulative allocation records precede CUDA. A review refusal,
missing launcher or absent sealed-suite evidence cannot be turned into approval.
Do not reuse v1 declarations or same-frozen-evaluation recovery privileges as
new-training authority. No new allocation has yet been created or consumed.

## Resource amendment

[ACCEPTED] The subsequent owner instruction raises the new polar experiment's
GPU allocator ceiling to 20 GiB (21,474,836,480 bytes). This is a ceiling, not
a reservation or a requirement to fill memory. Original v1 declarations and
2 GiB historical ceilings remain unchanged. TITAN RTX remains the only target.

Parallel seeds were considered. This bounded comparison uses serial seed
workers and serial matched CPU timing: prior small-network CUDA computation
was a minor part of wall time, and concurrent seeds would need an aggregate
VRAM guard rather than three independent 20 GiB ceilings. No concurrent model
or CPU benchmark work may contaminate the matched latency measurement.
Budgets, model, sampling, thresholds and all-seed tuning stop remain unchanged.

© 2026 techrote. All rights reserved.
