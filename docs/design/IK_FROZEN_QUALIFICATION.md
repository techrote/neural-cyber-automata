---
title: Fresh qualification of frozen CPU IK inference
kind: guide
summary: New model-free sealed suite and single-use qualification for unchanged checkpoints and thresholds.
owns: [design.ik_frozen_qualification]
verified: 2026-10-28
---

# Frozen CPU inference qualification

[PREPARATION] [D-037](../decisions/2026-10-28-frozen-ik-qualification.md)
commissions a new qualification after the [tuning-only CPU improvement](../evidence/2026-10-28-ik-cpu-development.md).
The original [final no-go](../evidence/2026-10-28-ik54-final-qualification.md)
and all its artifacts remain unchanged and excluded from this evaluation.

## Fixed candidates and inference

Keep seeds 41011/42013/43019 and selected epochs 755/755/915. Their checkpoint
SHA-256 values are respectively
`436c0499cd4eab533ed10315b63c229c6c52ad5af5ef61e278b91ad759f0f609`,
`d43ee34c7b541f11fecd06f833b43c2dc7999b1bb71c9672393ac6d8ce93832e`,
and `45bb2d0e4260279f99425d7f04e0790fcdefb7eb6da038ee8f5fac90471958b5`.
Authenticate the full completed training ledgers, all 600 tuning reports,
selection event membership and actual initial checkpoints. Rebuild the candidate
freeze against the new configuration, seal, archive and registry identities.

Both trained and actual initial v3 controls use the unchanged
`runner-polar-cpu-frozen/v1` adapter. Bind its file hash, the local Torch tracing
source hash and the exact runtime. A separate train/tuning-only proof checks
both models on all 2,020 tuning rows per seed, preserving statuses and joint
parity <= 2e-5 radians. Its finite allowance is 8,080 scheduled predictor calls
plus 16 prepaid synthetic graph forwards per seed, 3,600 whole-process seconds
and 8 GiB process commitment. Record observed process exit before packet freeze.

Graph construction occurs before final opening and outside measured decisions.
Two graph builds have six observed eager-hook forwards plus internal JIT
replays: the reviewed pinned implementation uses at most ten synthetic model
evaluations, conservatively prepaid as sixteen. Record eager-hook counts
separately rather than calling them all construction work.

## New model-free suite

`FrozenFinalConfig` is an explicitly registered final-only recipe, rather than
an unrecognized subclass of the consumed recipe. It fixes generation seed
53029 and family namespace `runner-frozen-cpu-final-v1`. Ordinary hip lanes
are (.54,.58) within each grid bin, withheld lanes (.96,.98), and boundary
stress uses the (.975,.98) ankle strip. Counts remain 2,000 nominal points,
40 complete twenty-step trajectories and 140 stress cases: 2,940 total,
including at least 200 withheld point queries. Teacher settings are unchanged.

Before any teacher call, freeze the recipe and verify all original 7,280
training/tuning tasks against the new geometry: no shared task/input/family/
trajectory identity, root point and complete piecewise-linear path clearance
strictly greater than .0005 cells, and joint-path clearance greater than .002
radians. The independent reviewers verify this isolation evidence. Namespace,
seed and nominal ranges differ from prior recipes; there is no claim of a
whole-prior-final stress/path guard and no consumed payload is read or regenerated.
Model-free deterministic conformance may verify old recipe identity hashes.

Preparation/isolation has its own 3,600-second/1-GiB hard envelope. Generation
has one attempt, no retries, 2,940 teacher reservations, 188,160 worst-case
iterations, 900 whole-process seconds and 512 MiB per-process commitment.
The custodian imports no Torch and sees no checkpoint/model outputs. It journals
reservations before teacher calls, publishes only a sealed archive and metadata,
and independently hashes its own archive publication before declaring completion.

## Separate qualification allocation

The new packet permits one group attempt, zero optimizer work, no reselection
and no retry. All three seeds reserve 78,240 final/timing decisions plus sixteen
synthetic construction forwards before graph work or opening. Five official
comparators retain the existing binary64 scorer; archived trained v1 is an
additional diagnostic. The new archive is opened atomically only after two
genuine exact-packet admissions, owner execution record, runtime/CI/custody
checks and all reservations. Opening is read back before archive hashing or
decoding. Preparation/evaluator preopening inspection reads metadata/stat only.

Retain 24 hours for the whole process with a 7,200-second work-stop reserve,
8 GiB Windows process cap, 32 GiB startup/post-preflight free host commitment,
16 GiB free disk, and 4 GiB continuous host/disk floors from attempt claim.
The 20 GiB aggregate TITAN watchdog is UUID-targeted; V100 is prohibited.
Keep immutable failure evidence, terminal readback, watcher shutdown, ownership
release and observed OS exit as completion requirements.

Accuracy remains p95 <= .06 cells and <= .01 L, worst <= .18 cells and <= .03 L,
>=95% success, plus finite/bounds/protocol/continuity/branch gates. Learned
contribution and CPU practical utility retain their existing definitions.
Timing remains one thread, batch one, 100 warmups and 10,000 measured complete
decisions per comparator, including independent bounds/FK checks. The cost
ratio limit stays <= .8 against the same-quality numerical reference, with
analytic cost reported. There is one fixed measurement per seed; no best-round
substitution. Passing software scoring is not export or physical acceptance.

© 2026 techrote. All rights reserved.
