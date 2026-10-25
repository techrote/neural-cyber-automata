---
title: 4LY-004 experiment-admission package
kind: evidence
summary: Exact public split manifests, pinned CPU runtime, finite whole-attempt resource enforcement, and external custody/review handoff for Trainer flight learning.
owns: [flight.learning.admission]
verified: 2026-10-25
---

# 4LY-004 experiment-admission package

This record follows merged bridge-qualification PR #85 at
`c76c8b3b385b85567b6c735b6811a6c6f01cd42d`. It prepares the remaining
prospective admission boundary for issue #74. It does **not** collect teacher
data, run behavior cloning, issue an experiment allocation, open learner-final
bytes, initialize CUDA or score a trained policy.

## Frozen public experiment inputs

The public case sets are now explicit repository artifacts:

- development: `configs/flight/4ly004-development-v1.json`, 44 cases,
  68,400 maximum physical ticks, 11,400 maximum decisions, canonical SHA-256
  `8b68faa6d3bf6172d0ba1dcf6d2752aae6ff9e83d521cfe33cf537cbbad9e5e7`;
- tuning: `configs/flight/4ly004-tuning-v1.json`, 40 cases,
  67,200 maximum physical ticks and 11,200 maximum decisions per checkpoint,
  canonical SHA-256
  `0295a669549bc7cf852fd12934d85506ef1f55ceec07340531f801084e08b3df`.

Both bind accepted Cybersand #249 host merge
`9935d5696721d2c72a8273ea70454ab0c5b11361`, build identity
`55e5ba334b6683adf2ba0d8fc93fb7defba109cd4a319d2f366a2a86e4dfdf84`,
course `4ly-flight-course/v1` and split geometry v3. No final geometry or
hidden final seed is present in Git.

The admitted learner runtime contract is Linux x86_64 on GitHub-hosted
`ubuntu-24.04`, Python `3.12.10`, Torch `2.8.0+cpu`, one Torch thread and
no CUDA. `tools/flight_admission.py` refuses runtime drift and computes an
exact clean-source closure for the flight adapter, transport, accounting,
learning/admission source and public split manifests on the checked-out commit.

## Finite whole-attempt resource policy

The admission package converts the previously-null whole-attempt resource
fields into explicit fail-stop ceilings:

| Resource | Ceiling |
|---|---:|
| whole-attempt wall time | 18,000 s |
| process-tree resident memory | 4 GiB |
| retained attempt storage | 8 GiB |
| publication bytes | 4 GiB |

These are **allocation caps, not performance predictions**. The retained
final-head bridge qualification measured 166.3225 confirmed physical ticks/s
on its slower confirmation run. At the complete 1,475,280-tick ceiling that
physical-only projection is 8,869.997 s; the 18,000 s cap leaves more than
2x that physical-host projection for learner/update/evidence work while still
terminating a nonconvergent attempt. It does not infer Torch throughput from
host throughput.

The 8 GiB storage cap covers the 63-publication checkpoint topology even at the
existing 32 MiB per-checkpoint software cap plus capture, tuning/final traces,
journals, manifests and readback. The 4 GiB RSS and publication limits are
deliberately conservative hard bounds. Exceeding any cap makes the attempt
fail-stop/inconclusive; it does not authorize a larger retry.

`FlightResourceGuard` samples the root process plus Linux `/proc`
descendants every 0.25 s, checks attempt-root regular-file storage without
following symlinks, tracks publication byte charges and refuses further
reserved work after a violation. Production declarations using the accepted
resource-enforcement evidence cannot open or resume a
`FlightAccountingLedger` unless the continuous guard is already running and
bound to the same declaration ceilings. Resource policy canonical SHA-256 is
`41bb1d5bd67e80ce479f8ec7c94abffd4fc9fbd97f8b69cd77c1ccbab17365c8`.

Stage accounting remains exact and seed-isolated. Capture, tuning and each
`clone:<seed>` stage retain their existing ceilings; final-stage work remains
disabled before a separately recorded final-open barrier.

## Final custody and independent review

The repository now defines, but does not self-satisfy, the two genuinely
external gates.

`configs/flight/4ly004-final-custody.example.json` specifies the metadata an
external custodian/private reviewer must return. The executor receives only a
suite SHA-256, byte count, exact 60-case family/scenario composition and
attestations that the payload is privately disjoint/feasible, remains outside
Git, was not opened by the executor and does not reuse the teacher-final suite.
The actual suite bytes are neither generated nor requested by this package.

`configs/flight/4ly004-independent-review.example.json` defines the separate
execution review. To avoid circular self-approval, the reviewer signs a stable
`review_binding_sha256` over the source closure, public split hashes, external
custody digest, runtime, host/build, resource policy, stage-accounting evidence
and `final_access=false`. Only an accepted independently authored record
upgrades the declaration's `independent_execution_review` dependency.

Without custody, the candidate state is
**READY FOR EXTERNAL CUSTODY**. With custody but no review it is
**READY FOR INDEPENDENT EXECUTION REVIEW**. Only both can produce an
`ADMITTED` prospective declaration, and even that declaration keeps
`final_access=false`.

## CI and evidence boundary

`.github/workflows/flight-admission.yml` is nonexperimental. It installs the
pinned CPU runtime, runs all focused flight software tests and emits an
exact-head source/split/admission candidate artifact. Routine CI explicitly
requires the candidate to remain nonadmitted because repository CI does not
possess external learner-final custody or independent execution-review records.

No failed or missing external gate is represented as success. No teacher-final
payload is reopened. No experimental capture, optimizer update or trained
checkpoint is produced by this admission package.

© 2026 techrote. All rights reserved.
