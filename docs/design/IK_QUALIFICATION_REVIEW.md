---
title: IK54 independent qualification review and versioned repair
kind: guide
summary: Two non-author refusals of the v3 packet, a separate large-run ledger repair and fresh exact-packet admission before any training.
owns: [design.ik_qualification_review]
verified: 2026-10-25
---

# IK54 qualification review

[OWNER SCOPE] The 24 October instruction commissions two independent non-author
reviews, remediation and more generous general training/resource rules. It
requires finals to remain excluded and stops before training. [D-033](../decisions/2026-10-24-training-resource-policy.md)
records the general policy; the original [development results](../evidence/2026-10-24-ik-dev-sweep.md)
and v3 declaration remain unchanged.

## Original exact-packet findings

Two independent reviewers refused source `8a0cd49e21497ebab2c5d8290169f158a5bb8969`
and original declaration file SHA-256
`db037796d101c35470c1cb1f2722ce2192b312dced3d777a60472b5b5e4d1867`.
Full execution creates 86,802 event files per seed. Even a minimal inventory
requires more than 9.3 MB, exceeding the historical ledger's 8 MiB JSON bound.
The actual publication function refuses the synthetic lower bound. No training
could make that packet complete successfully.

The 60-second paid publication reserve also lacked scale evidence. An independent
2,000-pair synthetic ledger measured approximately 17.54 seconds for one 4,000-event
inspection. Linear projections are warning evidence, not a full-scale measurement.
Both reviewers required concurrent scale-matched I/O and complete process readback.

## Versioned repair

The historical ledger and v3 launcher/configuration are preserved. New ledger v2
uses a distinct schema, bounded 64 MiB metadata, a 1 MiB checkpoint cap and support
for finite local allocations up to 72 hours. The new v4 qualification path retains
the same frozen hyperparameters, seeds, full training partition, tuning cadence,
one attempt, selection rule and acceptance. Its prospective resource declaration
uses twenty-four hours per invocation and a charged 7,200-second worker publication reserve.
Work admission stops with 10,800 seconds remaining. The versioned recipe is
[preregistered](../../configs/ik/experiments/runner-qualification-v4.json); preparing
a new external packet grants no execution authority.

The hard wall timer is independent of potentially blocking telemetry queries.
Board/thermal and host-commit monitoring continue through supervisor finalization;
the inherited Windows Job enforces per-process memory and kills children on exit.
Terminal admission requires completed status and every exact declared counter.
Supervisor readback authenticates every tuning report, selected checkpoint and
earliest complete minimum without making another model call.

## Owner stop — 24 October 2026

The owner instructed: "Don't worry about the review for now. Record and stop."
Implementation is stopped. A subsequent owner instruction authorizes committing
and pushing this preserved work and the resource rules to the existing draft PR.
The qualification repair remains unfinished and unadmitted; this publication
authorizes no further reviews or training.
No v4 recipe or declaration exists, no fresh final-source admission was obtained,
and no qualification training was started. The original v3 packet remains unchanged.

The completed CPU scale fixture sealed all three full-size ledgers, with terminal
inventories of 9,636,149–9,636,160 bytes. Three-worker process lifetime was
1,224.763 seconds; the serial supervisor tail was 213.379 seconds. A separate
CPU serialization/parser fixture measured a maximum 114,816-byte checkpoint,
0.0162-second serialization and 0.0666-second real-shaped report parse/gate.
The hard-exit/child-kill fixture completed in 1.355 seconds. These are storage,
serialization and shutdown measurements, not training throughput or acceptance.
The scale fixture predates later timing/admission guard changes; it does not
qualify the preserved unfinished source. External artifacts retain exact bindings.

The last completed CPU suite passed 524 tests with four skips in 130.480 seconds.
The recorded CPU result is not exact-head CI admission for this repair. Resuming would require completing
and freezing the recipe/packet, final-head checks and the required admissions;
these steps were not performed after the stop instruction.

## Resumption - 25 October 2026

The owner's "Continue" resumes the unfinished preparation and two independent
reviews. It does not execute Part B: the separately required owner go remains
absent. The preserved v3 refusal, previous stop and publication remain historical.
The v4 recipe restores every original training family, fixed tanh/cosine/1,000
recipe, all three original seeds and five-epoch tuning cadence. Per seed:
43,000 optimizer steps, 5,260,000 presentations, 404,000 tuning calls, 201
checkpoints, one attempt, zero final calls and no extension or re-selection.
All original acceptance thresholds and gates remain unchanged.

The prospective 24-hour ceiling replaces the unfinished eight-hour draft before
any v4 packet is frozen. Existing v3 and consumed allocations remain unchanged.
It is a ceiling, not a commitment to spend all of it. The slower historical
three-worker matched harness took 1,245.062 seconds for 40 epochs; multiplying
by 25 gives a conservative cross-harness warning of 31,126.55 seconds, already
beyond eight hours. This extrapolation is not measured v4 throughput. The actual
development trainer completed the winning 1,000-epoch runs in at most 2,510.438
seconds under up to eight workers. Scaling the entire duration by the largest
full/dev count ratio (2,020/787), adding the complete 1,224.763-second storage
fixture and 10,800-second reserve gives about 18,469 seconds plus preflight.
Both are planning estimates with retained limits; a hard stop enforces the ceiling.

A fresh independent cold-disk replay took 543.296 seconds for one ledger,
versus the saved supervisor's approximately 70 seconds. That variability is
retained. A 10,800-second work reserve leaves 3,600 seconds before the new
7,200-second paid publication reserve. The earlier full-scale fixture measured
first inventory about 464 seconds, worker finish 658 seconds and serial tail
213 seconds. The fresh cold replay suggests allowing substantially more I/O
headroom. The only ledger change since that fixture raises the allowed reserve
upper bound; hashing, inventory and readback semantics remain unchanged.
A focused no-model fixture checks the new reserve and rejection above it.
The supervisor timer tightens to the earliest paid seed envelope and remains
active through interpreter exit. Startup requires 16 GiB free storage and
32 GiB available host commitment; monitoring retains 4 GiB disk/host floors.
Fresh reviewers assess transfer, later guards, storage and composite costs.
No successful fixture guarantees a later run.
No new learner or CUDA probe is part of this preparation.

The public [resource artifact index](../evidence/ik54-qualification-v4-resources-20261025.json)
records exact retained fixture hashes and measurement limits.

## Current blocker - target metadata unavailable

At source `da0a3763e9fa7d60c59dfd629dfcd184674507b6`, actual packet
preparation invoked the existing target-only visibility check. The metadata
query reported a lost GPU and required a system reboot. Preparation refused
before creating its packet root, allocations or attempts. No CUDA context,
training, ownership claim, owner execution record or final call was created.
The [append-only refusal record](../evidence/ik54-v4-preparation-refusal-20261025.json)
binds the exact recipe, source and retained error log. No gate was bypassed,
no alternate device was queried or used, and no host repair was performed.

The final CPU suite passed 526 tests with four skips in 151.158 seconds.
This completes software validation locally, not hardware readiness or packet
admission. After target metadata is recovered, new clean-source preparation,
measured startup validation, exact-head CI and two fresh exact-packet admissions
remain necessary. The owner explicitly requires stopping before the long GPU run.

## Dated recipe annotation corrections - 25 October 2026

The frozen v4 recipe retains the inherited admission prose referring to a
60-second publication reserve. That sentence describes the refused v3 packet
and is stale for v4; it does not set an executable limit. The v4 numeric
`resource_policy.worker_publication_reserve_seconds` and source both specify
7,200 seconds, with a 10,800-second work reserve and 24-hour ceiling. This
additive correction preserves the frozen recipe bytes and their refusal binding.
Fresh reviewers must use those exact numeric fields and this correction.

The recipe also preserves the inherited 168-entry `executable_source_files`
table from v3. It is historical. The new 170-entry `executable_files` table
and `executable_source_sha256` bind v4 and match the current `source().files`
closure. The external declaration must bind that exact current closure and head;
the old table cannot substitute for it. Both tables remain preserved unchanged.

## Resource proof and final review

[PREPARATION ONLY] A new CPU storage fixture exercises three concurrent workers,
each with 43,000 synthetic optimizer reservation/completion pairs, 200 tuning
pairs and 201 checkpoints. It uses inert 1 MiB checkpoint files and valid inert
8 MiB report files at the declared storage caps. Real model/optimizer, CUDA,
corpus and final calls are zero. The real ledger validates byte hashes, state,
inventory, both worker terminal readbacks and the serial supervisor tail. Only
the checkpoint writer and synthetic score predicate are fixture replacements.
This measures storage/accounting cost, not learner throughput or learned quality.

The final packet and clean source must receive two fresh independent reviews.
The reviewers independently verify source/declaration hashes, exact-head CI,
authenticated training/tuning identity and schedules, finite allocations,
selection, failure holds and the complete resource envelope. A new owner go
remains required even if both admit. No final access is included.

© 2026 techrote. All rights reserved.
