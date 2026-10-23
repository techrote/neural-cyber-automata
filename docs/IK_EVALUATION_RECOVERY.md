---
title: Frozen IK evaluation recovery operations
kind: guide
summary: Same-candidate recovery with retained opening and failures, original budgets, accounted historical work and fresh admissions.
owns: [ik.evaluation.recovery.operations]
verified: 2026-10-23
---

# Frozen IK evaluation recovery

[PROSPECTIVE D-026] The owner explicitly authorizes recovery of the same frozen
candidates after the [recorded archive-layout failure](evidence/2026-10-23-ik-titan-continuation.md).
Implementation and review alone do not admit execution. No retraining is allowed.
The original opening remains opened/development-only and every original failure,
overrun, declaration, checkpoint and charge remains unchanged.

Use the qualified Windows environment and set `CUDA_VISIBLE_DEVICES` to the
single private TITAN RTX UUID. Publish only its fingerprint. V100 initialization
is prohibited. CPU preparation and header inspection must import no Torch and
extract no final or pending payload.

The archive member prefix is `runner-ik-corpus-v1-20261022/corpus/`.
Authentication checks the unchanged archive hash, safe unique regular member
headers and declared sizes before selecting payloads. The same frozen set has
semantic SHA `e9e1ea0e1bb4831b4323a78fef14a9a1012c97c12a61eae8833f5933c0161b85`.
Candidate selection, corpus, normalization, model, optimizer, comparison suite
and numerical thresholds cannot change.

Before touching actual ledgers, complete CPU tests, independent source review,
exact-source CI and a full metadata/accounting rehearsal on faithful copies of
the failed run and registry. The rehearsal is not an experimental invocation:
no model, prediction, final payload or CUDA operation occurs. Verify that real
run and marker bytes remain unchanged throughout.

Real preparation must finish read-only corpus/history/freeze validation before
appending any charge. It then records explicit historical failed work of at
least 131 seconds per seed, plus measured recording overhead and a bounded
publication reserve, inside each original 7,200-second allocation. Accounting
consumes the failed attempt slots; actual recovery uses attempt 4 for seed 41011
and attempt 3 for seeds 42013/43019, within the original maximum of four.
Original operation counters and checkpoint bytes do not change.

```powershell
python -m nca.evaluation.ik_recovery prepare --run ORIGINAL_RUN_DIRECTORY --failed-amendment FAILED_EVALUATION_AMENDMENT_JSON --failed-amendment-sha256 FAILED_EVALUATION_AMENDMENT_FILE_SHA256 --corpus ORIGINAL_CORPUS_DIRECTORY --registry ORIGINAL_REGISTRY_DIRECTORY --archive ORIGINAL_ARCHIVE --output NEW_RECOVERY_DIRECTORY
```

The recovery declaration binds the exact retained failure/opening/binding files,
same candidates, source/runtime/UUID fingerprint, original allocations and paid
reconciliation. Freeze source after preparation. Publish exact declaration and
source hashes and obtain two genuine fresh independent admissions. Historical
reviews cannot authorize recovery. The aggregate schema is
`nca-ik-frozen-recovery-admission/v1`, with the five original checks and verdict
`admit_declared_frozen_final_recovery`.

```powershell
python -m nca.evaluation.ik_recovery run --run ORIGINAL_RUN_DIRECTORY --amendment RECOVERY_AMENDMENT_JSON --amendment-sha256 RECOVERY_AMENDMENT_FILE_SHA256 --failed-amendment FAILED_EVALUATION_AMENDMENT_JSON --failed-amendment-sha256 FAILED_EVALUATION_AMENDMENT_FILE_SHA256 --corpus ORIGINAL_CORPUS_DIRECTORY --registry ORIGINAL_REGISTRY_DIRECTORY --archive ORIGINAL_ARCHIVE --review RECOVERY_ADMISSION_JSON
```

Only exact historical failures backed by durable paid accounting may be
reconciled. Ordinary readers and launchers still refuse them. Recovery-only
corpus access cannot authorize optimization. The existing opening and binding
are read back unchanged; a separate recovery binding records new execution.
New uncertain work is not automatically reconciled or retried.

Independent terminal verification must check unchanged model/optimizer/buffers,
all original charges and ceiling compliance, every comparator denominator,
100 warmups and 10,000 measured CPU calls per comparator, raw timing summaries,
source/checkpoint/ledger hashes and the aggregate verdict. Report failed,
no-go and inconclusive results honestly; no final-driven retuning is permitted.

© 2026 techrote. All rights reserved.

Historical observed outcome: [launcher refusal and required source/accounting rebind](progress/2026-10-23-ik-recovery-refusal.md). No recovery evaluation result exists.


## Source rebind after the pre-supervisor launcher refusal

The owner commissions a fresh declaration rebind for the same experiment.
This uses `python -m nca.evaluation.ik_recovery rebind` with the original run,
refused parent amendment and file hash, failed evaluation amendment and hash,
original corpus and registry, pinned refusal observation and a new external
output directory. It does not rerun `prepare` or overwrite either declaration.

Read-only preflight authenticates the exact refused parent, paid historical
records, original candidates and unchanged ledgers before any debit. The new
immutable declaration retains the old paid authority and separately names the
new clean execution source. An append-only overhead envelope pays ten seconds
for the measured 5.962-second launcher refusal, measured recording work and a
sixty-second publication reserve per seed. Attempts and all operation counters
remain unchanged. Later attempt starts include this debit in their prior wall
and parent chain; dropping it invalidates continuation.

Rehearse the full rebind/load on faithful copies first. After exact-head CI,
prepare the real rebind once, publish its file and semantic hashes, and obtain
two fresh independent declarations of admission. Use the rebound amendment in
the same `run` command above. Historical admissions cannot authorize new source.
No automatic retry, training, new allocation, candidate selection or reset is
permitted. Preserve every refusal and account its overhead before further work.
