---
title: IK compact CUDA runbook
kind: guide
summary: Exact authenticated corpus preflight, independent admission, supervised seed training and CPU conformance commands.
owns: [ik.training.operations]
verified: 2026-10-23
---

# IK compact CUDA runbook

**Current result:** [same-candidate recovery](evidence/2026-10-23-ik-frozen-evaluation-recovery.md)
completed final scoring and CPU timing with accuracy/utility no-go and
contribution pass. All ledgers are completed within original allocations.
Historical failure/opening records remain immutable; no training or reset
occurred. Commands below preserve operations and do not authorize replay.

The later explicit [D-026 recovery commission](decisions/2026-10-23-ik-evaluation-recovery.md)
permits a separately reviewed [same-candidate recovery workflow](IK_EVALUATION_RECOVERY.md).
It preserves the failure/opening records and original allocations, accounts
historical costs and requires two fresh admissions; it grants no retraining.

[PARTIAL] This operational continuation of the [IK training reference](IK_TRAINING.md)
preserves the accepted-corpus CPU preflight, prospective three-seed CUDA gate,
and focused software validation. The [measured continuation](evidence/2026-10-23-ik-titan-continuation.md)
completed all three training schedules; its initial failed evaluation is preserved.
The recovery linked above supersedes only the incomplete evaluation status.

## CPU preparation, then the GPU boundary

Install the already pinned optional runtime using [runtime operations](IK_RUNTIME.md).
Keep the extracted archive, original registry and new run directory outside the
source checkout. **TITAN RTX is the only authorized target for this training
programme.** The previously used V100 has a reported cooling fault: it is
**out of service and prohibited for all CUDA initialization, training, smoke
tests and performance qualification**. Its earlier #51 result remains historical
provenance, not a TITAN qualification or a performance baseline.

The prospective target is Windows 11 / CPython 3.12.10 / Torch 2.8.0+cu126 /
NumPy 2.3.5 / TITAN RTX (compute capability 7.5, 72 SMs) /
driver 581.80. The TITAN's **actual** Torch-reported memory is measured during
the admitted GPU invocation, not copied from the V100. At least 20 GiB must
be reported; the unchanged allocator budget is 2 GiB.

**Mandatory CPU-only device isolation before preparation:** set
`CUDA_VISIBLE_DEVICES` to the TITAN RTX **GPU UUID alone**. Obtain the true UUID
from existing authorized Windows hardware records (a metadata-only inventory
is permissible); do not substitute `0`, `1`, a GPU index list or any V100 UUID.
For example in PowerShell, replacing the placeholder with the actual TITAN UUID:

```powershell
$env:CUDA_VISIBLE_DEVICES = 'GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'
nvidia-smi -i $env:CUDA_VISIBLE_DEVICES --query-gpu=name,uuid,driver_version --format=csv,noheader
```

The command must report **TITAN RTX** and driver **581.80**. It targets that
UUID for a read-only metadata query; it does not run CUDA kernels. The
`prepare`, supervisor and worker independently enforce the single-UUID binding
and exact driver. `cuda:0` thereafter means **the isolated TITAN**, never a
physical motherboard enumeration index. Its UUID is not published; a SHA-256
fingerprint binds the private execution declaration to the selected card.

Preparation uses CPU corpus validation without importing Torch or initializing
CUDA. It checks the original corpus, retained labels, training-only normalization,
and the source/runtime/visibility binding before producing a declaration:

The Windows label compatibility inventory permits only exact, enumerated
original/replay label digest pairs after original-byte authentication; see the
[training reference](IK_TRAINING.md#authenticated-corpus-and-deterministic-sampling).
Task compatibility likewise requires exactly one of 17 pinned whole-task digest
pairs, with original task inputs retained. The full Windows CPU reader now
passes task/label/split checks and exact training-only normalization; the
clean-source `prepare`, final CI and prospective admissions remain required.
Do not create an admission record or run CUDA while any of these gates refuses.

```sh
python -m nca.train.ik_experiment prepare --corpus CORPUS_DIRECTORY --registry ORIGINAL_REGISTRY_DIRECTORY --output NEW_RUN_DIRECTORY --device cuda:0
python -m nca.train.ik_experiment inspect --run NEW_RUN_DIRECTORY --declaration-sha256 DECLARATION_FILE_SHA256
```

Use the existing activated PowerShell environment on the designated Windows
machine. Preparation refuses different OS/Python/Torch/NumPy metadata or absent
TITAN-only visibility *before* CUDA initialization. Preparation publishes exact
source, TITAN UUID fingerprint, environment, corpus/normalizer, three seeds,
all allocations and the unchanged acceptance identity in an immutable declaration.
A historical V100 declaration or review **cannot** authorize a TITAN run;
obtain fresh prospective TITAN-specific reviews against this new executable
source and declaration.
Independent admission binds that exact declaration and source and checks corpus,
resume/accounting, scoring/selection, CPU checks and resource feasibility. Use
the declaration's semantic `sha256` and `source.source_sha256` in the review record;
the command's `--declaration-sha256` instead names the exact declaration file
hash reported by preparation. The review schema is:

```json
{
  "schema": "nca-ik-compact-admission/v1",
  "declaration_sha256": "DECLARATION_SEMANTIC_SHA256",
  "source_sha256": "DECLARED_SOURCE_SHA256",
  "reviewers": [],
  "checks": {
    "source_and_cpu_checks": false,
    "corpus_and_splits": false,
    "allocation_and_resume": false,
    "scoring_and_selection": false,
    "environment_and_memory_feasibility": false
  },
  "verdict": "review_pending"
}
```

This deliberately incomplete example cannot admit execution. Record at least
two distinct public reviewer identifiers and their completed checks before
setting the verdict to `admit_declared_CUDA_comparison`. Both reviewers must
specifically accept the TITAN-only UUID binding, compute capability, selected
CUDA architecture route, available memory, finite costs and the absence of
V100 initialization. The supervisor preserves a canonical copy of that review
and binds its semantic hash to every attempt.

**Stop before the next command when no GPU execution is authorized.** It is the
actual device-dependent step. The recorded invocation has run once and paused;
its publication does not authorize repeating this command:

```sh
python -m nca.train.ik_experiment run --run NEW_RUN_DIRECTORY --declaration-sha256 DECLARATION_FILE_SHA256 --seed 41011 --corpus CORPUS_DIRECTORY --registry ORIGINAL_REGISTRY_DIRECTORY --review ADMISSION_REVIEW_JSON --pause-after-batches 1
```

That one-batch boundary spends the existing seed's allocation and retains its
real forward/backward/Adam and placement receipts. It is not a new seed or
free calibration run. A later explicitly authorized continuation uses the exact declared source,
same declaration and output; the other two seeds retain their original
allocations. The CUDA path refuses a CPU substitution. It reports synchronized
host/CUDA training durations and allocator memory separately from total cost.

After actual training, retain all three histories, select/freeze checkpoints
from tuning only, perform the candidate-bound final opening, score every
comparator and publish independent readback. The prepared scoring APIs support
that later stage; this continuation does not claim frozen trained candidates,
final scoring, learned quality, CPU export or physical qualification.

## Admitted continuation of the original allocation

D-025 authorizes a separate source amendment for the completed one-batch run.
The original declaration and review stay immutable. The continuation runner
accepts only a narrow executable source delta, authenticates the same retained
corpus and normalization, and inventories every original run file. Its new
amendment records the actual clean source, runtime, TITAN UUID fingerprint,
consumed charges and unchanged three-seed allocations. Preparation imports no
Torch and opens no final payload.

With the original environment activated and the single TITAN UUID already in
`CUDA_VISIBLE_DEVICES`, run these PowerShell commands using explicit paths:

```powershell
python -m nca.train.ik_continuation prepare --run ORIGINAL_RUN_DIRECTORY --corpus CORPUS_DIRECTORY --registry ORIGINAL_REGISTRY_DIRECTORY --output NEW_AMENDMENT_DIRECTORY
```

Publish the amendment's exact file hash and semantic hash, original declaration
identity, complete source inventory, checkpoint baseline and remaining budgets.
CI on that head and two genuine independent continuation admissions precede
CUDA. The fresh review uses schema `nca-ik-continuation-admission/v1`, fields
`amendment_sha256`, `source_sha256`, `reviewers`, `checks`, `verdict`, and the
same five Boolean checks as the original review. Its affirmative verdict is
`admit_declared_CUDA_continuation`. Do not copy the historical V100 or first
batch admission. Preserve the fresh review in the amendment directory.

```powershell
python -m nca.train.ik_continuation train --run ORIGINAL_RUN_DIRECTORY --amendment AMENDMENT_JSON --amendment-sha256 AMENDMENT_FILE_SHA256 --seed 41011 --corpus CORPUS_DIRECTORY --registry ORIGINAL_REGISTRY_DIRECTORY --review CONTINUATION_REVIEW_JSON
python -m nca.train.ik_continuation train --run ORIGINAL_RUN_DIRECTORY --amendment AMENDMENT_JSON --amendment-sha256 AMENDMENT_FILE_SHA256 --seed 42013 --corpus CORPUS_DIRECTORY --registry ORIGINAL_REGISTRY_DIRECTORY --review CONTINUATION_REVIEW_JSON
python -m nca.train.ik_continuation train --run ORIGINAL_RUN_DIRECTORY --amendment AMENDMENT_JSON --amendment-sha256 AMENDMENT_FILE_SHA256 --seed 43019 --corpus CORPUS_DIRECTORY --registry ORIGINAL_REGISTRY_DIRECTORY --review CONTINUATION_REVIEW_JSON
```

Each training command resumes the existing ledger and checkpoint; it cannot
grant new steps, presentations, attempts, time, memory or checkpoint slots.
Verify each supervisor and complete forty-epoch history before proceeding.
The final phase freezes all three tuning-selected identities before opening
final shards from the original hash-pinned archive. The pending journal remains
unread. It uses one supervised worker, retaining the authenticated training
view across opening, and reserves each seed's full 65,200 final/timing calls
before inference. Whole-process overhead is conservatively charged to every
seed. New checkpoints retain original allocation metadata and add actual
execution/amendment provenance. Unknown work blocks continuation.

Publish all histories, frozen candidates, raw comparison and CPU timing
receipts, aggregate verdicts, costs and independent readback. A measured no-go
or inconclusive result must be reported honestly; neither permits retuning on
the opened final suite. This workflow claims no CPU export or physical result.

### Evaluation-only amendment after completed training

The three training commands above describe the admitted training phase at
`a9c1c837c17d7f042117db22d036b11b8735c236`; they are not instructions to repeat
the now-completed training. CPU readback found that full tuning reports exceed
the final reader's original 1 MiB parser bound. The repair explicitly permits
up to 16 MiB only for authenticated tuning reports. Small registry/opening/event
receipts retain the 1 MiB bound; malformed, oversized or hash-mismatched reports
still refuse. Corpus bytes and scoring/selection thresholds do not change.

```powershell
python -m nca.train.ik_continuation prepare-evaluation --run ORIGINAL_RUN_DIRECTORY --prior-amendment TRAINING_AMENDMENT_JSON --prior-amendment-sha256 TRAINING_AMENDMENT_FILE_SHA256 --corpus CORPUS_DIRECTORY --registry ORIGINAL_REGISTRY_DIRECTORY --output NEW_EVALUATION_AMENDMENT_DIRECTORY
```

This CPU-only preparation requires all three inactive, verified complete
forty-epoch training ledgers, all 120 authenticated tuning reports, unspent
evaluation calls and no final opening. It embeds the exact parent amendment
and admission, all completed run file hashes, original remaining allocations,
old `training_execution` and the actual tuning-only frozen candidates. The new
source can differ from its parent only in the three evaluation repair modules.
Both training entry points reject an evaluation-only amendment before opening
an attempt or CUDA context.

Publish this immutable child and exact-source CI, then obtain two fresh actual
independent reviews. Their aggregate uses `nca-ik-evaluation-admission/v1`, the
same five checks and source/amendment/reviewer fields, and affirmative verdict
`admit_declared_final_evaluation`. The earlier evaluation admission is suspended;
its historical training approval cannot admit the repaired final phase.

```powershell
python -m nca.evaluation.ik_continuation_eval run --run ORIGINAL_RUN_DIRECTORY --amendment EVALUATION_AMENDMENT_JSON --amendment-sha256 EVALUATION_AMENDMENT_FILE_SHA256 --corpus CORPUS_DIRECTORY --registry ORIGINAL_REGISTRY_DIRECTORY --review EVALUATION_REVIEW_JSON --archive ORIGINAL_ACCEPTED_ARCHIVE
```

The worker rechecks the exact frozen candidates and old training provenance
before opening. Its final checkpoints preserve the old `training_execution`
and separately record actual new evaluation source. It changes no optimizer
steps, model parameters, normalization or allocations. Preserve failures and
unknown outcomes; no final-driven retry or retraining is permitted.

## CPU verification

The existing optional-learning workflow runs the new modules in its pinned CPU
runtime. Linux/Windows base-package discovery still works without Torch.
Focused commands are:

```sh
python -m unittest discover -s tests -p "test_ik_corpus.py" -v
python -m unittest discover -s tests -p "test_ik_run_state.py" -v
python -m unittest discover -s tests -p "test_ik_training_eval.py" -v
python -m unittest discover -s tests -p "test_ik_experiment.py" -v
python tools/check_docs.py --strict
```

The tests cover authentication/refusal boundaries, deterministic group sampling,
actual tiny CPU optimization and equivalent interrupted/restored trajectories,
cumulative reservations/failures, independent scoring, missing denominators,
three-seed freeze and final-opening mismatch controls. They do not execute the
accepted corpus comparison or a GPU workload. The owning PR records exact-source
observed results and explicit skips.

## Bounded polar follow-up

The [D-027 commission](decisions/2026-10-23-ik-polar-followup.md) authorizes one
new target-feature experiment under the [fixed design](design/IK_ACCURACY_FOLLOWUP.md).
Use a new directory and allocation. Preserve the completed v1 run, archive,
registry, normalizer, opened final and accounting. Original final geometry is
an exclusion set; its labels and outputs cannot supervise or select this model.

After clean exact-head CI, an independent CPU custodian first declares and
generates the fixed 2,940-row final-only suite. `$isolation` names the exact
independent production geometry receipt, whose SHA-256 is pinned by the sealer.
Do not print or inspect its archive payload during model development.

```powershell
python -m nca.train.ik_followup_seal declare --root $fresh --isolation $isolation
python -m nca.train.ik_followup_seal generate --root $fresh
python -m nca.train.ik_followup_seal inspect --root $fresh
python -m nca.train.ik_followup prepare --run $polarRun --corpus $corpus --registry $registry --fresh $fresh --archived-run $originalRun
python -m nca.train.ik_followup inspect --run $polarRun --declaration-sha256 $polarDeclarationFileSha
```

Preparation remains CPU-only. It verifies original labels and normalization,
derives the versioned polar training-only normalizer, authenticates the fresh
seal and original tuning-selected archived candidates, and creates three zero-
consumption ledgers. UUID-isolate TITAN RTX as described above. The new allocator
ceiling is 20 GiB; seeds and CPU measurements execute serially. This ceiling
does not reserve memory. The original 40 epochs, 128 capacity, three seeds,
losses, schedule, 7,200 seconds per seed and acceptance criteria remain fixed.

Both fresh independent reviews must approve this exact declaration, executable
source, completed hosted CI and sealed-suite evidence. Publish their provenance
and actual findings before constructing the source/declaration-bound admission
record. Existing v1 approvals do not apply.

```powershell
foreach ($seed in 41011,42013,43019) {
  python -m nca.train.ik_followup run --phase train --seed $seed --run $polarRun --declaration-sha256 $polarDeclarationFileSha --corpus $corpus --registry $registry --fresh $fresh --archived-run $originalRun --review $polarReview
  if ($LASTEXITCODE -ne 0) { throw 'Preserve failed allocation and protective phase claim; no automatic retry' }
}
python -m nca.train.ik_followup run --phase finish --run $polarRun --declaration-sha256 $polarDeclarationFileSha --corpus $corpus --registry $registry --fresh $fresh --archived-run $originalRun --review $polarReview
```

The finish phase independently authenticates every tuning report and checkpoint
lineage. If any selected seed misses the original raw tuning quality gate,
it records a terminal no-go and leaves fresh final sealed. No extra epochs,
seed replacement or automatic second experiment follows. If all pass, final
opens once for all six declared controls, including archived trained v1.
Raw scores and one-thread batch-one timing use unchanged acceptance tests.
Every evaluation reservation is checkpointed without optimizer updates before
terminal closure. Preserve the phase claims, supervisor receipts and ledger
readback for independent audit. A crash or uncertain outcome retains the
run-wide protective claim; do not delete it to restart.
© 2026 techrote. All rights reserved.
