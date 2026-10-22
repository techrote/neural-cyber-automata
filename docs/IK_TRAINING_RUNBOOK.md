---
title: IK compact CUDA runbook
kind: guide
summary: Exact authenticated corpus preflight, independent admission, supervised seed training and CPU conformance commands.
owns: [ik.training.operations]
verified: 2026-10-22
---

# IK compact CUDA runbook

[PARTIAL] This operational continuation of the [IK training reference](IK_TRAINING.md)
preserves the accepted-corpus CPU preflight, prospective three-seed CUDA gate,
and focused software validation. No original-corpus training or sealed-final
scoring has yet occurred.

## CPU preparation, then the GPU boundary

Install the already pinned optional runtime using [runtime operations](IK_RUNTIME.md).
Keep the extracted archive, original registry and new run directory outside the
source checkout. The measured declaration names the already qualified Windows 11 /
CPython 3.12.10 / Torch 2.8.0+cu126 / V100 / driver 581.80 environment. Other CUDA
environments retain their separate qualification requirement. Preparation below
uses metadata and CPU corpus validation;
it does not import Torch or initialize a CUDA device:

```sh
python -m nca.train.ik_experiment prepare --corpus CORPUS_DIRECTORY --registry ORIGINAL_REGISTRY_DIRECTORY --output NEW_RUN_DIRECTORY --device cuda:0
python -m nca.train.ik_experiment inspect --run NEW_RUN_DIRECTORY --declaration-sha256 DECLARATION_FILE_SHA256
```

Use the existing activated PowerShell environment on the qualified Windows
machine. Preparation refuses different OS/Python/Torch/NumPy metadata before
CUDA initialization. Preparation publishes exact source, environment, corpus/normalizer, three seeds,
all allocations and the unchanged acceptance identity in an immutable declaration.
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
setting the verdict to `admit_declared_CUDA_comparison`. The supervisor preserves
a canonical copy of that review and binds its semantic hash to every attempt.

**Stop before the next command when no GPU execution is authorized.** It is the
first actual device-dependent step for this continuation, and it has not run:

```sh
python -m nca.train.ik_experiment run --run NEW_RUN_DIRECTORY --declaration-sha256 DECLARATION_FILE_SHA256 --seed 41011 --corpus CORPUS_DIRECTORY --registry ORIGINAL_REGISTRY_DIRECTORY --review ADMISSION_REVIEW_JSON --pause-after-batches 1
```

That one-batch boundary spends the existing seed's allocation and retains its
real forward/backward/Adam and placement receipts. It is not a new seed or
free calibration run. Continuing uses the same command without the pause option,
same declaration and same output; the other two seeds retain their original
allocations. The CUDA path refuses a CPU substitution. It reports synchronized
host/CUDA training durations and allocator memory separately from total cost.

After actual training, retain all three histories, select/freeze checkpoints
from tuning only, perform the candidate-bound final opening, score every
comparator and publish independent readback. The prepared scoring APIs support
that later stage; this continuation does not claim frozen trained candidates,
final scoring, learned quality, CPU export or physical qualification.

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

© 2026 techrote. All rights reserved.
