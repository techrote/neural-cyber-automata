---
title: Bounded actor learning operations
kind: reference
summary: Install the optional CPU learner, validate the frozen screen and retain reproducible training evidence.
owns: [learning.operations]
verified: 2026-10-19
---

# Bounded actor learning

The first screen compares the balanced Jolt controller, direct behavioural
cloning and recurrent residual PPO under one accepted physical profile.
[The declaration and evidence](evidence/2026-10-19-t105.md) own the task, reward,
partitions, budget, comparison gates and measured conclusions. The
[frozen configuration](../configs/first-skill/jolt-screen-v1.json) is the exact
machine-readable declaration.

## Install the optional CPU lane

Use an isolated Python environment as described in [package operations](PACKAGE.md).
The tested learning lane uses Python 3.12, NumPy 2.3.5 and PyTorch 2.8.0+cpu:

```sh
python -m pip install setuptools==84.0.0 wheel==0.45.1 numpy==2.3.5
python -m pip install torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-build-isolation -e .
python -m unittest discover -s tests -p test_learning.py -v
python tools/first_skill.py prepare
```

The commands work in an activated environment on Linux or PowerShell. The full
physical learning experiment is qualified separately on Linux; portable contract
tests do not establish Windows game or training performance. Routine CPU package
imports and registry discovery do not import PyTorch. The registered trainer is
`first-jolt-screen@1`; its construction explicitly invokes the bounded screen.

## Run an identified experiment

Prepare the pinned runtime using [the host guide](HOST_ADAPTER.md). Invoke the
screen from an isolated Git checkout whose installed sources match that checkout:

```sh
python tools/first_skill.py run --checkout HOST_CHECKOUT --godot GODOT_EXECUTABLE --output NEW_RUN_DIRECTORY --accepted-t102 ACCEPTED_COMMIT --accepted-t103 ACCEPTED_COMMIT --accepted-t104 ACCEPTED_COMMIT
```

Replace the uppercase arguments with the corresponding runtime, new artifact
destination and verified dependency commit identities. The command rejects a
previously used output directory, writes a pre-registration before the first
rollout and retains failed or capped attempts. It does not extend its budget when
results are weak. All six final learned checkpoints freeze before the sealed
evaluation begins.

The result separates trained policy selection, local control, task evaluation,
character metrics and runtime cost. A completed command can report a no-go or an
inconclusive result. It does not imply useful learning or owner feel approval.

## Inspect and retain evidence

```sh
python tools/first_skill.py inspect --checkpoint CHECKPOINT_FILE
python -m pip install pillow==12.3.0
python -m nca.train.clips RUN_DIRECTORY
```

The run includes its source/environment manifest, compressed consumed-action
episodes, initial/final weights, optimizer and RNG states, tuning and evaluation
results, numerical export and matched pose clips. Preserve complete artifacts
outside Git and commit small hash-indexed receipts under the repository's
[run rules](RUNS.md). Resume restores learner state at a fresh episode; no exact
mid-episode physical restore is claimed. Runtime export and deployment require
their separate qualification.

© 2026 techrote. All rights reserved.
