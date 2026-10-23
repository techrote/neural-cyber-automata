---
title: Optional IK runtime operations
kind: guide
summary: Isolated FP32 device setup, bounded smoke declarations, immutable attempts and training-boundary resume.
owns: [ik.runtime_operations]
verified: 2026-10-22
---

# Optional IK runtime operations

[IK-CUDA-002 / #51](https://github.com/techrote/neural-cyber-automata/issues/51)
adds a separate optional runtime for the accepted [Runner point task](IK_LAYER.md).
The runtime smoke exercises geometric inputs, forward/backward operations and
optimizer state. It is conformance evidence, not a useful learned IK result,
the independent reference solver in #52, or physical tracking qualification.

The software path and hardware evidence have separate status. The
[22 October Windows V100 qualification](evidence/2026-10-22-ik-cuda-qualification.md)
is **historical evidence only**. The V100 has a reported cooling fault and is
out of service; **do not initialize or use it again**. Current #54 training
requires only the TITAN RTX, selected by its single GPU UUID, with separate
prospective TITAN admission and exact-hardware checks in the
[compact IK runbook](IK_TRAINING_RUNBOOK.md). The old V100 qualification
exercises real CUDA forward/backward/Adam, placement and same-device resume,
with a matched CPU measurement. It qualifies that identified FP32 smoke lane;
Linux CUDA and useful learned IK remain unmeasured. The existing CPU package
and historical learner environments retain their pins.

## Install one isolated lane

Prerequisites are Git, Python 3.12 with virtual-environment support, and access to
the official package indexes. CUDA execution also needs an already installed
compatible NVIDIA driver and a device supported by the selected Torch build.
Setup changes no driver and starts no GPU workload. It refuses an existing
`.venv-ik` so that rerunning setup cannot silently replace an environment.
Use a fresh checkout for a different installation identity.

| Component | Version or route |
|---|---|
| Python | 3.12; record the actual patch release |
| Build tools | setuptools 84.0.0, wheel 0.45.1 |
| NumPy | 2.3.5 |
| CUDA candidate | Torch 2.8.0+cu126, official cu126 wheel index |
| CPU comparison alternative | Torch 2.8.0+cpu, official CPU wheel index |
| Computation | FP32, TF32 disabled, deterministic algorithms requested |

The official [PyTorch 2.8 installation routes](https://pytorch.org/get-started/previous-versions/#v280)
identify the versioned indexes. Successful installation is not device qualification.

On Windows, run from the repository using PowerShell and the Python `py` launcher:

```powershell
powershell -ExecutionPolicy Bypass -File tools/setup_ik.ps1 -Lane cuda
$IkPython = ".\.venv-ik\Scripts\python.exe"
& $IkPython -m nca.ik.runtime doctor --device cuda:0
```

Use `-Lane cpu` in a fresh checkout for the CPU-only wheel. An already installed
CUDA wheel can also execute the matched `--device cpu` workload; its exact build
remains part of that CPU result's identity.

On Linux:

```sh
bash tools/setup_ik.sh cuda
source .venv-ik/bin/activate
python -m nca.ik.runtime doctor --device cuda:0
```

Use `bash tools/setup_ik.sh cpu` for the separate CPU-only wheel route. The
`doctor` command reports the requested device and installed package facts. It
uses metadata only; device initialization and measured preflight occur inside
the declared smoke. A missing or unsupported explicitly requested CUDA device
fails that attempt; it never becomes a CPU result. Doctor success does not
qualify forward/backward/optimizer execution.

## Declare before executing

Work from a clean committed source checkout. Keep artifacts outside the Git
source tree. `NEW_RUN_DIRECTORY` below means a fresh destination; preserve it
for every continuation of this allocation.

```sh
python -m nca.ik.runtime prepare --config configs/ik/runtime-smoke-v1.json --device cuda:0 --output NEW_RUN_DIRECTORY
```

Preparation produces `declaration.json` with the exact clean source identity,
configuration, profile, workload/generator/split identities, seed, expected
runtime and finite allocation. It runs no training. Publish that declaration
and its SHA-256 on the task before invoking the measured workload. The explicit
hash supplied to execution binds the invocation to those prospective bytes.

The default fixture requests 32 optimizer steps with a batch of 32. Its
allocation remains bounded by at most **256 attempted optimizer steps and
120 aggregate wall seconds**, including startup and failed or resumed attempts.
The supervisor charges elapsed startup, worker and inventory time plus a
sealing/shutdown reserve of up to five seconds within the remaining allocation.
It verifies that sealing
and readback fit inside that charged reserve. Charged costs are conservative
and separate from measured worker/stage times; the reserve is not training time.
An unsealed attempt or retained active marker blocks resume rather than turning
unknown work into a fresh allowance. Completed or exhausted allocations refuse
further execution without creating another attempt.
Iteration, sample, wall-time and memory controls retain their declared values;
an OOM does not authorize a smaller batch or a fresh allocation.

With an activated environment, the remaining commands are identical on Linux
and PowerShell:

```sh
python -m nca.ik.runtime smoke --run NEW_RUN_DIRECTORY --declaration-sha256 DECLARATION_SHA256
python -m nca.ik.runtime inspect --run NEW_RUN_DIRECTORY
```

For Windows without activation, use `& $IkPython` in place of `python` in these
commands. No long training or throughput sweep is implied. A throughput pilot
requires its own prospective declaration, with no more than three fixed batch
sizes and 60 seconds per size as specified in #51.

## Pause, cancel and resume

A deliberately bounded first attempt can stop at a completed training boundary:

```sh
python -m nca.ik.runtime smoke --run NEW_RUN_DIRECTORY --declaration-sha256 DECLARATION_SHA256 --stop-after 8
python -m nca.ik.runtime resume --run NEW_RUN_DIRECTORY --declaration-sha256 DECLARATION_SHA256
```

`--stop-after` limits this attempt within the unchanged total workload. Resume
uses the same run directory and declaration. It restores the last complete
optimizer boundary, including model, optimizer, optional scheduler/scaler,
normalization, sampler/generator identities and cursor, Python/NumPy/Torch RNG
states, counters and remaining allocation. A checkpoint cannot reset costs
already charged by the immutable attempt ledger.

To request cancellation from another terminal:

```sh
python -m nca.ik.runtime cancel --run NEW_RUN_DIRECTORY
```

Cancellation stops admission of further work and retains the last complete
boundary. Failed or capped work and its consumed budget remain visible. A
partially written, corrupt or incompatible checkpoint is refused. Sample cursor
and optimizer state advance together at the declared publication boundary;
work attempted after that boundary is charged even if it cannot be resumed.

Source, configuration and batch size cannot change on resume. An explicitly
allowed environment change uses `--allow-environment-change` and records a
continuation with the old and new identities and its reproducibility limits.
It does not create a new allocation or a claim of bit identity across devices,
operating systems, driver builds or Torch versions. Unsupported environments
remain refusals. PyTorch's [reproducibility guidance](https://docs.pytorch.org/docs/2.8/notes/randomness.html)
also distinguishes repeatability within one platform from cross-platform claims.

## Read the result and compare costs

Retain each attempt, including refusals, OOM, cancellation and exhausted-budget
results. Completion requires finite gradients, changed model weights and checked
device placement for inputs, model, loss, gradients and optimizer state. A
missing or skipped CUDA check has no measured-success status. A finite smoke
loss does not establish geometric quality against [the IK acceptance gates](IK_ACCEPTANCE.md).

The preflight identifies GPU model/capability, compiled Torch architectures,
driver and Torch/CUDA build, OS/Python, precision/determinism settings,
free/total memory and source/configuration identities. CUDA timing uses explicit
synchronization. Device timing and allocated/reserved memory peaks are separate
from generation, loading, training, checkpoint and total host costs. Unknown or
inapplicable measurements are null, not zero. The allocator memory fraction
limits Torch-managed allocations; it is not a hard cap on all process or driver
memory. See [PyTorch 2.8 CUDA semantics](https://docs.pytorch.org/docs/2.8/notes/cuda.html).

Declare and execute a matching CPU fixture separately, with the same workload,
batch, steps, seed and normalization. Record every attempted CPU and CUDA cost;
do not replace a failed result with a selected faster repeat.

```sh
python -m nca.ik.runtime compare --cpu CPU_RUN_DIRECTORY --cuda CUDA_RUN_DIRECTORY
```

Comparison is read-only and requires compatible workload identities plus actual
successful measurements. It reports measured costs without promising a GPU
speedup. A comparison with absent CUDA evidence cannot qualify the device.

Checkpoint publication uses atomic immutable artifacts and content hashes. Keep
checkpoint bytes and larger artifacts outside Git, with retrieval locations and
SHA-256 identities in the [run ledger](RUNS.md). Model weights alone are not a
resumable training checkpoint; [PyTorch checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
explains the need for optimizer and other training state. Publish small receipts,
source/configuration identities, failures and evidence limits with the owning PR.

© 2026 techrote. All rights reserved.
