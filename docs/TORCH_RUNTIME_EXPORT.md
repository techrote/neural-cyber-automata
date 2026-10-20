---
title: Original-operator Torch CPU deployment
kind: reference
summary: Explicit optional Torch backend for an unchanged evaluated numeric policy, with original-source owner fences and active-duration qualification.
owns: [export.torch_runtime]
verified: 2026-10-20
---

# Original-operator Torch CPU deployment

The additive `torch-operators-cpu/v1` backend loads the **same evaluated BC1103
numeric weights** and executes the original batch-one Torch Linear/GRUCell
operators. Its artifact has a separate `nca-torch-cpu-policy/v1` envelope,
runtime/version/source identity and qualification receipt. It preserves the
original NumPy artifact and its failed trajectory result. Conversion alone
does not qualify either lane.

This backend is useful where preserving the original operator path matters
more than removing the Torch dependency. It imports no learner or checkpoint
loader and performs no optimization. The normal `ExportController` still owns
observation age, recurrence, source generation, actor/profile/mount changes,
fallback and real-owner cancellation. FP16/BF16 autocast, another Torch version,
a changed thread count or changed CPU arithmetic configuration are refused.
Construction preserves the caller's Torch RNG state.

The [prospective follow-up](../configs/export/torch-cpu-followup-v1.json) names
the exact original bundle, failed screen and passed fresh-episode resume/cost
receipts. It first applies all unchanged arithmetic and trajectory limits to
two original matched cases. Any failure in those gates, negative contracts,
profile/cadence refusal or realtime ownership stops before throughput.
Throughput then requires two continuously executing **300-second windows**:
one learned Jolt, followed by one learned Jolt with one authored Monowheel.
A fixed sample count or idle padding cannot satisfy the full ten-minute gate.
Median/tail inference, step/transport, reset and observe costs are separate.
Torch deployment memory is measured in a fresh process without learner imports.

The entire follow-up has one invocation, a 900-second wall ceiling, 25,000 owner
requests, 22,000 step requests, 132,000 requested step ticks and two host starts.
The throughput sublimit is 20,000 decisions and 660 seconds. Every owner request
is reserved before dispatch, and its complete response is preserved. A lost
reply remains an uncertain tail. Closing the owned process is recorded, and
request/step caps remain part of final acceptance. The realtime episode has an
independent fixed 600-tick horizon. Its last observed tick is a lower bound on
autonomous work; the full horizon is reserved as its upper bound, including
the unobserved close tail. Both bounds are reported separately from requested
stepped-mode ticks. The manifest records CPU, build and effective container
limits without local paths or machine names.

The runtime requires the declared Linux x86_64 Python 3.12.14, NumPy 2.3.5 and
Torch 2.8.0+cpu environment. Installing `.[torch-export]` declares the optional
dependencies; select the CPU Torch wheel as described in [learning setup](LEARNING.md).
The existing pinned host setup and verified original evidence archive are also
required. From the exact clean published follow-up source:

```bash
python tools/export_torch_policy.py \
  --original-root artifacts/t109-staged-export-publiceb95361 \
  --host "$NCA_HOST_CHECKOUT" --godot "$NCA_GODOT" \
  --source-revision "$PUBLISHED_FOLLOWUP_REVISION" \
  --render-python "$NCA_EVIDENCE_PYTHON" \
  --output artifacts/t109-torch-cpu-followup-v1
```

The output directory must not exist. Do not rerun a failed or interrupted
invocation under a new directory: preserve its records and declare any further
work separately. The [follow-up evidence](evidence/2026-10-20-t109-torch-cpu.md)
records source-only verification separately from actual qualification.

After conversion, explicit runtime use is:

```python
import torch
from nca.export import context_for, dispatch
from nca.export.torch_runtime import TorchRuntimeBundle, torch_export_controller

torch.set_num_threads(1)
# profile and host_identity come from the verified live owner.
context = context_for(profile, host_identity)
bundle = TorchRuntimeBundle.load("bc-seed1103-torch-cpu.json", context,
                                expected_sha256=artifact_file_sha256)
controller = torch_export_controller(bundle)
decision = controller.step(original_observation, goal)
result = dispatch(env, decision, original_observation)
```

The registry entry is `policy/jolt-torch-cpu-export/v1`. Multiple learned actors,
in-engine execution, Windows physical deployment, GPU acceleration, exact
mid-episode host resume and owner feel remain separate unmeasured lanes.
