---
title: IK CUDA qualification progress on 22 October
kind: evidence
summary: Prospective admission, retained CPU/CUDA smoke and independent readback of the remaining issue 51 hardware gates.
owns: [progress.ik_cuda_qualification_20261022]
verified: 2026-10-22
---

# IK CUDA qualification progress

## 20:49 UTC — Prospective publication

Reconciled main `95873260c2411da320cac40d28644df2a45e6738`, issue #51 and all
comments, its accepted contract/reference/data dependencies and parent #49.
PR #68 already owns the software implementation. Open PR #76's unrelated Jolt
changes remain untouched. The existing Windows host exposes a compatible V100;
no driver change, purchased compute or long training is commissioned.

The existing PowerShell setup installs the separate Torch 2.8.0+cu126 / NumPy
2.3.5 environment. Independent read-only admission review finds no blocker.
Both exact declarations and their raw hashes are [published before optimization](https://github.com/techrote/neural-cyber-automata/issues/51#issuecomment-6351007674).
Each lane retains its original 256-step / 8,192-sample / 120-second cap and
predeclared 8+24 execution. Source remains clean and identical during both runs.

## 20:54 UTC — Complete measured evidence and independent audit

Both production CLI lanes complete all 32 steps / 1,024 samples and restore
their step-eight checkpoints. Four completed attempts consume 24.421 charged
seconds CUDA and 19.282 CPU; no failures, repeats, sweeps or changed batches.
CUDA synchronized training takes 0.491718200 seconds, CPU 0.075995100 seconds.
The tiny checked workload shows no GPU speedup. CUDA peaks are 68,178,432
allocated / 69,206,016 reserved bytes; CPU RSS remains unmeasured.

Independent offline readback passes 7,418 checks without findings, covering all
131 source files, four attempts, 64 steps, 66 checkpoints, exact sample order,
finite selected-device receipts and Adam 8→9 continuation. Its 2.241623200
seconds are separate evidence processing, without CUDA or optimizer calls.
The [owning evidence page](../evidence/2026-10-22-ik-cuda-qualification.md) and
[run ledger](../RUNS.md) retain the exact environment, original costs, public
archive/hash and qualification limits. Bare and optional IK suites each pass
237 cases, with 37 / two explicit optional/manual skips respectively.

Final publication review, exact-head hosted checks and merge verification
remain the owning PR's final gates. The measured source identity is preserved
through documentation publication. Learned IK, CPU export, physical tracking,
Linux CUDA and other devices remain separately unmeasured.

© 2026 techrote. All rights reserved.
