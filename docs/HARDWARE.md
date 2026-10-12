---
title: Hardware and device lanes
kind: reference
summary: Which GPUs the owner has available for training, the known software limits of each, and what has actually been checked on them (nothing yet).
owns: [hardware.lanes]
verified: 2026-10-13
---

# Hardware and device lanes

Nothing on this page has been run or measured for this repository yet. Lanes become usable only after T-002 records a device report as evidence, and get numbers only after T-007.

## Devices available to the owner

| Device | Architecture (compute capability) | Memory | Notes | Checked for this repo |
|---|---|---|---|---|
| Tesla K80 (several boards) | Kepler GK210 ×2 per board (`sm_37`) | 12 GB per die | Passively cooled data-centre card; each board appears as two GPUs | not checked |
| Titan RTX | Turing (`sm_75`) | 24 GB | Shared with other projects | not checked |
| Tesla V100 | Volta (`sm_70`) | per installed model | Shared with other projects | not checked |
| Quadro RTX 4000 | Turing (`sm_75`) | 8 GB | Shared with other projects | not checked |
| Desktop CPUs | x86-64 | — | CPU lane for CI-sized tests | not checked |

## Known software limits (from vendor documentation, not yet confirmed here)

- **Kepler K80:** the R470 driver branch is the last to support Kepler, and CUDA 11.x is the last toolkit to compile for it; CUDA 12 removed Kepler support. Official PyTorch builds that include `sm_37` are from older releases; the exact last version must be confirmed by checking `torch.cuda.get_arch_list()` in the legacy environment (T-002).
- **Kepler has no fast half-precision maths.** Training on the K80 uses FP32 throughout. This suits NCA models, which are small.
- **Each K80 die is a separate device.** One board shares power and cooling between its two dies; benchmark numbers taken while the other die is busy are labelled as such.
- **Modern GPUs:** current PyTorch builds; half precision is available but not used in the shared code path (D-006, proposed).

## Machine assignment

Which machine hosts which lane is not decided. It is recorded here, without machine nicknames (C-003), once T-002 runs.

## Not checked

- Drivers, toolkits, Python and PyTorch versions on every machine.
- Whether a K80 behind a narrow PCIe link (for example x4) affects training speed. Expected to be negligible for NCAs because data stays on the GPU; not measured.
