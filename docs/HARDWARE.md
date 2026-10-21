---
title: Compute lanes and measurement requirements
kind: reference
summary: Measured CPU learning and host lanes, with explicit limits for unqualified devices and deployment.
owns: [hardware.lanes]
verified: 2026-10-21
---

# Compute lanes

The Linux CPU learning and physical-host lanes have executed the bounded
[T105 learning screen](evidence/2026-10-19-t105.md),
[host pilots](evidence/2026-10-19-t102.md) and subsequent immutable runs in the
[run ledger](RUNS.md). These establish scoped software and workload evidence.
The exact [Torch CPU deployment bundle](TORCH_RUNTIME_EXPORT.md) has composite
qualification from the original physical run and separate completed pose evidence.
GPU/accelerator lanes, Windows physical runtime and owner feel remain unmeasured.
Device availability alone does not establish compatibility or useful throughput.

| Lane | Role | Current evidence or remaining requirement |
|---|---|---|
| Linux CPU package and learning | Protocols, authored control, bounded training and inference | Pinned Python/Torch/NumPy, actual host runs and scoped cost measurements in the ledger |
| Linux Torch CPU deployment | Exact evaluated BC1103 external controller | 600.521 active seconds: 95.163 ticks/s with one learned actor; 77.673 with one learned plus one authored actor. Standalone full-decision p95 2.713 ms; continuous realtime pacing remains unqualified |
| Windows CPU package | Installation and protocol conformance | Package CI passes; physical Windows gameplay and pacing remain unmeasured |
| Compatible accelerator | Batched network updates after host profiling | Device/driver/runtime report, supported operations, actual smoke training and memory/time |
| Optional IK CPU/CUDA runtime | Fixed Runner inverse-workload smoke and optimizer-boundary resume | [CPU software conformance](evidence/2026-10-21-ik-runtime-software.md); actual CUDA device and matched throughput remain unmeasured |
| Legacy K80 | Optional independent experiments if a compatible stack is justified | Separate environment/architecture verification and workload benchmark |
| Target game platform | Final inference and feel qualification | Exact host/profile, measured latency and gameplay comparison |

D-006 records the historical two-environment idea. D-010 removes legacy hardware qualification from the actor-control critical path; it does not claim that any legacy lane works. T-101 provides the CPU baseline and environment report. T-109 qualifies the exact Linux Torch CPU bundle separately from learning and package conformance; both original physical and NumPy no-go records remain unchanged. Additional device qualification requires its own evidence.

Record host stepping, reset, transport, learner updates and inference separately. Use aggregate base ticks/s and decision cadence to estimate experiment duration. A small neural model does not imply a fast cellular/Rapier environment.

Use the exact versions and build hashes from each qualified lane’s manifest. The recent CPU runs use Python 3.12.14, Torch 2.8.0+cpu and NumPy 2.3.5; package CI has its separately pinned environment. “Unavailable” or “not measured” is explicit; a result on one device does not qualify another. No driver/BIOS changes are commissioned here.

The [IK setup and resume guide](IK_RUNTIME.md) creates a separate environment
with Torch 2.8.0+cu126 or the CPU comparison wheel. Availability is only preflight;
actual FP32 data/model/loss/gradient/optimizer placement, finite updates and
unchanged prospective budgets determine the smoke result. The initial software
environment exposes no NVIDIA device. CUDA memory/timing, driver compatibility
and Windows optional-runtime execution have no measured qualification here.
