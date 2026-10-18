---
title: Compute lanes and measurement requirements
kind: reference
summary: Proposed compute roles and the absence of qualified training or performance evidence.
owns: [hardware.lanes]
verified: 2026-10-18
---

# Compute lanes

No learner environment, GPU lane or training throughput has been qualified for this repository. Device availability does not establish software compatibility or useful throughput.

| Lane | Proposed role | Evidence required |
|---|---|---|
| CPU | Protocol tests, baseline control, small inference, hosted CI | Pinned environment, successful conformance and measured cost |
| Compatible accelerator | Batched network updates after host profiling | Device/driver/runtime report, supported operations, actual smoke training and memory/time |
| Legacy K80 | Optional independent experiments if a compatible stack is justified | Separate environment/architecture verification and workload benchmark |
| Target game platform | Final inference and feel qualification | Exact host/profile, measured latency and gameplay comparison |

D-006 records the historical two-environment idea. D-010 removes legacy hardware qualification from the actor-control critical path; it does not claim that any legacy lane works. T-101 provides a CPU baseline and environment report. T-109 measures the complete pipeline and optional lanes.

Record host stepping, reset, transport, learner updates and inference separately. Use aggregate base ticks/s and decision cadence to estimate experiment duration. A small neural model does not imply a fast cellular/Rapier environment.

Avoid prescribing a specific Python/PyTorch/CUDA version until installation and supported operations are verified for the chosen lane. “Unavailable” or “not measured” is explicit; no substitution is reported as success for another device. No driver/BIOS changes belong to this prepass.
