---
title: IK runtime software conformance and remaining CUDA evidence
kind: evidence
summary: Optional pinned device execution, complete checkpoint resume and bounded attempt controls; actual CUDA remains unmeasured.
owns: [evidence.ik_runtime_software]
verified: 2026-10-21
---

# IK runtime software conformance

[IK-CUDA-002 / #51](https://github.com/techrote/neural-cyber-automata/issues/51)
starts from main `0d9d74631012c60e0286784cb25634c04bcedbf7` and the accepted
Runner contract in [#50 / PR #63](https://github.com/techrote/neural-cyber-automata/pull/63).
The branch was reconciled with main
`5e06b2817b7b5a8d19bbdf46743b0b0cf6a48243` after PR #67 merged. Its Jolt
implementation and preserved failed-allocation evidence remain unchanged;
issue #62 remains open and morphology 16 remains blocked.

This increment implements the optional runtime and its software controls.
**Actual CUDA qualification and matched CUDA/CPU measurement remain unmeasured.**
The implementation environment exposes no NVIDIA device or driver query utility.
Issue #51 remains open for its identified-device acceptance; the programme's
CUDA, learned-quality and physical gates are unchanged.

## Implemented boundary

The [operator guide](../IK_RUNTIME.md) owns installation and commands. Base
imports use the standard library; the isolated optional lane keeps Torch 2.8.0
and NumPy 2.3.5. The CUDA candidate uses the official CUDA 12.6 wheel route.
Explicit `cuda:N` either executes that selected device or records failure.
Availability metadata cannot produce a qualification result.

The fixed 4→32→2 inverse MLP maps current posture and the Runner sole target to
two bounded desired joint angles. A 256-point lattice, shuffled CPU sampler,
FP32 nominal endpoint loss and Adam exercise actual forward/backward/update
operations. The compact FK agrees with independently implemented nominal anchor
geometry and its neutral Jacobian in small fixtures. This smoke does not claim
the general FK/reference implementation in #52, a useful trained policy, a sealed
quality comparison, host execution or physical tracking.

Preflight identifies source/configuration, Python/OS, exact packages/build,
precision/determinism, GPU model/capability/compiled architectures, driver and
memory. Inputs, model, geometric loss, outputs, every gradient and optimizer
tensor are checked on the selected device. CUDA event intervals and synchronized
host durations are separate from generation, loading and checkpoint costs.
Allocator peaks exclude driver/context and external allocations; CPU GPU-memory
fields are null. No speedup is promised.

## Checkpoint and allocation integrity

The existing immutable publication helpers are shared through `nca.runs`;
historical learner imports retain their aliases. Each tensor checkpoint has
an immutable hashed marker, complete model/optimizer state, optional scheduler
and scaler slots, normalization and dataset/split/generator identity, permutation
and cursor, Python/NumPy/Torch CPU/CUDA RNG state, counters and an allocation
snapshot. Loading verifies bytes before tensor-only deserialization.

A successful optimizer step becomes resumable only after complete checkpoint
and step-receipt publication. Failed or interrupted work after that boundary
remains charged. Resume restores the committed training sequence without
duplicating or skipping its samples; attempted replay is a separately charged
cost. The final immutable attempt ledger overrides earlier checkpoint budget
snapshots. Changed environments require explicit continuation and retain their
limitations across later resumes; cross-device bit identity is not claimed.

The standard-library supervisor starts accounting before optional imports,
reserves each batch before work, and enforces the remaining original allocation.
The supplied declaration requests 32 steps × 32 samples, bounded by 256 attempted
steps, 8,192 attempted samples and 120 aggregate seconds. A sealing reserve is
charged conservatively and checked through final readback. Cancellation, OOM,
failed/capped attempts and an unresolved process are not successful results.
Completed/exhausted allocations refuse more work; batch changes and corrupt or
incompatible checkpoints cannot silently restart an experiment.

## Software verification

Small conformance fixtures exercise CPU operations and synthetic failure paths.
They are not formal device-performance runs. An explicitly selected manual CUDA
fixture fails if its request is unavailable; the default skip reports CUDA as
unmeasured. Routine PR CI runs no long training.

| Area | Evidence |
|---|---|
| Optional import isolation | Bare interpreter tests block Torch/NumPy imports while using contracts, declarations and run inspection |
| Device controls | Strict selectors, incompatible builds/architectures, missing CUDA, deterministic FP32, actual CPU tensor placement |
| Checkpoint restoration | Uninterrupted/resumed model, Adam, sampler IDs/cursor, dropout and Python/NumPy/Torch RNG equality on CPU |
| Attempt controls | OOM reservation charging, cumulative bounds, source/runtime refusal, canceled final steps, corruption and unresolved attempts |
| Qualification integrity | Success metadata without complete step receipts is rejected; environment changes remain visible after later resumes |
| Public CLI integration | Actual prepare, pause, resume and inspect subprocesses preserve three committed steps and nine sample IDs; exhausted resume creates no extra attempt |
| Source provenance | Clean checkout plus HEAD-blob verification, raw source hashes, imported package JSON/profile matching and immutable declaration hashes |
| Existing learner compatibility | Original learner checkpoint/artifact tests pass with shared immutable helpers |
| Actual CUDA and matched CPU/CUDA costs | Unmeasured; no NVIDIA device exposed |
| Windows optional runtime and PowerShell setup | Instructions supplied; local execution unmeasured |

Independent reviews covered device arithmetic/placement, the checkpoint restore
boundary, and supervisor accounting/qualification. They found cancellation and
environment-history edge cases; these were corrected and protected by regression
fixtures before publication. The existing Jolt process-tree check initially
failed inside a sandbox with inconsistent process namespaces; its unchanged
14-case module passes outside that sandbox. No gate was weakened.

The initial hosted Windows check exposed a test fixture that committed CRLF
setup files before switching the temporary repository's global normalization
policy. The fixture now applies an explicit checkout rule only to its synthetic
Python source and retains a pre-existing CRLF setup-file control. Production
source verification remains unchanged and strict; final-head hosted checks
record the corrected fixture's platform results.

The integrated local optional CPU suite passes 857 cases: 799 passed and 58
explicit skips, including the unrequested manual CUDA fixture. This includes the
real CLI subprocess test and the repeated environment-refusal regressions. The
earlier bare-interpreter suite passed 737 cases (554 passed / 183 explicit skips)
before the final regressions and reconciliation with PR #67. Final source
identities, hosted test counts and check links belong to the PR's final-head
record; they remain distinct from these local conformance checks.

## Remaining acceptance

The software is prepared for a prospectively published, bounded actual-device
qualification. A compatible existing CUDA device must complete the declared
workload with finite gradients, changed weights, exact placement evidence,
identified driver/build, synchronized timing and separate peak memory. Its
matched CPU result uses the same source/configuration/workload identities.

Publish original declaration, all attempts, checkpoints and hashes with durable
retrieval before closing the remaining hardware-dependent items. Do not relabel
CPU fixtures, availability or simulated failures as actual CUDA evidence. No
compute acquisition, driver change or larger training allocation is included.

© 2026 techrote. All rights reserved.
