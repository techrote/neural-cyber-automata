---
title: Independent IK kinematics and numerical reference evidence
kind: evidence
summary: Source audit, numerical conformance, bounded reference performance and the remaining CUDA and physical limits.
owns: [evidence.ik-kinematics]
verified: 2026-10-21
---

# Independent IK kinematics and numerical reference

Issue #52 implements [the nominal geometry and reference interface](../IK_KINEMATICS.md)
on accepted main `a1d71a58e87042649d7483131118a6acff5f94bf`. The existing Runner
profile and acceptance identities are unchanged. No training, physical-host
experiment, physical IK tracking or CUDA qualification is part of this increment.

## Source and independent review

The pinned host remains Cybersand `f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497`.
Independent readback confirms the catalogue blob
`397ac8a3103134da276590c5a409b17429394836`, SHA-256
`9f526afbb352b1c4d065bf04a69a9e85202b3b23cf9df1b7b6665ce526979b12`, 33,076 bytes;
and rig blob `ba2f00d018fa7db849927bb0a8bfef2aaa875c03`, SHA-256
`db283406e3aa86f5e329df15bba7465b58bca9767492ef0b54c7888965443ec4`, 63,070 bytes.

Authored pivots, both local anchors, initial angles, rotary zeros, units, play
and hard/command limits agree with the accepted profile. The independent
Runner vector formula and nonzero-zero/displaced-anchor hand fixtures provide
an oracle outside the implementation's transform path. The analytic inverse
uses cosine-law geometry without calling the FK or numerical inverse solver.

All four defects identified by independent review are corrected and have focused
regression coverage:

- Float32 bounds round inward, preserving CPU/Torch validity for the same numeric
  input instead of admitting a nearest-rounded value outside the exact descriptor.
- The equal-link folded continuum retains a legal first angle when the current
  angle lies outside narrower solve bounds.
- Analytic continuation normalizes equivalent current angles before ranking
  branches, without taking a shortcut through hard stops.
- Delta emission preserves the exact `0.25-radian` bound despite subtraction
  rounding, resolves positions from the emitted deltas and recomputes their pose.
  Representation adjustments are reported separately; the contract is unchanged.

The numerical-reference source at this checkpoint has SHA-256
`67b434277e5702e913f627a9ca272af4a0340400e9e8c88b3dd226d5b9f50a9e`.
Its final accounting change records elapsed time for early input refusals rather
than leaving those durations at zero. Final source and hosted-check reconciliation
remain part of the acceptance record.

## Conformance scope

Focused tests cover independent point/angle/Jacobian calculations, finite
differences, CPU/Torch dtype parity, autograd and row isolation, strict masks,
limits, singular and multiple-branch fixtures, prismatic/mixed numerical chains,
bounded inverse outcomes and continuation. Optional imports remain isolated.
The numeric criteria were published prospectively in
[the issue checkpoint](https://github.com/techrote/neural-cyber-automata/issues/52#issuecomment-6328273866).

The pinned optional runtime produced these pre-benchmark checkpoint results:

| Check | Result |
|---|---|
| Focused IK conformance | 160 cases: 158 passed and two actual-CUDA checks explicitly skipped |
| Full optional local suite | 920 cases: 860 passed, 59 explicitly skipped and one failure |
| Failure reproduced on unchanged accepted main | The same existing Jolt process-isolation test failed at `a1d71a58e87042649d7483131118a6acff5f94bf` |
| Hosted checks on the final PR head | Pending |

The full-suite failure is
`test_jolt_recovery_v4_executor.RecoveryV4ExecutorTests.test_isolated_python_tree_closes_a_separately_sessioned_descendant`.
It fails the sandbox's consistent PID namespace check and reproduces without the
IK changes on the accepted base revision. This remains a recorded local
validation failure; it is not counted as an IK pass or silently skipped. Hosted
final-head CI and the final independent readback are still required before merge.
Skipped actual-CUDA checks establish no device qualification.

This source checkpoint does not claim completed final-head acceptance or an
already executed reference-performance result.

## Reference performance

The benchmark is a separate finite CPU allocation declared from a clean source
commit before execution. Its deterministic geometry-conformance cases are not
learned training/tuning/sealed partitions. Both numerical and analytic methods
are legitimate comparators; there is no requirement that the numerical method
be faster. Every attempted call and failure is retained.

At this source checkpoint no measured reference allocation has executed. The
declaration, immutable result, budget consumption and retrieval identities are
published with the measured evidence before learned comparisons.

## Limits

Actual CUDA parity is unmeasured; #51 retains actual-device qualification.
Optional Windows Torch execution is unmeasured unless separately recorded.
CPU numerical accuracy does not establish contact, balance, clearance, compliant
anchor/pose accuracy, host joint-target capability or physical usefulness.
Dataset generation, learned comparison, exact CPU export and physical tracking
remain separate tasks; the Jolt and morphology gates are unchanged.
