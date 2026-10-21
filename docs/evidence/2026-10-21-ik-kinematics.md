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
than leaving those durations at zero. The independent evidence review below
reconciles all 119 measured source files with their declared identities.

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
| Hosted checks on the final PR head | Pending at evidence publication; final CI and merge acceptance are recorded on [issue #52](https://github.com/techrote/neural-cyber-automata/issues/52) |

The full-suite failure is
`test_jolt_recovery_v4_executor.RecoveryV4ExecutorTests.test_isolated_python_tree_closes_a_separately_sessioned_descendant`.
It fails the sandbox's consistent PID namespace check and reproduces without the
IK changes on the accepted base revision. This remains a recorded local
validation failure; it is not counted as an IK pass or silently skipped. Hosted
checks must pass on the final PR head before merge; issue #52 records that
acceptance separately from this evidence-publication checkpoint.
Skipped actual-CUDA checks establish no device qualification.

These conformance results do not claim completed final-head acceptance. The
separately declared reference measurement below retains its own frozen source
and allocation identity.

## Reference performance

Run `ik-reference-5d10108f7cf44cbc967d5ffd9091dae3` completed its single declared
CPU allocation on 21 October 2026. The [prospective issue declaration](https://github.com/techrote/neural-cyber-automata/issues/52#issuecomment-6328585303)
was published and read back at **23:09:01 UTC**, before the worker started at
**23:09:30 UTC**. Execution used clean source
`f801577b38cafe4a9f1a0e00ab5d6bcd67935972`, tree
`c026a669d4b3ebb5ad7e6c28269f387061ea4a3f`, source inventory SHA-256
`b1e12471fcce346fac2c372d607720c5e04599596ddb4ec576b312bba28596a0`.
Later documentation and evidence-publication commits do not replace that measured
source identity.

The [declaration](ik-reference-20261021/declaration.json) binds configuration
`fa1151cfb81d1d7fcad4744d4da401958a61300183258f4eed7bab1c7eec9474`, the unchanged
profile/acceptance, and Linux x86-64, CPython 3.12.14, AMD EPYC 9V74 CPU execution.
The environment exposes nine logical CPUs; the solver uses one Python execution
thread and no Torch or NumPy arithmetic. Its deterministic 9-by-9 Runner target
grid uses three fixed initial configurations, with no random seed or learned
training/tuning/sealed partition. Cases run in fixed order, numerical then
analytic, after eight warmup calls per method.

All **502 attempted calls passed** the declared finite/bounded/solved and
independent endpoint checks: **243 measured plus eight warmups per method**,
with zero failures. The acceptance ceiling is `1e-6 cell`; numerical stopping
uses the prospectively declared tighter `1e-7 cell`. The full
[result](ik-reference-20261021/result.json) preserves every call and failure
population. Percentiles use nearest rank over all measured calls.

| Method | Measured accepted / attempted | Median core ms | p95 core ms | Maximum core ms | Worst independent endpoint residual, cells |
|---|---:|---:|---:|---:|---:|
| Bounded damped least squares | 243 / 243 | 0.642176 | 1.628497 | 7.481853 | `9.977520442597764e-8` |
| Independent analytic two-link | 243 / 243 | 0.092699 | 0.177437 | 1.651767 | `2.808666774861361e-15` |

Timing values above are rounded milliseconds; exact seconds remain in the
result. The analytic reference is faster in this measurement and remains a
legitimate baseline for learned comparisons. Core timing includes each solver
call and diagnostic construction. It excludes target generation, independent
endpoint checking, target/lifecycle validation, model inference, transport,
host motion and publication. It is not an exported-runtime or physical latency
qualification, and shared CPU timing is not a universal deployment guarantee.

The allocation consumed **1,598 numerical iterations of 16,064 allowed** and
retained **148 numerical projection events**, including warmups. There are zero
calls with missing iteration/projection diagnostics. The append-only journal
contains 502 starts and 502 finishes with no incomplete call. No retry or resume
is admitted by this consumed allocation.

The [supervisor receipt](ik-reference-20261021/result.json.supervisor.json)
passes all fifteen source/environment/configuration, endpoint, journal,
accounting and closure checks. The child exits zero without timeout. Recorded
durations are **1.409384175 seconds** for the child process and **1.632817047
seconds** through supervisor readback; the latter excludes its own final receipt
write. The [outer CLI receipt](ik-reference-20261021/supervisor-cli.json) records
**1.671319562 seconds** through supervisor process completion, within the declared
**30-second** aggregate ceiling. The declaration reserves five seconds for
publication/readback and caps the child at twenty-five seconds.

The [publication index](ik-reference-20261021/index.json) binds byte-preserved
originals and retrieval paths. Exact file SHA-256 values include:

| Original file | File SHA-256 |
|---|---|
| Declaration | `3291e760d06db8127c073790640a9c03695621b4ae056533db448a3ceecb00d1` |
| Result | `57a381e314c8a7cb918acb94053889d7eb5ca5e253529f466b54ebe2e7764dce` |
| Supervisor receipt | `e1d7bc8d3ae73cc246744360340d8af5dace8097eed27105108390f9582129cf` |
| Outer CLI receipt | `c13262f84d5c635d1161d4f636013a5b4bf25f9274188c77c4fda1024ee30e67` |
| Independent review | `86ab2f9809bce907badc45b8a1755086a57fab145ff34c68cf43a1d5e7d4fa66` |

These file hashes are distinct from the canonical semantic digests inside the
records. The worker's original result explicitly awaits a separate supervisor
receipt; its bytes are preserved alongside that completed receipt.

The [independent acceptance readback](ik-reference-20261021/independent-review.json)
passes **29 of 29 checks with no findings**. It verifies all 119 source files,
the independently retrieved prospective declaration, all 502 calls, the 1,005
journal events, exact timing percentiles, 1,598 iterations, 148 projections and
receipt/digest consistency. Endpoint checks use an independent
complex-vector calculation without calling the implementation, worker or
supervisor. The publication index retains eleven artifacts totaling 1,029,830
bytes, excluding the index itself.

Hosted final-head CI was pending at evidence publication. Final CI records and
merge acceptance belong to [issue #52](https://github.com/techrote/neural-cyber-automata/issues/52)
and do not change the measured source identity above. No CUDA, learned or physical
qualification follows from the CPU geometric result.

## Limits

Actual CUDA parity is unmeasured; #51 retains actual-device qualification.
Optional Windows Torch execution is unmeasured unless separately recorded.
CPU numerical accuracy does not establish contact, balance, clearance, compliant
anchor/pose accuracy, host joint-target capability or physical usefulness.
Dataset generation, learned comparison, exact CPU export and physical tracking
remain separate tasks; the Jolt and morphology gates are unchanged.
