---
title: LEARN-JOLT-003 portable CPU review and normalized comparison checkpoint
kind: evidence
summary: Record owner-approved CPU hardware portability, exact software capability, two real calibrated CPU samples, unresolved memory evidence and prospective review status.
owns: [evidence.jolt003-runtime-portability-20261023]
verified: 2026-10-23
---

# Portable CPU identity checkpoint (D-022)

The owner explicitly relaxed *CPU model identity* matching between the
ephemeral GitHub Action runners and requested ballpark normalized figures.
The versioned [D-022 policy](../decisions/2026-10-23-jolt-cpu-portability.md)
retains full raw hardware identity in every diagnostic and declaration, while
pinning the Linux x86_64/Python3.12/Torch2.8.0+cpu/NumPy2.3.5/single-thread/
deterministic software contract for the source-bound admission review.

This is **implementation and measurement evidence, not prospective physical
experiment admission**, qualified learned-policy performance, GPU work or
a replacement experimental allocation.

## Measured reference and comparison

The first saved [reference CPU record](jolt-003-cpu-reference-20261023.json)
binds its full original memory report by SHA-256. The independent later
[reference-normalization receipt](jolt-003-cpu-normalization-20261023.json)
retains the exact original JSON from the second run without rewriting raw
observations.

| Field | Reference | Observed |
| --- | --- | --- |
| Original GitHub run | [39873284613](https://github.com/techrote/neural-cyber-automata/actions/runs/39873284613) | [39873915292](https://github.com/techrote/neural-cyber-automata/actions/runs/39873915292) |
| Original measurement source revision | `0ce035ec0440167abb878d90dda22c25ed8d1fe9` | `ebd67df26647ec672018e35dab21cb4125e4e27c` |
| CPU model | Intel Xeon Platinum 8573C | AMD EPYC 7763 64-Core |
| SHA-256 64-KiB proxy throughput | 1,643,390,192.84 B/s | 1,567,065,831.94 B/s |
| Relative speed vs reference | 1.0000× | 0.95356× |
| Raw observed memory-diagnostic duration | reference-specific (not compared) | 2.4540 s |
| Approximate duration on reference CPU | — | 2.3400 s |
| Maximum simultaneous synthetic-worker plus idle-native RSS | 639,930,368 B | 641,052,672 B |
| Portable pinned-software capability SHA-256 | `f65e3b731f44ce91511b5d51532f5b24cb622d586a26675c7150ec8988ba2406` | identical |

**Interpretation:** the observed EPYC calibration was about 4.64% slower
than the fixed Intel reference in this specific tiny SHA-256 task.
The output uses `t_ref = t_obs * R_obs / R_ref`, not the CPU marketing
name or advertised core count. The 2.34 s estimate is a *ballpark
reference-CPU projection*, not a measured duration on the reference
machine. The original 2.454 s and exact calibration factors remain
available for independent readback.

The second original report's SHA-256 is
`92ef503eec3f1df4510fe6964d1d13bc0164c56d383f2a861d12ea6faf03c72a`;
the comparison receipt has SHA-256
`88373a07842e4702f0c197985aec26262084c4b53bea5b49127031f4f24fbbf3`.
Both came from nonexperimental [GitHub Actions run 39873915292](https://github.com/techrote/neural-cyber-automata/actions/runs/39873915292).
Neither original file required replaying teacher episodes or trained models.

## Revised admission behavior

Current implementation in PR #80 defines a stable versioned
`nca-jolt-cpu-runtime-contract/v1` over pinned software, CPU mode,
platform architecture, deterministic algorithms and threading.
New `resource-admission/v2` and `source-admission-review/v2` bind its
SHA-256 rather than treating `cpu_model`, `kernel`, `libc` version,
`python` patch and available core count as exact-match admission keys.
The original declaration and observer receipt retain those raw hardware
facts; all original source/configuration/accepted host binary/teacher
and training/quality constraints remain strict.

The portable contract is specifically **not** permission to alter
policy weights, reward parameters, 39 operation ceilings, case splits,
training seeds, baseline gains or the raw per-case inference-p95 <=5 ms
acceptance. A CPU speed ratio must not be used as a backdoor to adjust
quality thresholds or qualify a training result.

The historical v1 source/resource reviews are not reinterpreted,
and the original 2026-10-22 memory prepass/unsigned review packets
remain unchanged observations. The prior EPYC 9V74 versus EPYC 7763
model mismatch is no longer an admission mismatch under D-022. It is
still recorded as metadata when comparing the older reports.

## Remaining non-CPU blockers

- The measured 0.64 GB peak samples include synthetic retained results
  plus a native capability/idle handshake, **not** full stepped native
  physics, every validated PPO transition, trained-policy lifecycle,
  independent audit or final publication peak.
- Four-times this synthetic/idle RSS is around 2.56 GB compared with
  an advertised 16 GB hosted runner, showing planning headroom.
  That figure is **not a proven hard upper bound** or proof that a
  kernel/cgroup will enforce a process-tree memory budget.
- Every memory admission still needs a justified prospective resource
  record, source-bound serialisation/byte/disk acceptance and two
  *genuinely distinct* accepted source/runtime reviewers.
  Earlier unsigned packets retain `accepted=false`, `reviewer=null`
  and `prospective_experiment_admitted=false`.
- A passing `freeze-v3` declaration and its independent readback
  must still precede the durable single-use allocation claim and
  the real `execute` mode. Neither operation has occurred.
- This document has no claim of independent signoff or experiment
  consumption. Original LEARN-JOLT-002 failure and morphology #16
  remain separate.

## Practical next step

Complete PR #80's final-head CI/review of the portable admission code,
then independently assess the memory-envelope evidence and sign two
new *version 2* source-admission records only if every non-CPU gate is
supported. Publish/read back the prospective resource and declaration
before any single-use LEARN-JOLT-003 invocation.
