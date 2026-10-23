---
title: LEARN-JOLT-003 memory feasibility and source-bound review checkpoint
kind: evidence
summary: Synthetic worker/native memory diagnostic, source/runtime identities, unresolved admission gates, and two unsigned reviewer packets.
owns: [evidence.jolt-003-memory-admission]
verified: 2026-10-22
---

# LEARN-JOLT-003 memory feasibility checkpoint

**Scope: nonexperimental preparation; prospective execution NOT admitted.**
The owner requested full-worker/native memory qualification and two independent
final-source reviews following the successful initial preflight. This checkpoint
documents measurements obtained without spending
`learn-jolt-003-20261022-a1`. It does **not** turn the nonexperimental
measurement into a full production-worker memory bound or impersonate
independent reviewers.

## Exact source, host and runtime

| Identity | Verified value |
| --- | --- |
| Memory diagnostic executable commit | `73f44409e799902e33519604a76a46464b300b60` |
| Complete v3 source inventory SHA-256 | `9431773c431562d052a420d9ebe8e08b9794eb68e4e7c4da2e0a04a30a7a5fda` |
| Version 2 experiment configuration SHA-256 | `cbedd0adc389bafa0f7d053737181f5e4fc2a34a0d6fa6b9b1ada8acb27a3c51` |
| Exact measurement runtime SHA-256 | `9acb06f2b987803206a11c9c95d65819040ff478a976e50127c3d2e7566a7a01` |
| Full pinned host identity SHA-256 | `dd6cbd59425a43497f41ed4bef98fef35eacd1d6cceba9ba11d484ebd721191e` |
| Host source revision | `f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497` |
| Runtime | Linux 6.17.0-1022-azure, AMD EPYC 7763, 4 logical CPUs, Python 3.12.10, Torch 2.8.0+cpu, NumPy 2.3.5, one Torch thread, deterministic algorithms |

GitHub [memory measurement run #39869437855](https://github.com/techrote/neural-cyber-automata/actions/runs/39869437855)
passed source-closure, runtime, pinned-host handshake, regression tests, explicit
non-admission checks, and independent readback of the report. Its original
`jolt-003-memory-report.json` is in the run's 90-day Actions artifact.
Raw report SHA-256 is
`3e23ae5ce187b0dac59a481d6d99b107af4fa17e271139495288a963799bc59d`;
the downloaded artifact ZIP SHA-256 is
`d93f00bd1c9629eec1fbc462d7935f373e9873f5b48d23a330aebbb52d35a53d`.
These receipts preserve the measured original bytes, but temporary Actions
storage is not durable archive publication.

## Observed memory

| Measurement | Bytes |
| --- | ---: |
| Peak sampled synthetic worker RSS | 491,495,424 |
| Peak sampled pinned native Godot RSS | 149,385,216 |
| Maximum simultaneous sampled RSS sum | 640,819,200 |
| Maximum simultaneous sampled PSS sum | 628,527,104 |
| Python worker `ru_maxrss` | 492,019,712 |
| Illustrative 4× peak sum (not a bound) | 2,563,276,800 |
| Advertised standard public Ubuntu runner memory | 16,000,000,000 |
| Measured `/proc/meminfo MemTotal` | 16,766,414,848 |
| Measured `/proc/meminfo MemAvailable` | 15,232,327,680 |

The synthetic worker retained 6,464 result rows (84 authored metric
envelopes), 2,560 compact teacher rows, 256 PPO one-transition proxies,
22 simultaneous GRU32 models, and two 64 MiB audit-reader buffers.
The real pinned Godot process was started for capability handshake and idle
residency **only**. No native reset, physical step, scored case, saved learned
checkpoint, synthetic-policy validation against sealed scenarios, or training
allocation was opened. Forty-nine process samples included nineteen native
samples. All six Linux cgroup v1/v2 current/limit/peak fields were unavailable
and are recorded as `null`, never zero or unlimited.

The earlier [read-only preflight #39866849277](https://github.com/techrote/neural-cyber-automata/actions/runs/39866849277)
reported 89,738,760,192 bytes of available disk against a required
16,895,424,646-byte complete artifact-and-reserve envelope.
**This disk reading and the memory reading belong to different machines and
cannot be combined into a single source/runtime-admitted declaration.**

## Admission and independence findings

- The simulated result rows, PPO proxies and two audit buffers do not
  encompass every real `JoltTransition` graph, native physics reset/step,
  full worker allocator history, complete audit deserialization and publication.
  The 4× comparison is illustrative headroom only, not an independent
  worst-case or a kernel-enforced peak.
- The earlier preflight identified **EPYC 9V74**, whereas the native memory
  diagnostic identified **EPYC 7763**. The v3 admission validator binds exact
  `runtime_identity()` (including CPU model and kernel) into the prospective
  SHA-256. Standard GitHub-hosted runners are ephemeral, so previously
  measured runtime identities are not automatically interchangeable with the
  runner allocated for a future `execute` dispatch.
- The configured GitHub-hosted six-hour lifetime remains a platform ceiling
  even though the internal v2 operational timeouts were removed. All
  39 exact operation limits, seed/case order, quality thresholds and refusal
  rules remain untouched.
- Two **distinct, source-bound review preparation packets** exist:
  [terminal learning](jolt-003-memory-20261022/terminal-learning-review-preparation.json)
  and [operation/resource envelope](jolt-003-memory-20261022/operation-resource-review-preparation.json).
  Both bind the executable source SHA, experiment configuration, measured
  runtime and host. Neither claims to be independently signed:
  `reviewer=null`, `accepted=false`,
  `prospective_experiment_admitted=false`, and
  `prospective_binding_sha256=null`. They do not satisfy the two required
  `nca-learn-jolt-003-source-admission-review/v1` records.

The memory gate is **unresolved**, not passed: its production full-worker/native
peak has no conservative hard bound or complete actual-process observation.
The source/runtime identity gate is also unresolved across future ephemeral
runners. The resource admission must not set `peak_resident_memory=true`
from these specimens, and no passing declaration may be manufactured.

## Required next acceptance evidence

1. Prospectively define an executable, independently justifiable full-worker
   plus native peak envelope (or a suitable kernel-enforced process-tree limit),
   including real stepped-native memory, Python retention, fitting, final
   audit and publication. Distinguish measured maxima from hard guarantees.
2. Resolve exact runtime binding on the **execution** runner without relaxing
   frozen host/version/source or integrity gates implicitly. Pin a stable
   environment or commission a separately versioned reviewed admission
   mechanism; do not treat two different EPYC models as identical.
3. Have two **genuinely independent authors** inspect the terminal-learning
   and operation/storage/resource gates. Each author must bind the final
   executable revision/closure, configuration, actual host and runtime,
   prospective declaration hash, provide reproducible original evidence and
   disclose any blocking findings.
4. Only after passing source-bound resource and review records and the
   independent freeze/readback should the single manual `execute` invocation
   be considered. Preserve the consumed LEARN-JOLT-002 failure and keep #62,
   owner feel, and morphology #16 gated.
