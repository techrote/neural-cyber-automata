---
title: LEARN-JOLT-003 D-023 resource-admission assessment and independent review handoff
kind: evidence
summary: Source-backed serialisation, disk and soft-memory risk assessment with two distinct unsigned v2 review requests; no physical allocation or approval.
owns: [evidence.jolt003-risk-resource-assessment]
verified: 2026-10-23
---

# LEARN-JOLT-003 resource-admission assessment (D-023)

**Disposition: PREPARATION COMPLETE; EXPERIMENT ADMISSION PENDING.**
The owner approved steps 1–3: an 8 GiB process-tree watchdog, an
independently read back source-bound resource assessment and two genuinely
independent final-source reviews. This report is the **implementer's**
assessment; it is not signed as an independent final experiment admission
and cannot substitute for either required reviewer.

## Original source and configuration basis

- The current reviewed implementation lives in
  [PR #81](https://github.com/techrote/neural-cyber-automata/pull/81).
  The first tested watchdog/reviewer-request executable revision is
  `638d59b49d5a70e7c439baffa88691a6f8b20600`, with exact
  261-file source inventory SHA-256
  `580608deade828f1668f357115ef7f3bc51ea850ce8551bf510c2db698bd2687`.
  This source identity was independently verified by
  [nonexperimental CI #39880610724](https://github.com/techrote/neural-cyber-automata/actions/runs/39880610724);
  the resource assessment original SHA-256 is
  `6d897a3280887a28b8c2725593e232f828dff6465cd8c4d2d3f5d2274074fd0d`.
  Later docs-only commits do not alter the source closure, but exact
  executable revision must still be reviewed and frozen intentionally.
  The approved preceding source baseline was
  [PR #80 merge \`9747ef0b\`](https://github.com/techrote/neural-cyber-automata/commit/9747ef0b610053b837cce1f57b62c6f880c8144f),
  merge tree \`e4dd3f939ae6e95338df97233e364794993155aa\`.
- Unchanged experiment configuration:
  \`configs/learning/jolt-learn-003-v2.json\`;
  stable content SHA-256 (canonical JSON)
  \`cbedd0adc389bafa0f7d053737181f5e4fc2a34a0d6fa6b9b1ada8acb27a3c51\`,
  allocation \`learn-jolt-003-20261022-a1\`. The old physical experiment
  remains unspent.
- Pin the Cybersand host revision
  \`f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497\`
  with its binary and Rapier provenance; the CPU software capability
  remains Linux x86_64, Python 3.12, Torch 2.8.0+cpu,
  NumPy 2.3.5, one Torch/interop thread, deterministic algorithms.
  CPU model is advisory under [D-022](../decisions/2026-10-23-jolt-cpu-portability.md).
- No new physics/training workload, baseline, teacher, seeds, case split,
  failure semantics, raw inference-latency acceptance or 39-operation ceiling
  is introduced by the resource/watchdog code.

## Resource gates and supporting evidence

| Gate | Assessment | Basis and remaining limitation |
| --- | --- | --- |
| Canonical reservation maximum | Static source **pass** | 39 exact counter limits, one attempt and four stage ceilings remain enforced by \`jolt_execution_v3.py\` and \`jolt_budget_v3.py\`; maximum 1,901,556 reservation records is a separate conservative count envelope |
| Reader limits | Static source **pass** | Capped record decoding, stream drains, SHA-256/EOF checks, checkpoint and JSON/gzip readers in \`jolt_storage_v3.py\` |
| Journal aggregate bytes | Static source **pass** | At most 115,110 records, each <=64 KiB; linked journal close/readback mandated |
| Total artifact bytes | Static source **pass** | 15,821,682,822 bytes (cumulative closed-form cap) across raw/sample episodes, journal, spools, reservations, checkpoints, receipts |
| Total file count | Static source **pass** | Up to 19,469 worker files plus lifecycle/observer originals; closed-inventory rejects extra files/dirs, links and nonregular data |
| Available disk | **Provisional** | 16,895,424,646 bytes required inclusive of 1 GiB reserve; earlier GitHub preflight measured ~89.7 GB free on a different runner. Exact execution-runner disk must be rechecked before the one-use claim |
| CPU / native host | Software contract **pass** | Pins CPU-only versions, source/config/host binary and deterministic mode. CPU model and kernel are recorded, not exact-match gates under D-022 |
| Peak resident memory | **Not yet accepted** | ~640–641 MB synthetic retained-worker + *idle* native RSS; 4x planning allowance ~2.56 GB. Neither that multiplier nor the new watchdog proves the peak of actual stepped native, all PPO data, independently reconstructed audit and publication |
| Execution process watchdog | **Implemented; CI verification pending exact final head** | Exactly 8,589,934,592 sampled bytes, every 0.25 s across original observer and all owned descendants; failure/monitoring error aborts original run and requires PID-safe revocation |
| Original wall-clock phase enforcement | **N/A (explicit null)** | Owner removed repository time limits in v2; original elapsed timing still recorded |
| Complete pipeline time feasibility | **N/A (explicit null)** | GitHub-hosted hard six-hour job maximum still applies independently |

## What the 8 GiB guard actually establishes

The outer original observer is already a Linux subreaper. The
[new source-bound memory-watchdog policy](../../src/nca/train/jolt_memory_watchdog_v3.py)
samples observer plus every identity-checked process-tree descendant.
Sampling is part of the actual bounded-stream wait loop, not a
post-completion file probe. Sampled RSS at/above 8 GiB, a changed
PID start identity or unreadable *live* process RSS throws a
machine-readable abort. The original observer retains stdout/stderr,
normal exit/cleanup evidence and a separate
\`-memory-watchdog.json\` receipt with sample count and peak RSS;
the workflow uploads this original receipt as a raw/partial artifact.
A failure cannot be silently retried and cannot acquire a positive
executor/observed-exit verdict.

**It is NOT a hard memory bound.** The kernel may OOM-kill the worker
between 250 ms samples, before evidence can be synchronized, or from
unrelated host memory pressure. Native physics, held PPO transitions,
source-bound audit object graphs and publication may differ materially
from idle/synthetic memory. The independent operations reviewer must
explicitly accept or reject that disclosed residual risk; this
implementer assessment keeps \`peak_resident_memory=false\`.

## New source-bound CI assessment

The nonexperimental
[\`tools/assess_jolt_resources_v3.py\`](../../tools/assess_jolt_resources_v3.py)
reads an original pinned-host/native synthetic memory report and
independently verifies exact executable source closure, reviewed config,
host and CPU software identity, 39 operation counters and the complete
\`storage_bounds_v3()\` byte envelope. It records the current diagnostic
runner's free disk, sampled peak RSS, fourfold illustrative allowance,
exact 8 GiB watchdog policy and unresolved risk.

The immutable CI artifact
\`jolt-003-resource-assessment.json\` has schema
\`nca-jolt-003-resource-assessment-draft/v2\`,
\`accepted=false\`, \`prospective_experiment_admitted=false\`,
\`peak_resident_memory=false\` and an empty independent-review list.
Its *negative* result is checked by the same source-bound CI job.
The [observed draft from run #39880610724](https://github.com/techrote/neural-cyber-automata/actions/runs/39880610724)
recorded 641,118,208 sampled RSS bytes, a nonbinding fourfold allowance
of 2,564,472,832 bytes, an 8,589,934,592-byte soft threshold,
89,737,969,664 bytes sampled disk free and
16,895,424,646 bytes required. Five static resource gates passed;
the diagnostic-runner available-disk sample passed; both historical
time-fit gates were explicitly null. The **peak resident memory gate
remained false**, four explicit blockers were preserved, and no
allocation or reviewer identity was consumed.
It is intentionally **not** a validator-admitted
\`nca-jolt-003-resource-admission/v2\` record. A green synthetic CI
run cannot upgrade this assessment to a passing resource declaration.

## Two independent final-source review requests

- [Terminal-learning and qualification reviewer request](jolt-003-resource-admission-20261023/terminal-learning-review-request.json)
  covers teacher prerequisites, v3 retirement/GAE, BC/PPO and
  case ordering, sealed holdouts and unchanged 5 ms inference p95.
- [Operations, serialization and memory reviewer request](jolt-003-resource-admission-20261023/operations-resource-review-request.json)
  covers 39 counters, native host provenance, 16.9 GB disk
  requirement, soft RSS watchdog/cleanup, original evidence and
  single-use allocation claim.

The requests have \`reviewer=null\`, \`accepted=false\` and
\`prospective_binding_sha256=null\` until **two different independent
reviewers** actually inspect the final source and the same prospective
resource declaration. Only those reviewers can author
\`nca-learn-jolt-003-source-admission-review/v2\` accepted records,
with zero unresolved blockers, original reviewer identities and exact
SHA-256 bindings. No agent-controlled duplicate JSON masquerades
as two independent reviews.

## Handoff and disposition

1. Require final-head source/CI identity from PR #81 and retain the
   original memory/resource assessment artifact with its own hash.
2. Have an independent terminal-learning reviewer and a different
   independent operations/resource reviewer each issue a source-bound
   assessment. For residual soft-memory risk, require an explicit
   accountable acceptance rather than a made-up hard-bound claim.
3. If a review refuses, retain it and keep the allocation unspent.
   If and only if a genuinely accepted resource v2 record and two
   independent v2 review records satisfy the unchanged validators,
   freeze/read back the declaration on the eventual actual runner,
   perform the current-run disk check, then require an explicit manual
   one-use \`execute\` dispatch.

**No LEARN-JOLT-003 teacher episode, BC/PPO fit, sealed comparison,
trained checkpoint or physical allocation was opened for this review.
Issue #62 and morphology #16 remain unaccepted.**
