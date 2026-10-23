---
title: LEARN-JOLT-003 operations
kind: guide
summary: Prospective admission, elapsed-time observation and immutable readback for the Jolt comparison without operational wall-clock limits.
owns: [learning.jolt003-operations]
verified: 2026-10-22
---

# LEARN-JOLT-003 operations

The [implementation contract](design/LEARN_JOLT_003.md), owner-issued
[task](design/LEARN_JOLT_003_PROMPT.md) and later
[D-020](DECISIONS.md#d-020--remove-operational-wall-clock-limits-and-rerun-ci)
govern this path. The allocation ID is
`learn-jolt-003-20261022-a1`; its actual commission is recorded in
[#62](https://github.com/techrote/neural-cyber-automata/issues/62#issuecomment-6347420584).
That commission is conditional on all final source/runtime/resource gates.
Do not run the commands below without the published passing declaration and
two passing independent final-source reviews.

**Current instruction (D-021): implement and verify an explicitly gated
long-running CI entry.** The owner reported CI green with no experiment run,
then authorized continuation. Green conformance alone does not satisfy
source/runtime/resource admission. Do not execute while the full-worker
memory and fresh final-source independent reviews remain unresolved.
The [completion evidence](evidence/2026-10-22-learn-jolt-003-completion.md)
and [refusal record](evidence/jolt-learn-003-completion-20261022/admission-refusal.json)
retain the historical finite-time admission refusal. Configuration v2 removes
all operational wall-clock ceilings and time-fit gates, including the whole
invocation, phase reserves, episode capture, host startup/transport, cleanup and
observation limits. Full-worker/native memory and final-source admission remain
unresolved. No passing execution declaration or experimental invocation is published.

## CPU portability and approximate benchmarking (D-022)

The owner has accepted [D-022](decisions/2026-10-23-jolt-cpu-portability.md):
do not use changing GitHub-runner CPU model, kernel, libc minor release,
Python patch or available-core count as an exact-match admission gate.
Record their original measured values in each declaration/receipt. Only
the pinned CPU **software-capability** contract is compared across jobs:
Linux x86_64, Python 3.12, Torch 2.8.0+cpu, NumPy 2.3.5, CPU-only,
one Torch and interop thread, deterministic algorithms. The exact source,
configuration, accepted Godot host, workloads, evaluation quality and native
artifact identities remain fixed.

The current portable resource-admission record is
`nca-jolt-003-resource-admission/v2`, with
`runtime_contract_sha256` and
`cpu_comparison_policy="cpu-hardware-observational-and-normalized/v1"`.
Each independently authored terminal and operation-envelope source review
must use `nca-learn-jolt-003-source-admission-review/v2`, bind the same
stable runtime contract hash plus full prospective-declaration binding, and
explicitly resolve every required non-time resource gate. Historical v1
review/resource records retain their exact-runtime interpretation.

The read-only preflight and memory diagnostic now publish an optional fixed
single-core SHA-256 calibration. For reference timing *planning only*,
`approx_reference_seconds = raw_seconds * observed_bytes_per_second /
reference_bytes_per_second`. The reference calibration must be a real saved
sample, not an assumed frequency or a CPU-model label. Raw timings and
hardware IDs remain visible; memory bytes, physical outcomes, stage counters
and the actual per-case inference-p95 <=5 ms gate are **never normalized**.
Normalizing ballpark throughput does not establish full-worker/native peak
memory or prospectively accept this allocation.

## Process-tree RSS watchdog and v2 resource review (D-023)

The owner accepted [D-023](decisions/2026-10-23-jolt-resource-risk.md)
to add a **sampled 8 GiB memory-stop threshold** and proceed with
source-bound admission review. It does not assert production peak-memory
qualification or authorize consuming a single-use allocation prematurely.

An accepted portable \`nca-jolt-003-resource-admission/v2\` record must
bind the exact \`memory_watchdog\` policy from
\`nca.train.jolt_memory_watchdog_v3.watchdog_policy_v3()\`:
8,589,934,592 bytes; 0.25 s between RSS samples; original observer plus
all subreaper-owned descendants (worker, independent verifier and native
Godot); fail-closed monitoring; no automatic retry. The original outer
observer samples concurrently with its child-wait loop. On a sampled
overage, unreadable live RSS or PID identity mismatch, that *same* original
attempt fails; the established process-tree reaper must revoke/reap it
and retain original stdout/stderr, exit, closure and watchdog receipts.

The raw Actions artifact includes
\`learn-jolt-003-20261022-a1-memory-watchdog.json\`, with separate
file-identity readback from the observed exit. This observer-derived RSS
threshold is **not** a kernel/cgroup hard limit: a transient memory surge,
an external process or runner-wide OOM may defeat it before the next sample.
GitHub's six-hour hosted-job expiry is unchanged.

Nonexperimental memory CI produces the original bounded synthetic/native
specimen, independent calibration and a
\`nca-jolt-003-resource-assessment-draft/v2\` with source/config/host
and portable software hashes. The draft **must** contain
\`accepted=false\`, \`prospective_experiment_admitted=false\`,
\`peak_resident_memory=false\` and the unresolved complete-worker peak.
It cannot replace a passing resource admission or independent review.
The source-derived artifact size ceiling includes a 1 GiB disk reserve;
free disk is sampled now but must be rechecked on the execution runner.

**Review A:** a genuinely independent terminal-learning reviewer checks
the exact clean executable source, safe-retirement/GAE masks and outcome
semantics, BC/PPO seeds and cases, frozen six-final barrier, all
qualification thresholds, and original teacher prerequisites.

**Review B:** a different independently authored operations/resource
reviewer checks the same source closure, CPU software contract and
pinned native host, 39 counters, journal/reader bounds, available disk,
sampled RSS limitations, soft watchdog abort/reaping, immutable evidence,
single-use claim and original observed exit.

Both final reviewer records must use
\`nca-learn-jolt-003-source-admission-review/v2\` and bind the exact
\`prospective_binding_sha256\`, same source/config/host/CPU capability
and zero unresolved blocking findings **before** \`freeze-v3\`.
Reviewer identities cannot be fabricated from unsigned preparation
documents or green CI. If either independent reviewer refuses, retain
that refusal without launching training.

## GitHub Actions: long-run dispatch

The [manual LEARN-JOLT-003 workflow](../.github/workflows/learn-jolt-003-long-run.yml)
must first be merged onto the repository's default branch alongside the
configuration-v2 dependency. It is **not** triggered by green pull-request CI,
ordinary push events or unit-test success. It runs on a pinned CPU-only Linux
runtime and invokes the actual learner only in its admitted `execute` mode.
The workflow explicitly passes `--unbounded` to `tools/host_setup.py`, removing
host download/import/setup timeouts in this manual lane while preserving the
previous bounded defaults for ordinary conformance CI.

1. Dispatch `preflight` to inspect the exact configuration, source closure,
   pinned CPU runtime, host identity, storage envelope, free disk and bounded
   memory telemetry. It does **not** start a physical experiment, open a model
   or consume allocation `learn-jolt-003-20261022-a1`. Reported cgroup memory
   is a sample, **not** proof of peak native/worker feasibility.
2. Resolve the resource admission, including peak worker/native memory and
   exact current storage/disk requirements. Publish a separate accepted
   resource record and two independently authored, final-source-bound reviews
   under `docs/evidence/`, with zero unresolved findings. Preserve all
   original bytes and the prospective binding. See the admission section below.
3. Only then dispatch `execute`, supplying a complete 40-character
   `executable_sha`, a complete 40-character `admission_sha` for the
   published reviews, and three exact repository-relative JSON paths:
   `resource_path`, `terminal_review_path`, `envelope_review_path`.
   Paths must be under `docs/evidence/`; these files are never invented
   by the workflow. The execution ref must be `main`, first attempt only.
4. The workflow runs `freeze-v3`, rereads the declaration with the existing
   source/runtime/review validators and current free-disk gate, then
   atomically claims the single allocation via a permanent comment on
   [#62](https://github.com/techrote/neural-cyber-automata/issues/62).
   The claim is independently reread before the observer can run. Existing
   claims, reruns or failed reviews stop execution. Concurrent dispatches
   use a shared non-cancelling concurrency group.
5. The **real** `tools/observe_jolt_v3.py` subsequently performs teacher
   collection, BC, recurrent PPO and paired comparisons. Its original
   stdout/stderr, observed exit, lifecycle and worker files stay outside Git.
   A clean observed exit permits a further independent inventory readback.
   Both success and failure paths attempt separate 90-day Actions artifacts
   for receipts/hash manifest and raw/partial original evidence.

A GitHub-hosted job still has the service's six-hour upper bound, which
configuration v2 cannot remove. Cancellation, runner loss and hard expiry may
prevent `always()` artifact-upload steps from running; missing uploads are
**not** evidence of a completed invocation. GitHub artifacts are temporary:
complete evidence still requires independent readback and durable
out-of-Git publication with original member hashes/EOF checks. Once the issue
claim exists, do not rerun even if the artifact upload fails. Preserve a
negative result and do not release morphology #16 without the original
learner-qualification requirements.

## Prepare the exact environment

Execution requires Linux with a consistent process and `/proc` PID namespace,
subreaper and pidfd support, the unchanged pinned Cybersand/Godot build,
Python 3.12, Torch `2.8.0+cpu`, NumPy `2.3.5`, one Torch thread and CPU only.
`tools/host_setup.py` prepares and verifies the existing accepted runtime; it
does not admit a physical experiment. A PID-isolated shell with a mismatched
`/proc` view cannot provide the required cleanup proof and is unsuitable.

Use the exact clean reviewed executable commit, including required checks on
that commit. Preserve newer unrelated main changes before freezing it. No
models are loaded from LEARN-JOLT-002. The old declaration and its output remain
consumed; the new output and its lifecycle/observer names must not exist.

The static configuration, complete scenario stream and comparison ordering are
already fixed. Check them without opening a host:

```sh
PYTHONPATH=src python -c 'from pathlib import Path; from nca.train.jolt_admission import load_config; load_config(Path("configs/learning/jolt-learn-003-v2.json"), root=Path.cwd())'
python tools/check_docs.py --strict
```

On Windows, configuration and documentation inspection can use an installed
CPU package and the same Python command. The v3 executor itself remains a Linux
lane; Windows preparation does not substitute for Linux process qualification.

## Freeze a reviewable prospective declaration

The resource record binds the exact source closure, configuration, host and
pinned software-capability contract while independently recording observed CPU
hardware. It includes `storage_bounds_v3()` byte/file totals and explicitly passes
serialization, readers, journal bytes, total artifacts/files, peak memory,
available disk. Under configuration v2, `original_clock_phase_enforcement` and
`complete_pipeline_time_feasibility` must explicitly be null; the seven other
gates must be true. A size ceiling or historical mean runtime does not satisfy every
resource gate. Synthetic conformance uses fresh random models and fabricated
inputs only; no undeclared physical or trained-checkpoint pilot is permitted.

Use `build_prospective()` in `nca.train.jolt_cli_v3` to construct the exact review
candidate from a clean executable checkout. Reviewers bind its
`prospective_binding()` hash, source revision/closure, configuration, host and
runtime. Publish their original review files and a resource record. Final
reviews set `offline_scope_only=false` and
`prospective_experiment_admitted=true` only if every gate passes.

```sh
python tools/learn_jolt.py freeze-v3 \
  --config configs/learning/jolt-learn-003-v2.json \
  --checkout "$NCA_HOST_CHECKOUT" --godot "$NCA_GODOT" \
  --output "$NCA_JOLT_OUTPUT" \
  --declaration "$NCA_JOLT_DECLARATION" \
  --resource-admission "$NCA_JOLT_RESOURCE_ADMISSION" \
  --review "$NCA_JOLT_TERMINAL_REVIEW" \
  --review "$NCA_JOLT_ENVELOPE_REVIEW" \
  --allocation-id learn-jolt-003-20261022-a1 \
  --authorization-ref 'https://github.com/techrote/neural-cyber-automata/issues/62#issuecomment-6347420584'
```

Reviews can reside in a later documentation checkout, with stable
`docs/evidence/` paths. Their exact original text is embedded and hashed in the
declaration; the clean executable checkout remains the reviewed source commit.
Publish and independently read back the declaration before opening the run.
If resource feasibility is unresolved, publish admission failure instead of
freezing a passing declaration or spending the allocation.

## Observe the one permitted invocation

The external observer starts the original clock before imports and child
creation, passes that unchanged origin to the executor, and retains actual CLI
stdout/stderr, observed exit code/time, cleanup and lifecycle verification.
It never infers process exit from a worker result flag.

```sh
python tools/observe_jolt_v3.py \
  --config configs/learning/jolt-learn-003-v2.json \
  --declaration "$NCA_JOLT_DECLARATION" \
  --checkout "$NCA_HOST_CHECKOUT" --godot "$NCA_GODOT" \
  --output "$NCA_JOLT_OUTPUT"
```

Retain the observer's original terminal output and actual exit as well. A lost
terminal stream, failed fsync, nonzero observed CLI exit, failed cleanup or
incomplete independent verification is not completion. Never rerun the command
to obtain cleaner evidence. Partial publication paths remain consumed.

The worker performs fresh teacher collection, fresh BC/PPO initialization and
the fixed comparisons. One persistent host and one worker share the declared
clock and counters. Safe failures advance the frozen schedule only after full
retirement evidence; all other failures terminate ordinary work.

## Read back, publish and interpret

The original executor performs inventory/cost verification without a wall-time
limit. Later read-only inspection can verify the retained
original stdout and independently observed CLI exit without a host or new
neural work:

```sh
python tools/learn_jolt.py lifecycle-v3 \
  --output "$NCA_JOLT_OUTPUT" \
  --terminal-stdout "$NCA_JOLT_ORIGINAL_STDOUT" \
  --observed-exit-code "$NCA_JOLT_OBSERVED_EXIT" \
  --observed-exit-elapsed-seconds "$NCA_JOLT_OBSERVED_ELAPSED"
```

Supply actual observed values from the immutable observer receipt, never the
worker's claimed return code. Later inspection does not repair a missing
original timing policy, receipt or journal suffix. Archive raw evidence and
checkpoints outside Git, verify complete archive EOF and every immutable file
identity, and publish concise hashes, costs, result, qualification and limits
in the repository. Preserve original failed and negative outcomes.

A complete negative comparison is a valid experiment result, with #62 still
open. An admission failure spends no experiment. An incomplete run admits no
retry, extension, new seed or checkpoint reuse. Owner feel and morphology #16
remain separately gated in every case.
