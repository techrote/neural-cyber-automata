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

**Current instruction: publish the time-limit removal, queue CI, then stop.**
Do not launch the experiment or merge while waiting for the owner's follow-up.
The [completion evidence](evidence/2026-10-22-learn-jolt-003-completion.md)
and [refusal record](evidence/jolt-learn-003-completion-20261022/admission-refusal.json)
retain the historical finite-time admission refusal. Configuration v2 removes
all operational wall-clock ceilings and time-fit gates, including the whole
invocation, phase reserves, episode capture, host startup/transport, cleanup and
observation limits. Full-worker/native memory and final-source admission remain
unresolved. No passing execution declaration or experimental invocation is published.

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
runtime; includes `storage_bounds_v3()` byte/file totals; and explicitly passes
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
