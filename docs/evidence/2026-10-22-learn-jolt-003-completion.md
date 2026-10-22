---
title: LEARN-JOLT-003 offline completion and admission refusal
kind: evidence
summary: Reviewed v3 software, preserved failures, exact source and synthetic pipeline limits without spending the conditional experiment.
owns: [evidence.jolt003-completion]
verified: 2026-10-22
---

# LEARN-JOLT-003 offline completion and admission refusal

The implementation and independent offline review requested in
[PR #73](https://github.com/techrote/neural-cyber-automata/pull/73) are complete.
**Prospective physical admission is refused.** Allocation
`learn-jolt-003-20261022-a1` remains unopened and unspent. Complete pipeline time
and full-worker/native peak memory have not been established. This is the
explicit refusal outcome permitted by the [commissioned task](../design/LEARN_JOLT_003_PROMPT.md),
not a learned comparison or a passing execution declaration.

[#62](https://github.com/techrote/neural-cyber-automata/issues/62) remains open:
no learned family is qualified, no sealed outcome was opened by this task, and
owner-feel and morphology [#16](https://github.com/techrote/neural-cyber-automata/issues/16)
retain their separate gates. The original
[handover](2026-10-22-learn-jolt-003-handover.md), failed LEARN-JOLT-002 evidence
and earlier synthetic reports remain byte-identical.

## Exact source and unchanged authority

The final executable source is
[`c3246acada9447ecc94b93c8c25425ecf7fa0214`](https://github.com/techrote/neural-cyber-automata/commit/c3246acada9447ecc94b93c8c25425ecf7fa0214),
tree `3c7ca2a126493e8fda45d6de0da28eb506b06420`, based on main
`f3412847cf1137529cf41d8e39d1c7acda74a5f8`, including the unrelated accepted IK work.
Later evidence/documentation commits preserve this exact source closure.

| Binding | Exact identity |
| --- | --- |
| Execution source inventory | 244 files; SHA-256 `c6113b08d7bf7416d29a4d1df578b0155ba8409e84765ac6603ce3a8171b3ecd` |
| Typed conformance inventory | The same 244 files plus `configs/morphology/profiles-v1.json`; SHA-256 `6b4a05a8331d75189ab2d5f098498164c24d1d134b7c3c634e0a7d2b40012496` |
| Configuration canonical SHA-256 | `239e2432d652414130217b51ffb8002aff67aeaa1694979d0081cfa3c341f353` |
| Configuration file SHA-256 | `76129b50e37774d20b46d34b82d4ca66b06c989838b44332ad57306453dddbf2` |
| Required host revision | `f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497`; no experiment host was started |
| Local offline runtime | Linux x86-64, Python 3.12.14, Torch 2.8.0+cpu, NumPy 2.3.5, CPU only, one configured Torch thread |

The [source freeze](jolt-learn-003-completion-20261022/source-freeze.json)
records every source hash and the full runtime identity. The
[preservation manifest](jolt-learn-003-completion-20261022/preserved-history.json)
verifies all 179 pre-existing evidence files and all 59 configuration files
against handover revision `4983c711ebe873ab04eecf10b8e72f678c4c622a`.
All 39 operation ceilings, fixed seeds, case/controller ordering, rewards,
loss masks and qualification thresholds remain unchanged. Stage limits remain
6,464 starts, 18,944 decisions and 113,664 requested ticks; the 20,000/120,000 aggregate
backstops grant no additional work. The executable whole-process ceiling
remains 7,200 seconds with a 600-second audit reserve.

## Software findings closed

The closed inventory now examines every filesystem entry before hashing
artifact contents. Only declared regular single-link files and their required
ancestor directories pass. Unexpected empty directories, FIFO/socket/special
entries, symlinks, hard links and noncanonical inventory names fail. The outer
verification wrapper validates the declaration file before its initial hash,
so a substituted FIFO cannot block that pre-inventory read. Lifecycle inventory
also rejects hard links. These checks cover static closed-tree substitutions;
they do not claim protection against a hostile concurrent filesystem mutator.

Hosted Windows testing found that `DirEntry.stat` supplies a zero link count
on that platform, incorrectly rejecting an unchanged valid inventory. The
final implementation uses `Path.lstat` for actual no-follow metadata. A new
portable regression emulates the Windows metadata omission and verifies both
valid-tree acceptance and real hard-link rejection. The original
[failed Windows job](https://github.com/techrote/neural-cyber-automata/actions/runs/39844933300/job/120055621359)
remains part of the evidence; the check was fixed, not skipped. This behavior
matches the [Python filesystem API contract](https://docs.python.org/3.12/library/os.html#os.DirEntry.stat).

Independent semantic testing found a second defect: quiet emergency recovery
can finish exactly at a PPO administrative cutoff. The final consumed physical
row is masked and has no model proposal, while the next-observation peek has a
real nonzero value. Delegating that value directly to the legacy sequence
validator rejected an otherwise valid v3 capture. The narrow v3 projection now
uses zero internal bootstrap for the masked final row at the administrative cutoff. The actual
peek value, observation hash, neural forward charges and raw physical evidence
remain intact; mask gaps already cut GAE. No extra action, value forward,
terminal penalty or critic target is invented. Legacy rollout validation and
the old execution v1/v2 interpretation remain unchanged.

Synthetic capture, persistence and independent neural replay cover ordinary
arrival, native task deadline, a task terminal coincident with emergency
duration, normal quiet recovery and quiet recovery exactly at administrative
cutoff. The tests preserve task-terminal priority and actual residual forward
counts. These tests use newly initialized networks and temporary synthetic
artifacts; they are not physical learning or checkpoint reuse from LEARN-JOLT-002.

## Verification and independent review

| Evidence | Source and observed result |
| --- | --- |
| [Full offline transcript](jolt-learn-003-completion-20261022/offline-tests.log) | Pre-portability executable source `0cba752e70194a437223682939edd36f25a2e1dc`: 1,181 tests, 1,122 passed and 59 expected environment skips, 139.259 seconds |
| [Focused transcript](jolt-learn-003-completion-20261022/focused-tests.log) | The same preceding source: 73 passed, zero skips, 46.157 seconds |
| [Final inventory/lifecycle transcript](jolt-learn-003-completion-20261022/inventory-portability-tests.log) | Final source `c3246aca…`: 26 passed, zero skips, 0.547 seconds, including the Windows regression |
| [Final executable source CI](jolt-learn-003-completion-20261022/source-ci.json) | All five required jobs passed on `c3246aca…`; hosted learning-CPU suite: 1,182 tests, 1,176 passed and six skips, 225.909 seconds |
| [Typed pipeline report](jolt-learn-003-completion-20261022/typed-pipeline-conformance.json) | Final clean source; fixed bounded synthetic sampling completed; no host, model or checkpoint executed by this helper |
| [Original helper output](jolt-learn-003-completion-20261022/typed-pipeline.log) and [observed exit](jolt-learn-003-completion-20261022/typed-pipeline-observed-exit.json) | Exit 0 without timeout; 23.145674724000855 seconds for the complete direct-child process |

The full and 73-test transcripts precede the final two-file Windows correction.
Only `jolt_cli_v3.py` and its inventory test changed afterward; semantic and
pipeline-helper bytes are identical. The final targeted checks cover that
delta. All five required jobs also passed on the final executable source,
including the hosted 1,182-test learning-CPU suite. The [validation record](jolt-learn-003-completion-20261022/validation.json)
retains exact transcript hashes, original failures, source attribution and
environment limits. The final merge gate is all five hosted jobs on the exact
publication head: docs, Ubuntu CPU, Windows CPU, optional CPU learning and
pinned-host conformance. The actual final job results and verified merge identity are recorded on
[PR #73](https://github.com/techrote/neural-cyber-automata/pull/73) after those
steps finish; this evidence-publication commit does not pre-claim a pending
CI result.

Three fresh review records bind the final source, configuration and runtime:

- [Terminal, learning and qualification review](jolt-learn-003-completion-20261022/semantic-review.json): independently reviews the production semantic correction and discloses authorship of regression tests.
- [Operation, inventory and envelope review](jolt-learn-003-completion-20261022/envelope-review.json): independently reviews inventory/counters/deadlines and discloses authorship of the synthetic helper and its tests.
- [Wrapper and pipeline cross-review](jolt-learn-003-completion-20261022/inventory-cross-review.json): independently reviews the wrapper correction and helper/evidence, and discloses inventory/Windows-test authorship.

Their acceptance is limited to offline software and truthful refusal evidence.
Every review retains `offline_scope_only=true` and
`prospective_experiment_admitted=false`. None substitutes for the two passing
prospective reviews required before a physical invocation.

## Complete pipeline assessment and budget decision

The typed helper accounts for 16 event categories totaling 115,110 journal
records, including 38,338 owner requests and 38,338 responses. A source walk
identifies nine reservation parse/replay passes and three complete journal-chain
passes across closure, audit, publication and external verification. Two extra
raw reservation identity hashes, after closure and final-publication writes,
remain separately untimed. Previous
reservation-only estimates omitted these other phase costs.

Sampling is fixed in advance: 16 rows per journal type, four read repetitions,
25,000 reservation rows, one- and forty-decision storage specimens at gzip
levels 1 and 9 with three repetitions, and an eight-MiB hash sample. Planning
projections multiply the slowest sample rate by the source count/byte bound and
four. Metadata and episode specimens are explicitly size/shape surrogates,
not valid physical trajectories. Filesystem caches were not evicted.

| Phase | Reserved phase increment | Measured-component planning estimate, excluding unresolved work |
| --- | ---: | ---: |
| Active work | 6,000 s | 18,864.256 s |
| Closure/pre-audit | 180 s | 1,323.666 s |
| Complete audit | 600 s | 8,058.123 s |
| Worker publication, before reserved cleanup | 108 s | 459.618 s |
| External verification, before reserved cleanup | 228 s | 1,370.063 s |

Each reserved increment belongs to the same original absolute clock. Earlier
entry into a phase can use preceding unspent time; these are not independent
per-phase timers.

These figures are conservative extrapolations from synthetic components. They
are **not observed experiment durations, universal upper bounds, proven minimum
times or a sufficient proposed larger allocation**. The audit component sum
contains 619.832 seconds for three reservation passes, 910.444 for a complete
journal-chain read, 331.655 for syntax scanning and 6,196.191 for two episode
decode/hash passes. Native operations, semantic replay and neural auditing are
not included. Earlier development and pre-portability observations are retained
in the evidence bundle with their own source/dirty status; no earlier report is
silently relabeled as final-source evidence.

The PR requested 1,200 seconds for reservation verification and 900 each for
journal, episode and neural verification, or greater, while retaining the
original whole-process ceiling. That is at least 3,900 seconds of audit. With
the other 600 tail seconds unchanged, only 2,700 active seconds remain within
7,200. The final component assessment does not support that schedule either.
Choosing a larger number cannot establish the missing native, semantic and
memory bounds. No guessed larger ceiling or executable reallocation was made.

The exact storage envelope is 15,821,682,822 artifact bytes and
16,895,424,646 required free bytes including the one-GiB reserve, with 19,469
worker files, 17 lifecycle slots and three original outer-evidence files.
Seventeen lifecycle slots include the complete sixteen-file inventory and at
most one mutually exclusive `terminal-output-error.json` or `late-exit.json`
receipt. Interrupted JSON remains at its already-counted final path. Observed available disk exceeded this bound.
The helper's 43,520,000-byte peak RSS is not full-worker/native peak memory;
shared cgroup snapshots are not attributable to this helper alone.

Unresolved work includes native startup/transport/cancellation and memory;
episode dataclass/retirement/metric semantics; fitting/inference/authentication
and neural audit; retained PPO and auditor copies plus allocator overhead;
full publication writes/fsync, raw identity hashes and small-file overhead;
cache/shape effects; and final reaping, stream sync, receipts and outer exit.
The original [CPU report](jolt-learn-003-cpu-conformance-20261022.json) is retained
as historical component evidence, not rebound to prove this entire pipeline.

The helper's 180-second guard covers sampling from script entry. Final report
publication and observed exit lie outside that guard; a separate direct-child
210-second timeout and original observed-exit receipt delimit this diagnostic.
Neither timer changes the experiment's clock or allocation.

## Refusal and remaining gate

The [admission-refusal record](jolt-learn-003-completion-20261022/admission-refusal.json)
publishes the exact unresolved gates. No passing declaration was frozen, no
experimental output identity was opened, no fresh physical teacher data or
trained policy was collected, and no open/sealed comparison was executed.
Hosted pinned-host CI remains ordinary software conformance and does not spend
or qualify this learning allocation. Synthetic tests and diagnostics are
separate from experiment costs.

Any physical continuation first requires a complete source/runtime-bound
time/storage/memory justification, matching executable phase enforcement,
fresh prospective reviews and a published passing declaration under the actual
owner authority. Original ceilings and all qualification thresholds remain
binding until an explicit authorized revision. Software acceptance does not
release #62, owner-feel or morphology gates.

© 2026 techrote. All rights reserved.
