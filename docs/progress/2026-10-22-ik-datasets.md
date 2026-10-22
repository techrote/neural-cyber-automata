---
title: IK dataset implementation checkpoints — 22 October 2026
kind: evidence
summary: Reconciliation, independent prospective admission, complete bounded CPU corpus and final integration gates.
owns: [evidence.ik-dataset-progress-20261022]
verified: 2026-10-22
---

# IK-CUDA-004 checkpoints

## Implementation and independent source review

The owner commissioned [#53](https://github.com/techrote/neural-cyber-automata/issues/53).
The initial main audit used `f23401eb793f9708040c4047a8146ab01ec3befc` and then
reconciled the separate docs-only PR #71 at `ab1706d47ccb12939673258d02901b54c84c828f`
before the measured source freeze. No overlapping implementation PR was replaced.
The work adds the branch-aware data module, finite recipes, bounded tooling,
tiny conformance fixtures and owning documentation in [PR #72](https://github.com/techrote/neural-cyber-automata/pull/72).

Independent review required and confirmed stricter status checking, exact boundary
geometry, actual malformed-input identities, complete semantic artifact readback,
early worker memory admission and complete-process timing. A source-only schedule
scan found 10,220 unique tasks, all 100 families, every accepted final population
and no violations across 3,363 relevant cross-partition point/path comparisons.
Synthetic branch/singularity fixtures stay separate from Runner evidence.

## 07:21–07:23 UTC — published prospective admission

The [complete declaration](https://github.com/techrote/neural-cyber-automata/issues/53#issuecomment-6335152671)
binds source `1e16c751939d7b276e8ced02b7fc371f34e9c0ee` and its 123-file closure.
Independent readback matched all 60,581 declaration bytes and passed seventeen
checks before any production teacher attempt. A 180-second planning ceiling was
revised before declaration to 900 seconds with a 120-second publication reserve;
10,220 tasks/attempts, 654,080 reserved iterations and 512 MiB worker memory stayed finite.

## 07:23–07:25 UTC — single complete CPU allocation

The process envelope started once at 07:23:48 UTC and completed at 07:25:10 UTC.
All 10,220 outcomes remain in 161 shards; 9,860 solved labels and 360 failed/refused
outcomes preserve their exact diagnostics. Actual work is 47,796 iterations,
82.480250064 seconds through generator exit and receipt readback, and 62.23828125
MiB worker peak RSS. There is no unknown or unfinished attempt and no retry.
The final suite meets accepted minima and remains unopened for candidate scoring.

## Publication and final integration

The [evidence record](../evidence/2026-10-22-ik-datasets.md) and
[run ledger](../RUNS.md) preserve original source, declaration, split/data hashes,
all clock boundaries and external artifact retrieval. Independent artifact review
checks the complete retained corpus without invoking the teacher again.

Initial hosted Linux CPU, pinned-host and optional CPU-learning conformance passed.
The Windows failure was one POSIX test mock referencing an absent signal constant;
the correction is test-only. Local testing also records the pre-existing sandbox
PID-namespace restriction. Applicable final-head checks, independent final review
and verified merge remain recorded in PR #72 and #53. No CUDA, learned-model,
export or physical IK qualification, Jolt change or morphology admission follows.
