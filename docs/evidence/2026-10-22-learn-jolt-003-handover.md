---
title: LEARN-JOLT-003 implementation handover and refused admission
kind: evidence
summary: Durable source checkpoint, synthetic feasibility blocker, review gaps, unspent allocation and next steps; not an experiment result.
owns: [evidence.learn_jolt_003_handover]
verified: 2026-10-22
---

# LEARN-JOLT-003 handover — 22 October 2026

## Handover status and provenance

**Implementation checkpoint; prospective admission refused. No physical experiment has been run.**
The owner-issued commission is recorded on [#62](https://github.com/techrote/neural-cyber-automata/issues/62#issuecomment-6347420584) and adopts the already reviewed [task prompt](../design/LEARN_JOLT_003_PROMPT.md). The fresh allocation identity is **learn-jolt-003-20261022-a1** and remains **unopened and unspent**.

The implementation branch is **impl/learn-jolt-003**, with executable checkpoint **21cc8f0d0874e9b00512294f967c21f7f5b74a06** built on main **f3412847cf1137529cf41d8e39d1c7acda74a5f8**. That main includes PR #70's reliability fix, PR #71's allocation plan and PR #72's independent IK work. The handover documentation commit is not a substitute for exact final executable-source admission. Do not treat this checkpoint as reviewed, passing, merged or admitted without new evidence.

Original qualified teacher: **authored-jolt-recovery@4**, physical source **46a22f98dce78be3c4ba534d955e58d5691c7d96**. Prepared but *not opened* runtime was Linux CPU, Torch **2.8.0+cpu**, NumPy **2.3.5**, pinned host revision **f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497**. The failed, consumed LEARN-JOLT-002 allocation is historical and cannot be resumed or reused.

## Implemented source checkpoint

The [v3 implementation contract](../design/LEARN_JOLT_003.md) and [runbook](../LEARN_JOLT_003_RUNBOOK.md) describe changes present in the above commit:

- Config **configs/learning/jolt-learn-003-v1.json**, separate execution-v3 identity, all **39 exact counter ceilings**, original workload/seeds/hyperparameters, and **7,200-second** shared-original-clock phases. Legacy execution v1/v2 and policy/action/feature v2 semantics remain distinct.
- Eligible **emergency_recovery_deadline** becomes a complete unsuccessful episode only after independently replayable 48-consumed-tick history, final observation and native cancellation evidence. Other integrity/authority/journal failures remain allocation-fatal. Ordinary attempts versus administrative-cutoff peeks and bootstrap accounting are explicit.
- Unchanged BC/PPO reward, masks, GAE cuts and no new terminal penalty. Fresh model initialization, **6,144** precomputed seeded PPO case entries, fixed comparison order for **32 open / 224 sealed rows** and explicit safe-failure denominators.
- Bounded streaming reservations, episode/sample artifacts, v3 journal and auditor, exact-clock executor, and an external observer designed to retain original stdout, stderr and actual CLI exit.
- Expanded opened-case inventory and [source/evidence-based proof](jolt-learn-003-unopened-sealed-proof-20261022.json) that the previously declared sixteen sealed candidates were not reached in the failed LEARN-JOLT-002 allocation. This does **not** repair that allocation's missing journal suffix or lifecycle receipts.

These describe source and synthetic behavior only, not actual physical outcomes, valid learned checkpoints or qualified policies.

## Refused admission: measured synthetic evidence

The [bounded storage conformance](jolt-learn-003-storage-conformance-20261022.json) reports a source-derived maximum of **1,901,556** reservation records, with sampled canonical row maximum **193 bytes**. Its predeclared factor-four projection gives **280.354 seconds per reservation-stream read** and **three independent audit reads**, for **841.063 seconds** **before** journal replay, episode reading, neural replay, hashing, closure and other audit work. The unchanged complete-audit interval is **600 seconds**. These projections are conservative capacity estimates, **not** proven hard execution times; they are nevertheless insufficient to admit the declared time envelope. Additional full-worker memory, disk and storage-throughput feasibility are unresolved. The storage report records **complete_pipeline_feasibility_established=false**, **prospective_experiment_admitted=false** and **experiment_allocation_consumed=false**.

The independent [CPU synthetic conformance](jolt-learn-003-cpu-conformance-20261022.json) covers newly initialized GRU32 models and synthetic tensors, not host/native work. It cannot override the storage/readback blocker or prove full process feasibility. No undeclared host pilot, physical ticks, teacher capture, trained checkpoint, model selection, open evaluation or sealed evaluation was performed for LEARN-JOLT-003.

The earlier task evidence reported a local full offline suite of **1,142 tests (59 environment-dependent skips, zero failures/errors)** and a later **89-test focused v3 regression pass**. These were reported before this handover, **not reproduced by the handover itself**; their original command transcripts and final-head hosted checks must be retained/reverified before any source acceptance. No passing final-source admission reviews or hosted CI receipts are asserted here.

## Review finding not yet closed

An independent integration review identified a gap in the final artifact inventory: validation may fail to reject an **unexpected empty directory** or a **special/non-regular file**. This finding must be reproduced, corrected and covered by negative controls; independently read back the resulting clean tree. Do not claim this is fixed by the current handover or treat a file-count-only inventory as complete.

Earlier integration reviews also identified startup-clock and authored-metric-size concerns; the source checkpoint includes corrections and targeted synthetic tests, but the final executable source still needs two *fresh*, independent, source/config/runtime-bound reviews: (1) terminal, learning and qualification semantics; (2) counters, bytes, deadlines, process closure and whole-envelope feasibility. Previous planning reviews do not admit the run.

## Exact resumption instructions

1. Continue **from the existing branch and checkpoint**, not a second implementation or a newly commissioned physical run. Reconcile any newer main changes without touching unrelated IK work or historical evidence. Preserve the measured refusal, original test results and all failure artifacts.
2. Close the **extra empty directory / special-file inventory** finding and rerun targeted negative and legacy tests. Publish original test outputs and hashes.
3. Resolve the **600-second complete audit feasibility** without changing the declared allocation, clocks, case stream, learning objective or semantic gates. Recalculate the *coupled* serialization, journal, reservation, neural readback, peak memory, disk and complete process envelope on an exact clean executable revision using bounded synthetic conformance. If still infeasible, keep admission refused and publish the failure; do not simulate or assume a pass.
4. Obtain both independent reviews of that final source, publish source/runtime/configuration bindings, pass strict documentation checks and all required final-head CI. Only then may prospective admission be reconsidered under the same commission and unspent allocation. Any material scope or budget change requires a **separately reviewed owner decision**.
5. Only if all prospective gates **actually pass** should the one original-clock CPU invocation be opened. Preserve success, negative, capped or incomplete outcomes immutably, audit the real stdout/exit and all original receipts, and never retry/resume/reseed/retune/extend. Keep **#62 open** unless the unchanged complete learned-actuation acceptance genuinely passes; owner-feel acceptance and morphology **#16** remain separately gated.

**Do not launch the physical experiment based on this checkpoint.** This handover records implementation and failed admission, not a qualified learned result.

© 2026 techrote. All rights reserved.
