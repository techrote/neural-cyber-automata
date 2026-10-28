---
title: IK54 final qualification declaration and custody
kind: guide
summary: Single-use final evaluation of the three fixed v4 candidates under separate accounting and admissions.
owns: [design.ik54_final_qualification]
verified: 2026-10-28
---

# IK54 final qualification declaration

[PREPARATION] The owner commissioned [D-035](../decisions/2026-10-28-ik54-final-qualification.md)
after all three original seeds passed the unchanged [tuning gate](../evidence/2026-10-28-ik54-qualification.md).
A new versioned evaluation-only adapter authenticates the completed v4 ledger
and all 200 scheduled tuning reports per seed, verifies exact selected checkpoint
events and epoch-zero controls, then freezes all three identities. Training
ledgers and old factories remain immutable; no model is resumed or optimized.

The pre-existing polar final was sealed before the later v4 training. Preparation
uses only its published, hash-pinned provenance embedded in the prior declaration;
it does not inspect the suite directory or read final data. Actual suite state is
checked after two independent exact-source/declaration admissions and the owner's
execution record. The original suite directory remains the single-use authority.
The evaluator writes and reads back a candidate-bound opening marker before
reading or decoding its archive; an existing opening or any mismatch refuses.

The five official comparators and the archived-v1 diagnostic use the unchanged
independent binary64 scorer and all 2,940 scheduled cases. CPU batch-one timing
uses one thread, 100 warmups and 10,000 measured calls per comparator. Each seed
has a conservative write-ahead reservation of 78,240 scheduled final and timing
calls: six comparators times 2,940 cases plus six times 10,100 timing calls.
Guarded refusals and actual predictor calls are counted separately. There are
zero optimizer calls, one group attempt, no retry and no candidate reselection.

The finite group ceiling is 86,400 seconds, with a 7,200-second work-stop reserve
and a hard timer through interpreter exit. An 8 GiB per-process Windows Job,
32 GiB host-commit and 16 GiB disk headroom are checked both at startup and
after preflight. From the attempt claim onward, 4 GiB continuous floors and
the targeted 20 GiB TITAN board watchdog protect execution; the process Job
and hard timer cover preflight as well.
Failures, openings and uncompleted reservations stay durable. Full reports and
latency samples remain outside Git; terminal receipts contain hashes and counts.

The tuning gate remains p95 <= 0.06 cells and <= 0.01 L, worst <= 0.18 cells and
<= 0.03 L, >= 95% success, and all finite, bounds, protocol, continuity and branch
gates. The final accuracy, learned-contribution and practical-utility criteria
are unchanged. The archived control is diagnostic and cannot change selection.
No physical host tracking or CPU export is claimed by this evaluation.

Exact-head CI, focused CPU custody/accounting tests, resource feasibility and two
genuine independent source/declaration/candidate admissions precede any final
access. This document is a prospective contract, not a result.

© 2026 techrote. All rights reserved.
