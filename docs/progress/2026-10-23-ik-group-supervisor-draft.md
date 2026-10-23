---
title: Matched IK supervisor draft and withheld admissions
kind: evidence
summary: CPU-tested supervisor work with explicit CLI refusal and unresolved admission blockers; no training or packet replacement.
owns: [evidence.ik_matched_supervisor_draft]
verified: 2026-10-23
---

# Matched supervisor draft

[INCOMPLETE, NOT ADMITTED] The owner requested a group supervisor and aggregate
memory watchdog, exact-head CI and two fresh TITAN admissions, without training.
The implementation is a disabled draft: its CLI refuses both run and worker
commands. Do not remove that refusal or treat preparation reviews as admission.

The draft implements targeted UUID-only TITAN/driver checks and board metadata
sampling; unknown, nonfinite and over-20-GiB readings stop the group. It limits
concurrent seed processes by declared mode, retains a group ownership claim on
failure, attempts cleanup of every active child, binds review/authorization
files and records worker results. Canonical child module invocation replaces
the incorrect CLI `__main__` identity. The fixed endpoint4 trainer factory is
connected behind an explicit source/config-bound hook. Both final suites remain
excluded and there is no final-opening entry point.

A separate fresh-matched-allocation ledger branch proposes a conservative
7,195-second coordination debit within the existing proposed 7,200-second bound.
It is a prepaid ceiling, not measured worker duration. It does not alter legacy
publication-reserve limits or any historical allocation/charge. This proposal
has not been exercised on an accepted corpus or admitted for execution.

Independent draft reviews identified blocking issues. Several local fixes now
have CPU negative controls, but final source-bound admissions are withheld:

- Worker preflight failures need a complete paid-accounting path, including
  failures before the attempt begins.
- The numeric-one/boolean admission regression is fixed with type-sensitive
  canonical validation; the independently observed failure is retained.
- Both-mode state transitions and interruption/cleanup need full mock lifecycle
  coverage and source-bound terminal verification.
- The declared cross-mode model/optimizer/RNG/sampler tensor comparison is not
  implemented. Partial resource counters do not complete the measurement plan.
- Aggregate resource feasibility is not established by prior low allocator use
  or metadata-only free memory. Neither reviewer has admitted three workers.

The earlier preparation packet is retained unchanged and is stale for this new
source. No replacement admission packet or affirmative review was fabricated.
No CUDA initialization, training, benchmark, accepted-corpus inference or final
payload access occurred. Original markers, charges and allocations are untouched.

The focused preparation/supervisor CPU checks pass (17 cases at draft review),
including runtime-free import, wrong target telemetry/refusals, canonical child
launch, immutable failure holds, authorization refusal and matched-only debit
controls. These fixtures establish neither successful real execution nor safe
parallel operation. Exact-head hosted checks are recorded on PR #78 separately.

© 2026 techrote. All rights reserved.
