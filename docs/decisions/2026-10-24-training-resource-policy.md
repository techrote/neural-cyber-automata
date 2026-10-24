---
title: D-033 generous development allocations and useful VRAM utilization
kind: decisions
summary: Owner-authorized general local development allowances, preserving explicit packet limits, sealed finals and strict qualification.
owns: [decisions.training_resources]
verified: 2026-10-24
---

# D-033 — Generous development, measured resource use

[ACCEPTED] 24 October 2026, citing the owner's instruction to “update repo rules
to be far more generous with training allocations and allowances in general”
and “encourage heavier use of VRAM during testing if beneficial”. The same
instruction requests two independent Part B reviews and stops before training.

An owner-commissioned training/development task authorizes routine local
experimentation in that scope. Do not turn historical preparation-only rules,
one-run recipes or conservative packet ceilings into universal permission loops.
A research-only task still does not commission training by implication.

## Development planning defaults

Plan enough updates and seeds to distinguish optimization length, learning rate,
capacity and data effects. When useful, allow initially up to 72 hours for a local
campaign and 24 hours for an individual run, rather than defaulting to tiny smoke
budgets. Larger finite budgets are appropriate when justified within the owner's
commission. These allowances are planning guidance, not compulsory expenditure.

Declare campaign ceilings, configurations or adaptive search rules, evaluation
cadence, finite checkpoint/storage budgets, seeds and bounded crash retries.
Prefer explicit extension room and up to three crash retries per development
configuration when useful. Routine changes within those prospective allowances
need no new permission. Keep every attempt and charge; do not overwrite evidence,
substitute a lucky seed or reclassify development results as qualification.
New work beyond a frozen allowance receives a new prospective declaration.

## Use the available hardware productively

Larger batches, cached tensors, concurrent workers and deliberate resource stress
tests are encouraged when they improve throughput, coverage or reliability.
Low allocator usage is not intrinsically desirable. Measure throughput, board
memory, host commitment and finalization/storage costs across the whole process.
Choose a useful operating point with explicit headroom and hard watchdogs. Do not
allocate unused padding merely to raise a VRAM number. Increasing batch size or
parallelism changes an experiment when it changes numerical or information
budgets; record that distinction and qualify the declared recipe separately.

## Boundaries retained

Existing declarations, ledgers and consumed allocations stay immutable. A new
runtime version may support more generous limits; do not retroactively relax a
historical validator. Explicit task limits and device prohibitions take priority.
For the current IK54 packet, TITAN-only isolation, the V100 ban, 20 GiB board cap,
fixed recipe and unchanged acceptance remain in force. No larger allocator is
required for this small model without demonstrated benefit.

Strict qualification keeps its frozen recipe, one-attempt policy, independent
admissions and separate owner execution instruction. Finals remain excluded
until a separate owner-approved declaration. This policy commissions no current
Part B training, final access, paid service, new compute acquisition or host edit.

The independently identified IK54 terminal-size/finalization defect is addressed
through an unfinished separate versioned ledger and prospective v4 packet. No v4
recipe or packet has been frozen or admitted. Its proposed eight-hour wall
ceiling and 3,600-second paid finalization reserve must be frozen before execution;
they do not change the winning recipe, consumed evidence or acceptance gates.
Resource proof and fresh exact-packet independent admissions precede any new
owner execution instruction. See [the review/repair record](../design/IK_QUALIFICATION_REVIEW.md).

© 2026 techrote. All rights reserved.
