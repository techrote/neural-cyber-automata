---
title: IK54 development sweep and strict qualification
kind: guide
summary: Owner-authorized training-only development lane, frozen family split and grid, followed by one separately gated qualification recipe.
owns: [design.ik_dev_sweep]
verified: 2026-10-23
---

# IK54 development sweep

[ACCEPTED SCOPE] D-032 records the owner's 23 October 2026 task instruction,
“#54 IK development sweep, then one strict qualification step”. Part A is
**development evidence; no acceptance claim**. It replaces the unexecuted
order-only proposal as the immediate experiment and preserves historical code
and evidence. Base is `560a04eb7757f44089dec9a803d64293a866b684`, containing
the matched infrastructure at `6a6961f1e9d86f109a2f00d4cb94aabb374f7a08`.

## Frozen development design

Hold out ten of seventy original training families by ascending SHA-256 of
canonical JSON `["ik54-dev-split/v1", family_id]`. Trajectories and stress rows
stay with their family. Freeze family membership and ordered sample hashes
before training. Verify the original 5,140-input training-only polar normalizer;
never fit a subset normalizer. Dev shares training hip lanes and measures only
within-lane accuracy. Original tuning lanes stay unscored until confirmation.

| Block | Configurations, three original seeds each |
|---|---|
| Core | tanh/interior-linear × constant/cosine LR × 40/200/1000 epochs |
| Order | interior-linear, cosine, 200 epochs, D-028 whole-batch permutation |
| Capacity | interior-linear, cosine, 200 epochs, width 128 |
| LR | interior-linear, constant 0.0003, 200 epochs |

Seeds remain 41011, 42013, 43019. Defaults retain width 64, LR 0.001, batch 128,
Adam, clipping 1.0 and `LossWeights()` (endpoint 1.0). Cosine reaches 0.00001
on the last update. Grouping, masks, polar inputs and normalizer remain v2's.
Version 3 adds an output-head checkpoint marker and refuses v2 loads in both
directions, even non-strict. Width 64 adds no parameters. The interior-linear
head knee is 0.62 hip / 0.22 ankle radians inside inward-FP32 command bounds;
its tails meet the linear interior with continuous first derivative.

Evaluate dev every epoch for 40-epoch recipes, every five epochs otherwise,
including the last. Save checkpoints only there. Per-step JSONL records loss
terms, preclip norm, epoch, stage, LR and batch identity. Dev records binary64
p50/p95/worst/success, continuity/branch counters, joint p95 and twelve-bin bias.
Point-stage RMS uses the last five point-batch endpoint terms each epoch;
for shuffled order these are in presentation order, not one terminal stage.

Rank by median across all three seeds of each seed's best complete dev p95.
Checkpoint ties choose earliest epoch. Recipe ties use maximum selected dev
worst across seeds, fewer epochs, width 64, stage order, tanh, then LR order
constant/cosine/constant-0.0003, then configuration ID. Missing seeds make a
recipe ineligible; retain every attempt. Freeze the top three before one
original-tuning confirmation per dev-selected checkpoint. No tuning reselection.

## Execution boundaries

New versioned paths use the owner's current development authority under D-032;
historical qualification launchers/admissions are not reused. Only the existing
authenticated train/tuning reader inspects required registry metadata. All final
payloads, predictions, scores, mixed archives and pending journals are excluded.

Use only the single-UUID TITAN RTX with the existing `_titan_visibility` gate.
V100 initialization is prohibited. At most eight concurrent workers, one CPU
intra/inter-op thread each, 1 GiB allocator, 8 GiB per-process Windows commit,
20 GiB aggregate board watchdog and host headroom checks. Twelve hours overall;
at most two crash retries per run, all retained. Short runs first; tanh-1000
has lowest priority and is dropped first if the remaining allocation cannot fit.
Keep all outputs outside the checkout, with frozen manifests and hashes.

## Baseline replication and results

The analyzer independently implements the review's receipt-RMS and binned-bias
equations; no third-party source is copied. Allowlist-authenticated readback
reproduces polar p95 0.322219084863 / 0.304910858351 / 0.282027377188 and
point-stage RMS averaged over epochs 36–40 of 0.140726283241 / 0.142421846632 /
0.145176795841 cells. A small original-receipt fixture tests the calculation.
Historical design pages append review §5 corrections without deleting old text.

[UNMEASURED] Part A has not started at this preparation revision. The final
report will retain every seed, learning curve, resource measurement and failure,
and test hip max absolute bin mean at most 0.015 rad, eta-squared at most 0.4,
and point-stage RMS below 0.10 cells. These are mechanism diagnostics.

## Strict qualification preparation

Freeze the top-ranked recipe and fixed epoch count for Part B. Retrain on all
5,260 original training rows. Select only by lowest complete raw tuning nominal
p95, earliest ties, at a declared cadence. Compute finite optimizer,
presentation, checkpoint and tuning-call ceilings from that recipe. One attempt
per original seed, no extension or reselection. Every seed retains p95 0.06
cells AND 0.01 L, worst 0.18 cells AND 0.03 L, success 95%, and every finite,
bounds, protocol, continuity and branch gate. A tuning pass permits no final
access here. Any later final requires a separate owner-approved declaration.
Commit declaration/source, get exact-head CI green, and stop for the owner.
Two fresh independent source/declaration/corpus/scoring/resource admissions
remain required before qualification execution.

© 2026 techrote. All rights reserved.
