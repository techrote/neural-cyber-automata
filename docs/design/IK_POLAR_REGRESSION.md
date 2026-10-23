---
title: Polar IK regression diagnosis and bounded batch-order proposal
kind: research
summary: Saved tuning evidence localizes the regression to hip and tangential fit; proposes one batch-order experiment without execution authority.
owns: [research.ik_polar_regression]
verified: 2026-10-23
---

# Polar IK regression and proposed next experiment

[DIAGNOSED, NOT A UNIQUE CAUSAL PROOF] The [polar follow-up](../evidence/2026-10-23-ik-polar-followup.md)
improved ankle/radial prediction but worsened hip/tangential prediction. Its
unchanged tuning gate correctly stopped it. The owner now authorizes saved-data
investigation and a bounded proposal only: no training, criterion changes or
sealed-final opening. Existing allocations remain stopped and cannot be reset.

## What changed in the actual errors

These are matched original tuning populations: 1,880 nominal rows in each
selected report, selected independently by the same forty-epoch p95 rule.
The diagnostic authenticated original train/tuning shards, all 240 saved reports
and 10,320 optimizer receipts across v1 and v2. Each seed used exactly the same
ordered sample IDs and complete batches in both runs. Epochs differ because
selection is unchanged; this comparison is not a paired same-epoch intervention.

| Seed | Endpoint p95, v1 to polar | Hip angle error p95, rad | Ankle angle error p95, rad |
|---|---|---|---|
| 41011 | 0.192758 to 0.322219 (+67.2%) | 0.035069 to 0.050555 | 0.180105 to 0.015326 |
| 42013 | 0.160719 to 0.304911 (+89.7%) | 0.027212 to 0.048266 | 0.097961 to 0.017289 |
| 43019 | 0.191002 to 0.282027 (+47.7%) | 0.041292 to 0.044159 | 0.176249 to 0.017646 |

Polar radial endpoint p95 falls to 0.01364 / 0.01396 / 0.01376 cells; tangential
p95 is 0.32222 / 0.30483 / 0.28200 cells, almost the entire residual. The saved
ankle-versus-target slope improves from 0.172 / 0.717 / 0.170 to approximately
1.01 / 1.01 / 1.02. The radius feature addressed the original ankle collapse.
Both point and trajectory endpoint populations regress; this is not a few
outliers confined to one population.

An independent geometric diagnostic substitutes the known tuning target joint
for one saved output coordinate, without executing any model. It is an error
attribution calculation, never a corrected policy, acceptance score or candidate:

| Seed | Polar saved p95 | Substitute exact hip only | Substitute exact ankle only |
|---|---:|---:|---:|
| 41011 | 0.322219 | 0.024533 | 0.317751 |
| 42013 | 0.304911 | 0.027675 | 0.300223 |
| 43019 | 0.282027 | 0.028247 | 0.278263 |

Hip error explains the present failure. In v1, incorrect hip and ankle errors
partly cancelled: for seed 41011, replacing only its hip by the known target
raises the diagnostic p95 from 0.192758 to 0.287919 cells. Better ankle accuracy
therefore need not improve the endpoint until hip fit also improves. No additive
decomposition of p95 values is claimed.

## Supported explanation and remaining uncertainty

The polar hip residual is systematic: negative and positive mid-angle errors,
with compression near the ends. A descriptive cubic regression of retained hip
error on target hip, its cube, target ankle and current joints explains
84% / 93% / 87% of its variance. This fits a persistent angular approximation
bias. It is consistent with the bounded nonlinear head and the optimization
schedule interacting with the new representation; it does not prove that
output tanh, width, learning rate or sampling is the unique cause. No fitted
diagnostic coefficients are used as a model correction or selection metric.

The fixed schedule is a concrete mechanism to test. Every epoch has 34 point
batches, one boundary batch, seven trajectory batches, then one other-stress
batch. The boundary batch has only 20 supervised rows; the final stress batch
has 20 endpoint-only rows and no supervised pose. Losses average over eligible
rows. Each therefore receives a complete optimizer step despite the smaller
label count. They remain included intentionally, and no label is discarded.
In the final ten polar epochs, mean gradient norms are about 0.035 for point
batches, 0.091-0.093 for boundary, 0.050-0.056 for trajectory and 0.043-0.048 for
final stress. The last nine updates always occur after every point batch.

Thus stage recency and unequal batch influence are plausible contributors to
the persistent hip curve. The shared sampler does not by itself explain why
v2 regressed: changing features changes gradients and their interaction with
this schedule. There are no saved within-epoch models/probe predictions to
establish the effect of the tail updates. Replaying training to obtain them
is outside this investigation. Smaller average joint loss is not a guarantee
of better endpoint p95: v2 epoch-40 point joint losses are approximately
0.00092-0.00107, versus v1's 0.01960-0.02984, while selected endpoint p95 worsens.

No gradient clipping occurred in either run. Saved FK/finite/placement audits
passed; raw Cartesian targets remain in the loss, and the fitted normalizer
uses only the original 5,140 inputs. Nominal bearings are over 1.12 radians
from the atan2 branch cut. Standard-library FP32 emulation gives maximum
normalized radius/bearing discrepancies about 1.1e-5 / 4.2e-7. These are small
magnitude diagnostics, not GPU parity or a network sensitivity bound. There is
no evidence supporting broader tolerances, a branch-cut repair or more data.

## Smallest next change: reorder the existing complete batches

[TENTATIVE] Proposal only. Keep the polar model, input transform, immutable normalizer,
weights/loss functions, Adam, learning rate, clipping, data and original seeds.
Change only the outer order of the existing 43 complete batches in each epoch.
First construct each batch exactly as before, including chronological whole
trajectories. Then sort them by SHA-256 of the canonical JSON list
`["ik-complete-batch-shuffle/v1", seed, epoch_zero_based, original_batch_index, ordered_sample_ids]`.
Use original batch index to break a hash tie. Do not repack rows, split
trajectories, oversample stress, choose an order from outcomes or add batches.

This retains every batch, per-batch loss and optimizer-work ceiling while
removing the guaranteed late-stage tail. A declared deterministic shuffle may
occasionally put stress last; the proposal does not promise a hand-picked
point-last schedule. Its schedule hash and exact per-epoch permutations must
freeze before training. The experiment tests order/recency; it leaves the
implicit batch-weighting objective unchanged. A negative result does not
authorize another automatic change.

Add diagnostic observations, with no optimizer work: freeze 256 training-only
point probes before execution, balanced over the 72 training bins (three per
bin plus one in 40 bins chosen by a fixed hash). Probe before and after the
boundary batch and the endpoint-only stress batch each epoch. Use the same
frozen probe in every seed; report raw endpoint errors, hip/ankle errors and
gradient/update diagnostics. Four probe passes per epoch give 40,960 sample
forward calls per seed. Charge them explicitly; they are neither new tuning
queries nor eligible checkpoints. Run probes without gradients and preserve
model/optimizer buffers and RNG state, with readback checks. This measures immediate batch effects on
the admitted training probe, not held-out quality or causal forgetting proof.

## Prospective finite envelope and stop rule

No allocation, declaration, generator, launcher or model is created by this
proposal. A future owner commission, versioned source, focused schedule/accounting
tests, exact-head CI and two independent prospective admissions are required.

| Quantity | Proposed ceiling per seed unless stated |
|---|---|
| Configurations / seeds | One batch-order change; 41011 / 42013 / 43019 |
| Epochs / original batches / batch capacity | 40 / 1,720 / 128; same complete batch membership |
| Optimizer attempts / optimization presentations | 1,720 / 210,400; all failures charged |
| Tuning calls / selection | 80,800; forty end-of-epoch reports, original lowest raw p95 / earliest ties |
| Training-only diagnostic forward calls | 40,960, explicitly charged; never used for selection |
| Attempts / checkpoints | At most 4 / 45; new immutable allocation, no old reset |
| Hardware / GPU allocator / cumulative wall | Same qualified Windows TITAN RTX only / 20 GiB / 7,200 seconds |
| Optional final comparators if the tuning gate passes | Original five plus archived v1 and polar v2, seven total |
| Final calls / matched CPU timing calls | 7 x 2,940 = 20,580 / 7 x 10,100 = 70,700 |
| Total diagnostic + evaluation-call ceiling | 132,240; 40,960 diagnostic and 91,280 gated final/timing |
| New final generation now or under this proposal | Zero |

Use the still-unopened suite only under a separately reviewed future declaration
that authenticates custody/isolation and binds all three newly selected candidates
and both archived controls. Its current seal is retained. No fresh payload,
label or score is read for this diagnosis or to choose the schedule.

After forty epochs, if any seed misses any unchanged raw tuning quality gate,
stop with final sealed. No extension, new checkpoint-selection opportunity,
best-seed substitution, new final set, loss adjustment or extra configuration.
If all three pass, final opening still requires the future admitted declaration.
All original accuracy, withheld, finite/bounds/continuity/branch, contribution
and matched CPU utility criteria apply. Improvement over either old model is
diagnostic and cannot substitute for acceptance. Failure or inconclusiveness
ends the bounded experiment; no automatic second modification follows.

## Evidence and limits

Maintainer-held bundle `ik-polar-regression-diagnosis-20261023` retains reproducible
standard-library scripts, full per-epoch and per-stage summaries, covariance
diagnostics and input hashes. It reads only original training/tuning shard
payloads, saved predictions and JSON receipts. No Torch, NumPy, inference,
optimizer, teacher, final shard or mixed pending journal was used. This is
retrospective analysis, not a new trained/evaluated result. The
[public diagnostic index](../evidence/ik-polar-regression-diagnosis-20261023.json)
binds scripts and full external result hashes with authorized retrieval details.

© 2026 techrote. All rights reserved.
