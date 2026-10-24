---
title: IK54 development sweep results
kind: evidence
summary: Full three-seed development grid and once-only shortlisted tuning confirmation; strict full-training qualification remains unexecuted.
owns: [evidence.ik_dev_sweep]
verified: 2026-10-24
---

# IK54 development evidence; no acceptance claim

[MEASURED DEVELOPMENT] D-032 commissioned 45 training/dev runs on the isolated
TITAN RTX. All 45 completed in one attempt each, with no learner failures,
retries or budget drops. Source `dcb1cb71951c3c32ddf995728c7b2cea501aa6ff`
contains the matched infrastructure; its base was
`560a04eb7757f44089dec9a803d64293a866b684`. Cybersand execution is unmeasured.

The [design](../design/IK_DEV_SWEEP.md), [run ledger](../RUNS.md) and
[immutable scalar/hash index](ik-dev-sweep-20261024.json) retain the scope and provenance.
External bundle handle `ik54-dev-sweep-20261023` contains Markdown/JSON reports,
all metrics, attempts, checkpoints and learning curves. Large artifacts stay outside Git.

## Frozen data and selection

Ten whole training families were held out: 4,473 training / 787 dev rows,
756 nominal dev rows. The original 5,140-input normalizer stayed unchanged.
Dev shares training hip lanes and measures within-lane accuracy. Original
tuning has 2,020 rows / 1,880 nominal rows and different hip lanes.

Split file SHA-256: `a4ca50980d5b0b80bed3acbb1b18cf381098a8e0f9a0470d7c0a9442253a5ec8`.

Selection-rule file SHA-256: `281f81511f38acc95d7bca3152af89e4fafb373fdf1ade1411c2d9b85d2088fb`.

Both were frozen before training. Ranking uses median across all three seeds
of each seed's best complete dev nominal p95, with the declared worst-error,
epoch and simplicity ties. The top three were frozen before tuning confirmation.

## All seed results

| Recipe | Seed | Selected epoch | Dev p95 cells | Worst cells | Success | Late point RMS cells |
|---|---:|---:|---:|---:|---:|---:|
| tanh-constant-e40-stage-w64 | 41011 | 37 | 0.343095 | 0.413202 | 12.963% | 0.152078 |
| tanh-constant-e40-stage-w64 | 42013 | 18 | 0.323722 | 0.395169 | 10.979% | 0.150946 |
| tanh-constant-e40-stage-w64 | 43019 | 38 | 0.325800 | 0.363592 | 11.905% | 0.157431 |
| tanh-constant-e200-stage-w64 | 41011 | 195 | 0.105614 | 0.142244 | 76.587% | 0.017295 |
| tanh-constant-e200-stage-w64 | 42013 | 195 | 0.119927 | 0.183807 | 65.476% | 0.021947 |
| tanh-constant-e200-stage-w64 | 43019 | 195 | 0.091974 | 0.149064 | 76.852% | 0.020754 |
| tanh-constant-e1000-stage-w64 | 41011 | 905 | 0.005064 | 0.014862 | 100.000% | 0.002267 |
| tanh-constant-e1000-stage-w64 | 42013 | 905 | 0.005153 | 0.015032 | 100.000% | 0.003802 |
| tanh-constant-e1000-stage-w64 | 43019 | 980 | 0.006697 | 0.026669 | 100.000% | 0.010480 |
| tanh-cosine-e40-stage-w64 | 41011 | 38 | 0.292611 | 0.425450 | 22.487% | 0.160377 |
| tanh-cosine-e40-stage-w64 | 42013 | 38 | 0.290525 | 0.417039 | 21.825% | 0.156702 |
| tanh-cosine-e40-stage-w64 | 43019 | 39 | 0.297342 | 0.402165 | 22.354% | 0.156839 |
| tanh-cosine-e200-stage-w64 | 41011 | 195 | 0.030413 | 0.069108 | 99.735% | 0.016336 |
| tanh-cosine-e200-stage-w64 | 42013 | 185 | 0.044059 | 0.069945 | 98.677% | 0.022313 |
| tanh-cosine-e200-stage-w64 | 43019 | 195 | 0.043296 | 0.080447 | 98.016% | 0.023068 |
| tanh-cosine-e1000-stage-w64 | 41011 | 970 | 0.004253 | 0.015498 | 100.000% | 0.001960 |
| tanh-cosine-e1000-stage-w64 | 42013 | 1000 | 0.004522 | 0.022250 | 100.000% | 0.002099 |
| tanh-cosine-e1000-stage-w64 | 43019 | 990 | 0.004816 | 0.021028 | 100.000% | 0.002118 |
| interior_linear-constant-e40-stage-w64 | 41011 | 26 | 0.170342 | 0.205518 | 44.180% | 0.044454 |
| interior_linear-constant-e40-stage-w64 | 42013 | 24 | 0.165503 | 0.199081 | 35.450% | 0.043826 |
| interior_linear-constant-e40-stage-w64 | 43019 | 11 | 0.167841 | 0.223123 | 48.545% | 0.039708 |
| interior_linear-constant-e200-stage-w64 | 41011 | 130 | 0.095468 | 0.126361 | 67.857% | 0.019294 |
| interior_linear-constant-e200-stage-w64 | 42013 | 160 | 0.121121 | 0.172648 | 65.873% | 0.019161 |
| interior_linear-constant-e200-stage-w64 | 43019 | 180 | 0.071994 | 0.112894 | 89.683% | 0.018695 |
| interior_linear-constant-e1000-stage-w64 | 41011 | 760 | 0.028835 | 0.048318 | 100.000% | 0.008984 |
| interior_linear-constant-e1000-stage-w64 | 42013 | 905 | 0.037038 | 0.054147 | 100.000% | 0.009649 |
| interior_linear-constant-e1000-stage-w64 | 43019 | 880 | 0.034149 | 0.053353 | 100.000% | 0.011907 |
| interior_linear-cosine-e40-stage-w64 | 41011 | 38 | 0.110717 | 0.188339 | 76.323% | 0.055541 |
| interior_linear-cosine-e40-stage-w64 | 42013 | 39 | 0.087440 | 0.201226 | 81.217% | 0.046795 |
| interior_linear-cosine-e40-stage-w64 | 43019 | 40 | 0.081004 | 0.151021 | 86.508% | 0.040675 |
| interior_linear-cosine-e200-stage-w64 | 41011 | 195 | 0.015914 | 0.040444 | 100.000% | 0.006784 |
| interior_linear-cosine-e200-stage-w64 | 42013 | 200 | 0.017512 | 0.033837 | 100.000% | 0.007044 |
| interior_linear-cosine-e200-stage-w64 | 43019 | 200 | 0.018621 | 0.041037 | 100.000% | 0.006810 |
| interior_linear-cosine-e1000-stage-w64 | 41011 | 995 | 0.017760 | 0.036651 | 100.000% | 0.006427 |
| interior_linear-cosine-e1000-stage-w64 | 42013 | 995 | 0.014974 | 0.034754 | 100.000% | 0.005499 |
| interior_linear-cosine-e1000-stage-w64 | 43019 | 1000 | 0.016589 | 0.031899 | 100.000% | 0.006946 |
| interior_linear-cosine-e200-shuffled-w64 | 41011 | 195 | 0.012929 | 0.028743 | 100.000% | 0.007177 |
| interior_linear-cosine-e200-shuffled-w64 | 42013 | 200 | 0.014366 | 0.025397 | 100.000% | 0.006890 |
| interior_linear-cosine-e200-shuffled-w64 | 43019 | 190 | 0.016204 | 0.038266 | 100.000% | 0.007080 |
| interior_linear-cosine-e200-stage-w128 | 41011 | 200 | 0.020316 | 0.041072 | 100.000% | 0.007166 |
| interior_linear-cosine-e200-stage-w128 | 42013 | 200 | 0.014659 | 0.033287 | 100.000% | 0.005974 |
| interior_linear-cosine-e200-stage-w128 | 43019 | 200 | 0.019143 | 0.036319 | 100.000% | 0.007110 |
| interior_linear-constant_3e-4-e200-stage-w64 | 41011 | 135 | 0.091514 | 0.126136 | 75.132% | 0.016293 |
| interior_linear-constant_3e-4-e200-stage-w64 | 42013 | 110 | 0.075937 | 0.091047 | 79.233% | 0.018966 |
| interior_linear-constant_3e-4-e200-stage-w64 | 43019 | 140 | 0.071962 | 0.091840 | 87.831% | 0.014557 |

Late RMS is the mean of the last five epochs' last-five-point-batch receipt
RMS, not a full training-set inference snapshot. The shuffled arm uses the
last five point batches in presentation order, not a contiguous terminal stage.

## Mechanism and shortlist

The interior-linear head accelerates the early-budget optimization and the
200-epoch cosine recipe meets both declared hip-bias diagnostics in all seeds.
Longer tanh training also learns the needed curvature and escapes the short-run
RMS floor. The review's approximately linear-logit mechanism is a conditional
explanation of the short run, not an intrinsic approximation limit.

Width 128 does not improve the median over width 64 in the matched 200-epoch
interior-linear/cosine comparison. Whole-batch shuffling gives a smaller gain.
The external report quantifies all matched LR/length/order/width contrasts
and every seed's bin-mean/eta-squared diagnostic. Three seeds do not establish
a population confidence interval. Longer recipes also have more selection points.

| Rank | Frozen recipe | Median best dev p95 cells |
|---:|---|---:|
| 1 | tanh-cosine-e1000-stage-w64 | 0.004522 |
| 2 | tanh-constant-e1000-stage-w64 | 0.005153 |
| 3 | interior_linear-cosine-e200-shuffled-w64 | 0.014366 |

Each dev-selected shortlisted checkpoint was scored once on original tuning
with the unchanged independent binary64 scorer and gate. No tuning reselection.

| Rank | Seed | Tuning p95 cells | Worst cells | Success | Gate |
|---:|---:|---:|---:|---:|---|
| 1 | 41011 | 0.017905 | 0.046517 | 100.000% | pass |
| 1 | 42013 | 0.012989 | 0.045722 | 100.000% | pass |
| 1 | 43019 | 0.012517 | 0.042826 | 100.000% | pass |
| 2 | 41011 | 0.006146 | 0.041125 | 100.000% | pass |
| 2 | 42013 | 0.014804 | 0.038943 | 100.000% | pass |
| 2 | 43019 | 0.013585 | 0.029774 | 100.000% | pass |
| 3 | 41011 | 0.011730 | 0.041036 | 100.000% | pass |
| 3 | 42013 | 0.011577 | 0.043075 | 100.000% | pass |
| 3 | 43019 | 0.011874 | 0.032554 | 100.000% | pass |

## Resources, conformance and limits

Group wall time: 9197.360 seconds; independently observed
whole launcher lifetime: 9200.000 seconds.
Time-weighted board utilization: 91.63%;
peak board memory: 5.769 GiB, below 20 GiB.
Board readings include pre-existing usage. Each worker used one CPU thread
and a 1 GiB allocator cap, with eight concurrent workers at peak.

Trained CUDA parity covered all 45 selected models with full dev batches
and 64 batch-one spot rows each; maximum joint difference was 1.78813934e-07
radians against the 0.00002 limit. Every CPU fixture row also passed parity.
All 18 tested constant-LR checkpoint prefixes were bit-exact in model,
optimizer and CPU/CUDA RNG state. This is not general parallel qualification.

The development manifest's `schedule_sha256` names the base stage sampler
even for shuffled order. Frozen config/source bind the permutation; all
22,200 ordered shuffled batch receipts were separately verified. Historical
manifests were preserved, with executed-order hashes retained in the bundle.

Two CPU software-equivalence updates on one retained training batch preceded
the grid and saved no checkpoint or quality report. They are separate from
the 45 GPU runs. Final software-equivalence tests use synthetic fixtures.
A process-observer inventory lookup initially refused two matching processes;
the parent-launcher correction made no learner/GPU call.

Post-sweep v3 conformance restores v2's finite-internal-logit refusal guard
and reuses the shared prediction type. The finite numeric mapping is unchanged.
The executed source remains separately identified; its records were not rewritten.

## Qualification remains prepared only

The [strict declaration](../../configs/ik/experiments/runner-qualification-v3.json)
freezes `tanh-cosine-e1000-stage-w64` from dev rank 1 regardless of tuning scores.
It restores all original training families, seeds 41011/42013/43019, fixed
epochs, declared tuning cadence, earliest-tie tuning selection and one attempt
per seed. No extension, retry, reselection or final access is authorized.

Two fresh independent exact-source/declaration/corpus/scoring/resource
admissions and a new exact-packet owner go remain required. Complete strict
ledger finalization under its 60-second publication reserve is unmeasured
by development JSONL logging and must be covered by those resource admissions.

All thresholds and protocol/continuity/branch gates remain unchanged. No
final data was accessed and V100 CUDA was not initialized. Historical evidence
and factory/config/launcher files were preserved. Physical-actor quality, learned contribution,
CPU utility, export and deployment remain unmeasured here.

© 2026 techrote. All rights reserved.
