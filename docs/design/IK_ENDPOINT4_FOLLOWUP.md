---
title: Endpoint-weight result and bounded parallel order experiment
kind: research
summary: Retained training and tuning diagnosis, actor-versus-method limits, and one preparation-only batch-order test.
owns: [research.ik_endpoint4_followup]
verified: 2026-10-23
---

# Endpoint-weight diagnosis and next experiment

[PREPARATION ONLY] The owner commissions investigation and a reviewed bounded
parallel experiment, with no training. Both final suites, final payloads and
saved final scores remain excluded. No old allocation, marker, charge,
checkpoint or candidate may be changed. This proposal is not an execution
admission or a launch-ready packet.

## What the completed intervention established

At source `6a6961f1e9d86f109a2f00d4cb94aabb374f7a08`, endpoint weight 4
replaced weight 1 with model, seeds, corpus, normalization and schedule fixed.
The [terminal record](https://github.com/techrote/neural-cyber-automata/issues/54#issuecomment-6369955562)
authenticates both complete trials: every paired model/optimizer/RNG/sampler/
trainer state and all tuning reports match exactly across serial and parallel.
Three-worker execution therefore did not cause this pair's quality failure.
Reported elapsed time improved from 2438.985 to 1245.062 seconds, a 1.959 ratio;
those fields precede terminal publication and cleanup. Peak board use was
2.682 / 3.105 GiB and per-worker reserved allocator peak was 66 MiB.
This is specific-host evidence, not general concurrency qualification.

Saved tuning and batch-receipt analysis gives:

| Seed | Selected epoch, both weights | p95 weight 1 / 4, cells | Hip absolute p95, rad, 1 / 4 | Ankle absolute p95, rad, 1 / 4 | Steps exceeding clipping threshold, 1 / 4 |
|---|---:|---|---|---|---|
| 41011 | 37 | 0.322219 / 0.318275 | 0.050555 / 0.047850 | 0.015326 / 0.021797 | 0 / 11 |
| 42013 | 28 | 0.304911 / 0.303365 | 0.048266 / 0.046758 | 0.017289 / 0.026656 | 0 / 11 |
| 43019 | 40 | 0.282027 / 0.273762 | 0.044159 / 0.041744 | 0.017646 / 0.025837 | 0 / 5 |

Endpoint p95 improves only approximately 1.22%, 0.51%, 2.93%, far short of
0.06 cells. Weight 4 shifts the fit: slightly better hip accuracy accompanies
worse ankle accuracy. Success remains 25.319%, 23.138%, 24.255%, versus 95%.
Replacing only the saved hip coordinate with its known tuning target reduces
diagnostic endpoint p95 to 0.034892 / 0.042669 / 0.041358 cells; replacing only
the ankle leaves 0.301094 / 0.292037 / 0.261599 cells. These substitutions are
error attribution, never corrected candidates, scores or a deployable method.
The dominant residual remains hip fit.

In the last ten epochs, point-batch mean preclip gradient norms rise from about
0.035 to 0.124-0.131; boundary norms rise from 0.091-0.093 to 0.335-0.341.
Point mean endpoint terms improve modestly while mean joint terms worsen.
These are unweighted loss components, not comparable contributions to total
loss. Weight 4 also introduces clipping (11 / 11 / 5 steps, versus zero).
Thus the intervention changed optimization, but did not remove the systematic
fit problem. Clipping is a measured consequence, not proven root cause.

The unchanged stage schedule always puts 34 point batches before one boundary,
seven trajectory and one stress batch. Small eligible populations still receive
whole optimizer steps. This supplies a concrete order/recency hypothesis;
end-of-epoch records cannot establish within-epoch forgetting. Network capacity,
bounded output parameterization, objective conflict and generalization remain
alternative explanations. No gradient-direction or training-set accuracy
measurement exists here to discriminate them conclusively.

## Actor or training method?

This #54 result is a synthetic fixed-leg inverse-kinematics mapping. It does not
run the physical actor, Rapier, contacts, locomotion or actuator dynamics. It
cannot diagnose a faulty physical actor. Existing independent FK checks and
finite optimizer/device receipts support a functioning geometric training path;
they do not prove every training-system choice correct.

Evidence currently favors investigating the model/training formulation: the
compact learned mapping retains angular error despite stronger endpoint loss.
It does not isolate model capacity from optimization, sampling, objective or
representation. There is no evidence justifying changing the actor mechanics,
loosening accuracy gates, adding final data or blaming the TITAN hardware.

## One fixed next test: whole-batch order at weight 4

Retain the completed endpoint-4 candidate as the historical control. Prepare
one new configuration with ONLY the outer order of the existing 43 complete
batches changed. Apply D-028's deterministic hash permutation:
`["ik-complete-batch-shuffle/v1", seed, epoch_zero_based, original_batch_index, ordered_sample_ids]`,
SHA-256 of canonical JSON, original index as tie-break. Preserve whole trajectory
membership and chronology, every retained label, refusal and batch mask. Do not
repack, duplicate or discard rows. Keep endpoint weight 4, all other losses,
model, Adam, learning rate, clipping and train-only normalizer unchanged.
This tests guaranteed stage recency at the current weight; it does not change
implicit weighting of small batches and does not implement D-028's weight-1
proposal or its extra training probes.

| Prospective ceiling | One new configuration, three seeds |
|---|---|
| Seeds / execution | 41011, 42013, 43019; at most three concurrent workers |
| Epochs / updates / presentations per seed | 40 / 1720 / 210400 |
| Tuning per seed | 80800 calls, 40 reports; lowest complete raw p95, earliest ties |
| Attempts / checkpoints per seed | One, no retry/resume; 41 expected, 45 ceiling |
| GPU / allocator / board | UUID-isolated TITAN RTX only; 5 GiB per worker, 20 GiB aggregate |
| Host / CPU | 8 GiB committed memory hard cap per process; intra/inter-op threads 1 |
| Group / per-seed prepaid time | Active deadline 7190 s; coordination charge 7195 s within 7200 s |
| Additional model/probe/teacher calls | Zero |
| Final / evaluation / CPU utility calls | Zero, even if tuning passes |

These are proposed new ceilings, not issued allocations. Neither consumed
matched packet may be reused. No new serial training control is proposed:
the completed same-weight serial/parallel control is already bit-identical.
Compare new schedules with the historical parallel control. Scheduling changes
are intended to change trained tensors; require identical initial tensors and
batch membership, not equality of trained tensors to the control.

Freeze permutations and their hashes before execution. Software fixtures must
prove 43 batches/epoch, unchanged batch multisets/masks, trajectory order,
1720 updates and all original sampler counts. Reject unknown work, stale
telemetry, over-budget startup/publication, child failure and any retry.
Bind a new candidate factory and sampler explicitly: the old matched launcher
hardcodes its candidate and MUST refuse this configuration. Do not pass this
proposal to that launcher or edit its consumed declarations.

The existing D-028 preparation already fixes the weight-independent permutations
for this exact sampler. Retain `proposed-schedule.json`, file SHA-256
`1cbdaec81e29b8123077afb3dcf2ecf0336841e600951f53118a31eae102ea60`,
in bundle `ik-polar-regression-diagnosis-20261023`. Per-seed canonical schedule
hashes are `df9a46d91a205b0bef1d65df0e6b70375b3db3852ed8cd1c1f1b0d45132c7abb`,
`29802ed22c091376241a08f2631c5b4edf57383f90105a8c4e80fd0843031e92`,
`bef05ba2d098b7461fd5ed5bba437ab8546fc77b361f0c7cb60174a7868a52be`
for seeds 41011 / 42013 / 43019. The future implementation must reproduce these
before admission; do not generate a replacement permutation from new outcomes.

Before any later execution: implement and review that separate versioned
launcher path, exact full-process and worker start/end timing, cleanup-inclusive
timing, time-weighted telemetry and explicit immutable resource receipts; pass
focused lifecycle/accounting/schedule tests and exact-head CI; authenticate the
unchanged corpus and normalizer; bind the existing resource evidence to the new
source or stop if insufficient; prepare fresh declarations and obtain two fresh
independent TITAN admissions plus the later owner execution instruction.
Design review here supplies neither admission nor execution authority.

Report all seeds and all forty tuning epochs, raw p95/worst/success, joint
residuals, gradient clipping by stage, tensor/checkpoint lineage, ordered batch
receipts, timing and board/allocator/host memory. Do not choose another order,
coefficient or seed from outcomes. All seeds must meet the original p95 limits
0.06 cells AND 0.01 leg lengths, worst limits 0.18 cells AND 0.03 leg lengths,
95% success, and integrity/finite/bounds/continuity/branch gates. Failure or
inconclusiveness stops the experiment; a pass also leaves BOTH finals excluded.
Improvement alone is diagnostic, not promotion. No automatic next experiment.

## Evidence custody

Maintainer-held bundle `ik-endpoint4-diagnosis-20261023` contains the standard-
library analysis script, recorded receipt/report hashes, authenticated archive/
shard pins and its result. The result
SHA-256 is `6b8546501022350d64a05e4a4a39b2cb696e0014ede9c90eca4a3a8a556e2977`.
The script reads saved training receipts and tuning reports plus authenticated
tuning shards only; no model/optimizer/teacher calls, Torch/NumPy import,
CUDA initialization or final payload access. Its event envelopes are checked;
terminal audits bind original checkpoint/report histories. Retrieve these
external bytes from the maintainer by bundle name and hash before independent
reanalysis. The [public diagnostic index](../evidence/ik-endpoint4-diagnosis-20261023.json)
binds those handles. Prior results remain immutable. The matched results index is
`a611c7db7d939d0a220dd53cbcd0901b3eb73cd434eaf4f479cf1d2aa1d613f3`.

© 2026 techrote. All rights reserved.

## Dated correction — independent review, 23 October 2026

The later independent training/tuning-only review corrects the earlier evidence
limits and interpretation; the historical text above is retained.

1. Every optimizer receipt contains pre-update per-batch `terms.endpoint`.
   Training RMS by stage and point-batch position is therefore measurable
   without replay. Polar end-of-point-stage RMS over epochs 36–40 is
   0.140726 / 0.142422 / 0.145177 cells (seeds 41011 / 42013 / 43019).
   Per-row training p95 and within-epoch snapshots remain unmeasured.
2. All endpoint4 clipped steps (11 / 11 / 5) occur in epoch 1, mostly point
   batches; seed 41011 has nine point and two trajectory clipped updates.
3. The roughly 3.6-fold gradient-norm rise reflects loss rescaling, not evidence
   of a new optimization pathology. Endpoint already dominates at weight 1;
   clipping is a startup transient and does not explain late underfitting.
4. The frozen D-028 permutation ends on a point batch in 40/40 epochs for
   41011, 29/40 for 42013 and 34/40 for 43019. The legitimate prospective
   treatment differs across seeds; report this with order comparisons.

The [D-032 development sweep](IK_DEV_SWEEP.md) is now the authorized immediate
experiment. These corrections and the review's output-head hypothesis do not
establish a trained improvement, acceptance or any physical-actor result.
