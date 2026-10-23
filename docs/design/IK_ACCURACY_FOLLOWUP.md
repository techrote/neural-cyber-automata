---
title: IK accuracy diagnosis and bounded representation follow-up
kind: research
summary: Saved-output diagnosis and owner-commissioned bounded polar follow-up with separate admission and unchanged criteria.
owns: [research.ik_accuracy_followup]
verified: 2026-10-23
---

# IK accuracy diagnosis and bounded follow-up

[MEASURED NO-GO] The D-027 polar hypothesis was tested in the independently
admitted [bounded follow-up](../evidence/2026-10-23-ik-polar-followup.md).
Every tuning-selected candidate failed the unchanged quality gate; fresh final
remains sealed and unmeasured. The earlier
[completed result](../evidence/2026-10-23-ik-frozen-evaluation-recovery.md)
remains accuracy/utility no-go and contribution pass. Its allocation is completed.
This analysis reads authenticated original shards, saved outputs, 120 tuning
reports and 5,160 update receipts. It imports no Torch, runs no predictors or
optimizers, generates no data and changes no original evidence.

## What the retained evidence establishes

All three selected candidates fail the same 0.06-cell raw p95 / 0.18-cell worst
and 95% feasible-success gates. The independent analytic and numerical controls
succeed on all 8,400 nominal queries. Terminal audits reproduced all saved scores
exactly against original inputs and verified unchanged checkpoint tensors.
A scorer/FK disagreement, missing data, nonfinite model, silently reset optimizer
or mislabeled successful reference is not supported as the explanation.

| Seed | Tuning-selected epoch | Tuning p95, cells | Final point p95 | Final trajectory p95 | Final withheld success |
|---|---:|---:|---:|---:|---:|
| 41011 | 18 | 0.192758 | 0.235804 | 0.147157 | 0.357% |
| 42013 | 40 | 0.160719 | 0.187662 | 0.111802 | 32.143% |
| 43019 | 15 | 0.191002 | 0.227034 | 0.157935 | 0.357% |

These are not failures restricted to a few withheld or boundary cases. Both
point and trajectory populations fail, and tuning already showed the problem.
The per-seed nominal final p95 values are 0.216149 / 0.176660 / 0.209830 cells.

The ankle mapping is especially poorly resolved:

| Seed | Hip absolute error p95, rad | Ankle absolute error p95, rad | Predicted ankle vs target ankle slope |
|---|---:|---:|---:|
| 41011 | 0.038560 | 0.186074 | 0.133 |
| 42013 | 0.030000 | 0.108830 | 0.654 |
| 43019 | 0.046469 | 0.182355 | 0.127 |

Slopes are descriptive covariance/variance over all 2,800 nominal saved outputs,
not causal estimates or a fitted replacement policy. The single admitted Runner
branch makes target configuration an unambiguous joint diagnostic; no second
admitted branch needs to be averaged. Radial endpoint p95 components are
0.152632 / 0.088175 / 0.150525 cells; tangential components are
0.178096 / 0.163951 / 0.173825 cells. Radial error alone does not explain the
whole failure. Both directions need improvement.

The [43-batch sampler](../../src/nca/train/ik_corpus.py) stage order repeats every epoch: point tasks, boundary stress,
whole trajectories, then other stress. Each epoch ends away from the main point
population. Saved average joint-loss terms fall from roughly 0.10-0.12 in epoch
one to 0.019-0.030 in epoch forty, while tuning p95 is nonmonotonic. Seed 41011's
p95 worsens from 0.192758 at epoch 18 to 0.293401 at epoch 40. Seed 43019's best
is epoch 15; seed 42013 improves through epoch 40. These are unweighted means
of batch receipt losses, not a measured complete training-set error distribution.
They suggest optimization/generalization and possible stage-recency effects;
they do not prove catastrophic forgetting. No within-epoch tuning checkpoints
exist to establish that mechanism.

## Smallest justified change to test

Propose a versioned **target-feature transform only** in the
[model input path](../../src/nca/ik/models.py): replace the network's two
Cartesian target channels with hip-centred squared radius and bearing. Retain
14 inputs, two 64-unit tanh hidden layers, two bounded outputs and 5,250 learned
parameters. Current/posture channels, masks, training rows, loss weights, Adam,
learning rate, gradient clipping, sampler, epochs and acceptance stay fixed.

For original root-local target `(x,y)` and fixed hip anchor `(3.6,2.8)`, compute:

```text
u = x - 3.6; v = y - 2.8
rho2 = u*u + v*v
theta = atan2(v,u)
```

This exposes the two geometrically relevant target directions directly. In the
admitted chain, hip rotation changes bearing while ankle motion also changes
hip-to-sole radius; the existing acceptance derivation establishes a nonsingular,
single-branch inner domain. The transform does not solve IK, use teacher angles,
return a reference, introduce numerical correction or grant evaluator inputs.
The bearing branch cut is outside nominal reachable targets; invalid/unsupported
requests retain the original guards and sanitized-mask behavior.

Keep raw Cartesian tensors in batch preparation and in the
[differentiable FK endpoint loss](../../src/nca/train/ik.py). Apply the new representation only inside the model's numeric
input transform. Otherwise reusing `features[:,2:4]` for the endpoint loss would
silently compare points against polar values, which is forbidden.

Fit a separate immutable v2 normalizer from exactly the same 5,140 retained
training-only inputs, binding transform version, feature names and original
input hashes. Preserve the original corpus and v1 normalizer byte-for-byte.
Publish new derived mean/scale hashes; fit neither from tuning nor final.

This is a testable hypothesis, not a proven sufficient repair or an observed
v2 result. Existing evidence cannot identify a unique cause. It favors testing
a better-conditioned representation before adding capacity, training longer,
changing losses or enabling a solver. Stage mixing is a plausible second
hypothesis but must not be changed in this experiment, so attribution stays
possible. Likewise do not simultaneously remove current/posture inputs.

## Fresh evaluation and finite follow-up proposal

This is a **new prospectively commissioned experiment**, not a continuation or
retry of the completed allocation. No generation, training or final scoring is
authorized by this document. Freeze implementation, generation specification,
budgets and exact source; pass CI and obtain two non-author admissions first.

Use original training/tuning rows and their current information status; old
final cases are opened development evidence and cannot qualify another model.
Do not train on old final cases or use them to select the new checkpoint.
Compare the one new representation with the archived, already-selected v1
candidate for each corresponding seed. No v1 control needs retraining.

The first metadata-only isolation proposal failed: narrowing the ordinary lane
alone still produced 267 spatial guard collisions, including 108 complete-path
collisions, and reused 15 old-final families. Retain that failed preflight; it
is not an unmeasured or admissible recipe.

The revised versioned `FreshFinalConfig` fixes seed 53023, ordinary hip lane
`(0.74,0.78)`, a separate withheld strip `(0.02,0.06)` and a new family namespace.
Independent replay of the production generator, without labels, teacher calls,
Torch or candidate outputs, found zero spatial collisions, exact-input duplicates
or shared family/trajectory IDs against all 10,220 original records. Recipe
semantic SHA-256 is `b7c59d037bffb00ecaba6537f865dad7303949cdaf5c5dcf4e89b7126c93b5b8`;
ordered task IDs SHA-256 is
`aab6fb764202204f397b5a73a0b0f6aa25e7ec5ca4fc841e4dba5891f0984b72`.
The original within-bin training/tuning/opened-final lanes remain unchanged.
Geometry feasibility does not seal labels, admit optimization or establish a
new held-out result. The old final geometry serves only as an exclusion set;
no old final labels or model scores determine new cases.

Preserve all 80 target bins, the original eight withheld bins and their
separate gates, at least 200 withheld queries, 2,000 point queries (25/bin),
40 whole trajectories of 20 decisions, and seven stress classes of 20 cases:
2,940 fresh final tasks total, including 2,800 nominal tasks. Family membership
and every transformed/derived variant stay together. Fresh cases in known
regions are not a claim of new morphology or an unseen task domain.

Have a separate reviewer/custodian seal the exact new archive, registry,
ordered task IDs, manifests and hashes before training; only metadata and
isolation receipts are available to model development. Retained final labels
and scoring stay sealed until all three tuning-selected candidates freeze.
No candidate-conditioned filtering, regeneration or missing-row deletion.
The original archive, registry and opened receipts are never overwritten.

| Quantity | Proposed hard ceiling |
|---|---|
| New training configuration | Exactly one: v2 target representation |
| Training seeds | Original 41011 / 42013 / 43019; no best-seed substitution |
| Epochs / batch capacity | 40 / 128; same complete-group stage schedule |
| Presentations / optimizer attempts per seed | 210,400 / 1,720 |
| Tuning calls per seed | 80,800; fixed original tuning only |
| Model attempts / checkpoints per seed | At most 4 / 45, cumulative failures included |
| Final-only dataset generation | 2,940 tasks/teacher attempts; at most 188,160 teacher iterations, 900 CPU seconds, 512 MiB; separate immutable ledger |
| Final and timing comparators | Original five plus archived trained v1 candidate |
| Final calls per seed | 6 x 2,940 = 17,640 |
| CPU timing calls per seed | 6 x (100 warmup + 10,000 measured) = 60,600 |
| Combined evaluation calls per seed | 78,240; every refusal/failed attempt charged |
| GPU allocator / elapsed per seed | 20 GiB / 7,200 seconds including startup, checkpoints and evaluation |

TITAN RTX only, UUID-isolated to `cuda:0`; same Windows/Python/Torch/CUDA/driver
pins unless separately qualified and reviewed. The feature transform's full
inference cost belongs in batch-one one-thread CPU timing. Archived v1,
untrained v2, learned-disabled, analytic and numerical controls use the same
fresh queries and information/capability budgets. Report all six, never select
the best of them as the new candidate after opening final.

Use the unchanged tuning-only lowest complete nominal raw p95 selection,
earlier checkpoint on ties. After forty epochs, if any seed fails the existing
raw quality checks on tuning, declare development no-go and leave the fresh
final sealed. No extra epochs, checkpoint resampling, loss adjustment or seed
replacement. Passing tuning does not imply passing final.

If that gate passes and all three candidates freeze, open final once. Apply
all original accuracy, withheld, continuity, bounds, nonfinite, contribution
and practical-utility criteria, including p95 <=0.06 cell and <=0.01L,
worst <=0.18 cell and <=0.03L, >=95% feasible success per seed, zero branch
switches, raw p95 <=half matched untrained, and same-quality CPU p95 <=0.8 of
numerical reference. Direct output remains zero refinement/fallback. Improvement
versus archived v1 is diagnostic; it cannot replace any original acceptance.
A fail or inconclusive result ends this bounded follow-up without an automatic
second modification or new final set.

## Diagnostic provenance

Maintainer-held external bundle `ik-accuracy-diagnosis-20261023` retains the
standard-library analysis script and full grouped/epoch summaries. Original
archive bytes and selected shards were SHA-256 verified; saved comparisons
were checked against the immutable recovery index. No pending journal was read.
Diagnostic JSON SHA-256:
`f7bb2c9feccbbc21ba3e3aeb40bca1636eeb758249ade3cf781560b63c47326b`.
This is a post-hoc diagnosis of opened evidence, not a new held-out result.

The polar model/trainer, versioned geometry recipe and tuning-stop helper are
implemented software components. The v2 checkpoint component has a distinct
identity and serialized feature-version marker; cross-version restore refuses
before mutation while the original v1 checkpoint identity remains unchanged.

The separate `nca.train.ik_followup` launcher binds the polar normalizer, new
finite allocation, original opened-registry receipts and new sealed-suite receipt.
Legacy launchers refuse its scoped original-training view. Selection is rebuilt
from all forty authenticated tuning reports/checkpoint event histories in each
seed ledger, including inside the finish worker; no mutable selection handoff
can grant final access. Its sole final payload reader rechecks all three gates
and preserves original training selections separately from fresh geometry.

The CPU custodian `nca.train.ik_followup_seal` declares a separate one-use
generation allocation, binds the independent full-path isolation receipt,
retains every teacher reservation and seals all scheduled rows. A hard OS memory
limit and process timeout enforce generation ceilings. Run-wide atomic phase
ownership prevents different-seed or evaluation overlap; uncertainty preserves
the claim and blocks automatic retry. Evaluation reservations close through an
unchanged trainer/optimizer/sampler checkpoint within the same 45-slot cap.

The independently admitted [TITAN execution](../evidence/2026-10-23-ik-polar-followup.md)
completed all three forty-epoch training runs. Every tuning-selected candidate
fails the unchanged raw quality gate; fresh final remains sealed and unmeasured.
CPU preflight authenticated all original training/tuning rows and derived the
new normalizer from the same 5,140 training inputs. No automatic second change,
retuning or fresh evaluation is admitted by this negative result. See the
[follow-up operations](../IK_TRAINING_RUNBOOK.md#bounded-polar-follow-up).

© 2026 techrote. All rights reserved.
