---
title: Endpoint-loss ablation and matched serial/parallel preparation
kind: research
summary: One fixed loss-balance hypothesis and two unspent training-only trial packets, with no execution authority or final access.
owns: [design.ik_endpoint_matched]
verified: 2026-10-23
---

# Endpoint-loss ablation and matched execution trials

[PREPARATION ONLY] The owner explicitly selected **prepare both; do not train
yet**. This commission prepares one candidate under serial and parallel
execution. It does not admit either trial, extend a consumed allocation or
permit either historical final suite in training, tuning or selection.
The [batch-order proposal](IK_POLAR_REGRESSION.md) remains a separate untested
alternative. These packets preserve the original stage order; they do not
combine a schedule change with a loss change.

## Evidence supporting the hypothesis

Two independent training/tuning-only audits authenticated the original shard
bytes and saved receipts. All 1,720 batch identities per seed match between
Cartesian v1 and polar v2, including row masks, retained labels and original
loss-weight identity. The selected tuning comparisons are:

| Seed | Endpoint p95 v1 / v2, cells | Hip error p95 v1 / v2, radians | Ankle error p95 v1 / v2, radians |
|---|---|---|---|
| 41011 | 0.192758 / 0.322219 | 0.035069 / 0.050555 | 0.180105 / 0.015326 |
| 42013 | 0.160719 / 0.304911 | 0.027212 / 0.048266 | 0.097961 / 0.017289 |
| 43019 | 0.191002 / 0.282027 | 0.041292 / 0.044159 | 0.176249 / 0.017646 |

These are independently selected end-of-epoch tuning candidates, not paired
same-epoch interventions. Polar v2 improves ankle fit substantially, but hip
error moves the endpoint enough to dominate the remaining error. Epoch-40 mean
joint loss improves while endpoint loss worsens. No saved gradient norm exceeds
the clipping threshold. Arithmetic checks reproduce the original 5,140-input
training-only polar normalizer and show no demonstrated anchor, branch-cut,
normalization or Cartesian-loss-coordinate defect. FP32 transform discrepancies
are small magnitude diagnostics, not a network sensitivity bound.

The next fixed hypothesis raises **only endpoint loss weight from 1.0 to 4.0**.
Four is a prospective, deliberately fixed intervention; the receipts do not
identify an optimum or prove that it will help. Do not search coefficients or
claim a causal fix before measurement. Preserve polar features, 5,250-parameter
network, normalizer, all other loss coefficients, Adam, learning rate, clipping,
seed set, whole-group stage sampler, forty epochs and batch capacity 128.
There are no extra probes, optimizer steps or tuning queries.

The candidate factory is separate from the historical v2 factory. A CPU fixture
checks bit-identical initial model tensors and unchanged optimizer settings,
with endpoint weight as the sole loss change. This is software conformance,
not evidence that a new accepted-corpus model has been trained.

## Matched work and resource proposal

| Item | Serial trial | Parallel trial |
|---|---|---|
| Candidate, runtime, train/tuning data and normalizer | Identical | Identical |
| Seeds | 41011, 42013, 43019 | Same three seeds, one process each |
| Maximum simultaneous seed workers | 1 | 3, subject to future resource admission |
| Epochs / optimizer ceiling / presentations per seed | 40 / 1,720 / 210,400 | Same |
| Tuning calls / checkpoints per seed | 80,800 / at most 45 | Same |
| Attempts / cumulative whole-process wall per seed | At most 4 / 7,200 seconds | Same |
| CPU intra/inter-op threads per worker | 1 / 1 | 1 / 1 |
| CUDA allocator ceiling per worker | 5 GiB | 5 GiB |
| Combined concurrent allocator ceiling | 5 GiB | 15 GiB |
| Combined GPU budget including monitoring margin | At most 20 GiB | At most 20 GiB |
| Evaluation / final / utility calls | 0 | 0 |

The 5 GiB nonallocator/monitoring margin leaves the aggregate proposal at
20 GiB. It does not prove CUDA contexts, libraries or unrelated GPU activity fit
that margin. The earlier approximately 66 MiB allocator peak does not qualify
three simultaneous workers. The two trial groups must not overlap. No extra
seed, repeat, retry or lower concurrency fallback is automatic; a problem stops
the group with failures, charges and unknown work retained.

Use only the privately identified TITAN RTX, isolated by its single UUID.
The public UUID fingerprint is
`e5df8b0db3a22718306c0f68e7ec85b69f1555759649550d74fc44f563c841d9`.
The expected metadata is TITAN RTX and driver 581.80. The V100 is prohibited
for every CUDA context, test, benchmark and training operation. Preserve the
qualified Windows/Python/Torch/NumPy environment in the original
[runbook](../IK_TRAINING_RUNBOOK.md). Preparation performs targeted NVIDIA
metadata inspection only and imports neither Torch nor NumPy.

## Metrics and decision rules fixed before execution

Each declaration contains null metric slots, explicitly **unmeasured**.
Future recording must include the following evidence for both approaches:

1. Whole group makespan, from supervisor entry through verified publication,
   including process startup, corpus authentication, tuning, checkpoints and
   cleanup. Record each worker's whole wall time and sum separately; overlapping
   worker times are not group elapsed time. Record worker start/finish times
   and overlap so a claimed parallel trial actually demonstrates concurrency.
2. Real CUDA event durations for forward/backward/optimizer work; per-worker
   allocated/reserved peaks; timestamped targeted-board utilization, used/free
   memory and temperature. A board monitor enforces the 20 GiB envelope in
   addition to per-worker allocator partitions, accounting for pre-existing
   board usage. Missing readings are unresolved, not zero. WDDM may not provide
   per-process board memory: do not invent it from allocator readings.
3. Per-worker CPU user/system time, process-tree peak resident memory, I/O and
   host contention; monitor overhead and failures/stalls. Configure both
   intra-op and inter-op thread counts and relevant BLAS thread environment
   before Torch import. No CPU inference utility benchmark runs concurrently.
4. Every optimizer/sample receipt, the complete forty tuning histories, exact
   configuration and ordered sampler hashes, and canonical hashes of model,
   optimizer, RNG and sampler tensors/state at matched checkpoints. File hashes
   alone do not prove tensor equality because execution metadata can differ.
5. Per-seed p95/worst endpoint error, nominal success, finite/bounds/continuity
   and branch results, selected epoch, failures, retained charges and unused
   allocation. Record initial and selected tensors as well as final epochs.

Report speedup as serial group makespan divided by parallel group makespan;
throughput as total accounted presentations divided by group makespan. Compare
per-seed CUDA/worker time and resource interference separately. Admission to
routine parallel execution requires all declared work complete, no unknown
work or resource violation, exact per-seed initial/sampler identities, and exact
matched model/optimizer/RNG/sampler tensor agreement. If concurrent execution
changes tensors or metrics, report the divergence and do not qualify consistent
parallel runs from this pair. One matched pair is limited evidence about this
specific configuration and host, not a universal speed or safety guarantee.

Tuning uses the unchanged lowest complete nominal raw p95, earliest epoch on
ties. Each of the three seeds must satisfy every existing raw quality gate:
nearest-rank p95 at most 0.06 cells **and** 0.01 leg lengths, worst error at most
0.18 cells **and** 0.03 leg lengths, and nominal success at least 95%, plus all
finite, bounds, protocol, continuity and branch requirements. The nominal
denominator remains 1,880 within 2,020 retained tuning rows. Failures/refusals
are retained. No best-seed substitution, relaxed threshold or extra epoch.

Feasible success is the number of usable, finite, in-bounds nominal outputs
within both raw p95 distance limits, divided by **every scheduled nominal
query**. A low feasible-success fraction means this same rate is below 95%;
there is no separate partial-credit grade. Missing, refused, malformed or
nonfinite predictions remain failures in that denominator. Improvement
over historical v1/v2 is diagnostic and does not establish qualification.
Missing complete rows or required outputs makes a report incomplete. A tuning
no-go terminates that trial. Even a pass cannot open either final suite under
these packets. Contribution, held-out accuracy and CPU utility remain unmeasured.

## Windows preparation and remaining launch gates

Keep accepted corpus, original registry and a **new** output directory outside
the checkout. Set `CUDA_VISIBLE_DEVICES` to the actual TITAN UUID from private
hardware records as described in the runbook; do not publish that UUID.
Use the existing pinned environment and a clean committed source revision:

```powershell
python -m nca.train.ik_matched_prepare prepare --corpus ORIGINAL_CORPUS_DIRECTORY --registry ORIGINAL_REGISTRY_DIRECTORY --output NEW_MATCHED_PACKET_DIRECTORY
python -m nca.train.ik_matched_prepare inspect --output NEW_MATCHED_PACKET_DIRECTORY
```

Preparation authenticates only original training/tuning payloads, retained
labels and training-only normalization. It writes two immutable declarations
and six new zero-attempt ledgers, hashes their source/runtime/schedules and
leaves actual admissions empty. It never opens any historical run ledger,
archive payload, final result or fresh-final directory. It refuses existing
output and corpus/registry overlap. The configuration file is included in its
clean committed source closure. It has **no execution command**.

Before any future invocation, obtain an explicit owner instruction to execute,
finish and review the source-bound group supervisor/board watchdog and its
fail-stop/accounting tests, pass exact-head CI, authenticate unchanged packets,
and obtain two fresh independent TITAN execution admissions covering aggregate
resource feasibility. This preparation's design reviews do not grant those
admissions. The current instruction expressly withholds training; it requires
no owner confirmation to complete preparation.

The [preparation evidence index](../evidence/ik-matched-preparation-20261023.json)
binds the training/tuning-only diagnostic bundles. Existing outcomes, ledgers,
failure markers, charges, candidates and both final protections stay intact.

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
