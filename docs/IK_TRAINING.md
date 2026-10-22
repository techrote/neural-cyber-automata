---
title: Compact IK training and GPU preparation
kind: reference
summary: Authenticated data, deterministic sampling, cumulative training state and independent comparison prepared through the CUDA execution boundary.
owns: [ik.training]
verified: 2026-10-22
---

# Compact IK training and GPU preparation

[PARTIAL] [Issue #54](https://github.com/techrote/neural-cyber-automata/issues/54)
now has the compact model, authenticated train/tuning corpus reader,
deterministic whole-trajectory sampler, cumulative attempt/checkpoint ledger,
training orchestration and independent comparison scorer. CPU conformance uses
small opened development fixtures. The accepted corpus has not been trained
or scored, its final registry remains unopened, and this model's actual CUDA
forward/backward/optimizer path remains unmeasured.

The [candidate specification](../configs/ik/experiments/runner-compact-v1.json)
and [fixed schedule](../configs/ik/experiments/runner-compact-schedule-v1.json)
are prospective software specifications. They do not by themselves admit a
measured invocation. The current owner continuation stops before GPU work.

The starting revision is `ccd4c16058f76f422e6a5159915a5e22e7bfd87e`.
Both accepted dependencies are present: the [CUDA runtime qualification](evidence/2026-10-22-ik-cuda-qualification.md)
and [complete geometric corpus](evidence/2026-10-22-ik-datasets.md).
That earlier CUDA result exercises the runtime smoke fixture; it does not
measure this model, differentiable-FK loss or optimizer path.

## Model and input boundary

The lazily registered policy is `runner-compact-ik@1`. Explicit construction
requires a selected `DeviceContext` and a typed training `Normalization`.
The fixed FP32 MLP is 14 → 64 → 64 → 2 with tanh hidden activations and bounded
absolute joint-position references. It conditions on the current configuration,
target and posture preference. It contains no recurrence, numerical correction
or full-reference fallback.

| Columns | Values | Treatment |
|---|---|---|
| 0–1 | Measured current hip and ankle | Training-fitted numeric normalization |
| 2–3 | Root-local sole target x/y | Training-fitted numeric normalization |
| 4–5 | Declared posture hip and ankle | Training-fitted numeric normalization |
| 6–13 | Presence, measurement, activity and task masks | Eight explicit unstandardized Boolean channels |

The first six means/scales come from the accepted train-only normalization.
The existing dataset's general `Normalization.apply` remains unchanged.
This model preserves mask bits directly instead of centering them as continuous
measurements. Missing numeric channels are sanitized before network arithmetic;
missing, inactive, partial-task, malformed and out-of-bounds inputs produce an
explicit invalid result. Their finite zero output is only a masked placeholder.

Raw references stay inside the inward-rounded command limits: hip ±0.72 and
ankle ±0.27 radians. These are distinct from hard measurement limits ±0.75 and
±0.30. Input bounds also use inward FP32 endpoints; a nearest-rounded decimal
endpoint outside its exact descriptor limit is refused. The focused boundary
fixture records this strict representation rule. The initial candidate uses
absolute references because a single bounded
0.25-radian delta cannot reach every full target in the existing point suite.
No silent delta clipping, projection or solver pass changes the scored output.
The returned tensor prediction is not an actionable host `IKResult`.

## Retained data and objective

`prepare_training_batch` accepts typed, validated training samples only.
It rejects tuning/final rows, duplicate task IDs, changed generator/config
identity and independently inconsistent retained labels. It preserves failed
rows and teacher-status counts. It does not regenerate numerical teacher labels,
fit normalization, load the external corpus or open its final suite.

Pose supervision uses only independently checked `pose_loss` labels.
Reachable endpoint supervision uses its separate `endpoint_loss` mask, including
eligible numerical-exhaustion cases without converting their failed candidates
into pose labels. Invalid specimens exercise refusal, not learned status
classification over fabricated measurements.

| Term | Weight | Definition |
|---|---:|---|
| Endpoint | 1 | Mean squared distance through accepted differentiable FK, divided by L² |
| Joint/reference | 0.1 | Wrap-aware coordinate error, scaled by command half-width |
| Posture | 0.00001 | Weak preference for the declared posture |
| Continuity | 0.0001 | Predicted reference change across genuine adjacent trajectory samples |
| Step | 0.01 | Hinge above 0.25 radians on those adjacent reference changes |
| Bounds | 1 | Command-bound violation penalty; structurally zero for valid model outputs |

L is 6.550781059358212 cells. Each term uses its own eligible-row denominator.
Continuity pairs share family, trajectory and augmentation, have consecutive
step indices and target change at most 0.10 cell. Random current/target pairs
are not continuity observations. Both members must be present in the prepared
batch; the whole-trajectory sampler preserves those chronological groups.

## Explicit update and software checks

The trainer registry entry is `runner-compact-ik-step@1`.
Construction creates components; `step(batch)` requests exactly one Adam update
at learning rate 0.001 and gradient-norm ceiling 1. The step primitive does not launch a training loop; explicit
orchestration owns repeated updates. It verifies FP32 device placement and finiteness for data, model
buffers, outputs, geometry, loss, gradients and optimizer state. The CUDA
optimizer uses capturable state on the selected device.

Receipts retain attempted/committed steps, sample presentations, ordered sample
identity, dataset/normalization/loss identities, teacher statuses, mask counts,
loss components, gradient norm and actual weight-change status. Failed attempted
steps poison that trainer instance. These process-local counters do not replace
the existing durable allocation ledger or authorize a resumed run.

Prepared tensor batches are trusted in-process objects. The model constructor
checks the normalization record, but does not authenticate it against an
external corpus manifest. A measured harness must bind the exact normalization
bytes and training provenance before construction and own the prepared tensors.

The focused fixtures exercise independent binary64 endpoint scoring, gradients,
actual small CPU updates, mask/refusal behavior, split guards, adjacency,
nonfinite-gradient rejection and existing checkpoint tensor restoration.
They do not score the corpus or establish useful learned IK. In the optional
CPU environment documented in [runtime operations](IK_RUNTIME.md), run:

```sh
python -m unittest discover -s tests -p "test_ik_models.py" -v
python -m unittest discover -s tests -p "test_ik_training.py" -v
python tools/check_docs.py --strict
```

The ordinary package suite skips optional tensor cases when Torch/NumPy are
absent. The optional-learning CI job installs the pinned CPU runtime and runs
them. Current PR checks own the observed validation result.

## Authenticated corpus and deterministic sampling

The reader in [ik_corpus.py](../src/nca/train/ik_corpus.py) binds the exact accepted
manifest, declaration, normalization and original registry. It verifies selected
training/tuning shard bytes, identities, scheduled rows, retained labels and the
train-only normalization provenance. It never reads final shards or the original
mixed-partition pending journal. An already opened final registry refuses further
training access. Tiny fixtures use a separate constructor and cannot acquire an
accepted-corpus identity.

The original archive and retrieval instructions remain in the
[published corpus index](evidence/ik-dataset-20261022/index.json). Obtain that exact
archive and verify its SHA-256 before extraction. Do not regenerate labels or
substitute freshly generated data. The preparation session confirmed the
archive's availability and recorded size; the unavailable workspace prevented
executing a new original-archive byte readback here. Its previous accepted
readback remains identified in the index. CPU fixture verification is not a new
readback of those 7,280 training/tuning rows.

The original corpus remains byte-identical. A 23 October 2026 Windows run-host
preflight authenticated the original archive but reported a binary64 arithmetic
difference in `training-00000.jsonl` row 4, analytic `points[0][1]`:
retained `7.1951845160020005`, Windows `7.195184516002` (one ULP).
This is not evidence of archive corruption or authorization to change labels.

The accepted-v1 reader now has a **narrowly source-bound compatibility rule**:
only after verifying the exact published manifest, original registry and each
training/tuning shard's retained byte size and SHA-256 can its independent
label replay admit a one-`math.nextafter` binary64 step in analytic FK
**point coordinates only**. Original label bytes, analytic configurations,
statuses, teacher results, branch identities, endpoint/error checks and all
supervision masks must remain **exactly equal**. A second ULP, any other
numeric difference, rehashed development fixture or altered identity still
refuses. Normal fixture replay remains strict. Neither the corpus nor the
generator, numerical teacher, normalization or final suite is changed.

The restored, original archive and its retained copy were separately read back
as 7,340,716 bytes, SHA-256
`97e3c547c7056e6ef727dc07437f469fde54b545bb091503a791aed85a4f99ce`.
An offline Linux binary64 forward computation of 7,120 solved permitted
training/tuning samples matched **all 14,240 stored analytic point coordinates
exactly**; no final payloads were inspected in this check. This does not
establish the unchanged original archive's full Windows semantic admission.

**Stop condition remains in force:** qualified Windows CPU preflight must run
again on the patched executable commit and succeed through normalization, with
two independent prospective source/declaration-bound reviews before the first
CUDA allocation. New source/CI evidence cannot be re-labelled as the
original blocked execution, and no CUDA or final-suite action is implied.

Sampling preserves each complete chronological trajectory group within a batch,
keeps the declared curriculum stage order and deterministically shuffles whole
groups by seed and epoch. Failed/refused rows remain in the schedule. A batch
with no eligible pose or endpoint labels charges its presentations, checks the
model boundary, and performs no optimizer update. Sampler state binds the entire
schedule and exact cursor; changing rows, seed, grouping or schedule on resume
is rejected.

## Frozen schedule and cumulative costs

The single direct-output configuration retains the three independent seeds
41011, 42013 and 43019, FP32 Adam at 0.001, the original masks, loss weights and
numeric acceptance. The new schedule fixes forty epochs, batch capacity 128,
an epoch checkpoint and full tuning pass after every epoch. There is no adaptive
learning-rate schedule, sweep, architecture expansion or final-informed tuning.

| Per-seed quantity | Prospective allocation |
|---|---|
| Training presentations | 40 × 5,260 = 210,400, including unlabelled rows |
| Optimizer attempts | Exact deterministic sampler batch count; no more than one update per batch |
| Tuning calls | 40 × 2,020 = 80,800 |
| Final comparison calls | 5 × 2,940 = 14,700 |
| Reserved CPU timing calls | 5 × (100 warmup + 10,000 measured) = 50,500 |
| Combined final/timing calls | 65,200 |
| Attempts | At most four, sharing the same allocation |
| Checkpoints | At most 45, including initialization, epochs and continuations |
| CUDA allocator memory | 2 GiB for allocator-managed memory; excludes driver/context allocations |
| Complete elapsed allocation | At most 7,200 seconds, including startup, checkpoints and evaluation |

The whole-process ceiling is inherited from #54. It is not the separate
LEARN-JOLT-003 timing policy. The declared counts are finite upper bounds, not
evidence that this hardware can finish them. Exact-runtime admission must assess
memory and elapsed feasibility before work. A capped or no-go result never
authorizes a smaller batch, longer allocation or replacement experiment.

[ik_run_state.py](../src/nca/train/ik_run_state.py) records immutable, chained
attempt events. Work is reserved before execution and charged even on failure.
Checkpoints retain model, optimizer, sampler, Python/NumPy/Torch RNG states,
normalization and all source/configuration/runtime/seed identities. The actual device environment
is also sealed into continuation state, and changing it is refused. Tuning
reports remain immutable external files; checkpoints carry their hashes and
compact progress state. Initialization is retained for the matched untrained
control.

Pause/resume uses the same directory and original limits. A live attempt,
unresolved reservation, checkpoint/accounting mismatch or unverified supervisor
outcome prevents replay. A previous checkpoint cannot erase newer attempted
work. Five seconds are conservatively reserved for final sealing/readback;
the parent supervisor independently checks complete process duration and kills
a worker at the remaining ceiling. Unknown interrupted work remains a visible
nonresumable failure, never a new allowance.

## Independent scoring and candidate freeze

[ik_training.py](../src/nca/evaluation/ik_training.py) scores raw references with
an independent binary64 two-vector equation, separately from tensor training FK.
The five comparators are the trained raw model, matched untrained initialization,
learned-disabled current reference, bounded numerical reference and independent
analytic reference. Direct mode admits zero numerical correction and zero full
reference fallback. Work counters and raw output remain visible.

Every expected task ID comes from the frozen partition schedule. Missing rows,
refusals, exceptions and nonfinite outputs remain in the denominator. Nominal,
withheld, trajectories and stress populations stay separate. Percentiles use
nearest rank. Completeness, bounds/continuity, accuracy, improvement over
untrained output and CPU practical utility have separate results; unavailable
timing is null/unmeasured. The fixed CPU timing helper includes feature
preparation, inference, FK and guards at batch one and one thread. No timing
benchmark runs in routine CI.

Selection uses the lowest complete tuning nominal raw p95, with the earlier
checkpoint winning ties. Failed/incomplete tuning cannot win by dropping rows.
All three selected checkpoint hashes must freeze before the existing dataset
opening protocol can export final rows. Final scoring validates the original
registry, experiment, suite, corpus and all three candidate hashes against that
opening receipt. The scorer never opens the suite on its own. The unchanged
[accepted numeric gates](IK_ACCEPTANCE.md) determine any later qualification.

## CPU preparation, then the GPU boundary

The [GPU execution runbook](IK_TRAINING_RUNBOOK.md#cpu-preparation-then-the-gpu-boundary)
contains the exact CPU-only original-corpus authentication, environment-bound
declaration, independent admission record and first supervised CUDA step.
No CUDA work is authorized by documentation alone, and this software
checkpoint did not execute the accepted corpus.

## CPU verification

The [GPU execution runbook](IK_TRAINING_RUNBOOK.md#cpu-verification)
lists the focused CPU conformance commands and their evidence limits. Those
checks are separate from original-archive authentication and CUDA qualification.

© 2026 techrote. All rights reserved.
