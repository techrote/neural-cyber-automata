---
title: Compact IK training foundation
kind: reference
summary: The first model, masked loss and explicit optimizer step for IK-CUDA-005, with measured execution still outstanding.
owns: [ik.training]
verified: 2026-10-22
---

# Compact IK training foundation

[PARTIAL] [Issue #54](https://github.com/techrote/neural-cyber-automata/issues/54)
has a compact model and a single-update training primitive. This is a software
foundation for the three-seed comparison. It has no measured-run launcher,
trained candidate, learned-quality verdict, export qualification or host
authority. The [candidate specification](../configs/ik/experiments/runner-compact-v1.json)
is explicitly not an execution declaration.

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
batch; the future sampler must preserve these pairs where required.

## Explicit update and software checks

The trainer registry entry is `runner-compact-ik-step@1`.
Construction creates components; `step(batch)` requests exactly one Adam update
at learning rate 0.001 and gradient-norm ceiling 1. There is no loop or automatic
experiment. It verifies FP32 device placement and finiteness for data, model
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

## Continuation to a measured comparison

The remaining work in #54 is:

1. Read back the exact external corpus from its [published index](evidence/ik-dataset-20261022/index.json), retaining the original registry and teacher outcomes.
2. Bind the exact training normalization and freeze curriculum, pair-preserving sampling, schedule, checkpoint cadence and tuning-only selection.
3. Implement the cumulative attempt ledger, complete optimizer/sampler/RNG checkpoints and same-allocation continuation, including failed and capped attempts.
4. Implement independent raw scoring and the matched numerical, analytic, untrained and learned-disabled controls, plus CPU latency and CUDA throughput/memory recording.
5. Publish and independently review exact executable source, environment, all three seeds and finite presentations, steps, tuning/evaluation calls, memory and whole-process wall allocations before execution.
6. Freeze all three selected candidates before candidate-bound final opening; retain every scheduled denominator and issue separate accuracy, learned-contribution and practical-utility verdicts.

Seeds remain 41011, 42013 and 43019. The candidate selects direct output and the
CPU p95 utility comparison against a matched numerical pipeline at the same
acceptance quality. The [accepted numeric gates](IK_ACCEPTANCE.md) remain
unchanged. Final scoring remains sealed; actual execution of this new CUDA path,
trained weights, useful quality and CPU export remain unmeasured.

© 2026 techrote. All rights reserved.
