---
title: IK CPU utility diagnosis and frozen inference development
kind: evidence
summary: Frozen batch-one inference preserves tuning outputs and reduces CPU p95 cost under the unchanged development utility comparison.
owns: [evidence.ik_cpu_development]
verified: 2026-10-28
---

# CPU utility diagnosis and inference improvement

[MEASURED DEVELOPMENT] Under [D-036](../decisions/2026-10-28-ik-cpu-development.md),
a CPU-only campaign tested the three already selected checkpoints on the original
2,020-row tuning partition. No consumed final directory, payload, prediction,
raw timing or scoring call was used. The [published final summary](2026-10-28-ik54-final-qualification.md)
supplied the historical failure observation; its no-go remains unchanged.

## Diagnosis

The original seed 43019 failed the unchanged <= 0.8 CPU p95 ratio: 0.8358 ms
learned versus 0.9811 ms numerical, ratio 0.8519. Its untrained control also
slowed to 0.8241 ms, compared with 0.5296 ms in the first seed's measurement;
the numerical reference rose from 0.8528 to 0.9811 ms. This is evidence of a
broader timing change, not proof of a seed-specific mathematical defect.
The precise historical cause (for example scheduling, background load or clock
variation) cannot be identified from these published summaries.

New tuning measurements show that the eager implementation has limited cost
margin across seeds: each seed exceeded 0.8 in one of the two rounds. Seed 43019
passed at 0.7428 in round one and failed at 0.8057 in round two. Identical
model structure and inputs therefore do not produce a stable seed-specific
pass/fail classification under these measurements. Both rounds are retained.

Profiling 300 seed-43019 predictor calls attributed 0.138 of 0.151 seconds to
the eager model forward, while the three dense layers together took about
0.008 seconds. Numerous small tensor operations implement mask, finite and
bounds checks, sanitization, polar features and output checks. Python/module
dispatch and tensor-operation overhead are substantial in this batch-one path.
The profile diagnoses the implementation; its timings are separate from the
complete-decision benchmark below.

## Versioned improvement

[`inference_v1.py`](../../src/nca/ik/inference_v1.py) copies the existing v3 model
and freezes a TorchScript tensor graph for CPU [1,14] FP32 inference. Structural
shape/dtype/device checks stay at the callable boundary; every row-dependent
mask, finite, bounds, polar-transform and output-refusal calculation remains in
the graph. It uses the same weights, normalizer, output head and feature order,
without answer caching, shared mutable input buffers, numerical correction or
fallback. This is an opt-in in-process adapter, not a portable export bundle.

For all 6,060 tuning rows across the three checkpoints, eager/frozen joint
difference was exactly zero, with identical refusal statuses and accuracy gates.
CPU tests additionally cover both heads and widths, nonfinite inputs and network
outputs, missing/nonbinary masks, input bounds, tensor shape/dtype refusal and
snapshot isolation from later model mutation.

Use `freeze_batch_one(context, model)` after authenticating/loading the original
checkpoint, then pass its result to the existing `model_predictor(context, ...)`.
The historical factories, predictor and scorer remain unchanged.

## Complete-decision measurements

Measured on an AMD Ryzen 5 5600X, Windows 11 Pro, CPython 3.12.10 and Torch
2.8.0+cu126, with CUDA visibility empty and no initialized CUDA context. Each
measurement retains the original one-thread/batch-one protocol, 100 warmups and
10,000 complete calls per comparator including independent bounds/FK checks.
The five comparators rotate within each measurement. Round two reverses both
the seed and eager/frozen order. The matched untrained timing control is a
deterministic fresh initialization shared between variants; this campaign makes
no qualification-control claim about that initialization.

| Seed | Eager p95 rounds, ms | Frozen p95 rounds, ms | Median p95 reduction | Eager ratios | Frozen ratios |
|---|---|---|---:|---|---|
| 41011 | 0.6977 / 0.5661 | 0.4768 / 0.4771 | 24.5% | 0.8273 / 0.6914 | 0.5014 / 0.5340 |
| 42013 | 0.7887 / 0.7221 | 0.4927 / 0.3737 | 42.7% | 0.7509 / 0.8105 | 0.4690 / 0.4526 |
| 43019 | 0.7811 / 0.7186 | 0.4801 / 0.4972 | 34.8% | 0.7428 / 0.8057 | 0.5130 / 0.5219 |

All six frozen measurements meet the unchanged <= 0.8 development cost check.
The observed margin is useful; two rounds do not establish performance on every
CPU or under all host loads. No threshold, thread setting, batch size or scorer
semantics were relaxed. Results do not supersede the consumed final no-go or
establish CPU export, physical tracking or a new final qualification.

## Reproducibility and limits

Executed source: `871462498858e3c3123238d7adb82d9e399a1fd1`, closure
`9d708d8fb696c6017b9fb590ea92570d205c3be729d02f8d613541a3428e3861`.
The external declaration was frozen before corpus loading and measurement:
`1752ebb65c8e275f61d2c9b093b22c4cb62e63d972ac831b0ff2499b153bbcd4`.
The campaign exited zero after 237.719 seconds, with 619,560 predictor calls,
zero optimizer/final calls and no failed attempts. Its two-hour/one-million-call
ceiling and 6 GiB process cap remained in place. CUDA was disabled; no V100 or
TITAN compute was used. All previous declarations, ledgers, reports and factories
remain preserved.

The [public result index](ik-cpu-development-20261028.json) includes all twelve
measurements, checkpoint hashes, parity and raw-artifact hashes. Full timing
samples and profiles remain outside Git in the maintainer-held development
bundle. Reproduce with `python -m nca.evaluation.ik_cpu_dev_v1 --root <new-output>
--training <original-training-packet>` and `CUDA_VISIBLE_DEVICES` empty, using the
original authenticated training packet. There is no final-suite argument.

Validation: `python -m unittest discover -s tests -p "test_ik_*.py" -v`
passed 538 tests with four expected skips, with CUDA visibility empty. All six
CI jobs passed on the executed implementation head. The strict documentation
checker passed 133 files without errors or warnings, and `git diff --check`
passed. No test uses the consumed final suite; final-lifecycle tests use their
own synthetic fixtures.

© 2026 techrote. All rights reserved.
