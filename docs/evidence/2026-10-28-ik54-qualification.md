---
title: IK54 strict qualification training and tuning
kind: evidence
summary: Three admitted full-training seeds pass the unchanged tuning gate; finals remain excluded.
owns: [evidence.ik54_qualification]
verified: 2026-10-28
---

# IK54 strict qualification: training and tuning

[MEASURED TUNING PASS; NO FINAL ACCEPTANCE] The owner authorized the frozen
[D-034 execution](../decisions/2026-10-28-ik54-qualification-execution.md) after
target metadata recovery and two independent exact-packet admissions. All three
full-training seeds completed and passed the unchanged tuning gate. No final
suite was opened or scored, and physical/runtime acceptance remains unmeasured.

| Seed | Selected epoch | p95 cells | p95 / L | Worst cells | Worst / L | Nominal successes | Tuning gate |
|---|---:|---:|---:|---:|---:|---:|---|
| 41011 | 755 | 0.012699107 | 0.001938564 | 0.037526447 | 0.005728545 | 1880/1880 (100.0%) | pass |
| 42013 | 755 | 0.009745300 | 0.001487655 | 0.046657128 | 0.007122376 | 1880/1880 (100.0%) | pass |
| 43019 | 915 | 0.008803401 | 0.001343870 | 0.027150277 | 0.004144586 | 1880/1880 (100.0%) | pass |

The fixed gate requires p95 <= 0.06 cells and <= 0.01 L, worst <= 0.18 cells
and <= 0.03 L, at least 95% success, and all finite, bounds, protocol,
continuity and branch gates. Every selected report is complete; each has
760/760 continuity pairs and zero admitted branch switches or over-limit pairs.
The independent binary64 scorer applies nearest-rank percentiles and the
original 1,880-row nominal denominator. No final acceptance is inferred.

## Executed contract

- Learner source `7c65e5920d577695b3012b2b3c4391574329c6fc`, executable closure `4e369f0e5691e8d3e8657a51fbaa247a44cbaac580545750fc77c538fc80462a`;
  host geometry revision `f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497`.
- Declaration SHA-256 `5e0ce7c4e36bbcbebe7543f8d230c5df166cf33d291a01d847efc6ed9ad2c4f8`; frozen recipe SHA-256
  `6279994faeedbca4c7639035415926727af227d218a8e7e5b497ac0ad0492da4`; corpus identity `dd6394f603f069295abfd37991deb1beb84d864e16e031f30c0330deafeed76a`;
  original training-only normalizer `9ce68b359737af8abd65223cd01f766d16c574c5c07d484018d27e75f6257d00`.
- Profile `6cb15832e240f82f4bcd582d2c26015ef986e44f335da683abaf9746e31503c4` and acceptance file `cea8642ed41313c9c6fc8f27f288e90aa8743146e5f61a2f0ab2ea86b874aa8b`.
- Windows 11 AMD64, Python 3.12.10, Torch 2.8.0+cu126, NumPy 2.3.5,
  NVIDIA driver 581.80; UUID-isolated TITAN RTX, fingerprint
  `e5df8b0db3a22718306c0f68e7ec85b69f1555759649550d74fc44f563c841d9`.
- Full original 5,260-row, 70-family training partition and original 2,020-row
  tuning partition. Polar features, original normalizer, bounded tanh head,
  width 64, batch 128, original stage grouping/order, 1,000 epochs; Adam with
  original loss weights, clip 1.0 and cosine learning rate from 1e-3 to 1e-5.
- Every five epochs, tune once and checkpoint. Select each seed by lowest
  complete raw nominal p95, with the earliest epoch breaking ties.
- Per seed: one completed attempt, 43,000 committed optimizer steps, 5,260,000
  presentations, 404,000 tuning calls, 201 checkpoints and zero final calls.
  All three terminal ledgers were sealed and supervisor-verified, with no
  unresolved work, retry, failure marker or retained GPU ownership.

## Resource and custody evidence

The supervisor recorded 3617.417 seconds before
its final result write. Process exit was observed by 3627.135 seconds;
that upper bound includes polling and observation delay. Per-seed wall charges
are prepaid accounting, including the 7,200-second publication reserve:
41011: 10450.229 s, 42013: 10434.526 s, 43019: 10440.151 s.
They are not measured worker durations. The group ceiling was 86,400 seconds;
work admission stopped with 10,800 seconds reserved. Peak sampled board use
was 2.196 GiB and maximum temperature was 29 C.
Mean sampled GPU utilization was 41.41% across 1,678 samples;
it is an unweighted sampled mean including readback, not a kernel-overlap claim.
Minimum sampled host commit headroom was 61.11 GiB;
minimum sampled disk headroom was 47.50 GiB.
The 20 GiB aggregate board cap, 1 GiB per-worker allocator and 8 GiB
per-process host limits remained in force. The V100 was not used.

Selected tuning report and checkpoint SHA-256 values, by seed:

- `41011` report `607bdbba7f32df8760afe676a0d7c2124ef54fe558b25a8b95af65246858f484`;
  checkpoint `436c0499cd4eab533ed10315b63c229c6c52ad5af5ef61e278b91ad759f0f609`.
- `42013` report `0a0090e18e392eb6d99bb668d25633703091f970a2863e11bf42c4ca9cfcfb72`;
  checkpoint `d43ee34c7b541f11fecd06f833b43c2dc7999b1bb71c9672393ac6d8ce93832e`.
- `43019` report `203a4ba671d13cb209a78fa6f217b68f05a0f74c36edf550ae5449dcfca9f270`;
  checkpoint `45bb2d0e4260279f99425d7f04e0790fcdefb7eb6da038ee8f5fac90471958b5`.

External machine-readable terminal summary SHA-256: `514e86e65d4d756df03f42b3ae70e7c76cf6e8b0db6ce9451350275958c04983`.
Immutable group result SHA-256: `bff8553856aa7c9b98fe5fbbd3a51399e18396ab1612eb559b4078baa7f6d7f7`.
Owner execution record SHA-256: `ed5a527c8401c20b5e3f8487aba7345043f85fcc225ac3b250f95158d38adcc3`.
Independent admission A: `a1f52ecdd292b984088fc5b854eac31dfb3a65b1612fa53e235536ad8194789c`.
Independent admission B: `299c6b5d2e521961912152e1a8f8160496d40639808f8bb8bd50cd9c1cfc66b6`.
Original development results, historical packets and their evidence remain
unchanged. Large checkpoints, detailed ledgers and reports remain in the
owner-approved external evidence bundle. The still-sealed final suite requires
a separate owner-approved declaration and admissions before any opening.

© 2026 techrote. All rights reserved.
