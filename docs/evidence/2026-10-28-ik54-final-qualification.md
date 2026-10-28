---
title: IK54 candidate-bound final qualification result
kind: evidence
summary: Single-use final evaluation passes geometric accuracy and learned contribution but fails aggregate CPU practical utility.
owns: [evidence.ik54_final_qualification]
verified: 2026-10-28
---

# IK54 final qualification result

[MEASURED NO-GO] The separately declared [D-035](../decisions/2026-10-28-ik54-final-qualification.md) final evaluation ran once at source revision `a916f07a0c81e109d7b9f46a7b24c45cdaf877f0` after all six exact-head CI jobs and two independent non-author admissions passed. The three tuning-selected candidates were fixed before opening the original polar final. Its 2,940 cases per seed were evaluated with the declared five comparators and an archived-v1 diagnostic. The original suite is now opened and future use is development-only.

| Seed | Nominal p95, cells | Nominal worst, cells | Nominal successes | Withheld p95, cells | Accuracy | Learned contribution | CPU utility |
|---|---:|---:|---:|---:|---|---|---|
| 41011 | 0.031303 | 0.067032 | 2799/2800 | 0.009373 | pass | pass | pass |
| 42013 | 0.024306 | 0.071588 | 2796/2800 | 0.012738 | pass | pass | pass |
| 43019 | 0.019297 | 0.049603 | 2800/2800 | 0.014120 | pass | pass | no-go |

All three pass the unchanged accuracy and learned-contribution gates. Each has 760 complete continuity pairs and zero admitted branch switches. The aggregate practical-utility gate is **no-go**: seed 43019's CPU p95 was 0.8358 ms versus 0.9811 ms for the independent numerical reference, a ratio of 0.8519 against the unchanged maximum 0.8. The other two ratios were 0.6310 and 0.6183. This outcome is not an acceptance, CPU export, physical tracking or host qualification claim; independent post-run acceptance remains external.

The evaluation exited zero, sealed and read back its terminal record, released its ownership claim and has no failure or resource-failure marker. Prepublication elapsed time was 297.892 seconds; the targeted board watchdog observed a maximum 892,338,176 bytes and 27 degrees Celsius. Each seed reserved 78,240 decisions, performed 77,880 predictor calls including timing, and recorded 360 guarded refusals. No optimizer calls, retraining, reselection or retry occurred. No V100 was used.

Declaration file SHA-256: `ee1096c95320ea3ac5b8250b4f4e673751c23400143fad1bfd54ba31cc3c9de1`. Independent admission file SHA-256 values: `61e96572c44ae8ef480c83ed2f771d083e66afca5a6ba671e9f749a6c89e5134` and `89e46b009db3751876750f061735a658eeece3f505a1ac218a1f2a0f2d3de2f2`. Opening semantic SHA-256: `d2ea2db3e8d0222eebd673784b6e6fde6f65a87c54360aa30ea6a64f6e20ad4d`. Terminal file SHA-256: `357fe801c168c655f4bc50666c0db84413d63bf44ad02de6a5bbe5d2d83be8d0`. Aggregate file SHA-256: `33779d6f2870e943ac95412f03207c334bea246ae0fc0f580700b5b86bedb54b`. Observer file SHA-256: `59fc0f2f78517fbacc9b2c0d41409715d79b98fb82311d9728de80e3c35b746c`. Full per-seed reports, raw latency measurements and receipts remain in the maintainer-held evidence bundle outside Git.

© 2026 techrote. All rights reserved.
