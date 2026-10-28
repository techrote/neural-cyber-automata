---
title: Fresh frozen-inference IK qualification result
kind: evidence
summary: Frozen inference meets CPU cost ratios but all three seeds fail mandatory withheld accuracy on the new sealed suite.
owns: [evidence.frozen_ik_qualification]
verified: 2026-10-28
---

# Fresh frozen-inference qualification

[MEASURED NO-GO] The [D-037](../decisions/2026-10-28-frozen-ik-qualification.md)
qualification ran once at `80415bdc5ae3ad8e5ed660b06c52208f07b3dc82`, after all
six CI jobs and two independent exact-packet admissions passed. The same three
trained checkpoints and actual epoch-zero controls used the unchanged frozen
CPU inference path. No training, selection, threshold change or retry occurred.

## Outcome against unchanged gates

| Seed | Nominal p95, cells | Worst, cells | Nominal successes | Withheld p95, cells | Withheld successes | CPU cost ratio |
|---|---:|---:|---:|---:|---:|---:|
| 41011 | 0.009369 | 0.155005 | 2775/2800 | 0.114646 | 255/280 | 0.441431 |
| 42013 | 0.007981 | 0.151340 | 2775/2800 | 0.103742 | 255/280 | 0.531452 |
| 43019 | 0.006241 | 0.142137 | 2775/2800 | 0.100379 | 255/280 | 0.481342 |

Every seed fails the mandatory withheld p95 <= .06 cells and <= .01 L and
withheld success >=95% gates: success is 91.07% in that population. Nominal
p95, worst and success pass; all 760 continuity pairs are complete with zero
admitted branch switches. There are no nonfinite, bounds or protocol defects.
Learned contribution passes for all seeds. These are geometric software results,
not export, physical tracking, host acceptance or promotion evidence.

The frozen route meets the unchanged cost-only <= .8 ratio on every seed, but
the official practical-utility gate also requires accuracy. It therefore remains
**no-go** for every seed and in aggregate. Trained/numerical CPU p95 milliseconds
are 0.3727/0.8443, 0.4858/0.9141 and 0.6011/1.2488. The analytic comparator is
also reported: p95 0.0862, 0.1079 and 0.1267 ms. All timing uses the declared
one-thread, batch-one, 100-warmup/10,000-measurement complete-decision protocol.
No extra timing round or best-seed substitution was performed.

## Independent new suite and immutable candidates

The new model-free custodian froze generation seed 53029, a distinct family
namespace and the [new final-only recipe](../design/IK_FROZEN_QUALIFICATION.md)
before teacher calls. Two independent audits verified separation from all 7,280
training/tuning rows, including full root/joint trajectory-path guards. Counts
are 2,000 points, 800 trajectory decisions and 140 stress cases, including 200
withheld point queries. Generation made 2,940 calls and 13,299 teacher iterations
in one supervised attempt: 10.204 seconds, approximately 61.3 MB peak, no filtering.
Preparation used 17.218 seconds and approximately 96.6 MB peak in its own envelope.

The same selected checkpoint hashes remain
`436c0499cd4eab533ed10315b63c229c6c52ad5af5ef61e278b91ad759f0f609`,
`d43ee34c7b541f11fecd06f833b43c2dc7999b1bb71c9672393ac6d8ce93832e`,
and `45bb2d0e4260279f99425d7f04e0790fcdefb7eb6da038ee8f5fac90471958b5`.
The original normalizer and all training ledgers/checkpoints remain unchanged.
The tuning-only proof authenticated the full completed ledgers and all 600
tuning reports, then measured exact zero joint difference for both trained and
actual-initial models on every tuning row. It exited zero after 1,409.090 seconds
with 685,920,256 peak process bytes, within its 3,600-second/8-GiB ceiling.

New-suite geometry and its withheld lane differ from prior recipes; accuracy
numbers are not a matched same-input comparison with earlier finals. The earlier
final no-go remains unchanged. No older consumed suite, payload, saved prediction
or raw timing was reused. The new suite was opened once and is now development-only;
it cannot support another qualification attempt.

## Source, admissions and completion

Preparation/proof ran at `e2aec1f`; the [explicit source amendment](../design/IK_FROZEN_QUALIFICATION.md#source-amendment-before-qualification)
preserves their exact artifacts while binding the reviewed execution revision.
Production closure is `7456adfd18a11504e5a5d7cfedfc334915acd1db63289228dbb3be35ce7802d3`.
Frozen inference file SHA is `a849b12b47d767636753e346a6c970d388b4b14d37625606fbeaef8ccbbf23f4`.
No numerical generation, isolation, inference or scoring code changed in that
amendment. The two CI compatibility fixes retained the historical Windows/Linux
replay contract and Jolt's 16 KiB provenance cap; failures and corrected synthetic
fixture attempts remain preserved. No live Jolt qualification follows.

Declaration file SHA: `a9bf7d7ae4cd3ad367323d1c0c153774a54cf8336f076b7f529ad9f54d16d41e`.
Independent admission file SHAs:
`856900488b6fa32a6833ee656d48db1d8f23796eee0480cfd5f6de72dff0582a`
and `4ef1c555d4ec9cfe60f3061a392bd513bd2e09da354cb804aff80f36c7e910ac`.
New seal semantic SHA: `f5c26e1ef353816cec632ded3d039592ef485b64b1f61ef0903a155408742984`;
archive SHA: `92989ef4d7326cde639b680b91aa80e1371776d99be7b351b2dbdc0ded975705`.

The process exited zero, sealed and read back all reports/terminal/completion,
and released its ownership claim. No execution/resource failure marker remains.
Prepublication elapsed time was 309.836 seconds. All three seeds prepaid 78,240
final/timing decisions and sixteen synthetic graph forwards. Each records 77,880
actual predictor calls and 360 guarded refusals; two graph builds record six
eager-hook forwards and conservatively reserve internal tracing work separately.
No optimizer calls occurred. The 132 board samples peaked at 8,590,983,168 bytes
and 30 C, below the unchanged 20-GiB aggregate cap; these board readings include
unrelated background use. Inference was CPU-only; no V100 was used.

Both independent terminal audits reproduced all 52,920 saved comparator rows,
population/continuity/gate arithmetic and all 180,000 saved latency samples.
Independent FK error differed by at most 3.553e-15 cells. They confirmed the
no-go and all hashes, counters, reservations, telemetry and completion bindings,
with no integrity finding, model/teacher/timing calls or archive reopening.
Audit SHAs: `f77bcfbdf9aa788dcc9464b91a8c51a2e1a37ef98ec7165760a23fcbe87455f7`
and `164f6aaa456aecb9030c04757730d06d8cb367f5d6e3944abddc09d6f306b7d4`.

Terminal file SHA: `5b97074eb026691ddc7fb97da5a1e56599e164413e0ccffee61b65d6d619c297`.
Aggregate file SHA: `adcf7f71bd86208b6951619595311dac141b197e4e1bf073f6ae9ea7d382b8f3`.
Observer file SHA: `5802b74936fa684b0fa1582a59ae3b3a4ea3e4fa90f814e52a92af169b87605f`.
The [public result index](ik-frozen-qualification-20261028.json) records complete
per-seed comparator summaries, budgets and artifact hashes. Full artifacts remain
in the maintainer-held `ik54-frozen-qualification-20261028` bundle outside Git.
Existing evidence ledgers remain immutable; STATE and the documentation map link
this separate new result.

Validation: `python -m unittest discover -s tests -p "test_ik_*.py" -v` passed
553 tests with four expected skips; all fifteen focused new tests passed. The
affected synthetic Jolt fixture groups passed, and all six CI jobs were green on
the exact execution head before launch. The full CPU-learning CI suite passed
1,575 tests with ten expected skips. `python tools/check_docs.py --strict` passed
136 files without errors or warnings, and `git diff --check` passed. This
documentation-only outcome record performs no merge or runtime promotion.

© 2026 techrote. All rights reserved.
