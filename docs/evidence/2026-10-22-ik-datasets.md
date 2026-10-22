---
title: Runner IK corpus — one complete CPU allocation
kind: evidence
summary: Exact source and data identities, complete 10,220-task corpus, retained failures, isolated final population and finite generation cost.
owns: [evidence.ik-datasets]
verified: 2026-10-22
---

# Runner IK corpus — 22 October 2026

[IK-CUDA-004 / #53](https://github.com/techrote/neural-cyber-automata/issues/53)
produced one complete nominal geometric corpus under the
[dataset contract](../IK_DATASETS.md). All **10,220 scheduled outcomes** are
retained in **161 immutable shards**. The final population meets the accepted
coverage minima and remains sealed for candidate scoring. This establishes data
quality and provenance; no learned, CUDA, export or physical IK result follows.

## Prospective source and admission

The implementation reconciled main `ab1706d47ccb12939673258d02901b54c84c828f`,
including the separate Jolt allocation-planning documentation, before freezing
[source `1e16c751939d7b276e8ced02b7fc371f34e9c0ee`](https://github.com/techrote/neural-cyber-automata/commit/1e16c751939d7b276e8ced02b7fc371f34e9c0ee),
tree `1e968cecf421cd3e4e398ab3c410fde76e99377c`. The declaration contains the exact
123-file package/configuration/generator/process-envelope closure, SHA-256
`240433e97cafe2992cc217a747b8a0cca418241bc8dac665e3c8e94c3328a3b5`.

The [complete declaration](ik-dataset-20261022/declaration.json) was
[published before generation](https://github.com/techrote/neural-cyber-automata/issues/53#issuecomment-6335152671)
at 07:21:50 UTC. Independent readback matched all 60,581 bytes, revalidated the
clean committed source and current environment, and passed
[17 admission checks](ik-dataset-20261022/source-admission-review.json).
No teacher was called during schedule/publication admission. The registry's
process claim begins at 07:23:48 UTC; the completed receipts end at 07:25:10 UTC.

| Frozen identity | SHA-256 |
|---|---|
| Declaration content | `126889a65bc889c11bfe7021de50371cfbb1e288669434e74dd5f66a42d110dc` |
| Declaration file | `58d8c83b713ddeb538883a7859b280818a0c1a2c04393989db32ba80c2cff0e2` |
| Configuration | `fb8614412a62f7495519f463e3052ab060b0701701a3dfdb7265f0ad64488b51` |
| Split plan | `d641b2bea1b6810d63c0eed3d4423b369202f63754a5ed077634970d1c7b03ad` |
| Ordered task identities | `c187c72cd8ae6bd71bfd52d5909922c6cc11c1bab06b22f3766d2c62ea8e24a9` |
| Ordered task values | `97c837465d1900570064e746cff67cb1533da35499e82abcdeb1697a51d5fc8f` |
| All retained sample values | `2af514d793dafc8dd38476a9650804b1bd8b8ae5486cf7ec64de4f687e807a82` |
| Final input suite | `aa32e10577e40fb3ab29ba2e1f8ebb8c8324356572da5484609e82f4deba40a1` |

The environment was Linux x86-64, CPython 3.12.14, Python binary64 arithmetic
and one execution thread. NumPy 2.3.5 was installed but was not used for teacher
arithmetic; Torch was absent. The accepted Runner descriptor retains host pin
`f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497`; that host was not executed by this
allocation. Generation seed is 53021 and family split seed remains 61003.

## Complete population and retained outcomes

The [coverage report](ik-dataset-20261022/coverage.json) keeps nominal and stress
denominators separate. All 100 declared families occur in their assigned split.

| Population | Training | Tuning | Final |
|---|---:|---:|---:|
| Families used | 70 | 15 | 15 |
| Nominal points | 4,320 | 1,080 | 2,000 |
| Complete trajectories | 40 × 20 | 40 × 20 | 40 × 20 |
| Nominal denominator | 5,120 | 1,880 | 2,800 |
| Stress rows | 140 | 140 | 140 |
| All rows | 5,260 | 2,020 | 2,940 |
| Solved labels with pose supervision | 5,140 | 1,900 | 2,820 |
| Retained failed/refused outcomes | 120 | 120 | 120 |

Every final grid bin has 25 point queries; eight withheld regions provide
200 point queries. The 40 final trajectories include twenty smooth sweeps and
twenty reversals. Final nominal strata contain 90 rest-neighbourhood cases,
280 withheld-region cases including trajectories, and 1,800 transformed-frame
cases. Each of the seven stress populations contributes twenty final rows.

All 9,800 nominal tasks and all sixty inner-limit boundary tasks solved. The
remaining **360 outcomes remain explicit**: 120 numerical best-effort results
on hard-limit or geometrically infeasible targets, sixty iteration-exhausted
results, and sixty each of missing measurement, unsupported profile and invalid
input. Independent analytic labels distinguish command-limit failure from
geometric infeasibility; numerical best effort is not an infeasibility proof.
No retry, analytic replacement or success-only filtering occurred.

The maximum independently checked endpoint error recorded among solved labels is
`9.990437242002138e-8` cells. The 9,860 valid solved labels permit pose loss;
9,920 supported reachable rows permit endpoint loss, including the sixty
deliberately iteration-exhausted rows. Parser/guard statuses are retained for
explicit handling, not invented deployable neural inputs.

Runner has zero admitted singularities and zero second-branch cases. Synthetic
two-link fixtures separately test those conditions; they do not inflate this
corpus's Runner coverage. All admitted branch labels follow the actual link
vectors, not the sign of the ankle coordinate alone.

## Finite cost and complete process boundaries

The allocation remained one attempt, with no resume or relabel sweep.
An early planning ceiling of 180 seconds was increased to 900 before declaration
or production teacher work to cover worst-case teacher calls and readback.
The frozen allocation reserved 120 seconds for publication, 10,220 tasks and
attempts, 654,080 numerical iterations, and 512 MiB worker memory.

| Recorded cost | Result |
|---|---:|
| Started / completed tasks | 10,220 / 10,220 |
| Unknown attempts or additional work | 0 |
| Actual / conservatively charged numerical iterations | 47,796 / 47,796 |
| Reserved numerical iterations | 654,080 |
| Worker module through manifest preparation | 46.96489484199992 s |
| Complete worker subprocess | 48.32807375000266 s |
| Inner supervisor through semantic readback | 82.39503638500173 s |
| Complete generator process plus receipt readback | 82.48025006399985 s |
| Outer module through receipt readback | 82.59141588900093 s |
| Worker peak RSS | 62.23828125 MiB |
| Worker hard address-space ceiling | 512 MiB, POSIX `RLIMIT_AS` |

These clocks overlap; their durations must not be summed. The
[inner receipt](ik-dataset-20261022/supervisor.json) passed all seven checks and
the [process envelope](ik-dataset-20261022/process-envelope.json) passed all five.
The outer clock includes generator startup/exit and the inner receipt write.
Its own final receipt write is explicitly excluded from its recorded duration
and checked against the ceiling before return. Parent-process memory, Windows
peak memory and CUDA memory were not measured by this allocation. Later offline
review, packaging and CI are outside generation cost and perform no relabeling.

## Independent readback and runtime separation

The [publication review](ik-dataset-20261022/publication-review.json) passed all
34 independent checks and binds the
original declaration, manifest, receipts, registry and corpus hashes. Verification
does not invoke the numerical teacher: it checks retained results and independently
calculable geometry. All required files, shard boundaries, partitions, task/sample
identities, pending order, journal reservations/results, work totals, loss masks
and coverage are checked rather than trusting rehashed summaries.
Its direct endpoint calculation differs from retained endpoint values by at most
`3.972054645195637e-15` cells; frame conversion error is at most
`7.047505473780188e-13` cells. All 3,363 relevant cross-partition point/path
comparisons pass the guard, with zero violations.

The [training normalization](ik-dataset-20261022/normalization.json) uses exactly
5,140 replay-validated training inputs, selected by declared task classes rather
than teacher success. Its input digest is
`00bf161af280724ac0629547c1cf2b797a47cdcd71dd52f15a0fee9bcb67dfeb`.
No tuning/final row participates. Exact target configurations, teacher answers,
analytic feasibility, branch labels and split metadata remain outside the
fourteen permitted runtime features.

The final registry has no opening receipt. Structural data-quality review has
not evaluated or selected any learned candidate. The public deterministic
generator is not a secrecy boundary: bypassing the audited opening API or
discarding its registry invalidates a pristine-holdout claim. #54 must retain
the registry, freeze every candidate and publish its single candidate-bound
opening before scoring. Later use of an opened suite is development data.

## Retrieval and integration checks

The [publication index](ik-dataset-20261022/index.json) names the complete external
archive, exact byte/hash inventory and retrieval route. Obtain that archive from
the maintainer's issue-completion handoff, verify its SHA-256, and extract it
outside the checkout. Preserve the declaration and registry alongside `corpus`.
Do not regenerate and substitute different teacher timings/statuses for these bytes.

Use the documented Linux or PowerShell environment setup, then run
`python tools/ik_dataset.py inspect --artifact <extracted>/corpus --registry <extracted>/registry`.
This performs semantic readback without opening final candidate scoring. Only
the small manifests, summaries, normalization and independent reviews enter Git.
The raw shards and journals remain in the external archive.

Source conformance comprised 77 dataset tests, including adversarial rehashed
inventory/partition/journal/normalization attacks and process-budget failure tests.
Initial hosted Linux CPU, pinned-host and optional CPU-learning checks passed;
Windows exposed a POSIX test mock missing `SIGKILL`. The subsequent change is
test-only and mocks that constant explicitly. All generator/configuration bytes
remain those of the measured source. The local full suite's separate existing
PID-namespace restriction is recorded; it is not hidden or waived. The
[implementation PR](https://github.com/techrote/neural-cyber-automata/pull/72) and
[issue](https://github.com/techrote/neural-cyber-automata/issues/53) record the
applicable final-head checks, independent integration review and merge verification.

Actual CUDA qualification remains #51. Learned comparison is #54; host handoff,
exact export and physical tracking retain #55–#57. Jolt, morphology and historical
allocation gates are unchanged.
