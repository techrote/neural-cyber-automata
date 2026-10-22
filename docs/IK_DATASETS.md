---
title: Branch-aware geometric IK datasets
kind: reference
summary: Deterministic Runner tasks, isolated trajectory families, bounded teacher labels, immutable data and single-use final evaluation.
owns: [ik.datasets, ik.dataset_splits, ik.dataset_features, ik.final_suite_lifecycle]
verified: 2026-10-22
---

# Branch-aware geometric IK datasets

[IK-CUDA-004 / #53](https://github.com/techrote/neural-cyber-automata/issues/53)
adds a CPU dataset path on the accepted [geometry and references](IK_KINEMATICS.md).
The source is [`data.py`](../src/nca/ik/data.py); the
[generation tool](../tools/ik_dataset.py) owns immutable artifacts and final-suite
access. These are nominal geometric tasks, **not ACT004 physical episodes**.
CUDA training, learned quality, export and physical tracking retain their separate gates.

## Frozen recipe and curriculum

The [prospective configuration](../configs/ik/datasets/runner_v1.json) is
`runner-ik-corpus-v1`. Its exact configuration, source closure, profile,
teacher settings, environment, split plan and scheduled identities are hashed
in the declaration before a measured invocation. The
[accepted numerical criteria](IK_ACCEPTANCE.md) and their original JSON are unchanged.

| Population | Training | Tuning | Final |
|---|---:|---:|---:|
| Nominal point queries | 4,320 | 1,080 | 2,000 |
| Complete target trajectories | 40 × 20 | 40 × 20 | 40 × 20 |
| Seven stress classes | 7 × 20 | 7 × 20 | 7 × 20 |
| Total unique tasks | 5,260 | 2,020 | 2,940 |

The complete recipe contains **10,220 tasks**. Final points cover all 80 bins
of the accepted 10 × 8 hip/ankle grid with 25 queries per bin. Eight withheld
regions supply 200 of these point queries. Rest neighbourhood, interior points,
withheld regions, smooth sweeps, reversals and root-frame transformations have
separate coverage counters. Rest neighbourhood means target-configuration norm
at most 0.05 radians; it does not mean the measured current pose is already there.

Curriculum order is fixed: interior points, exact joint boundaries, moving and
reversing targets, then explicit failures. The six accepted stress populations
are inner-limit boundary, hard-limit boundary, certified geometric infeasibility,
unsupported profile/effector, missing measurement/inactive degree of freedom,
and nonfinite/malformed input. Iteration exhaustion is a seventh explicit class.
Both joints and both signs appear in the boundary populations. Stress attempts
remain outside the nominal reach-success denominator.

The [tiny configuration](../configs/ik/datasets/tiny_v1.json) contains 254 opened
development tasks. It supports portable CI and does not satisfy final qualification
coverage. The [independently calculable fixture](../tests/fixtures/ik/dataset-independent-v1.json)
also remains opened development data. No corpus shards enter Git.

## Families, regions and continuous paths

Split seed **61003** sorts 100 distinct generation-family identities into
70 training, 15 tuning and 15 final families. Families represent generator/start
conditions and own complete point groups or trajectories. A family can cover
multiple target regions; family identity and workspace region are different axes.
Assignment happens before task sampling, current-pose variants or augmentation.

The nominal q domain is hip `[-0.6,0.6]`, ankle `[-0.2,0.2]`. Bin index is
`8 * hip_bin + ankle_bin`. Withheld bin indices are
`6,15,24,33,46,55,64,73`; their full interiors are absent from training and tuning.
Every exact region is a q rectangle mapped by the accepted Runner FK. The mapping
is one-to-one on this domain. An axis-aligned endpoint box is not substituted for it.

Other bins have separated hip-fraction lanes: training `[0.06,0.50]`, tuning
`[0.62,0.70]`, final `[0.82,0.94]`. Ankle fractions stay within `[0.08,0.92]`.
Withheld final patches use `[0.08,0.92]` on both axes. The configuration requires
positive exclusion bands, initially 0.002 radians, and a 0.0005-cell minimum
cross-partition endpoint/path separation. Actual sampled points stay further
inside each patch. Configurable guards cannot overlap the declared lanes.

Interpolation labels describe the separated patches in ordinary bins. The
withheld-region label means spatial generalization into excluded holes within
the nominal domain, not extrapolation beyond the global convex hull. Boundary
stress outside the nominal q domain is separately tagged `outside_nominal`.

Complete trajectory targets are defined by deterministic bounded q curves,
sampled at the declared steps. Their deployable target sequence is the resulting
root-local points, with piecewise-linear interpolation between decisions.
Validation compares all points and every consecutive line segment across
partitions; a path cannot cross another partition's sampled point or path
between its stored points, within the declared guard.
It also checks the exact replayed q-patch membership and canonical frame.

`augment_task` provides two declared root transformations. It retains original
family/group/partition and canonical target. Copies cannot become fresh holdouts
by changing frame, sample ID or labels. Duplicate permitted inputs are hashed
without arbitrary IDs/partition names; contradictory supervised poses are rejected.
Malformed tasks hash their original malformed request, not their finite metadata.

## Teacher outcomes and independent checks

Each geometric task carries measured current q, target, permitted masks and a
declared posture preference. Moving cases use a perturbed previous prescribed
configuration as the synthetic current state; this is a nominal continuation
fixture, not measured host feedback or a successful previous teacher answer.
The preference is current posture, never the known target-generating configuration.

One bounded numerical solve labels each supported task. The default has 64
iterations, 0.05 seconds, eight backtracking trials, 0.25-radian trust norm and
`1e-7`-cell solve tolerance. The original candidate, residual, status, projection,
correction, work and elapsed time remain intact. No random retry or analytic
replacement relabels a failed numerical attempt. Hard-limit targets can be
joint-limited under the narrower command bounds and retain that result.

A separate two-vector endpoint calculation and the independent analytic
comparator verify endpoints, admissible bounds, residuals and branch identity.
The elbow branch uses the angle between geometric link vectors. For Runner,
the bend is `qa + atan2(1.25,1) - pi/2`, so both positive and negative admitted
ankle coordinates belong to the same negative-bend branch. Runner has no admitted
singularity or second inverse branch. Those tests use separately named synthetic
two-link fixtures and never enlarge claimed Runner coverage.

| Label/mask | Permitted use |
|---|---|
| Raw `teacher` | Original bounded outcome and all consumed work; never rewritten by the checker |
| Independent `analytic` | Reachability/branch evidence; numerical failure is not an infeasibility proof |
| `pose_loss` | Solved, independently verified, branch-consistent labels only |
| `endpoint_loss` | Valid supported reachable input, including a retained numerical failure where geometry loss is meaningful |
| `status_loss` | Status/refusal evaluation population; not an automatic neural training feature |
| All scheduled rows | Coverage and evaluation denominators, including failed, refused and interrupted work |

Malformed specimens contain the exact original invalid JSON. Parsing that input
is the attempt; the finite surrounding fields are metadata and are not submitted
to the numerical teacher. Malformed and unsupported-profile rows supervise
parser/guard handling. They cannot become neural status examples over invented
placeholder inputs. Missing measurements stay null with false measurement masks.

`validate_sample` checks outcomes without invoking the teacher again. It rejects
false solved labels, converged residuals labelled best effort, iteration exhaustion
before its ceiling, time exhaustion before its limit, and rejected inputs carrying
an actionable candidate. Failed configurations remain inspectable but receive no
pose-regression weight.

## Runtime features and normalization

`RuntimeInputs` is a strict record. Unknown fields fail instead of being silently
dropped. The deployable vector contains current hip/ankle, canonical target x/y,
declared posture hip/ankle and separate presence, measurement, activity and task
masks. Exact profile identity is checked. Null q uses a numerical placeholder
only beside its false measurement mask; it is not an observed zero.

Generating target q, teacher solution/status/residual/iterations, analytic
feasibility, branch answer, split, region and curriculum metadata are excluded.
Passing a full sample or evaluator dictionary to `runtime_features` is refused.

`fit_normalization(samples, config)` requires training records validated against
their exact frozen generator state. Relabelling a final row as training fails.
Selection uses the predeclared nominal/inner-boundary task classes, independently
of teacher success. It records the exact sorted training-input digest, count,
means and scales. Constant columns use scale one. This immutable fit is reused
for tuning, final evaluation and export; tuning/final data cannot refit it.

## Replay boundary and finite execution

Counter hashing uses generator version, seed, task address and named channel;
batch size, iteration order and worker scheduling do not alter runtime input
geometry. Full task/group identities also bind the entire configuration, so a
changed configured batch size creates a different identity with the same inputs.
State records recover IDs, families, current pose, targets, masks and frame
variants. Generator arithmetic is Python binary64; sampled q and ordinary
endpoints round to twelve decimal places. Exact boundary FK bytes are retained
to avoid moving a boundary point outside its legal inverse interval by rounding.
Same-source/environment replay is exact. Cross-platform trig implementations
are only expected within `2e-11` cells/radians and require their own verification.

Teacher elapsed time is not deterministic. A scheduling delay can change a
wall-limited teacher status. The immutable recorded result is authoritative;
replaying inputs does not promise byte-identical labels or authorize relabelling.
Declaration preparation hashes the deterministic schedule without calling the
teacher. It is separate from the measured one-attempt label-generation allocation.

The prospective ceiling is 10,220 tasks/attempts, 654,080 reserved numerical
iterations, 512 MiB memory and 900 seconds, with a 120-second publication
reserve. Every attempt reserves its maximum iterations durably before execution.
Unknown interrupted work remains charged conservatively. Raw completed results
are flushed before completion acknowledgments; partial shards and failures remain.
The supervisor kills an over-budget worker and publishes its own receipt.

The tool reports the exact scope of every clock. The
[process envelope](../tools/ik_dataset_supervise.py) includes generator interpreter
startup and teardown, semantic readback and the inner supervisor's final receipt
write. Its own receipt write remains explicitly excluded from its recorded elapsed
field and is checked against the ceiling before returning. Windows memory
enforcement uses a Job Object; Windows peak RSS remains unmeasured. The hard
512 MiB ceiling applies to the worker's address space on POSIX or committed memory
on Windows, not the entire operating system. A skipped device is unmeasured.

## Operations and final-suite opening

From a clean checkout with Python 3.11 or newer, install the package or set
`PYTHONPATH=src` (`$env:PYTHONPATH = "src"` in PowerShell). Outputs and the durable
registry must be outside the checkout. Declare first:

```text
python tools/ik_dataset.py declare --config configs/ik/datasets/runner_v1.json --registry ../ik-data-v1/registry --output ../ik-data-v1/declaration.json
```

Publish and independently read back that exact source/configuration/partition/
budget declaration before consuming an assigned allocation. Then:

```text
python tools/ik_dataset_supervise.py --declaration ../ik-data-v1/declaration.json --registry ../ik-data-v1/registry --output ../ik-data-v1/corpus
python tools/ik_dataset.py inspect --artifact ../ik-data-v1/corpus --registry ../ik-data-v1/registry
```

Keep declaration, registry, all shards, journals, normalization, coverage, worker
manifest, supervisor and process-envelope receipts together. Publication records exact bytes/hashes
and retrieval instructions in the [run ledger](RUNS.md). No retry or resume is
authorized by an incomplete result. A new allocation needs its own declaration
and retains the original consumed cost.

The final suite freezes before model training. Structural data-quality review
checks coverage/geometry/lineage without evaluating or selecting a candidate.
Ordinary inspection returns no raw final rows. `open-final` requires an experiment
ID and all frozen candidate SHA-256 values; it records consumption durably before
exporting data. A repeated opening, including a copy of the dataset, is refused
against the same retained registry. Later access requires an explicit development
label, and a later experiment needs a fresh final suite.

This is an auditable single-use evaluation discipline, not encryption or a claim
that public deterministic geometry is secret. Bypassing the supported API or
discarding the registry invalidates pristine-holdout claims. #54 must preserve
the registry and publish the candidate-bound opening with its evaluation.

Run tiny conformance with `python -m unittest discover -s tests -p "test_ik_data*.py"`.
Synthetic branch fixtures, development diagnostics and production data-quality
evidence have distinct identities. No result here qualifies a learned model,
CUDA execution, a host target lane, collision-free motion or physical tracking.
