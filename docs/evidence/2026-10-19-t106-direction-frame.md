---
title: Fixed Jolt direction-frame prerequisite follow-up
kind: evidence
summary: Separate commissioned per-rig BC experiment with fresh policy seeds and sealed geometry, unchanged utility gates, exact source and checkpoint admission, and a fixed CPU allocation; no result before execution.
owns: []
verified: 2026-10-19
---

# Jolt direction-frame prerequisite follow-up

The renewed instruction to complete the repository authorizes this separately declared useful-skill
follow-up. The [original T105 result and T106 entry decision](2026-10-19-t106.md) remain immutable and
blocked. The new experiment has no measured result yet. Its purpose is to test one coordinate
interpretation within the existing Jolt cloning implementation; it does not itself answer the
shared-morphology research question.

The complete [declaration](../../configs/first-skill/jolt-direction-frame-v1.json) has canonical SHA-256
`af936549f71bc72bc7fabfc8c75b2d04b58a8348635b5f47ddec50ea904a7687`. It binds the original task,
reward, teacher source, host, Jolt profile, bridge, runtime and utility thresholds. Any change to this
declaration requires a new experiment before further collection or fitting. The runner rejects changed
criteria, all previously known holdout geometry and incomplete paired outcomes.

## One explicit coordinate hypothesis

Both candidates use the same 26 original features, 32-wide GRU, six cloned action channels, critic,
teacher observations, loss, 40 BC epochs and fixed final checkpoint. For each policy seed their initial
numeric weights are identical; weights are independently initialized across seeds. They receive the same
complete eligible authored training sequences, including unsuccessful teacher episodes. There is no
success filtering, extra mirrored episode, privileged observation, reward change or tuned teacher.

| Candidate | Coordinate interpretation | Actual wire request |
|---|---|---|
| `direction_frame_bc` | Fixed sign of caller target X minus initial observed root X; coincident X uses +1 | The same sign restores predicted move X and trim to world coordinates |
| `world_frame_bc` | Original world coordinates | Original move X and trim |

The direction sign is chosen once per invocation and stays fixed after overshoot. It multiplies only
target displacement X, velocity X, sine of root rotation, angular velocity, previous move X and previous
trim. Presence/measurement masks, vertical features and all other history channels are unchanged. The
teacher's move X and trim labels receive that same transform. The original world-goal aim mapper is
shared by both candidates. The controller retains the actual restored world request for subsequent
history, then transforms its two signed channels on the next observation.

This is an inductive-bias hypothesis. No reflection symmetry of the full rig, contacts, terrain,
materials, host assists or gameplay plant is asserted. Algebraic involution and reflected-input tests
establish the coordinate implementation only. The actual-host comparison must determine whether it
helps on the declared cases.

The [controller and checkpoint contract](../../src/nca/controllers/goal_frame.py) reuse the original
episode, owner, mount, part, goal, deadline and feedback-order fences. Frame-aware checkpoints carry a
separate explicit experiment specification; the original checkpoint reader rejects them. A normalized
checkpoint cannot be relabelled as the original world-frame input contract.

## Fixed allocation and unopened cases

The policy seeds are **7103, 7207 and 7301**. The eight authored training geometries remain floor targets
at ±24, ±36, ±48 and ±60 cells, with three newly declared host seeds per geometry. All 24 teacher
episodes are collected before fitting. No tuning split or PPO update is present.

The eight new evaluation cases are absent from every original T105 training, tuning and evaluation
geometry and from this follow-up's training geometry:

| Case pair | Initial vertical placement | Horizontal goal distance | Host seeds |
|---|---|---:|---|
| Floor right / left | Original supported floor placement | 27 cells | 625001 / 625002 |
| Floor right / left | Original supported floor placement | 63 cells | 625003 / 625004 |
| Drop right / left | 12 cells above supported placement | 45 cells | 625005 / 625006 |
| Drop right / left | 28 cells above supported placement | 57 cells | 625007 / 625008 |

All six final epoch-40 checkpoints and all six corresponding initial checkpoints are byte-hashed and
frozen before any new case opens. Each case evaluates the authored reference, idle and both phases of
every family/seed: **112 matched evaluation episodes**. The family priority is direction frame, then
world frame, with no best-seed selection. Neither early heldout fitting nor case-dependent checkpoint
selection is permitted.

| Phase | Maximum game decisions |
|---|---:|
| 24 authored teacher episodes | 960 |
| 112 paired evaluation episodes | 4,480 |
| Complete scheduled worst case | **5,440** |
| Absolute experiment ceiling | **6,000** |

The aggregate wall ceiling is **1,800 seconds**, one CPU thread, zero GPU, paid-service, transfer or
adaptation work. Each episode has at most 40 six-tick requests and the original intrinsic 240-tick
deadline. Host startup is included in the allocation. An uncertain RPC consumes its reserved request;
an interrupted fit retains its exact completed optimizer-update count. A capped or failed experiment
cannot pass the utility gate, and the runner never extends its budget or retries automatically.

Wall checks are cooperative, before each dispatch or optimizer window and after completion. One
in-flight operation may cross the ceiling; its measured overrun remains reported and disqualifies the
run. If the executor disappears before summary finalization, completed immutable artifacts form an
explicitly incomplete prefix. Missing attempt/update receipts leave an unknown tail; that prefix cannot
be relabelled as a completed or fully accounted capped experiment.

## Unchanged decision and evidence boundaries

The [pure prerequisite evaluator](../../src/nca/train/goal_frame_config.py) requires complete exact
pairing, finite outcome evidence, mean family success of at least **0.75**, every seed at least **0.50**,
mean gain over its own initialization of at least **0.125**, and a strictly positive lower 95% bound
from all 27 length-three policy-seed resamples. Every seed retains the original character, vigor,
ordinary-trim, inference, integrity, admission, lifetime and uncensored-evaluation requirements. The
inference limit remains five milliseconds at p95 on every final evaluation scenario. Both candidate
families and every failure remain in the report whether either qualifies or not.

The [runner](../../src/nca/train/goal_frame.py) admits every executed source plus the launcher and
declaration only when tracked and byte-identical to the cited published Git tree. It saves the complete
input manifest before opening Godot. Captured episodes must match that source identity and the pinned
actual Jolt profile. Every original packet, admitted/consumed input, six-tick interval, terminal owner
revocation and same-tick cancellation is audited. Native terminal action readback remains explicitly
unavailable; no post-revocation actuation is invented. Cancellation preserves measured physics while
clearing requested motor targets.

Checkpoint files are rechecked against their exact size, file hash, policy hash and complete seed,
phase, source, dataset and frame metadata before load and holdout opening. Evaluation leaves the policy
state unchanged and rechecks original artifacts and source bytes afterward. Partial episodes,
unconfirmed requests, unknown tick counts, capped fits and errors remain separate in the immutable
summary. Neither a missing row nor malformed source evidence becomes a passing family.

Preparation is read-only and imports neither Torch nor a host:

```sh
PYTHONPATH=src python tools/direction_frame.py prepare
```

After the complete declaration and source are public, the single coordinated CPU invocation is:

```sh
PYTHONPATH=src python tools/direction_frame.py run \
  --host ../cybersand-host \
  --godot ../godot-runtime/Godot_v4.7-stable_linux.x86_64 \
  --output jolt-direction-frame-evidence
```


## Formal run result

The single declared actual-host invocation ran on GitHub Actions run
`39323820599` from published source `f0c4c1aa4ad30e972ebd406ca8eba78f96ef99ed`.
It completed without an execution error in **385.969869986 seconds**, recording
**4,660 confirmed game requests**, **27,960 physical ticks**, one host start and all
six fixed 40-epoch fits (**11,760 optimizer updates**). There were zero unconfirmed
requests, zero unknown tick counts, no automatic budget extension and **zero
morphology-transfer evaluations**.

The prerequisite result is a **no-go**. Direction-frame BC reached **0.3333333333**
mean success with **0.25** minimum seed success. Its mean gain over its own initial
policies was **0.25**, with a 95% interval of **[0.1520833333, 0.3479166667]**, but
it failed the unchanged 0.75 mean / 0.50 every-seed thresholds and the aggregate
character/runtime gate: seed 7103 failed orientation and seed 7301 failed launch
character. World-frame BC reached **0.25** mean success with **0.0** minimum seed
success and a gain interval crossing zero. The authored baseline scored 0.50 and
idle scored 0.0 on the same sealed cases.

Accordingly the immutable evaluator records
`mechanical_utility_prerequisite_passed=false`, `selected_family=null` and
`learned_morphology_transfer_measured=false`. The original T105 entry result is
unchanged. No family was promoted, no transfer run was opened, and owner feel remains
unmeasured. This is the declared research outcome, not a reason to retune, retry or
weaken the prerequisite.

The complete 433-file evidence artifact is Actions artifact `12220749995`
(`t106-direction-frame-evidence`, **18,749,182 bytes**, ZIP SHA-256
`fb008148c65f1e51887b31e4fae2987752fa58ea5c67fc13659bd5ec134b7510`).
The exact `summary.json` file SHA-256 is
`6098e8114de26c9b17d0b6c13fee73b40697549f02311ab61b326e7173e777e5`;
its canonical self-hash is
`5b721e6f6b2373a18247ab86869d1106279a152df66859fd5f8e8906452df3ff`.
A compact immutable receipt is retained at
[t106-direction-frame-result-20261019.json](t106-direction-frame-result-20261019.json).

## Preparation validation and remaining work

The pinned Python 3.12.14 / Torch 2.8.0+cpu / NumPy 2.3.5 environment passes **85 tests** across the
original learning, morphology and new direction-frame suites, including 15 new coordinate, checkpoint,
source, receipt, budget and whole-family gate checks. Two already-recorded original T105 **training**
episodes also passed the new read-only boundary audit: one 15-decision arrival plus cancellation and one
40-decision intrinsic terminal. That compatibility check collected no new episode and read no sealed
evaluation outcome. Fixtures and unit fits are software evidence, not a useful-skill result.
Independent source review confirmed the fixed sign/history and holdout ordering and identified the
untracked-source and checkpoint-file substitution paths. Both now refuse before their evidence can be
admitted; the focused regression suite covers those corrections.

A future separately commissioned prerequisite would need to pass the unchanged gate before it could support a separate explicit morphology continuation declaration. It
would still require a qualified semantic waypoint teacher, the unchanged conditional padded/graph
comparison allocation, actual zero-shot and adaptation evidence, and independent result review. The
original admission runner stays blocked. Unsupported independent geometry/damage perturbations,
Windows runtime qualification and owner feel remain unmeasured.

© 2026 techrote. All rights reserved.
