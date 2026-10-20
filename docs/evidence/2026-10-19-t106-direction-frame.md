---
title: Fixed Jolt direction-frame prerequisite follow-up
kind: evidence
summary: Separate commissioned per-rig BC experiment with fresh policy seeds and sealed geometry; the designated local run and later concurrent GitHub run both failed the unchanged prerequisite.
owns: []
verified: 2026-10-20
---

# Jolt direction-frame prerequisite follow-up

The renewed instruction to complete the repository authorizes this separately declared useful-skill
follow-up. The [original T105 result and T106 entry decision](2026-10-19-t106.md) remain immutable and
blocked. The designated primary experiment is complete and failed the unchanged gate. The
[combined provenance record](t106-direction-frame-result-20261019.json) retains both executions and
their aggregate budget deviation. Its purpose was to test one coordinate
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

The completed invocation used this command shape after its declaration and source were public:

```sh
PYTHONPATH=src python tools/direction_frame.py run \
  --host ../cybersand-host \
  --godot ../godot-runtime/Godot_v4.7-stable_linux.x86_64 \
  --output jolt-direction-frame-evidence
```


## Formal run result

### Designated primary execution

The designated formal invocation was the original local run from public source
`1222d2e8dc30b87bb58de2326711fc3df361ac5e`. It was already running before a
later push-triggered GitHub workflow was introduced. The local invocation completed
with confirmed exit 0 and a complete immutable summary: **136 rows**, **12 frozen
initial/final checkpoints**, **4,693 attempted and recorded game requests**,
**28,158 confirmed physical ticks**, zero unconfirmed requests, zero unknown tick
counts, one host start and all six fixed 40-epoch fits (**11,760 optimizer
updates**). Its internal measured wall time was **943.605670498 seconds**, within
the unchanged 1,800-second ceiling.

The primary prerequisite result is a **no-go**. Direction-frame BC reached
**25% mean success / 12.5% minimum-seed success**; world-frame BC reached
**29.17% mean / 0% minimum**. Both are below the unchanged **75% mean / 50%
every-seed** reliability gate, and retained style/latency failures prevent an
independent character/runtime pass. No family is selected and **zero
morphology-transfer evaluations** are authorized from this result. The original
T105 entry decision remains unchanged.

The complete primary archive is `t106-direction-frame-evidence-v1.tar.gz`,
**18,708,060 bytes**, SHA-256
`291924eeedea0ba23ec365a9fb0f6e7ae8560eb614e53586070b3c6e64274fd4`.
Its archive index records `run/summary.json` SHA-256
`574a51108903484977629804bc9be9a54d348b8bf929c594eeee169584a30f78`,
summary-content SHA-256
`ed1c999a935a5a308f6d63ca64676b15000e4ecee511ccb8df10a0f0fe23a5e9`,
and an index self-hash of
`11b94d6d53c77cdc6d1efdf452d9ca1f0d867907b6224d70429d306b6e103443`.
The archive contains **438 verified members**.

The [primary-only detailed index](t106-direction-frame-result.json) retains the complete original
family evaluation and all twelve checkpoint receipts. Direction-frame mean gain over initialization
was 0.166667 with lower 95% bound 0.027083; world-frame gain was 0.208333 with lower bound -0.070833.
Every final policy seed failed its per-scenario latency gate, and one seed in each family failed
purposeful-launch retention. Source, owner, integrity and uncensored-evidence checks passed.
The primary local run overlapped a separately coordinated staged export-candidate evaluation, so no
uncontended latency qualification or isolated timing cause is claimed. The reliability thresholds
failed independently of latency. Removing a latency concern would not admit either family.

A 20 October read-only recovery audit found the available TAR copy truncated at
18,451,815 bytes. Its observed hash differs from the original container identity
above; that damaged copy is not a valid complete archive. Every one of the
438 originally indexed member files survives and matches its original byte
length and SHA-256. The original index and summary identities also match.

The separate transport archive `t106-primary-evidence-recovered-20261020.tar.gz`
contains those unchanged 438 files, the unchanged original index and a new
transport-recovery receipt: 440 members, 18,730,255 bytes, SHA-256
`c6a857e7fdc036939b47a8861d48bb0bae2cc57935fd878cb2c082c193b31fb1`.
The [compact recovery record](t106-primary-archive-recovery-20261020.json)
preserves both container identities and the complete member audit. This is
packaging recovery only, with zero host starts, game requests or optimizer
updates. It does not change the primary selection, failed result, original
budget accounting or any morphology admission gate.


### Concurrent alternate execution

A later concurrent change added a push-triggered GitHub workflow after the primary
local run had already begun. That separate invocation ran as Actions run
`39323820599` from publication
`f0c4c1aa4ad30e972ebd406ca8eba78f96ef99ed`. It is retained as a separate observed
execution, not as a restart, continuation or replacement of the primary run.

The alternate recorded 4,660 requests / 27,960 ticks in 385.969869986 seconds.
It also returned a no-go: direction-frame BC 33.33% mean / 25% minimum-seed
success and world-frame BC 25% / 0%. Its complete Actions artifact is
`12220749995`, 18,749,182 bytes, ZIP SHA-256
`fb008148c65f1e51887b31e4fae2987752fa58ea5c67fc13659bd5ec134b7510`.
The alternate result was not used to select, tune or override the designated
primary outcome.

The two executions consumed **9,353 confirmed requests and 56,118 ticks**, with
**23,520 completed optimizer steps and two host starts**. The request total
exceeds the original **6,000-decision allowance**. No additional allocation was
declared by the original execution thread before the concurrent run. The two
internal timers total 1,329.575540484 seconds, but omit separately scoped startup
and outer-process overhead; they do not establish complete aggregate wall-budget
compliance. Each original per-invocation receipt remains unchanged. The extra
execution is a budget deviation, not evidence that the original allocation was
respected or a basis for selecting a different prerequisite.

Both retained executions record the same bounded conclusion:
the unchanged direction-frame prerequisite was **not met**. No morphology family
is promoted, no transfer experiment is opened, and owner feel remains unmeasured.

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

## Guarded comparison source

The [conditional morphology preparation](2026-10-19-t106-learning.md) now includes distinct complete
local-input capture, masked recurrent BC/PPO, immutable checkpoint lineage, a separate semantic waypoint
teacher, a guarded orchestrator and the exact 148-row comparison summary. Its expanded software suite
passes 99 tests with no skips. The development admission verified the actual primary files, source
and episode/metric/checkpoint links, then returned the expected failed-gate refusal in 282.603 seconds.
The subsequent public-source admission at `eb488674475e3d13e8e6e4511ddfbe66192552bb` completed in
217.304178604 seconds with the same refusal, CLI exit 2 and zero host, model, checkpoint, policy, game
or optimizer work. Its exact source and receipt hashes are recorded in the
[conditional preparation evidence](2026-10-19-t106-learning.md). Torch and the morphology capture,
learner and checkpoint modules remained absent. Source verification imported generic
`nca.data.capture` and host adapter definitions; it constructed no host.

The separate [selection declaration](../../configs/morphology/prerequisite-selection-v1.json) binds the
designated public `1222d2e8` source, manifest and exact original summary bytes. A valid concurrent repeat
cannot replace that primary input, and no outcome-based choice or CLI bypass exists. The source
preparation changes no original experiment record or allocation and starts no morphology qualification,
teacher collection, optimizer, transfer or adaptation. The measured acceptance remains unmet.

© 2026 techrote. All rights reserved.
