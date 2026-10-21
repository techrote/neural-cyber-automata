---
title: LEARN-JOLT-002 — bounded execution runbook
kind: reference
summary: Versioned learned Jolt contracts, fixed static splits, prospective counters and the single-invocation freeze/run procedure.
owns: [learning.jolt002-operations]
verified: 2026-10-21
---

# LEARN-JOLT-002 runbook

The [#62 design](design/LEARN_JOLT_002.md) and [admission checklist](design/LEARN_JOLT_002_IMPLEMENTATION.md) define the unchanged scientific and acceptance requirements. This page preserves the separate v2 implementation and original bounded execution procedure. Published source reviews and the prospective declaration admitted one invocation at clean source `a68e687ced52bef778375a2753b1362dea3704fa`, with 182-file closure SHA-256 `d9754ca327bae965c7a50d213bb44171fdce4212a71ba13afb943c8126bdc727`. Later evidence-publication commits preserve that physical source identity.

**The original allocation was consumed and failed.** Its [retained evidence](evidence/2026-10-21-jolt-learn-002.md) includes five of six final checkpoints and no open or sealed evaluation rows. Independent readback found 6,720 original owner-journal records against the worker's claimed 24,195. Complete physical evidence and terminal execution receipts are missing, paired learned reliability is unmeasured, and no learned family is qualified. #62 stays open and blocked; morphology #16 stays blocked. Source tests, archive preservation and later publication do not supply those missing qualifications or reopen the consumed declaration.

## Admission records and source ownership

| Record | Purpose |
| --- | --- |
| [Experiment configuration](../configs/learning/jolt-learn-002-v1.json) | Architecture, losses, seed rules, exact cases, stage caps and every operation ceiling |
| [Teacher prerequisite](../configs/learning/jolt-prerequisite-v1.json) | Accepted v4 source/specification, supervisor, declaration, original evidence and independent review identities |
| [Opened inventory](../configs/learning/jolt-opened-inventory-v1.json) | Historical source-declared cases and conservative exclusion of old layouts and starting regions |
| [Static split](../configs/learning/jolt-static-split-v1.json) | Realized initial geometry, local contact fingerprints and start/approach identities for 64 training, four open and 16 sealed cases |
| Execution declaration produced by `freeze` | Exact clean source commit and file closure, configuration, runtime, host, two independent reviews and the single allocation |

The teacher remains `authored-jolt-recovery@4` at physical source
`46a22f98dce78be3c4ba534d955e58d5691c7d96`, with specification SHA-256
`2a04924885e3147881fef5ed3421c7c0e43457faa83e5c13b8e3c351c0384038`
and original canonical declaration SHA-256
`bc3cf6843d670a9305fa2f70bfa7531c865fa05363f3d55c88f8c554e520d30c`.
Its [accepted teacher result](evidence/2026-10-20-jolt-recovery-v4.md) includes
two leftward failures and no learned, sealed or owner-feel acceptance. Earlier
arrival and recovery v1/v2/v3 records retain their outcomes and consumed costs.

[Admission](../src/nca/train/jolt_admission.py) checks those byte identities and
all split invariants before host construction. The
[runner](../src/nca/train/jolt_screen.py) additionally binds all Python/GDScript
source, tools, tests, learning configuration and `pyproject.toml` to the frozen
source inventory. Review records and execution declarations are outside that
inventory to avoid a self-reference; their exact hashes are independently bound
by the execution declaration. Each accepting review needs a distinct identity
and must accept the same complete source inventory.

## Full learned control contract

The [new controller and bundle](../src/nca/controllers/jolt_learned.py) preserve historical source, artifacts and contracts.
One architecture uses 48 local features, a 32-wide `tanh` encoder, one 32-wide GRU
cell, seven actor outputs and one value output. No architecture or checkpoint search is admitted.

| Contract | Frozen behavior |
| --- | --- |
| Features | `jolt-local-goal-dual-history/v2`: local target delta, own motion/attitude, measured support/resources/rays/contact indicators with availability masks, original remaining time, goal pace/tolerance and separate previous proposal/dispatch histories |
| Continuous action | `jolt-five-continuous-two-buttons/v2`: `tanh` movement/trim/aim coordinates, sigmoid effort, then unit aim; aim norm at most `1e-8` maps to `(0, -1)` |
| Buttons | Independently learned primary and secondary; deterministic evaluation uses logit `>= 0`; native held/released edge semantics remain unchanged |
| Sampling | `jolt-pretransform-normal-bernoulli/v2`: five original Normal draws and two Bernoulli draws; likelihood is evaluated before the deterministic wire transformations |
| Invocation | Original caller goal, start/deadline ticks, actor lifetime, authority generation and topology identity survive all recurrent windows and interventions |
| Bundle | `nca-jolt-learned/v2`, including exact feature/action specification, policy and frozen-base identities; mismatched historical bundles are rejected |

Feature scales are constants; no evaluation observations fit normalization.
The network receives no evaluator success, progress, terrain map, future receipt
or teacher phase. Ordinary movement, aim, launch timing, quiet holding and
contact recovery are learned. No authored teacher runs inside the residual.
Auxiliary self-right belongs only to the shared physical supervisor: fresh
absolute wrapped angle at least 1.20 radians or absolute angular speed at least
4.5, with the unchanged bounded emergency declaration. Outcomes retain autonomous
and emergency-assisted counts separately.

The native task remains a four-second, 240-tick directed hop at pace 0.8 and
four-cell tolerance, requiring support, speed at most 6, angle at most 0.32 and
twelve consecutive quiet base ticks. Six-tick decisions use a twelve-tick helper
lease capped by the original observation and goal deadline. A sampled quiet
estimate is advisory; only the caller's per-tick task result retires intent.

## Evidence, history and numeric comparisons

[Versioned sample records](../src/nca/data/jolt_samples.py) retain the full source
observation and invocation, ordinary proposal or its absence, supervisor reason,
transmitted packet, admission, original draw and every native consumed receipt.
Proposal and actual dispatch have separate previous-action values and availability,
intervention and admission masks. An absent proposal supplies no invented neutral
target. Overrides, refusals, mixed or unverified consumption and deterministic
evaluation supply no ordinary PPO actor likelihood. Unsuccessful episodes and
abstention remain evidence and remain eligible when their ordinary labels are valid.

Native Vector2/JSON numeric action readback uses an absolute **`1e-6` tolerance**
per scalar. This tolerance applies only to original-input native action receipt
comparisons. Buttons and packet metadata remain exact; original policy draws,
proposal bodies, hashes and source/history identities are not rounded. The
per-tick `applied.action_available` and `applied.action` are required. A terminal
suspended view may expose neutral current input; it cannot replace the earlier
original-input actuation receipt or manufacture eligibility.

Physical episode order stays intact across fixed windows. Missing observations
and interventions apply explicit recurrence/loss masks. Padding performs no
state update; each window rebuilds its full preceding physical prefix under
current weights without gradients. Recurrent segmentation cannot renew time.
The residual and frozen BC base have independent recurrent states. Reward and
GAE use actual confirmed duration; task terminal and invalid final states do not
bootstrap, while an administrative rollout cutoff needs a valid received final
observation and value. A new update starts from a fresh physical episode.

## Frozen learning procedure

Policy seeds are `190103`, `190207` and `190301`, with independent initialization,
optimizer and RNG streams. BC initialization/window order use the policy seed;
residual initialization uses seed +100,000, action sampling seed +200,000 and
training-case order seed +300,000. PPO chooses only exact declared teacher cases
and their host seeds, with replacement. All families share the recorded teacher
dataset. Neither open nor sealed outcomes alter fitting or checkpoint choice.

| BC rule | Fixed value |
| --- | --- |
| Endpoint / windows | Epoch 40; physical episode windows at 0, 16 and 32 as present; masks stay inside windows |
| Optimizer | Adam 0.001, gradient norm cap 0.5 |
| Loss | Movement/trim/effort MSE weight 1; normalized aim MSE weight 0.5 on ordinary requested-primary rows only; button BCE weight 0.25 |
| Class weights | Training negatives / `max(positives, 1)`, clamped to [1, 10]; no-positive secondary has weight 1 |
| Label limits | All eligible failures and abstention retained; zero positive secondary labels recorded explicitly; no demonstrated rebound-skill claim |

Each residual uses its own seed's frozen BC-final. Initial deterministic actor
corrections are exactly zero, preserving BC-final action parity. Continuous
corrections add `(.5, .4, .6, 1, 1) * tanh(residual)` to base latent values.
Base button logits are clamped to [-4, 4], then receive `8 * tanh(correction)`;
the residual can initiate or withhold either button throughout that range.
Initial Normal log standard deviation is -1 and its learned bounds are [-5, 0.5].
Stochastic exploration is separate from the zero deterministic correction.

| PPO rule | Fixed value |
| --- | --- |
| Endpoint / rollout | Eight updates per seed, 256 physical decisions per update; four epochs, unroll 16 |
| Optimizer | Adam 0.0003, gradient norm cap 0.5 |
| Objective | Ratio clip 0.2, value weight 0.5, entropy weight 0.001 in raw Normal5 plus Bernoulli2 space |
| Duration discount | Gamma 0.997 and lambda 0.99 per confirmed base tick |
| Identity replay | Authenticate original draws, base weights, source, features and complete recurrence before each update |

The [training module](../src/nca/train/jolt_learning.py) exports the exact BC/PPO
specifications; execution rejects a configuration that differs. Initial and
fixed-final artifacts are retained for every family/seed. Intermediate PPO
artifacts are evidence, not selectable candidates. All six final hashes freeze
before any sealed outcome, including authored and idle. Four open cases then
compare authored, idle and the six finals; sixteen sealed cases compare authored,
idle and all twelve learned initial/final controls.

## Static split scope and limitations

The accepted host has eleven fixed terrain rectangles and only five seeded
sand-pile heights, each in [3, 9]. Initial actor placement is available; arbitrary
platform edits, forced support removal and injected disturbances are unavailable.
The split claims separation of initial material layouts and local encounters
inside this one authored lab. It does not claim broad terrain generalization,
new mechanics or unseen dynamic trajectories solely from a new seed number.

The opened inventory binds 115 historical JSON sources, 186 source-declared case
entries and 6,242 conservatively enumerated old seeds, producing 5,195 distinct
five-pile layouts. It includes failed/partial, export, cellular, direction-frame,
arrival and every recovery diagnostic. Cases whose start is not localized use
a conservative exclusion region. This is a source-declared initial-geometry
inventory; it does not reconstruct every historical trajectory from raw archives.

Candidates were selected in ascending order from the finite seed range
1,950,000–1,953,999 using the pinned Godot RNG and static source only. No physical
rollout outcome was used. All 84 new whole layouts differ from one another and
the retained old layout set. A 96-by-48 composed-material crop, normalized around
the encounter anchor, additionally rejects seed-only or distant-pile differences.
Overlapping fixed sand beds are composed before hashing, so a buried pile change
cannot manufacture a distinct contact family.

| Split | Local geometry and starts | Coverage |
| --- | --- | --- |
| Training: 64 | Third pile, heights 3–6; start x 344–416 | Floor approach, elevated drop/landing and pile-edge starts |
| Open: 4 | Fourth pile, heights 7–8; start x 432–492 | Floor approach, drop/landing and pile-edge start |
| Sealed: 16 | Fifth pile, heights 7 and 9 over the fixed six-cell sand bed; start x 520–602 | Floor/bed approach, drop/landing and bed-edge starts |

The respective splits contain four, two and two local material families. Their
start regions and approach identities are disjoint, with a minimum sixteen-cell
separation criterion between split starts. These static coverage labels identify
prospective arrangements; actual landing, contact, recovery and success require
the native receipts. The admission code recomputes every fingerprint before any
host construction.

## Prospective costs and counter meanings

One CPU invocation, one persistent host and one Torch thread share an immutable
**1,800-second whole-process deadline** beginning at external executor entry.
The 120-second cleanup reserve ends ordinary work by second 1,680. Startup,
admission, collection, fitting, serialization, fsync, cancellation, termination,
independent replay and final inventory/readback all use the original clock.
This is a hard ceiling, not a prediction that all stages finish.

| Stage | Episode-start cap | Six-tick decisions | Requested ticks |
| --- | ---: | ---: | ---: |
| Teacher | 64 | 2,560 | 15,360 |
| PPO | 6,144 | 6,144 | 36,864 |
| Open | 32 | 1,280 | 7,680 |
| Sealed | 224 | 8,960 | 53,760 |
| Sum of stage caps | 6,464 | 18,944 | 113,664 |
| Aggregate backstop | 6,464 | 20,000 | 120,000 |

The conservative PPO episode cap permits one decision per episode. Unused
aggregate margin grants no additional cases, updates, retries or stages. Costs
are reserved before dispatch or computation and never refunded after uncertain
work. The durable owner journal precedes transmission. Unknown pending work
blocks further ordinary requests; closure operations retain separate capacity.

| Counter | Ceiling and unit |
| --- | --- |
| Attempts / host | One invocation, one host-start attempt/start and one termination |
| Owner requests | 38,338 total: 18,944 step + 6,464 reset + 6,464 profile + 6,465 cancel + one close |
| Configure / observe | Zero separate transport calls; native reset performs its internal configuration |
| Episode status | At most 6,464 started, completed or aborted; retained status determines the relevant terminal count |
| Tick evidence | Requested, confirmed, refused and uncertain tick counters each have a 120,000 ceiling; these dimensions can overlap and are not added as distinct physical time |
| Checkpoints | 36 writes, 48 loads, 15,744 attempted learned decision calls; a residual/base decision pair counts once even when supervision prevents its forwards |
| Inference / bootstrap / audit | 31,488 ordinary module forwards, 12,288 bootstrap module forwards and 37,888 audit module forwards |
| BC | 23,040 optimizer calls, 307,200 forward rows, 368,640 burn-in rows and 307,200 eligible sample uses |
| PPO | 24,576 optimizer calls and actor/critic sample uses; 24,576 forward rows and 958,464 burn-in rows per residual/base module; 6,144 pre-update replay rows per module |

BC's optimizer bound is `3 * 40 * 64 * ceil(40/16)`. PPO's conservative bound is
`3 * 8 * 4 * 256`; its burn-in bound permits 39 prefix rows per nonempty window.
The actual dataset/window shape is recorded before fitting. Full reservation
history and final artifact counts/bytes remain auditable; a failed or capped
invocation retains its charged work and partial artifacts.
The 15,744 learned-call ceiling conservatively exceeds the 14,784 possible stage
calls; it does not enlarge the independently enforced physical stage caps.

## Freeze and launch once

**The 2026-10-21 declaration has already been consumed. Do not launch or resume it.** The commands below document its original procedure. A new directory, publication commit or recovered artifact does not grant another attempt. Any further physical invocation requires a separately commissioned, prospectively admitted allocation; the original no-retry and no-extension limits remain in force.

Use the pinned [Linux host setup](HOST_ADAPTER.md#linux-setup) and the isolated
[CPU learner installation](LEARNING.md#install-the-optional-cpu-lane), with
Python `3.12.14`, Torch `2.8.0+cpu` and NumPy `2.3.5`. The declaration also binds
the public CPU model, kernel, machine, libc and thread settings. Linux ownership requires `/proc`,
pidfd support and child-subreaper cleanup. Portable software tests do not qualify
Windows physical execution. Store large output and the execution declaration
outside the source checkout. Replace uppercase arguments below with actual
runtime locations and new destinations; no private machine paths belong in Git.

After the exact source and two independent review records are published and its
CI passes, use the verified clean checkout to create its prospective declaration:

```sh
python tools/learn_jolt.py freeze --checkout HOST_CHECKOUT --godot GODOT_EXECUTABLE --declaration NEW_DECLARATION_FILE --review docs/evidence/IMPLEMENTATION_REVIEW.json --review docs/evidence/ADMISSION_REVIEW.json
```

`freeze` reads identities and constructs no physical host or fitted model.
Both reviews must contain `accepted: true`, distinct reviewer identities and
the exact current `source_sha256`. Keep the declaration unchanged.

Publish the declaration and CI receipt in a later documentation-only commit.
Keep the execution checkout at the clean source commit recorded by `freeze`;
later publication commits never replace that physical source identity.

Launch through the external supervisor, whose clock includes worker imports:

```sh
python tools/learn_jolt.py run --checkout HOST_CHECKOUT --godot GODOT_EXECUTABLE --declaration NEW_DECLARATION_FILE --output NEW_RUN_DIRECTORY
```

The output directory and its sibling ending in `-lifecycle` must both be new.
The internal `worker` mode belongs to the supervisor. Existing evidence is not
resumable, and a new pathname does not authorize another allocation. The parent
checks owned-process termination and independently reads back the closed output
inside the same wall limit. A nonzero return or incomplete receipt is retained;
there is no automatic retry, budget extension, seed replacement or split tuning.
The authoritative completion verdict is the executor's final JSON on standard
output, checked together with its exit code. It binds `executor.json` and
`finalization.json` by hash and measures wall time after their publication and
fsync. An earlier worker `completed` status or `completion_candidate` alone is
insufficient. Preserve this terminal output with the lifecycle receipts.

## Qualification and publication

[Qualification](../src/nca/evaluation/jolt_learning.py) requires every declared
paired result and verified evidence, then applies the unchanged gates:

| Gate | Required result |
| --- | --- |
| Final reliability | Mean arrival at least 75%; every policy seed at least 50% |
| Own-initial improvement | Mean gain at least 0.125 with strictly positive lower 95% bound from all 27 three-policy-seed resamples |
| Original authored assessment | Success gain at least 0.125 with positive lower bound; alternatively authored success at least 0.875, capped-time gain at least 0.15 seconds and 10%, no seed success regression and positive lower time-gain bound |
| Integrity | Complete pairs, zero failed admissions or authority/lifetime/integrity regressions, confirmed native and process closure |
| Character, every seed | Ordinary trim reversals at most authored +0.5/s; mean angle at most authored +0.1 radians; peak toward-speed ratio 0.7–1.5; purposeful-airborne fraction loss at most 0.125 |
| Runtime | Every final case's inference p95 at most 5 ms |

Report each numeric input and interval, actual impulse/torque, airborne motion,
button requests and physical confirmations, support changes, recovery progress,
latency, chatter, interventions and complete wall cost. Aggregate actuation does
not establish unavailable passive/contact-force decomposition. Prefer the whole
BC family when it passes; residual needs its own initial-to-final improvement.
A completed experiment can still be a negative result. Owner feel, desktop
pacing and deployment remain separate assessments.

Retain the complete raw archive outside Git and publish small hash-indexed
manifests, reviews, failures, checkpoints/dataset identities and costs under the
[run rules](RUNS.md). A qualified learned family still needs independent result
acceptance and a new T-106 prerequisite/continuation record. Teacher qualification
or source tests alone do not admit morphology #16.

© 2026 techrote. All rights reserved.
