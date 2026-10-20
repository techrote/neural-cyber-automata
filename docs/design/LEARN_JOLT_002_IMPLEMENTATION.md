---
title: LEARN-JOLT-002 — implementation admission checklist
kind: reference
summary: Conditional checks for complete learned actuation, honest action labels, fresh physical holdouts and bounded learning costs.
owns: [learning.jolt002-implementation]
verified: 2026-10-20
---

# LEARN-JOLT-002 implementation checklist

This appendix makes the [experiment design](LEARN_JOLT_002.md) operational.
It proposes implementation checks and budget arithmetic. It authorizes no
collection, fitting, checkpoint execution or transfer. Bind accepted recovery
teacher and complete evidence results, then publish the final source, runtime,
split and compute declaration before any learning execution.

## Admission and scope

- [ ] Accept the complete teacher and evidence gates for recovery
  [#48](https://github.com/techrote/neural-cyber-automata/issues/48), evidence
  [#43](https://github.com/techrote/neural-cyber-automata/issues/43) and advisory
  [#60](https://github.com/techrote/neural-cyber-automata/issues/60). Bind exact
  controller/supervisor specs, declaration, source, raw evidence and audit hashes.
- [ ] Preserve the failed recovery v1 experiment. The advisory v2 source approval
  alone supplies no physical reliability, character or evidence qualification.
- [ ] Retain one 32-wide GRU family, three seeds and the authored/BC/residual/idle
  comparison. Give changed feature, action, sample and bundle semantics new
  versions; keep historical implementations and declarations replayable.

## Six interface risks to close before collection

### 1. Full learned aim and button authority

The [existing mapper](../../src/nca/controllers/learned.py) derives aim from the
goal; its residual changes three continuous channels and copies authored buttons.

- [ ] Define five continuous outputs (`move_x`, `trim`, `effort`, `aim_x`,
  `aim_y`) and learned `primary`/`secondary` buttons. Freeze ranges, transforms,
  unit-aim normalization, zero-vector rule, thresholds and previous-action values.
- [ ] Test that both aim outputs reach serialized requests, including near-vertical
  recovery aims. The new mapper must not replace them with a goal-derived aim.
- [ ] Test initiating, withholding, holding, releasing and reinitiating primary.
  Respect native edge semantics without a scripted ordinary launch schedule.
  Test that secondary remains an independently selectable native request.
- [ ] Keep ordinary launch timing, contact recovery, posture and quiet holding in
  learned actuation. The wrapper supplies no teacher trajectory or hop watchdog.
- [ ] Reject old bundles under the new loader and new bundles under old contracts.
  If a direction frame is chosen, version its aim/history transformations too;
  the [old transform](../../src/nca/controllers/goal_frame.py) leaves aim unchanged.

### 2. Invocation time must survive recurrent segmentation

The [imitation converter](../../src/nca/train/imitation.py) sets `start_tick` at
sequence resets; [features](../../src/nca/controllers/learned.py) use that value
for remaining time. An intervention cut can therefore renew the deadline feature.

- [ ] Store immutable invocation identity, original start and absolute deadline
  separately from recurrent segment/window starts. Test cuts without time renewal.
- [ ] Match all authority, goal, actor/profile and lifetime fences. A fresh
  observation alone grants no takeover; retired contexts cannot be resurrected.
- [ ] Bind dispatch to its original observation and cap repeat by both remaining
  observation lease and goal deadline, including delayed dispatch/current clocks.
- [ ] Preserve the [v2 advisory lifecycle](../../src/nca/controllers/jolt_recovery_v2.py):
  sampled quiet estimates cannot retire intent. Only the caller validates the
  unchanged per-tick task outcome/current invocation and cancels local/native intent.
- [ ] Retain the aliasing regression: quiet samples 222/228/234 coexist with a speed
  violation at 223, leaving eleven quiet ticks through 234. Do not infer tick 235.
  Test valid advisory leases and renewed correction after fresh disturbance.

### 3. Proposal, dispatch, history and loss masks must agree

[Capture events](../../src/nca/data/records.py) distinguish replacement and
on-policy eligibility. The new adapter must preserve that distinction while
representing supervision and reconstructing recurrent history.

- [ ] Record policy-visible observation/context, ordinary proposal or its absence,
  supervision reason/source, dispatched packet, admission, consumption and physical
  receipts separately. Bind complete observation bodies, not only observation IDs.
- [ ] Define whether each previous-action feature means proposal or dispatch.
  Give separate fields/masks if both are supplied. Reconstruct exactly the same
  history in teacher conversion, BC, PPO, deployment and replay after an override.
- [ ] Freeze one shared emergency declaration predicate and
  [supervisor contract](../../src/nca/controllers/recovery_supervisor.py).
  Auxiliary belongs only to an explicitly declared, observed, bounded emergency.
  Report autonomous and emergency-assisted outcomes separately.
- [ ] Mask supervisor overrides from ordinary imitation and PPO actor loss;
  never label actual impulses or substituted commands as the original policy draw.
  An absent ordinary proposal must not become an invented neutral training target.
- [ ] Preserve ordered physical observations, previous-action availability and
  explicit reset/loss masks. Do not concatenate across missing observations.
  Padding never advances recurrence; burn-in rebuilds the actual prior history.
- [ ] Use actual consumed durations for reward/GAE. Distinguish task terminal,
  administrative cutoff and invalid final state; test their bootstrap masks.
- [ ] Retain unsuccessful episodes, failed launches, refusals and partial tails as
  evidence. Eligibility removes unsupported labels, never inconvenient outcomes.

### 4. Residual control needs a learned base and honest initial comparator

The current [runtime](../../src/nca/controllers/learned.py) instantiates an authored
base. The new residual must use each seed's independently trained, frozen BC-final.

- [ ] Prove that ordinary residual execution never calls a scripted controller.
  Maintain distinct BC/residual hidden states with declared lifecycle resets and
  history semantics. Freeze the base artifact before that seed's PPO updates.
- [ ] Initialize deterministic corrections to exactly zero. Verify initial
  base-plus-residual action parity with BC-final on identical observation histories.
  Freeze this initial artifact; its trained BC base is not an untrained control.
- [ ] Freeze continuous correction ranges and discrete-logit combination rules.
  Demonstrate both initiation and withholding across the admitted base-logit range;
  bounded corrections to saturated logits can otherwise leave buttons ineffective.
- [ ] Store actual continuous latent draws and Bernoulli decisions before aim
  normalization, clipping or admission. Freeze their joint likelihood space.
  Version the [three-latent transition/density path](../../src/nca/train/recurrent.py).
- [ ] Exclude deterministic means, replacements, mixed sources, refusals and invalid
  intervals from actor likelihoods. Negative controls must detect altered latent,
  button, mapping, base-checkpoint, source and recurrent-history identities.

### 5. Sparse launches and absent rebound labels need explicit treatment

The [recovery teacher](../../src/nca/controllers/jolt_recovery.py) always requests
`secondary=False`; scarce primary launches also make an undifferentiated aim loss
vulnerable to domination by arbitrary no-launch aims.

- [ ] Freeze continuous/button loss weights and class weighting using training data
  only. Train the complete eligible sequence, including button abstention.
- [ ] Declare whether aim loss is conditional on requested primary; fix the rule
  before collection. Never select aim labels by future impulse or arrival success.
- [ ] Record zero positive rebound demonstrations explicitly. BC can learn this
  abstention; useful rebound behavior would require residual exploration within
  its existing budget. Do not claim demonstrated rebound skill or add a fallback.
- [ ] Freeze residual sampling variance, Bernoulli exploration and entropy rules.
  Nonzero stochastic exploration may coexist with zero deterministic correction.

### 6. Epoch labels and a decision counter do not bound all work

The [screen loop](../../src/nca/train/screen.py) and
[window updates](../../src/nca/train/recurrent.py) make optimizer work depend on
sequence shape and host resets depend on episode length.

- [ ] Publish the exact batching, optimizer-call, episode-start and per-operation
  owner-request ceilings below before collection; unset ceilings block execution.
- [ ] Reserve costs before work/transmission, including refused and uncertain
  requests. Count all invocation attempts and enforce stage caps independently.
- [ ] Include startup, fitting, serialization, durability, replay and final
  verification in the whole-process ceiling, with a fixed cleanup reserve.
  A capped outcome stays incomplete without automatic retry or budget extension.

## Fresh geometry, starts and three independent seeds

- [ ] Freeze initialization, optimizer, action RNG, scenario-order and host-seed
  rules for policy seeds `190103`, `190207`, `190301`. Teacher data may be shared;
  policy/optimizer/RNG streams remain independent. No seed/checkpoint selection.
- [ ] Hash an inventory of every opened geometry/start/target case: T-103/T-104/T-105,
  direction-frame, export, cellular and **all arrival and recovery diagnostics**,
  including failed recovery v1, advisory v2 and every partial/capped attempt.
  Reject sealed overlap before host construction; old evaluation labels do not seal.
- [ ] Freeze whole-layout and contact-relevant encounter fingerprints, starting
  regions/poses and target approaches. A new seed or distant changed pile alone
  is insufficient. The host's five variable pile heights constrain the geometry
  claim; declare realizable disjoint landing/edge/contact coverage before collection.
- [ ] Establish geometry from static host layout inspection. Freeze up to 64 demos,
  four open cases and sixteen sealed pairs before outcomes; narrow an unrealizable
  claim before execution. No arbitrary platform or support-removal API is assumed.
- [ ] Fit normalization only on training data or use declared constants. Inputs
  contain permitted local observations/masks and caller context, never evaluator
  success, privileged terrain, future receipts or teacher phase/progress telemetry.
- [ ] Freeze all six final checkpoint hashes before any sealed outcome, including
  authored/idle runs. Open cases are diagnostic only, with no tuning or selection.

## Proposed exact cost arithmetic

| Stage | Maximum six-tick decisions |
| --- | ---: |
| Teacher: 64 episodes x 40 | 2,560 |
| PPO: 3 seeds x 8 updates x 256 | 6,144 |
| Open: 4 cases x (authored, idle, 6 finals) x 40 | 1,280 |
| Sealed: 16 cases x (authored, idle, 12 initial/final learned) x 40 | 8,960 |
| Planned stage total | 18,944 |
| Aggregate backstop | 20,000 |

The plan requests at most 113,664 ticks within a 120,000-tick backstop. The unused
aggregate margin grants no extra cases/retries. Use one CPU invocation, one host,
one Torch thread and a proposed 1,800-second whole-process ceiling; no GPU/new service.

- [ ] For BC, fix epoch 40 and unroll 16. With fixed windows over each complete
  physical episode (at most 40 decisions), and masks inside windows, the optimizer
  cap is `3 x 40 x 64 x ceil(40/16) = 23,040`. Publish resolved window/sample counts
  before fitting. Splitting interventions into new windows changes this bound and
  requires a prospectively recomputed cap; do not silently reuse this arithmetic.
- [ ] For PPO, fix eight 256-decision updates/seed, four epochs and unroll 16.
  With one optimizer call per nonempty window, each rollout has at most 256 windows:
  `3 x 8 x 4 x 256 = 24,576` is a conservative cap. A tighter bound needs declared
  episode/batching rules. Count optimizer calls and actor-eligible samples separately.
- [ ] Bind all remaining counters: started/complete/aborted episodes; reset, profile,
  configure, cancel, close and step requests; host starts/terminations; requested,
  confirmed/refused/uncertain ticks; checkpoint writes, loads and executions; BC/PPO
  forward, burn-in, bootstrap and audit replay work; thread count and cleanup reserve.
  Freeze exact ceilings from the implemented operation inventory before execution.

## Unchanged gates and result acceptance

Preserve the [entry declaration](../../configs/morphology/entry-gate-v1.json),
[direction-frame gate](../../configs/first-skill/jolt-direction-frame-v1.json) and
[original authored assessment](../../configs/first-skill/jolt-screen-v1.json).
Bind the new experiment through a separate versioned prerequisite record.

| Requirement | Threshold |
| --- | --- |
| Mean final arrival / every seed | At least 75% / at least 50% |
| Mean final-minus-own-initial arrival gain | At least 12.5 percentage points |
| Gain uncertainty | Strictly positive lower 95% from all 27 policy-seed resamples |
| Paired outcomes | Every declared case/seed, no censoring or duplicate pairs |
| Integrity, failed admissions, authority/lifetime regressions | Zero |
| Each seed's mean ordinary trim reversals | At most authored +0.5/s |
| Each seed's mean absolute angle | At most authored +0.1 radians |
| Each seed's mean peak toward-speed ratio | 0.7-1.5 of authored |
| Each seed's purposeful-airborne fraction loss | At most 0.125 against authored |
| Every final case's inference p95 | At most 5 ms |

- [ ] Retain the separate authored-improvement assessment: 0.125 success gain,
  or its original completion alternative: authored success at least 0.875,
  capped-time improvement at least 0.15 seconds and 10%, and no seed's success
  regression. Preserve a strictly positive lower 95% bound on the chosen measure.
  Report numeric inputs and both assessments; neither replaces the other.
  Independent policy seeds are the uncertainty unit, not timesteps.
- [ ] Independently bind executing audit/controller/model source, specs and weights
  to the manifest; replay proposals, supervision, outcomes and costs from original
  receipts. Require durable journals, refusal records and verified owner closure.
- [ ] Report actual impulses/torque, airborne ticks, button requests/confirmations,
  recovery progress/outcomes, latency, chatter, interventions and whole wall cost.
  Keep aggregate actuation distinct from unavailable passive/contact decomposition.
- [ ] Select the whole BC family first when qualified. Residual must pass its own
  initial-gain gate; a strong BC base grants no exception. If neither family passes,
  retain the negative result. Only a fully accepted learned result permits a new
  T-106 continuation with additional profile and unseen part/connection qualification.
