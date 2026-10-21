---
title: LEARN-JOLT-002 — learn a useful Jolt skill after recovery qualification
kind: reference
summary: Conditional imitation and residual-control pilot with learned launch authority, fresh geometry/start holdouts and unchanged reliability gates.
owns: [learning.jolt002-design]
verified: 2026-10-21
---

# LEARN-JOLT-002

This is the experiment design commissioned by D-015 and tracked in
[LEARN-JOLT-002 / #62](https://github.com/techrote/neural-cyber-automata/issues/62).
The [v4 recovery result](../evidence/2026-10-20-jolt-recovery-v4.md) satisfies its
[Jolt teacher #48](https://github.com/techrote/neural-cyber-automata/issues/48)
and [complete diagnostic evidence #43](https://github.com/techrote/neural-cyber-automata/issues/43)
dependency. Versioned implementation, fresh splits and source/runtime/budget
admission preceded the single allocation at clean source
`a68e687ced52bef778375a2753b1362dea3704fa`, with a 182-file source closure.
Later evidence publication preserves that original execution identity.

**The allocation was consumed and failed.** The [retained evidence](../evidence/2026-10-21-jolt-learn-002.md)
contains collection and fitting records, five of six final checkpoints, and zero
open or sealed evaluation rows. Independent readback found only 6,720 original
owner-journal records against the worker's claimed 24,195. Complete physical
evidence and terminal execution receipts remain missing; paired learned
reliability is unmeasured and no family is qualified. #62 remains open and
blocked. No retry, resume, tuning or extension is admitted by that declaration.

The original design and acceptance requirements below remain unchanged. Earlier
open diagnostic cases remain open, and the authored teacher result supplies no
learned qualification or replacement for the failed
[morphology prerequisite](../evidence/2026-10-19-t106-direction-frame.md).
Morphology #16 remains blocked. A further physical experiment would require a
separately commissioned prospective allocation.

The [implementation checklist](LEARN_JOLT_002_IMPLEMENTATION.md) binds complete
learned actuation, action/history masks, immutable invocation time, zero-correction
residual parity and the historical prospective cost and split checks.
The separate [execution runbook](../LEARN_JOLT_002_RUNBOOK.md) records the v2
contracts, exact counters, static split limitations and original freeze/run
procedure for the consumed declaration.

The first [recovery diagnostic](../evidence/2026-10-20-jolt-recovery-v1.md)
stopped after 24 of 96 rows. A sampled settled estimate retired its lease before
the per-tick task had twelve quiet ticks. Its seven complete matched pairs also
show restored short-left airborne launches but a peak-speed ratio of 0.58924;
the full character/reliability gate remained unmeasured in that attempt.
The [separate complete v2 result](../evidence/2026-10-20-jolt-recovery-v2.md)
verifies all 96 rows and repairs the advisory lifecycle, but fails teacher
qualification: 25/32 arrivals versus 27/32, two historical rescues instead of
three, the unsupported/drop failure unresolved and four regressions instead of
at most one. Airborne character is restored in 32/32 cases, and the complete
0.727556 vigor ratio passes. Motion response is slower despite sub-millisecond
worst-case inference p95. These findings prioritize arrival timing and physical
recovery over architecture work. The [v3 result](../evidence/2026-10-20-jolt-recovery-v3.md)
improves to 28/32 arrivals but retains only two historical rescues, one regression
and failed final inventory verification. These earlier results remain unchanged.

The separate fourth diagnostic for [#64](https://github.com/techrote/neural-cyber-automata/issues/64)
passes all thirteen unchanged teacher checks: **30/32 arrivals versus 27/32**,
three historical rescues, zero success regressions and 32/32 purposeful-airborne
cases. The [original result](../evidence/jolt-recovery-v4-result-20261020.json)
and [independent complete review](../evidence/jolt-recovery-v4-review-20261020.json)
bind one complete 96-row physical invocation. Two leftward failures remain;
`reference-floor-left-63` ends 12.503336 cells from target versus v3's 8.131480.
The [four-case comparison](../evidence/jolt-recovery-v4-case-comparison-20261020.json)
does not establish consistent useful landing progress after the second coast.
Owner feel remains unaccepted. All 32 cases were already open; these results
release only the teacher dependency and establish no sealed learned skill.

The qualified reference is `authored-jolt-recovery@4` at frozen physical source
`46a22f98dce78be3c4ba534d955e58d5691c7d96`, with candidate specification SHA-256
`2a04924885e3147881fef5ed3421c7c0e43457faa83e5c13b8e3c351c0384038`
and canonical diagnostic declaration SHA-256
`bc3cf6843d670a9305fa2f70bfa7531c865fa05363f3d55c88f8c554e520d30c`.
The prospective learner must bind these exact teacher, supervisor and evidence
identities from the [manifest](../evidence/jolt-recovery-v4-manifest-20261020.json)
and complete review. A later publication commit does not replace the frozen
physical source. This teacher qualification does not relax the learned reliability,
gain, character, runtime or morphology continuation requirements below.

## Question and comparison

Can a small recurrent policy learn expressive directed hops and contact recovery
with sufficient arrival reliability to satisfy the existing per-rig gate?
Compare a fixed improved authored reference, behavioural cloning (BC), and
recurrent residual PPO. Retain one 32-wide GRU architecture family. New input
and output channels implement the required control authority; this is not an
architecture search. Choose the whole BC family first when it passes. Additional
residual computation needs a measured improvement over its own initial control.

The existing residual implementation only modifies movement, trim and effort;
it copies authored launch buttons. Its fixed aim mapper also cannot reproduce
the recovery teacher's steeper pulse directions. Preserve those v1 source files,
artifacts and results. Add a separate versioned learner, action mapping and
bundle contract rather than silently changing historical semantics.

## Policy and physical supervisor responsibilities

| Ordinary learned control | Minimal physical supervisor |
| --- | --- |
| Direction, speed, posture and momentum correction | Invalid or unavailable state |
| Launch timing, launch direction and bounded effort | Lost authority, retired context and expired commands |
| Contact/edge recovery and landing adjustment | An explicitly requested, observed and bounded self-right emergency |
| Expressive movement and imperfect settling | Neutral refusal and required native cancellation at a boundary |

The supervisor must not supply a target trajectory, ordinary hop schedule or
contact-recovery action. It cannot renew the source observation or goal deadline.
Use the same frozen supervisor, sensors, cadence, native limits and assists for
matched controllers. Record each requested action, any supervisor override,
native admission and actual actuation. Override samples remain useful evidence;
they do not receive an ordinary policy likelihood or an unmodified imitation
label. Report autonomous outcomes separately from emergency-assisted outcomes.

The proposed direct action schema has five continuous channels (`move_x`,
`trim`, `effort`, `aim_x`, `aim_y`) and two learned button channels (`primary`,
`secondary`). Define the finite range, unit-aim normalization, zero-vector rule
and deterministic evaluation thresholds before collection. Auxiliary self-right
is emitted only by the declared emergency supervisor. The policy receives only
permitted own-actor observations, availability masks, its previous requested
action and explicit caller goal/deadline. Add measured contact/edge inputs with
their masks; no evaluator progress, success flag, terrain map or future receipt
may enter the network.

BC learns the complete eligible control sequence, including unsuccessful
trajectories and the absence of pulses. The residual controller corrects a
frozen, independently trained BC base for the same policy seed, including aim
and launch decisions. It does not call the scripted recovery teacher at runtime.
Freeze continuous residual ranges and the discrete-logit correction authority;
the latter must actually permit withholding or initiating a launch. Preserve
the sampled continuous latent and Bernoulli decisions for PPO likelihoods before
normalization, clipping or host admission. Keep recurrence, padding, burn-in,
duration and intervention rules explicit and covered by negative controls.

## Three independent training seeds

Use three independently initialized policy/optimizer/RNG streams, with proposed
policy seeds `190103`, `190207` and `190301`. Freeze the final seed rules in the
execution manifest. Both learned families may use the same fully recorded
teacher dataset; no successful-seed selection or best-checkpoint search is
permitted. Maintain independent recurrent state for the frozen BC base and
the residual module and clear both at every relevant lifecycle boundary.

BC starts from its recorded random initialization and ends at epoch 40.
Residual PPO starts from its recorded BC base plus the initial residual module,
then ends at the predeclared final update. Its base is already trained; do not
call this an untrained policy. Freeze residual initial artifacts before any PPO
updates. Report initial-to-final improvement for each family/seed. BC-final
also provides the direct learned/zero-residual comparison. Freeze all six final
checkpoint hashes before opening any sealed-case outcome, including the authored
or idle reference on those cases.

## Fresh geometry and starts

Create a hashed inventory of all previously opened geometry/start/target cases,
including failed, partial, export, cellular and diagnostic runs. At minimum this
covers T-103, T-104, T-105, the direction-frame follow-up, the original arrival
diagnostic and every failed or completed recovery diagnostic, including v1/v2/v3/v4.
All 32 v4 cases belong in this opened inventory, including the eight historically
labelled evaluation cases.
An old evaluation label does not make a case sealed again. Reject any overlap
before host construction and record the inventory/source identities.

The accepted host varies only five small sand-pile heights through `layout(seed)`
and accepts initial actor positions. A new numeric seed alone is not a new
geometry family. Hash the realized layout and contact-relevant local terrain,
then hold out whole layout/encounter families and new starting regions. Separate
whole starts/approaches as well as target distances; do not create a held-out
claim from different names for the same physical setup. Keep drop/landing and
edge/contact cases in both the task definition and the declared coverage matrix
while using disjoint physical arrangements across splits.

Prospectively allocate up to 64 demonstrations, four open development cases and
16 sealed paired evaluation cases. Exact cases, source-of-randomness rules,
realized geometry fingerprints and split membership must be frozen before
collection. Establish candidate geometry from read-only host layout inspection;
do not open sealed rollout outcomes to check whether they are easy. If the
existing API cannot supply the declared geometric separation, record that
limitation and narrow the claim or commission a separate host task before
execution. Arbitrary platform mutation, forced support removal and injected
disturbances remain unavailable in this host version.

## Prospective compute envelope

This proposed allocation is for the subsequent learning commission, not a
second Jolt diagnostic or an implicit running experiment. The final declaration
must bind the accepted teacher/evidence hashes, exact cases and environment.

| Work | Maximum six-tick decisions |
| --- | ---: |
| 64 teacher demonstrations, at most 40 decisions each | 2,560 |
| PPO, three seeds at 2,048 decisions each | 6,144 |
| Four open cases, eight fixed controllers, 40 decisions each | 1,280 |
| 16 sealed cases, authored + idle + initial/final for both families and three seeds | 8,960 |
| Planned total | 18,944 |
| Hard aggregate ceiling | 20,000 |

Use one CPU invocation and one persistent actual host, at most 120,000 requested
simulation ticks and a proposed 1,800-second whole-process ceiling. This is a
ceiling, not a prediction of completion. Startup, training, serialization,
durability and final verification are inside it. No GPU, new compute, paid
service, automatic retry or budget extension is included. An incomplete outcome
stays incomplete. Counters must include every invocation and uncertain request.

The execution specification must also fix optimizer-call caps, epochs, unroll,
minibatches, host-request and episode ceilings, thread count, and cleanup reserve.
Begin from the existing BC epoch-40 / PPO eight-update, four-epoch configuration;
derive exact optimizer-call ceilings from the frozen data shape. Reserve native
configure/reset/profile/cancel/close calls independently from simulation ticks.
These unresolved declaration details are implementation gates, not unbounded
work. Preserve failed and partial data together with source and budget receipts.

## Unchanged qualification requirements

The [original entry declaration](../../configs/morphology/entry-gate-v1.json)
and [direction-frame gate](../../configs/first-skill/jolt-direction-frame-v1.json)
remain unchanged. Bind the later experiment through a new versioned prerequisite
record; never edit their historical source or substitute this teacher result.

| Requirement | Threshold or rule |
| --- | --- |
| Mean final arrival across three policy seeds | At least 75% |
| Arrival for every seed | At least 50% |
| Mean gain over the same family's initial checkpoint | At least 12.5 percentage points |
| Policy-seed gain uncertainty | Strictly positive lower 95% bound from all 27 three-seed resamples |
| Complete paired outcomes | Every declared case and seed; no censoring |
| Integrity, failed admission and authority/lifetime regressions | Zero |
| Mean ordinary trim reversals, each seed | At most authored +0.5/s |
| Mean absolute angle, each seed | At most authored +0.1 radians |
| Mean peak toward-speed ratio, each seed | 0.7–1.5 of authored reference |
| Purposeful airborne launch fraction, each seed | Loss no greater than 0.125 against authored |
| Inference p95, every case | At most 5 ms |

Retain the original improvement-over-authored assessment separately, including
its declared high-baseline capped-completion alternative. A better teacher does
not waive learned improvement, reliable control or character. Show each gate's
numeric inputs, not just a passing Boolean. Use independently trained policy
seed as the uncertainty unit; demonstration repeats and individual time steps
are not additional independent policies.

Report actual impulse and torque, airborne ticks, primary/rebound requests and
physical confirmations, landing/support changes, recovery outcomes and progress,
response latency, chatter, supervisory interventions and complete wall cost.
An entered recovery phase is not a recovered case. The host reports aggregate
actor actuation but no separate passive/contact force decomposition; retain that
availability distinction. Owner feel and desktop pacing remain separate
assessments.

## Result and morphology continuation

If teacher/evidence admission fails, or neither learned family passes all gates, retain the
negative result and its complete evidence; do not select another seed, change
the split, extend the budget or quietly run again. The design can be revised
through a separately commissioned increment with a new prospective allocation.

If a whole learned family passes, publish exact source/runtime/dataset/checkpoint
identities, independently recompute the gate and bind the accepted result into
a separate T-106 prerequisite/continuation declaration. Only then resume
[morphology #16](https://github.com/techrote/neural-cyber-automata/issues/16),
using the already prepared masked recurrent and graph comparison with unseen
part/connection arrangements. Preserve its independent profile qualification,
matched budgets, adaptation/recovery/latency metrics and simpler-policy preference.
