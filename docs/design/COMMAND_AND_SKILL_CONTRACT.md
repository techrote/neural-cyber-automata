---
title: Command sources, skills and actor execution
kind: guide
summary: Proposed extension of the existing Thunderbowl control boundary for human, learned and language-model control of articulated Cybersand actors.
owns: [design.command-skills]
verified: 2026-10-18
---

# Command sources, skills and actor execution

[PROPOSED] This is a research and interface design for the rechartered project.
It does not establish an implemented skill runtime, live model service, trained
controller or successful Rapier integration. ACT-004 mechanics remain owned by
[Cybersand issue #178](https://github.com/techrote/cybersand/issues/178).

The design targets plausible, stable and expressive gameplay. The common
authored actuator, assist and constraint profile may include simplified gyro
assistance, damping, compliant approximate connections and controlled pseudo
flex/play. It need not conserve finite gyro momentum or perfectly represent a
mechanism. Human and policy control receive the same configured behavior;
qualification checks that behavior against the target game, not physical reality.

## Preserve the existing boundary

[CURRENT external snapshot] The
[ACT-003 reference at b31a897](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/docs/systems/thunderbowl-control-reference.md)
already separates actuation lane, decision placement, perception privilege,
coordination, controller family and training privilege. Retain those independent
axes: a central planner need not be an oracle; shared weights need not share memory.

Its version-1 commands bind run, actor, epoch, sequence and issued observation.
Their 1–18-tick leases start at the observation tick; the helper requests 12 ticks.
Humans update at 60 Hz, scripted/reflex/NCA decisions at 10 Hz, and the motor and
material step at 60 Hz. The mock smart link allows one outstanding request, starts
no more often than every 30 ticks, and expires 120 ticks after the source observation.
Admitting a late reply never renews that age. Its fresh reflex issues motor packets.
Cancel disarms charge; hold and expired plans coast under passive physics, without
an implicit brake, hover or path solver. File polling is not asynchronous I/O.

[PROPOSED] Extend these contracts explicitly; do not reinterpret version 1.
The current four-channel metal profile and sampled motor are reference arms,
not already the articulated actuator manifest or Rapier evidence.

## One execution path, several decision sources

[PROPOSED] Separate five responsibilities:

1. A source interprets the task and proposes an intent, skill or permitted direct input.
2. Admission validates identity, observation provenance, capability and authority.
3. A local skill uses fresh permitted observations to generate wire or metal commands.
4. The actor owner resolves requests through the authored gameplay profile.
5. Completed physics/material steps publish observations and separate evaluator records.

Every source reaches the same actor owner. Controllers request behavior rather
than directly mutating bodies, terrain or hit results. The profile defines gains,
coordinate frames, authored effort limits, optional resource costs, recoil,
stabilization and connection behavior. Compensating torque or simplified grip
assistance is allowed when it belongs to that shared profile. Keep requested
channels, admitted channels, applied effort/assistance and resulting motion
distinct. Clamping is an observable result, not proof of command success.

| Source | Permitted entry | Interpretation |
|---|---|---|
| Human | Wire input, skill request or configured metal input | Manual decisions use the same gameplay profile |
| Script/behavior tree | Registered skills or configured direct policy channels | Auditable baseline and fallback |
| Learned local policy | Declared observations/history to configured channels | Small recurrent controller may run continuously |
| Decision model/team planner | Bounded actor-specific intents | Shared process does not confer extra perception |
| Frontier LLM | Asynchronous typed skill/mission proposals | Local feedback executes while the model is unavailable |

For example, a human clicking a visible object and an LLM choosing that same
observed object both submit `approach_and_grip(target_ref, mount_ref, tolerance)`.
The same implementation uses the configured drive and aim, checks authored
clearance rules and actuates the grip. Any compliant attachment or simplified
grip assistance is shared. A human can instead control the configured metal channels;
that is a different declared actuation comparison, not equal assistance.

[SUPPORTED] The options framework models a temporally extended behavior through
its initiation conditions, policy and termination rule. This supports a shared
skill abstraction independently of whether the skill is hand-authored or learned.
[Sutton, Precup and Singh, 1999](https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf).

## Minimal versioned extensions

[PROPOSED] Reuse existing fields before adding new ones. If the present epoch
already invalidates a lifetime, use it; add another revision only for a change
that can occur independently. Advertise a bounded actuator manifest per physical
rig, with stable part/joint/channel references and unsupported-channel rejection.
Do not infer actuator count from the eight-part budget or silently pad vectors.
Controller ownership is an internal routing record, normally an ID and generation;
it does not require a user permission workflow or a separate authorization service.

| Extension | Purpose |
|---|---|
| Source/session identity and owner-issued grant | Identify who controls which actor/resources; source text cannot assign its own priority |
| Grant fencing generation | Invalidate buffered commands and late replies after transfer or revocation |
| Actor incarnation | Reject a command for a destroyed/reused actor slot if epoch alone does not cover it |
| Morphology/profile revision | Reject outputs bound to another part, joint or actuator arrangement |
| Mount/object generation | Reject grip, release or trigger requests targeting replaced equipment |
| Skill instance ID, version and parameters | Bind progress, cancellation and final result to one invocation |
| Maximum run duration and dependency conditions | Bound a skill separately from the observation-age limit for starting it |

Validate at receipt and again when consuming the action at the simulation boundary.
One actor owner commits the final actuator frame per actor per tick. Start
with exclusive actor control; concurrent locomotion/aim/tool grants need explicit
resource footprints and conflict tests because joints and recoil couple them.
Keep the configured wire/metal lane fixed for the actor lifetime as version 1
requires; takeover does not silently change it.

Use exact invocation IDs and payload hashes for idempotent retries. The same ID
with different content is rejected; a repeated valid request returns its existing
receipt/result instead of firing again. Sequence checks also apply to buffered
outputs. Record the actual accepted order for replay.

## Clocks, freshness and long reasoning

[PROPOSED] Distinguish three quantities:

| Quantity | Clock and meaning |
|---|---|
| Observation/command age | Simulation ticks since the issued observation; preserve source anchoring |
| Skill duration | Simulation ticks since execution began; timeout even with healthy transport |
| Source liveness/service latency | Receiver monotonic wall time; never trust a remote clock for authority |

A paused or accelerated world makes these quantities diverge. Declare whether
wall-time liveness is active during pause and record that configuration. Training
must not silently pause physics while a slow model thinks and call it real-time control.

Keep the existing 120-tick smart-plan expiry. A future persistent `TaskSpec` is
a separate schema: distinguish observation-dependent requests from a standing
goal, specify latest start/expiry, and recheck relevant dependencies using fresh
permitted observations. Traveling to a last-seen point can remain meaningful;
assuming a target is still there and firing cannot inherit that assumption.
An adapter must not relabel a stale model decision with a fresh observation ID.

The physical loop never waits for inference. A service worker exchanges bounded
messages, while the existing request cap remains the initial baseline. Drop late
results even if remote cancellation fails. An invalid plan yields an explicit
fallback state. Active stabilization may use the configured simplified assist;
it must not appear implicitly for only one source or silently change hold semantics.

Local timeout, reset or takeover retires a request identity independently of remote
completion. Resolve its outstanding-slot state explicitly: report unavailable or
retry after bounded timeout/backoff. Keep actual in-flight backend work bounded;
retiring a local request must not accumulate unlimited abandoned model calls.
Fresh local control continues while the backend is hung or unavailable.

[SUPPORTED] ROS 2 separately specifies message lifespan, deadline and liveliness;
these are useful distinctions, not a requirement to adopt ROS or its middleware.
[ROS 2 QoS design](https://design.ros2.org/articles/qos_deadline_liveliness_lifespan.html).

## Skill outcomes and cancellation

[PROPOSED] Each registered skill declares prerequisites, observation profile,
resource footprint, success/failure predicates, progress measures, deadline,
interruption behavior and cleanup. Parameters are typed and bounded; the LLM
cannot supply executable predicates or arbitrary runtime code.

| Receipt/state | Exact meaning |
|---|---|
| `rejected` | No execution; return a specific identity, age, capability or precondition failure |
| `accepted` | Scheduled; no claim that an action was applied or the task succeeded |
| `running` / `degraded` | Executing; degradation identifies limits, missing observations or fallback |
| `canceling` | Future requests are revoked; bounded cleanup/result publication remains |
| Terminal result | `succeeded`, `failed`, `timed_out`, `canceled`, `preempted` or `invalidated` |

“Completed” means terminal, not necessarily successful. Preserve separate reason
codes such as target lost, no progress, authored effort limit, clearance blocked,
mount changed and world reset; energy failures apply only to profiles with that
resource model. Publish progress with its tick,
last applied action and affected effect/object IDs.

Takeover is an owner event applied at a simulation boundary: revoke the previous
grant, advance its fence, clear queued actions and charge, then admit the new
controller. Cancellation wins over a still-unconsumed trigger in that boundary.
Human takeover does not require waiting for an LLM acknowledgement. Resumption
requires a new grant and fresh observation; an old plan never resumes implicitly.

Usables distinguish held input, trigger edge, charge, release, grip and detach.
Canceling charge must not become release-to-launch. Cancellation stops future
tool requests; it cannot erase an already-fired projectile, undo deposited
material, or remove recoil already applied. Existing autonomous timers continue
under their own gameplay rules and remain visible in the result. Cleanup follows
the configured grip/release behavior; it must not invent an undocumented attachment
or correction that is available only to one controller source.

Cancellation clears pending input and filter targets at the next owner boundary;
it does not reset body velocity, loaded compliance, stance or play state. Those
follow the configured fallback/grip rules. A world reset separately reconstructs
the entire episode, including mechanical and recurrent initial state.

Success belongs to a predicate over observed execution, not the model's narrative.
Report `already_satisfied` for a no-op goal when appropriate. Score task-instance
completion once, so repeated success replies or reopening an identical skill
cannot earn reward repeatedly. Keep a policy-visible skill estimate distinct
from the privileged evaluator's task verdict; feeding that verdict back grants
an additional observation channel.

[SUPPORTED] ROS 2 actions distinguish acceptance, execution, cancellation and
terminal results. Its lifecycle is a useful starting point for this proposal.
[ROS 2 actions](https://design.ros2.org/articles/actions.html).

## Partial observations, assistance and training records

[PROPOSED] Carry observation identity, sampling tick, field validity and remembered
track age into model requests. Filter before summarization, scene-graph construction
or team aggregation. Hidden actors must not continue updating in remembered tracks.
Camera centering, overlays, reward feedback and success detectors count as sensors.
Privileged evaluator snapshots never become an undeclared planner input.

Offer assistance as explicit modes: manual, user-selected skill, consented shared
assistance, or autonomous. Begin with exclusive control and rapid takeover.
Later continuous blending is an experiment: blend compatible actuator requests
before the same authored profile, log both proposals and the mixture, and never average
discrete fire/grip/release events. Confidence alone must not seize authority.

[SUPPORTED] Shared-autonomy research models uncertain user goals and shows that
faster task completion can trade against perceived control authority. Evaluate
both assistance performance and human intervention burden.
[Javdani et al., RSS 2015](https://roboticsproceedings.org/rss11/p32.html).

Log source/model/checkpoint and configuration hashes, observation profile,
training privilege, source proposal, admitted plan, local policy output, applied
actuation, human correction, termination reason and task outcome. Preserve
intervention and model-override labels; a blended trajectory is not automatically
an expert demonstration. Reset recurrent state on the corresponding lifetime
change, including actor reuse and incompatible morphology changes.

[SUPPORTED] SayCan grounds skill choice with contextual feasibility estimates;
RT-H uses an intermediate language-motion layer that supports human corrections;
Inner Monologue studies environment feedback for replanning. These justify
experiments with grounded skill selection and correction, not claims that a
frontier model can pilot these rigs successfully.
[SayCan](https://proceedings.mlr.press/v205/ichter23a.html),
[RT-H](https://rt-hierarchy.github.io/),
[Inner Monologue](https://proceedings.mlr.press/v205/huang23c.html).

## Qualification experiments

[PROPOSED] Preserve the 60/10 Hz baseline and measure articulated-control stability
before changing cadence. Pin the target game's mechanics and assist profile for
qualification; exact profile matching does not require simulation-level fidelity
to reality. Inject response delays of 0, 6, 30, 120 and 240 ticks
(0, 0.1, 0.5, 2 and 4 seconds at 60 Hz), plus jitter, outage, duplication and
reordering. Boundary tests must establish the exact expiry comparison. Measure
wall-clock p50/p95/p99 inference time separately from simulated observation age.

Test takeover during queued motion, charging, firing and gripping; reset during
inference; slot reuse; mount replacement; stale observations; consumed/repeated
commands; and lost targets behind changing terrain. Verify bounded refusal and
fallback without undeclared assists, profile changes or direct policy terrain writes.
Replay recorded applied actions with pinned runtime/configuration and declared
tolerances; do not claim cross-platform bit identity from a deterministic policy.

Compare identical source/rig/lane/perception budgets using script selection,
learned selection and mock slow-model selection. Report success, false-success
rate, no-progress loops, recovery, intervention count/duration, cancellation
latency, applied-action age, resource saturation, step-time overhead and model
calls per task. Live model efficacy requires a later authorized measured trial.

[SUPPORTED] Real-time chunking addresses continuity between asynchronously
generated action chunks; REMAC additionally identifies mismatch between the
executed chunk and current perception. Therefore, future action-chunk policies
need delay-aware training/evaluation, interruption and queue invalidation, not
merely a larger buffer. These findings concern their benchmarks, not Cybersand.
[RTC, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/300ccb2187dedd4edcc07f7e76d8e553-Abstract-Conference.html),
[REMAC, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c35834443b7881e782e52b3519fe27c7-Abstract-Conference.html).
