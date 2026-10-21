---
title: Authored actor controllers and response fixtures
kind: reference
summary: Jolt and Vector feedback skills, versioned contact recovery, separate physical supervision and explicit evidence limits.
owns: [controllers.authored, controllers.fixtures]
verified: 2026-10-20
---

# Authored actor controllers

`AuthoredController` provides Jolt (`jev`) `directed_hop`, Vector (`astra`)
`burst_traverse`, and their declared `recover_level` mapping. The balanced Jolt
controller is the initial learning reference. Firm and compliant variants remain
available in [the resolved configs](../configs/baselines/); no variant changes
the game's authored gyro, joints, grip, resource costs or physics owner.

The controller reads only copied local feedback, keeps a goal invocation and
returns a bounded `WireAction`. It does not have access to evaluator snapshots.
Unknown actor/profile mappings require an explicit implementation. An arrival
result in the original controllers ends an invocation; it does not promise
sustained holding or hovering. The recovery v2 extension below keeps its local
settling estimate advisory until the caller retires the invocation.

The separate `authored-jolt-arrival@1` candidate in
[`jolt_arrival.py`](../src/nca/controllers/jolt_arrival.py) adds a single
room-aware Jolt hop, ground approach and supported settling. It preserves the
original `authored-feedback@1` implementation and source identity. Its
[fixed development diagnostic](evidence/2026-10-20-jolt-arrival-reference.md)
tracks whether it improves the teaching trajectories. It is not a learned
policy and cannot satisfy the morphology prerequisite by itself.

The additive `authored-jolt-recovery@1` controller in
[`jolt_recovery.py`](../src/nca/controllers/jolt_recovery.py) implements the
contact/progress follow-up commissioned by
[JOLT-RECOVERY-001 / #48](https://github.com/techrote/neural-cyber-automata/issues/48).
The two earlier controller implementations retain their exact original bytes.
Its software tests establish decision and refusal behavior; actual recovery,
airborne character and arrival rates require the separately frozen paired
diagnostic. Teacher results do not qualify a learned policy or morphology.

## Jolt contact recovery

`JoltRecoveryController` owns ordinary directed movement, posture, launch timing,
aim and correction. It requests at most two primary pulses per invocation. A
six-tick anticipation unloads traction before a launch. Short goals get steeper
aim to preserve an upward impulse without spending excessive horizontal room.
The room estimate includes current velocity, conservative downward-height
flight, braking, response time, ceiling clearance and time to settle before
the original deadline. These are admission heuristics over the fixed gameplay
profile; the controller never integrates or moves a body itself.

An extra pulse needs fresh physical evidence. A supported watchdog requires
18 ticks of close-spaced observed support, directed drive and little actual
progress. The host's `blocked` flag is a delayed motion heuristic, so it is
reported as a reason without being treated as a wall sensor. Unsupported edge
recovery requires a measured near-floor, slow, stationary window. Unexpected
support loss requires a fresh support transition, sustained descent and the
actor below its caller-provided goal. A bounded airborne pulse uses Jolt's
existing primary ability; ordinary free-air movement requests remain zero.
An unsupported edge recovery can supply the first pulse if the initial stance
never acquired raw support.

For a second pulse, fresh observed positive cooldown must also acknowledge the
first request. This local acknowledgment is not an actual-impulse measurement.
The controller preserves the recovery witness through one six-tick unload
interval, then checks the current physical state and pulse bounds again. New
motion, missing support/floor information, sample gaps or changed authority
disarm that witness. Cooldown expiry alone never triggers another launch.

Landing resumes ordinary feedback after the fresh support observation; there
is no unconditional rebound. The v1 local arrival estimate requires raw
support, speed at most six cells/s, wrapped angle at most 0.32 radians and a
12-tick span of quiet samples with gaps no larger than six. These samples cannot
prove that every intervening physical tick was quiet. Deliberate posture lean
and shared request shaping retain bounded play; they do not promise perfectly
still settling.

`specification()` and `specification_sha256` expose the fixed policy, supervisor,
constants and preserved dependencies. `diagnostic_state()` copies the latest
local pulse/recovery reason, progress window, support state, recovery epoch and
supervisor intervention flag. Its pulse count counts requests. Actual impulses,
airborne duration and whether a recovery worked come from native receipts.

## Advisory arrival in recovery v2

[`JoltRecoveryControllerV2`](../src/nca/controllers/jolt_recovery_v2.py), registered
as `authored-jolt-recovery@2`, changes only the sampled-arrival lifecycle. It
inherits the preserved v1 movement, launch, contact-recovery and emergency
supervisor behavior. Both v1 source modules retain their published bytes and
remain registered for historical replay; no kinetic constants or original
diagnostic declaration change.

The first recovery diagnostic stopped in `demo-floor-right-36-1`: samples at
ticks 222, 228 and 234 met the local quiet predicate, but the retained physical
speed at tick 223 was about 7.599 cells/s. The unchanged task therefore counted
only eleven consecutive quiet ticks by 234. V1 retired its invocation from the
sampled estimate; the dispatcher correctly refused its zero lease before any
new transport. The failed experiment and its unconsumed terminal decision
remain evidence. No unobserved continuation establishes arrival.

V2 reports `phase="settle_hold"`, `reason="supported_arrival_advisory"` and
`status="running"` after the same sampled quiet span. Its neutral request keeps
the valid original-observation lease, capped by the original goal deadline.
`diagnostic_state()["arrival_advisory"]` explicitly marks this estimate. Fresh
disturbance, unknown support or a sample gap ends the advisory; ordinary
correction resumes. Invalid state, stale input, lost authority, explicit
emergency and deadline boundaries keep their original refusal behavior.

The policy receives no task outcome as an observation. After independently
establishing its required outcome and validating the current invocation and
native ownership, the caller uses `controller.cancel()` and cancels the native
owner. Local cancellation returns a zero lease; neither operation fabricates a
physical outcome or removes momentum. A delayed result from a previous physical
lifetime cannot authorize completion of the current invocation merely because
its goal values happen to match. There is no new goal-only completion API.

The small retained kinematic excerpt and synthetic lifecycle checks in
[`test_jolt_recovery_v2.py`](../tests/test_jolt_recovery_v2.py) exercise this
boundary without consuming physics. The correction does not establish teacher
qualification, preserved vigor, or success in any unobserved case. Those claims
require their separately declared complete physical comparison.

## Contact-transition movement in recovery v3

[`JoltRecoveryControllerV3`](../src/nca/controllers/jolt_recovery_v3.py), registered
as `authored-jolt-recovery@3`, adds a fresh initial contact-transition launch.
An exactly six-tick supported-to-unsupported anticipation may launch only on a
current observation with measured floor within 16 cells, displacement at most
0.40 cells and speed at most six cells/s. Current energy, cooldown, ceiling,
room and original deadline are rechecked. Unknown support, motion outside that
envelope, gaps or authority changes discard the witness without renewing a lease.

The unchanged first-pulse calculation is followed by a separately solved
shorter recovery vector: signed existing velocity, bounded lift and horizontal
flight/reaction/stopping room, caller effort, minimum recovery effort 0.40,
horizontal aim at most 0.82 and resulting upward speed at least 20 cells/s.
These are admission heuristics; contact forces and landing progress remain
native measurements. Fresh recovery windows and at most two edges still apply.

V3 immediately holds a supported quiet sample inside tolerance and counter-drives
horizontal speed above six cells/s. Arrival stays advisory. After two pulses,
a fresh supported stall can authorize twelve ticks of stronger ground traction;
motion, support uncertainty/loss, gaps and authority boundaries clear it.
The preserved [v3 evidence](evidence/2026-10-20-jolt-recovery-v3.md) and
[source tests](../tests/test_jolt_recovery_v3.py) retain the detailed contract,
28/32 arrivals, 32/32 airborne cases and failed inventory/third-rescue gates.

## Interior settling in recovery v4

[`JoltRecoveryControllerV4`](../src/nca/controllers/jolt_recovery_v4.py), registered
as `authored-jolt-recovery@4`, implements the ordinary settling follow-up in D-017.
It inherits v3's exact initial-contact and pulse-vector methods, and preserves
all older source bytes, gyro/profile limits, task thresholds and supervisor.

Current raw support inside the caller's Euclidean tolerance starts capture.
Small outward disturbances retain capture until distance exceeds 1.25 times
that tolerance. Outside the inner region, the controller requests bounded 0.10
inward traction; within it, horizontal speed above two cells/s requests bounded
counter-traction. The existing Schmitt/play/slew shaping remains active.
Neutral holding requires distance at most 0.75 times tolerance, total speed at
most two cells/s and the existing 0.32-radian posture bound. These are policy
choices, not tighter evaluator thresholds. Unknown/lost support, sample gaps,
cancellation, authority changes, emergency and deadline clear capture.

Exactly six ticks after the second pulse, a consecutive current observation
with acknowledged positive cooldown, raw support and upward speed at least
six cells/s may request one six-tick traction coast. That action's original-source
lease is narrowed to six ticks. Later, delayed, nonrising or uncertain observations
cannot extend this coast; no inferred contact state or new primary edge is added.

`diagnostic_state()` adds `settling_capture`, `interior_quiet_hold` and
`post_pulse_coast`. Sampled arrival stays advisory with `status="running"`;
continuous native outcome and cancellation belong to the caller. The unchanged
tick-223 regression still prevents sparse samples from proving completion.
The [v4 tests](../tests/test_jolt_recovery_v4.py) and
[offline retained prepass](evidence/2026-10-20-jolt-recovery-v4.md) establish
software decisions only. New physical outcomes and teacher acceptance require
the separately frozen diagnostic and complete original final verification.

## Minimal physical recovery supervisor

[`PhysicalRecoverySupervisor`](../src/nca/controllers/recovery_supervisor.py)
is separate from ordinary motion. It checks observation/profile validity,
monotonic observation and current clocks, lifetime, control generation, goal
identity and the original deadline. It passes ordinary actuation through and
does not read a goal target to choose a movement or trajectory.

The only non-neutral override is an explicitly declared `unstable_attitude`
emergency, admitted at absolute wrapped angle at least 1.20 radians or angular
speed at least 4.5 radians/s. The Jolt policy makes that declaration from its
permitted observation. The supervisor requests the existing auxiliary
self-right assist and bounded trim, with no primary pulse or traction. It
returns control after 12 fresh quiet ticks and cancels after at most 48 ticks
without recovery. Ordinary leaning, landing and contact recovery remain policy
decisions. A future learned controller must retain that ordinary authority.

The returned lease is anchored to the original observation and capped by the
original goal deadline. Supplying a later `current_tick` does not renew its
age. Dispatch must use the remaining lease and preserve native owner checks;
a new goal ID is an explicit caller invocation, not a grant of actor ownership.
In synchronous stepped mode, after constructing a goal and observing the host:

```python
from nca.controllers import JoltRecoveryController, FeedbackObservation

controller = JoltRecoveryController()
frame = controller.step(FeedbackObservation.from_host(view), goal)
if frame.status != "running":
    env.cancel()
else:
    lease = controller.diagnostic_state()["action_lease_ticks"]
    packet = env.command(frame.action.to_dict(), observation=view, lease_ticks=lease)
    view, reward, terminated, truncated, info = env.step_packet(packet, ticks=min(6, lease))
```

On cancellation, invalid feedback, takeover or failed dispatch, retire local
intent and revoke the native owner using `controller.cancel()` and
`env.cancel()`. Neutral output alone does not revoke an existing lease or remove
physical momentum. After an episode reset, reset local controller memory too.

## Use an authored skill

After [host setup](HOST_ADAPTER.md), with a reset Jolt `CybersandEnv`:

```python
from nca.controllers import AuthoredController, FeedbackObservation, SkillGoal

view = env.observe()
controller = AuthoredController("jev")
goal = SkillGoal("hop-1", "directed_hop",
                 target=(view["position"][0] + 48, view["position"][1]))
while True:
    frame = controller.step(FeedbackObservation.from_host(view), goal)
    if frame.status in {"succeeded", "timed_out", "canceled"}:
        env.cancel()
        break
    view, reward, terminated, truncated, info = env.step(frame.action.to_dict())
    if terminated or truncated:
        break
```

On takeover, call `controller.cancel()` and `env.cancel()` together. The first
clears controller intent; the second revokes pending owner input and issued
views. Neither operation removes existing motion or material effects. A canceled
goal cannot silently restart; create a fresh invocation ID. On episode reset,
reset controller memory as well. The registered policy ID is
`authored-feedback@1`, constructed with `actor="jev"` or `actor="astra"`.

## Repeat the matched screen

Prerequisites: the CPU package and pinned host from the host guide. Matched GIF
generation additionally uses the optional Pillow dependency. In a configured
Python environment, these commands work in Bash or PowerShell:

```text
python -m pip install --no-build-isolation ".[evidence]"
python tools/control_baseline.py --output artifacts/baseline-screen-1
```

Set `NCA_HOST_CHECKOUT` and `NCA_GODOT` using the host guide. Add `--skip-clips`
to run without Pillow. The output directory must be new. The fixed screen has
18 episodes: two actors, three tunings and step/pulse/reversal fixtures, seed
1703, 1,080 decisions and 6,480 base ticks, capped at 600 seconds. It does not
train a policy. Failures retain completed traces and an explicit failure record.

The output includes compressed per-base-tick receipts, a summary with source,
host, profile, tuning and artifact hashes, and six matched pose-replay GIFs.
Each GIF shows recorded body poses and observed local cells with the same camera
across tunings. These are pose replays; actual shaded game footage and owner
feel review remain separate evidence.

## Fixtures and measurement

The [scenario definitions](../configs/fixtures/) also declare neutral disturbance,
support loss/return, soft stops, recoil/payload changes and recovery. Request
shaping tests cover hysteresis, deadbands, damping, time-scaled slew, bounded play,
cancellation, mapped soft stops and recovery feedback. Response metrics separate
onset, arrival, overshoot, useful progress, settling and corrective chatter.
Intentional launch, flight, recoil and recovery motion is excluded from the
corrective-trim chatter count.

The actual public arena has no arbitrary force/torque injection, support-removal,
payload-replacement or joint-profile override call. Those requested fixture
operations explicitly refuse. The software fixtures and ordinary launch/contact
observations establish no controlled physical perturbation result for these
missing facilities. Native usable recoil is a separate actual capability used
through its declared mounted-tool interface.

The [evidence report](evidence/2026-10-19-t103.md) preserves tuning alternatives,
the compliant Jolt reversal failure, Vector's drift after arrival, original and
corrected metric results, and the distinction between measured behavior and
owner feel. [Run records](RUNS.md) identify the executed screens.
