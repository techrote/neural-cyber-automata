---
title: IK joint-target host handoff
kind: guide
summary: Required Cybersand capability, authority, timing and receipts for the separately scoped IK host bridge.
owns: [ik.host-handoff]
verified: 2026-10-24
---

# IK joint-target host handoff

This is the host contract handoff for
[IK-CUDA-006 / #55](https://github.com/techrote/neural-cyber-automata/issues/55).
The [IK layer](IK_LAYER.md) defines the first profile and geometric interfaces;
[IK acceptance](IK_ACCEPTANCE.md) defines prospective numerical and physical
criteria. This document specifies the physical boundary now implemented by the
separate NCA joint-target adapter and Cybersand ACT-IK-001 companion. The
[exact-head adapter evidence](evidence/2026-10-24-ik-host-adapter.md) records
the source/runtime binding and actual-host conformance without qualifying a
learned policy.

## Implemented joint-target lane

NCA PR #87 adds a strict `physical_joint_target/v1` client, a separately
pinned actual-host launcher and an NCA-owned loopback bridge. It does not
replace the accepted historical ACT004 adapter. The candidate client binds one
source grant and one issued observation to each single-use `ik_submit`, fails
closed on unknown transport outcome until owner-confirmed cancellation, and
requires continuous per-base-tick receipts.

The companion implementation is Cybersand #210 / PR #224. NCA's hosted
conformance authenticates its exact source/tree plus Godot, native and Rapier
bytes before exercising authority, contact, receipts, cancellation, takeover
and expiry. Final acceptance remains a cross-repository merge gate; fixture or
source-only passes do not substitute for the companion's own hosted checks.

## Existing capability and separate ownership

The [accepted host adapter](HOST_ADAPTER.md) pins Cybersand
`f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497`. Its
[physical session](https://github.com/techrote/cybersand/blob/f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497/godot/scripts/physical_actor_session.gd)
accepts `physical_wire_action/v2`: semantic movement, aim, effort, trim, reel
and buttons. It does not advertise independent joint-position targets.
The [rig driver](https://github.com/techrote/cybersand/blob/f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497/godot/scripts/physical_actor_rig.gd)
derives joint motor requests from authored behavior. Its rotary `joint.target`
is motor velocity, whereas its slide `joint.target` is displacement.

The existing generic NCA `direct` schema and ACT003 metal channels do not add
this capability to ACT004. A new Python packet cannot make an unsupported
host operation available. Cybersand code remains unchanged in IK-CUDA-001.

The bridge implementation must link a corresponding Cybersand task, reviewed
host PR and accepted source/build identity. Cybersand owns the new target lane,
motors, bodies, contacts, assistance and native material schedule. NCA owns its
client, inference, copied data, conformance and evidence. Specification and
offline fixtures can proceed before the host change; physical completion of
#55 requires its accepted capability and actual-host checks.

## First profile and target semantics

The first task follows Runner's right sole through `hip_1` and `ankle_1` to
`foot_1`. The whole actor remains physically simulated. The geometric solver
requests two joint angles for a two-dimensional point target; it does not gain
an independently controlled foot orientation or the ability to move the root.

The profile carries stable part, joint and effector identifiers, topology,
parent/child anchors, zero-angle offsets, authored bounds, masks and hashes.
The pilot preserves authored angular play of **0.055 radians** and places
requested targets **0.03 radians inside** the declared joint limits. The
resulting target bounds are `hip_1` in **[-0.72, 0.72] radians** and `ankle_1`
in **[-0.27, 0.27] radians**. A `joint_delta` result changes the original
observed configuration by at most **0.25 radians** per joint. An absolute
`joint_position` result may describe any nominal solution inside the inner
bounds, including one farther from the current pose. Its geometric validity
does not admit an immediate physical jump: host tracking separately bounds
slew and motor response and records any interpolation or rate limiting. Result
mode must match the target's requested mode; a large delta cannot be relabelled
as an absolute result to satisfy that request.

The interior margin limits requests; it does not remove host compliance or claim
that measured joints always stay within the requested interval. Actual
tracking and geometric tolerances remain separately defined in acceptance.

| Quantity | Meaning at this boundary |
|---|---|
| Root-local effector target | Desired point in the coordinate frame declared by the issued observation |
| Joint-position reference | Desired rotary angle relative to the profile's joint zero; future prismatic references need explicit travel units |
| Joint delta | A bounded proposal resolved against the same observed configuration before dispatch |
| Wire aim or movement | Existing semantic input; it is not a joint reference |
| Motor velocity or effort | Host tracking output under the authored response and caps |
| Measured angle, travel or motion | Physical feedback; it is not proof that a reference was admitted or followed |

Root-local coordinates are canonical for the pilot. A world target must be
converted using the root transform from the **same issued observation** as the
joint configuration. Preserve that observation's identity and tick through
inference, correction, queuing and dispatch. Never combine an old configuration
with a newer root transform or stamp an old result with a fresh observation.
Root movement may change the world location of a root-local goal. Holding a
world point requires a new local target derived from a new valid observation.

Joint IDs and their ordering are bound to the profile; array position alone is
insufficient. Presence, measurement validity and actuator activity remain
separate masks. An absent or unmeasured value is unavailable, not zero. A
passive joint cannot become an actuator merely because its pose is observed.

## Required capability negotiation

The joint-target handshake identifies the exact host source/build, packet and
observation schema versions, profile/topology/effector hashes, supported joint
IDs and units, target limits and masks, timing rules and receipt availability.
The client must refuse an absent, unknown or incompatible target capability.
Preserve ACT003 and `physical_wire_action/v2` without reinterpreting fields.

The target envelope must bind the run, actor, episode/epoch, physical incarnation,
control generation, source identity, grant ID/generation, policy digest, goal identity and
deadline, profile/topology identity, sequence and original observation. Include
mount identity when it can affect the chain or its exclusive resources. Reuse
existing host lifetime fields where they express the required identity; a new
field must have a defined owner and invalidation rule.

Policy identity records provenance and cannot grant authority. An owner-issued
source grant authorizes one actor and the declared joints. A changed goal,
policy, source, topology or profile invalidates incompatible pending decisions,
branch/history/filter state and retained targets. Old generations cannot
become current through cancellation or a fresh client object.

## Timing, dispatch and cancellation

Use the existing **60 Hz** base schedule and **six-tick** policy decisions.
The helper defaults to a **12-tick** lease; declared leases remain bounded to
**1–18 ticks**. This pilot's original-observation age limit is **12 ticks,
exclusive**. A legal lease duration cannot extend observation freshness.

For an observation at tick `s`, each consumption tick `t` must satisfy:

```text
s <= t
t < s + 12
t < s + lease_ticks
t < goal_deadline_tick
```

The effective drive deadline is the earliest of these exclusive upper bounds.
Receipt, retry, target filtering or inference completion never renews `s`.
For example, a twelve-tick grant observed at zero can drive the interval
11 to 12, but cannot drive 12 to 13. A shorter goal deadline wins. Revalidate
at owner consumption, including each base tick covered by a repeated action.

Bind every pending decision to its producing controller and current invocation.
Only that decision may dispatch, once, under a strictly advancing sequence.
Duplicate/replayed/foreign decisions cannot produce another actuation or lease.
If transport supplies idempotent receipt lookup, it may return the original
receipt but cannot reapply the action. One ordered request remains in flight;
ambiguous dispatch failure requires confirmed owner revocation before further
drive, rather than an automatic resend under new sequence or observation IDs.

Cancellation, expiry, takeover and terminal suspension revoke future drive and
clear obsolete targets, filter requests and armed discrete actions at the owner
boundary. Canceling a charged action must not release it. Preserve current
velocity, contact response, compliant loading and authored mechanical play.
Only an explicit world reset reconstructs physical state. A reset invalidates
all earlier actor/goal/grant/observation identities and queued decisions.

## Exclusive tracking and unchanged mechanics

One explicit mode owns the commanded joints. The authored gait and IK cannot
both drive `hip_1` or `ankle_1`. The host task must declare behavior of every
non-commanded joint and of movement, gyro assistance, buttons and mounted
equipment during IK mode. Mode changes clear obsolete drive and armed inputs;
they do not secretly reset an actor or supply a stabilizing pose.

The host tracks admitted joint references through its bounded motors, with
declared speed/force caps, slew, damping, deadzones and authored play. Reference
and learned solvers receive the same response profile. Neither may assign body
transforms, suppress collisions, gain stronger assistance or pin the root.
All bodies and the native material/Rapier causal schedule remain active.

Declare fallback for invalid, stale, nonfinite, unsupported and unsolved
requests. Geometric best effort and solver iteration exhaustion are distinct
from proven infeasibility. Any permitted numerical correction, reference
fallback or emergency recovery has its own recorded identity and contribution.
Fallback cannot silently become the ordinary controller while the result is
reported as learned success. Contact blockage is not solved by increasing caps.

## Per-tick receipts and unavailable evidence

Retain the causal chain for every consumed base tick, not only one summary per
six-tick decision. Aggregate counts cannot recover a missing owner exchange.

| Receipt group | Required record |
|---|---|
| Requested | Original envelope, observation/root frame, joint references, masks, source/policy/goal and sequence |
| Admitted or rejected | Actual owner tick, identity checks, accepted references, clamp/slew/limit changes and explicit reason |
| Applied | Actual selected tracking mode, motor requests, limiting and fallback flags, and observable actuation |
| Observed | Completed tick, copied part/joint poses and motion, measured tracking error and available contact state |
| Integrity | Actual versus requested ticks, terminal boundary, missing/dropped/unknown trace counts and confirmed close |

Requested references, admitted references, commanded motor response and measured
motion are separate quantities. Never infer actual force or torque from a
target field or call passive/contact motion a learned action. The accepted
ACT004 host has aggregate actor impulse/torque and part motion but lacks a
separate passive-wrench decomposition. Preserve unavailable fields explicitly.

The joint-target handshake states which contact sensors and per-joint receipts
exist. Available local contacts are observations, not a complete contact-force
oracle. Evaluator-only state cannot enter policy preprocessing. Arbitrary torque
injection, support removal, individual body replacement, assist mutation or
damage cannot be tested until the host explicitly supports those operations.
Fixture coverage and reset-based layouts do not qualify unavailable operations.

## Host acceptance handoff

The separately scoped host and NCA bridge work must establish:

- Capability/version refusal and unchanged legacy wire/metal behavior.
- Exact source-frame, profile, ID-ordering, mask and numeric validation.
- Actor/source/goal/epoch/sequence fences at receipt and per-tick consumption.
- Exclusive expiry boundaries, replay refusal and confirmed cancellation after
  ambiguous transport failure.
- Single actuator ownership through mode changes, takeover and charged inputs.
- Equal bounded mechanics, visible compliance and continued whole-actor physics.
- Actual per-tick receipts, availability labels and regression coverage for
  existing Jolt, authored and usable-control paths.

Publish the accepted host task/PR, source/build identities and actual conformance
results before physical qualification. Preserve historical experiment pins.
The exact exported CPU bundle is qualified later through
[IK-CUDA-008 / #57](https://github.com/techrote/neural-cyber-automata/issues/57);
geometric tests and this handoff do not qualify CUDA training, real control,
desktop pacing, owner feel or the separate morphology gate.

© 2026 techrote. All rights reserved.
