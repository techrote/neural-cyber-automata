---
title: Runner point-IK contracts
kind: reference
summary: Registered nominal geometry, source-bound IK requests and results, and the boundary to CUDA training and physical tracking.
owns: [ik.contracts, ik.runner_profile, ik.nominal_geometry]
verified: 2026-10-21
---

# Runner point-IK contracts

[IK-CUDA-001 / #50](https://github.com/techrote/neural-cyber-automata/issues/50)
implements the first contract milestone of [the CUDA IK programme](ROADMAP.md#cuda-ik-programme).
`runner-right-sole@1` is a registered, dependency-free nominal profile. It
describes a two-joint point task, strict observation/target/result records and
small hand-checkable geometry fixtures. The separate [kinematics/reference
increment](IK_KINEMATICS.md) implements independent FK, Jacobians, bounded inverse
solving and analytic comparison. Learned weights, CUDA execution and physical
joint-target capability remain separately qualified stages.

The separate [IK runtime](IK_RUNTIME.md) now supplies optional device support,
a bounded fixed-profile inverse-workload smoke and complete optimizer-boundary
resume. Its [CPU software conformance](evidence/2026-10-21-ik-runtime-software.md)
does not establish actual CUDA execution, a general inverse solver or IK quality.

The [acceptance declaration](IK_ACCEPTANCE.md) freezes prospective numerical,
runtime and physical gates. The [host handoff](IK_HOST_HANDOFF.md) specifies the
separate capability needed to track these references through the actual game.
These are distinct evidence stages; a valid nominal result cannot qualify any
later stage. D-016 records the owner scope.

## Checked source and exact pilot

The profile derives from Cybersand
`f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497`, the revision pinned by the
[host adapter](HOST_ADAPTER.md). It records full Git blob IDs, raw content
SHA-256 and byte counts for the audited
[catalogue](https://github.com/techrote/cybersand/blob/f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497/godot/scripts/physical_actor_catalog.gd)
and [rig construction/driver](https://github.com/techrote/cybersand/blob/f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497/godot/scripts/physical_actor_rig.gd).
The JSON is a source-derived binary64 description, not a live host-exported
profile. Godot scalar rounding and compliant physical motion remain unqualified.

| Item | First profile |
|---|---|
| Descriptor | [`runner_right_sole_v1.json`](../src/nca/ik/runner_right_sole_v1.json) |
| Parts in stable slot order | `chassis`, `leg_-1`, `foot_-1`, `leg_1`, `foot_1` |
| Joints in stable slot order | `hip_-1`, `ankle_-1`, `hip_1`, `ankle_1` |
| Solver joint order | `hip_1`, `ankle_1` |
| Effector | `right_sole`, part `foot_1`, local point `(0, 1)`, outward normal `(0, 1)` |
| Required task components | Root-local point x/y only |
| Capacity | Eight part slots, seven joint slots; unused slots remain absent |
| Conservative target generator | `qh = 0.6 u`, `qa = 0.2 v`, with `u,v` in `[-1,1]` |
| Hard angular limits | Hip `[-0.75,0.75]`, ankle `[-0.30,0.30]` radians |
| Desired-angle limits | Hip `[-0.72,0.72]`, ankle `[-0.27,0.27]` radians |
| Authored angular play | `0.055` radians |

The sole point is the centre of the authored bottom face. Rotation can make
another point lower; the declared effector does not promise actual contact.
The generator defines a curved reachable set through forward kinematics.
Its axis-aligned bounding rectangle is not a reachability certificate.
Collision clearance, support and dynamic reachability require physical tests.

All five parts and both left joints remain in the descriptor. Left joints are
passive **for this solver**, not physically frozen or known to be unactuated.
Presence, measurement and solver activity are separate observation masks.
The current task needs both right joint measurements and the issued root pose;
unavailable left measurements are null, never synthesized as zero. A missing
part or joint changes this fixed topology and is refused.

The profile admits this two-rotary-joint open path through a rooted tree.
Closed loops and prismatic joints are rejected. A future slide profile needs
an explicit parent-local unit axis, zero/travel bounds and cell-valued
coordinates and deltas; radians cannot silently become travel. The eight/seven
capacity does not establish support for arbitrary mechanisms or whole-body IK.

## Frames and hand checks

Coordinates use game cells, radians and 60 Hz base ticks. X points right and
Y down. `R(a)(x,y) = (cos(a)x - sin(a)y, sin(a)x + cos(a)y)`, so positive
rotation appears clockwise on screen. The root is the observed chassis frame;
it is never a solver degree of freedom.

Each local anchor is `R(-initial_angle) * (authored_pivot - initial_position)`.
For a rotary joint, `q = wrap(child_angle - parent_angle - zero_radians)` in
`[-pi,pi)`, with the authored initial child-minus-parent angle as zero.
Thus the right hip's `+0.18` and ankle's `-0.18` references both give `q=0`
at the authored rest pose. With zero reference, child `-pi+0.05` and parent
`pi-0.05` give `q=0.10`, not a near-full-turn error.

For the selected sole, the independent compact nominal formula is:

```text
h = (3.6, 2.8)
p = h + R(qh) (0, 4.95) + R(qh + qa) (1, 1.25)
foot_orientation = qh + qa
```

The implementation traverses local anchors and initial zero references;
tests compare it to this independent formula. Do not add the initial `0.18`
again in the compact formula. Foot orientation is a diagnostic, not a third
independently controlled objective.

| qh | qa | Sole x | Sole y |
|---:|---:|---:|---:|
| 0 | 0 | 4.6 | 9.0 |
| 0.1 | 0 | 3.9760369821 | 9.0688592414 |
| 0 | 0.1 | 4.4702123945 | 9.0935886232 |
| 0.2 | -0.1 | 3.4867992070 | 8.9949181836 |

At neutral the point Jacobian is `[[-6.2,-1.25],[1,1]]` with determinant
`-4.95`. Across the hard ankle interval its determinant is
`-4.95 (cos(qa) - 1.25 sin(qa))`, whose absolute value is at least
`2.9003843424` square cells. There is no admitted Runner singularity to invent
or train around. Future generic singular/branch cases require explicitly
synthetic fixtures or a separately declared profile.

For world intent, `point_root = R(-root_angle) * (point_world - root_position)`
uses the **same issued observation** that supplies the joint configuration.
For example, root `(10,20,pi/2)` maps world `(1,24.6)` to local `(4.6,9)`.
A new moving-root transform cannot be combined with an old joint sample or
renew an old target. A world-fixed task needs a fresh root-local request from
each fresh observation. The canonical local target travels with the root.

## Records and validation

[`contracts.py`](../src/nca/ik/contracts.py) uses the existing frozen `Record`
codec: unknown fields, duplicate JSON keys, boolean numbers and nonfinite
values are refused. Typed constructors validate the same data as JSON decoders.

| Record | Meaning |
|---|---|
| `IKProfile` | Exact source, convention, graph, geometry, limits, participation, workspace and unavailable host lane |
| `IKObservation` | Original tick/lifecycle, all profile identities, issued root pose, distinct presence/measurement/activity masks and measured q |
| `IKAuthority` | Trusted current tick, lifecycle, source, goal, policy hash, owner-issued grant identity/generation, consumed sequence and revocation |
| `IKTarget` | Exact observation hash, source/profile/topology identities, effector/joint order, original and canonical target, posture preference, output mode and deadlines |
| `JointReference` | Desired angle or bounded angle delta, with explicit joint IDs and radian units |
| `IKResult` | Immutable target hash, solver identity/status, nominal endpoint/orientation/residual, work/fallback accounting and optional infeasibility proof |

`target_from_observation` binds the original facts and converts the frame.
`validate_against` checks a target or result against the profile, issued
observation and **trusted current authority**. It grants no authority and sends
no packet. The caller must look up the issued observation and grant itself;
accepting a client-supplied authority record would defeat these checks.

All lifecycle fields matter: run, actor, epoch, incarnation, control generation,
morphology version and mount generation. Source/goal/grant/policy changes,
cancellation and already-consumed sequences reject the pending result. A
source name is not a policy identity, and a policy hash cannot issue a grant.
Recheck the bindings when consuming a result, after any asynchronous work.

The lease begins at the **original observation tick**, defaults to twelve
ticks and is bounded to 1–18 ticks. Freshness is independently less than twelve
ticks. Effective expiry is the minimum of observation tick plus lease,
observation tick plus twelve, and the goal deadline; every upper bound is
exclusive. Receipt time cannot extend it. Host enforcement and actual receipts
remain part of the separate handoff implementation.

Absolute results are nominal desired configurations inside the inner joint
limits. Delta mode is bounded to `0.25` radians per joint and resolves against
the original measured configuration. Changing output mode is rejected. An
absolute geometric solution can be farther away than that increment; the host
must track it using its own declared bounded motors and slew/interpolation.
Neither form is a motor-velocity request, an effort command or measured motion.
Observed compliant angles are retained even when outside desired intervals.

## Status and evidence boundaries

| Result status | Required interpretation |
|---|---|
| `solved` | Recomputed nominal endpoint meets this target's tolerance |
| `best_effort` | Bounded candidate reference is valid but misses the requested tolerance |
| `geometrically_infeasible` | Independently checked geometric proof; currently only outside the unrestricted outer radial bound |
| `iteration_budget_exhausted` | Numerical budget ended; no claim of geometric infeasibility |
| `invalid_input` | Invalid solve input; no candidate action |
| `unsupported_profile` | This solver cannot handle the request/profile |
| `missing_required_measurement` | Required data unavailable; no fabricated state |
| `nonfinite_result` | Numerical failure; no nonfinite action is serialized |

Only `solved` and `best_effort` carry references. Failure results cannot carry
an action payload. A full result check independently recomputes its nominal
endpoint, orientation and residual. The radial proof uses hip centre `(3.6,2.8)`
and outer radius `4.95 + hypot(1,1.25)`; being inside that circle does not prove
feasibility. Constructor/CLI validity alone is not contextual validity.

The caller's requested tolerance governs result status. It cannot relax the
separately frozen evaluation gates. A status string, `host_observation` label
or matching hash does not prove where data came from, actual actuation,
learned contribution, GPU execution or physical task success.

## Local verification

From a repository checkout with the [CPU package installed](PACKAGE.md), these
commands work in PowerShell or a conventional shell:

```text
python -m nca plugins
python -m nca validate ik-profile src/nca/ik/runner_right_sole_v1.json
python -m unittest discover -s tests -p "test_ik_*.py" -v
python tools/check_docs.py --strict
```

`nca validate` also accepts `ik-observation`, `ik-target` and `ik-result`.
It performs structural and record-local checks, not context/authority or
host admission. Each record exposes `json_schema()` for structural interchange;
the Python validators additionally enforce semantic bounds and identities.
The profile JSON ships in the wheel and imports without NumPy or PyTorch.

These fixtures establish nominal geometry and contract conformance. The
[batched FK/reference implementation](IK_KINEMATICS.md) and its independent
checks belong to #52; CUDA qualification to #51; dataset/training to
#53/#54; host tracking to #55; exact CPU export to #56; and physical qualification
of that exported artifact to #57. No learning or physical experiment runs here.
