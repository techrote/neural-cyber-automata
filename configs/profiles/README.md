---
title: Implemented profile examples
kind: guide
summary: Strict profile examples, their source provenance, and the boundary between a recorded fixture and game qualification.
owns: [profiles.examples]
verified: 2026-10-19
---

# Profile examples

The JSON files decode with `nca.contracts.GameplayProfile.from_json`. Constructors
and JSON decoding share the same checks: finite numbers, explicit versions,
part references, unique identifiers, 1–8 part capacity, channel masks, limits,
sensor visibility and declared response mechanics. Each resolved profile has a
canonical SHA-256; changing assistance, compliance, play, cadence or mapping
changes that identity.

| File | Meaning | Evidence limit |
|---|---|---|
| [recorded-rig-v1.json](recorded-rig-v1.json) | Two-part schema fixture, a compliant slide and bounded gyro settings | Authored test values; no simulated or tuned game mechanics |
| [ripper-metal-v1.json](ripper-metal-v1.json) | ACT003 track drive, chassis torque, brake and a reserved zero slot | Sampled legacy mapping, not ACT004 |
| [copter-metal-v1.json](copter-metal-v1.json) | ACT003 collective, cyclic, torque and brake | Sampled legacy mapping, not ACT004 |
| [gemini-metal-v1.json](gemini-metal-v1.json) | ACT003 left/right thrust and two reserved zero slots | Sampled legacy mapping, not ACT004 |

The legacy mappings and 32 receipt-field indices are derived from
[actor_rig_actuation.gd at f2e34cf](https://github.com/techrote/cybersand/blob/f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497/godot/scripts/actor_rig_actuation.gd).
Legacy receipt sensor ranges are broad finite storage bounds, not training
normalization. Fields marked `host_declared` retain their source semantics;
an experiment needs an explicit normalization before feeding them to a policy.

The Python factories `recorded_fixture_profile()` and
`legacy_metal_profile(rig)` produce these profiles without a file-system or
optional trainer import. `nca.registry.Registry` provides versioned factory
registration and rejects duplicate IDs, incompatible versions and missing
capabilities before invoking a factory.

## Requests, admission and motion

`Command` and `Observation` are experiment records with a lifecycle stamp and
profile digest. They are not alternative host packets. `admit_command` records
a refusal or admitted request; `consume_admission` rechecks the profile,
observation, lifetime, grant, lane and expiry at the owner boundary. Its
`ActuationReceipt` retains requested action, admitted/consumed action, actual
applied effort and passive motion separately. A sequence watermark is supplied
by the owner, which remains responsible for exactly-once consumption.

The expiry tick is exclusive: a request observed at tick 0 with a 12-tick lease
can drive transition 11→12, but cannot drive 12→13. A late admission never
renews the source observation's age. Actor replacement, reset, mount changes,
morphology changes and control-generation changes invalidate older requests.

`Observation` admits only local/global policy sensor fields and a validity mask.
Unavailable values are `null`, never invented zeros. `EvaluatorRecord` is a
different type and cannot be decoded as an observation. Central decision
placement, coordination and training privilege are independent capability axes.

## Legacy packet compatibility

`LegacyCommand`, `LegacyWireAction` and `LegacyMetalAction` retain exact ACT003
v1 field names, including `movement`, the four `actuators`, and `tag_aim`.
The decoder accepts integer-valued JSON floats in the legacy envelope because
the host v1 validator does. Unknown fields, v2 packet versions and invalid
cancel/charge combinations are refused. V1 diagonal component values remain
legal; they are not silently reinterpreted as already normalized v2 vectors.

`LegacyMetalAction.admitted_actuators(rig)` exposes the owner's separate clamp
mapping and rejects nonzero reserved slots while retaining the original request.
`WireAction` describes already bounded ACT004 v2 values with `move` and the newer
usable buttons. Actual host receipt, authority and consumption checks belong to
the host adapter and Cybersand owner.

The [T-101 evidence](../../docs/evidence/2026-10-19-t101-contracts.md) records
the focused conformance checks and exact candidate source identity.

© 2026 techrote. All rights reserved.
