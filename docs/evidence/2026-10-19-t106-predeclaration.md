---
title: Morphology experiment prerequisite declaration
kind: evidence
summary: Fixed mechanical utility prerequisite for morphology experiments, declared before opening first-skill holdout results.
owns: []
verified: 2026-10-19
---

# Morphology experiment prerequisite

[T-106 / #16](https://github.com/techrote/neural-cyber-automata/issues/16) requires a useful learned skill on one rig and accepted additional actor profiles before transfer training starts. This record fixes the mechanical utility criterion before the sealed evaluation in [the frozen first-skill screen](https://github.com/techrote/neural-cyber-automata/blob/745578947a6f1ac2159830c4ae4de5de5593f32f/docs/evidence/2026-10-19-t105.md) opens. It authorizes no additional training, budget increase or change to that screen.

The machine-readable declaration is [entry-gate-v1.json](../../configs/morphology/entry-gate-v1.json). It refers to T105 candidate source `745578947a6f1ac2159830c4ae4de5de5593f32f` and the already frozen screen configuration `8e40c1f65a2fe32597ecc97d10bd6f4414f96762bdc198f5063a56acad68616c`.

## Fixed utility criterion

A whole candidate family must meet every condition using its three independently trained policy seeds and all eight predeclared holdout scenarios:

- Mean successful-arrival rate is at least 0.75, with no seed below 0.50.
- Mean paired success gain over that same family's initial checkpoints is at least 0.125. The lower 95% policy-seed bootstrap bound must be strictly positive. The calculation uses all 27 resamples of the three per-seed paired gains, with linear interpolation for the 2.5 and 97.5 percentiles.
- Every seed has complete paired coverage, zero integrity failures, zero refused admissions, zero ownership or lifetime-fence regressions, and no censored evaluation.
- Every seed passes the original T105 mechanical and runtime bounds against the authored controller: ordinary trim reversals within +0.5/second; mean absolute angle within +0.1 radians; mean peak useful speed ratio in [0.7, 1.5]; purposeful-launch fraction within -0.125; and inference p95 at most five milliseconds in every scenario.

The fixed family priority is BC, then residual PPO. Selection always retains the three-seed family; it does not select a lucky individual checkpoint. Missing, invalid or incomplete evidence closes the prerequisite. No threshold, seed set, checkpoint selection or training budget changes after evaluation opens.

This prerequisite measures usable learned behaviour relative to initialization. The separate T105 decision still requires its original improvement over the authored controller. A family may meet neither, one or both criteria; the records must report these separately. Owner feel judgement also remains separate and pending unless the owner provides it.

## Additional morphology prerequisites

A positive per-rig result would permit planning a bounded comparison of a padded, masked recurrent policy with a shared graph policy. It would not establish transfer. That comparison still needs accepted additional actor profiles, matched wire capabilities, sensor and cadence budgets, declared parameter and compute budgets, and whole heldout connection arrangements and geometric parameters.

The accepted host permits selecting existing authored profiles and initial positions. It does not expose arbitrary part geometry, mass, assist strength or perturbation setters through the accepted public experiment API. A configured identity label cannot substitute for a changed physical arrangement. Supported mount changes must retain real connection generations and invalidate or reset recurrent state at lifetime boundaries. Unsupported geometry or damage cases remain unmeasured.

The accepted evidence already supplies the following useful profile inventory. The physical part counts and button meanings come from the [pinned catalog](https://github.com/techrote/cybersand/blob/f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497/godot/scripts/physical_actor_catalog.gd); the actual selected-profile integration limits are recorded in [T102](2026-10-19-t102.md) and [T103](2026-10-19-t103.md).

| Profile | Parts and connection arrangement | Primary / secondary meaning | Preparation implication |
|---|---|---|---|
| Jolt (`jev`) | Impact core and two yielding pad hinges; three parts | Pulse / rebound | T105's one-rig learned-hop candidate |
| Vector (`astra`) | Chassis and two compliant thrust pods; three parts | Pulse / brake | Authored feedback is qualified on selected fixtures; movement dynamics and braking differ from Jolt |
| Monowheel | Weighted cab, tumbling rim and axle; two parts | Spin / hop | Actual two-part integration is accepted; a hop mapper must declare that the secondary channel carries the impulse |
| Ripper | Chassis, two sliding bogies and two driven rollers; five parts | Cut / brake | Actual five-part integration is accepted; a primary pulse label would be false for this rig |

These profiles have different physical capabilities. A shared wire field name cannot establish a matched action budget. A transfer pilot must declare which task is physically supported on every selected train and holdout profile, how capability masks map its requests, and which arrangements or parameters are absent from training. The legacy `ripper-metal-v1` example is an ACT003 sampled profile and cannot be substituted for the ACT004 five-part actor.

No morphology training or transfer result is reported in this declaration. The utility calculation and the additional-profile acceptance gate must be recorded before any such experiment starts.

© 2026 techrote. All rights reserved.
