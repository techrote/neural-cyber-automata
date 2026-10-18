---
title: Authored gameplay control profiles
kind: guide
summary: Proposed response design for lively, stable actor assistance, compliant connections and shared human/policy mechanics.
owns: [design.gameplay-profile]
verified: 2026-10-18
---

# Authored gameplay control profiles

[PROPOSED] This page makes the gameplay-emulation direction in [D-011](../DECISIONS.md#d-011--plausible-gameplay-through-emulation-with-deliberate-flex-and-play) concrete. It specifies candidate behaviour, not implemented mechanics or measured settings. The [source audit](../evidence/2026-10-18-source-prepass.md) identifies existing host capabilities; [control methods](../research/CONTROL_METHODS.md) explains the wider controller comparisons.

The target is a machine that responds intelligibly, moves with character and survives awkward contacts. Gyro-like assistance needs no stored angular momentum. Connections may flex, rattle within controlled limits or use emulated constraints. Acceptance concerns the authored response and gameplay usefulness, not a mechanically perfect real-world counterpart.

## One profile describes the shared mechanics

Give each actor a versioned profile that defines its actuator mapping, target ranges, assistance, damping, deadzones, rate limits, contact modes, joint play and recovery rules. Keep the resolved profile with rollout and evaluation records. A numerical setting is a tuning choice until its response is checked in the game.

Human, scripted, learned and model-directed commands pass through the same configured mechanics. A matched comparison cannot secretly give the human extra balance assistance or the policy stronger grip. Direct and assisted control remain separately named lanes; their different mappings are explicit.

Publish the requested target, filtered target, applied effort/assist and limiting flags where relevant. Include profile identity and control-mode changes. Stateful smoothing, stance latches, backlash take-up and recovery timers require reset rules and either useful observations or an explicit declaration that the policy must infer them. A profile change invalidates assumptions in old training data.

## Damped tracking and bounded gyro assistance

A useful first attitude controller combines wrapped target-angle error with damping from measured angular velocity, then limits the assist output. Target angle can follow movement, aim or an explicit self-right command; it need not always point upright. Publish which target is active.

Tune correction strength, damping, maximum assistance and target-turn rate independently. Strong correction with weak damping overshoots; heavy damping can make every turn feel reluctant. Permit a deliberate amount of overshoot if it settles consistently and suits the rig. Avoid integral accumulation initially: a machine pinned under a ceiling should not store an escalating demand and kick violently when released.

The gyro is an authored torque assist. An optional effort bar, cooldown or reduced assistance during another action is a game mechanic, not a rotor-momentum ledger. Preserve visible recoil and tipping by allowing deflection before recovery. Instant pose locking would undermine the intended response.

## Deadzones and hysteresis provide quietness

Use an input deadzone for small device noise and a separate settled-target band for control error. They solve different problems: the first avoids accidental requests; the second prevents repeated corrective taps around an acceptable posture.

Use two thresholds when entering and leaving a settled, gripping or stressed state. For example, enter settled mode only inside a small angle/speed band and resume active correction outside a larger band. A short dwell can reject brief contacts, provided its delay is recorded.

This is purposeful tolerance, not random jitter. If an actor needs anticipation or restless personality, express it through a bounded animation/control rhythm tied to action or state. Do not inject arbitrary noise into a joint target merely to hide chatter.

## Contact-aware stance and grip

Treat stance as a small state machine: seeking support, supported, slipping and recovering are useful candidate states. Derive transitions from fresh available contact or terrain observations, with hysteresis. A changing surface must be able to invalidate support.

Grip can be an assisted attachment with an authored reach, hold strength, compliance and release rule. It need not model microscopic friction. Its anchor still belongs to valid geometry, and release must not create a stale attraction toward a deleted surface. A brief grace interval after contact loss is acceptable if deliberately specified, rather than an accidental consequence of delayed sampling.

Reduce drive or redirect it when the rig cannot use the requested contact. Bracing may trade speed for steadiness; a lunging stance may briefly unload one attachment. Learned control chooses among the same available behaviours as human input.

## Soft end regions, compliance and play

A soft end region begins slowing motion before a nominal travel boundary. Increase opposing damping or reduce further target travel as the part approaches it. The final response may be a compliant stop, provided the intended movement envelope remains understandable and collision handling remains valid.

Tune connection stiffness, damping, angular/linear slack and load-dependent deflection separately. A loose linkage can yield under impact and return toward its rest configuration without demanding zero positional error. Emulated constraints are acceptable when their visible and collision behaviour follows the declared profile.

Pseudo-backlash should have memory: after reversing demand, allow a small bounded take-up interval before the linkage transmits the new direction strongly. Keep the offset bounded and make engagement gradual. Randomly displacing the joint each frame is not backlash.

Changing a spring rest target or tightening a grip can itself create a kick. Ramp those changes or make the kick an explicit action. Release, break and reconnect behaviour must specify how stored profile state is cleared.

## Smoothing without removing vigour

Input filtering and output slew limits trade abruptness for delay. Filter noisy analog targets lightly; limit implausibly sharp changes in assistance or motor demand. Use actual simulation duration, so a setting describes a response time rather than an unexplained number of frames.

Different phases may use different ramps: decisive launch, softer braking, quick first steering response and slower settling. Avoid cascading several smoothers without measuring their combined lag. A machine can look stable while feeling disconnected from the player.

Cancellation bypasses normal target smoothing at the next owner consumption boundary so old movement requests stop driving the controller. Clear obsolete command/filter targets and armed input; do not erase loaded compliance, stance or play state. Coast, grip release or authored recovery follows the explicit fallback profile. Full episode reset separately reconstructs mechanical state. Do not queue delayed button edges through a continuous filter; fire, detach and takeover retain their [command-contract](COMMAND_AND_SKILL_CONTRACT.md) semantics.

## Reduce corrective aggression under stress

Detect stress through available signals such as repeated large constraint errors, excessive correction, persistent saturation or unusually rapid contact changes. These are candidate diagnostics, not claims that all such telemetry currently exists.

When stress persists, reduce target-correction strength or target speed, allow more compliance, and prioritize a small recovery action. Apply hysteresis to entering and leaving this mode. Reducing controller aggression must not silently disable collisions or native material authority.

If a linkage can break, define a predictable condition and visible state transition. A reconnecting mechanism needs a valid anchor and a clear rearming rule. Otherwise, prefer a bounded recovery or explicit stuck state. Numerical explosions, NaNs, frame-dependent snapping and growing oscillations remain failures; intentional flex stays bounded, repeatable and legible.

## Three candidate actor profiles

| Rig | Proposed response and personality | Useful challenge |
|---|---|---|
| Ripper | Fast drive onset, short controlled braking slide, modest chassis lean and a damped attitude assist. Contact mode reduces useless airborne drive. A little suspension/play absorbs rough terrain without flattening every impact. | Reverse at speed on rough ground, enter a confined passage and fire a recoiling mounted item. Retain decisive direction changes without a prolonged settling wobble. |
| Scrabbler | A low compliant stance, contact-aware grips and a committed lunge. Leg targets share a predictable sequence; attachment transitions have hysteresis. Self-righting can use bounded assist with visibly yielding legs. | Remove support during a crouch and interrupt a lunge with a wall. Recover or release clearly, rather than rapidly alternating grip states. |
| Copter | Responsive thrust with limited attitude assistance and noticeable tilt/recoil. Damping increases near the requested attitude, while a wider manoeuvre band preserves demanding flight. | Apply short alternating steering pulses, brush a ceiling and change payload. Stay unstable enough to be interesting but recoverable without target snapping. |

These are tuning hypotheses. The actor may need a different response once its actual parts, collision footprint and available control channels are established.

## Candidate acceptance clips and measurements

Start with short, repeatable fixtures and inspect the resulting motion alongside measurements.

| Fixture | Record |
|---|---|
| Step, pulse and reversal requests | Initial response delay, rise, overshoot, braking distance and settling time within the authored tolerance |
| Neutral rest and a gentle disturbance | Uncommanded oscillation, corrective sign changes and recovery without unnecessary activity |
| Rough contact and disappearing support | Slip, grip transitions, stale-anchor refusal, stuck time and recovery |
| Full travel, recoil and payload changes | Soft-stop behaviour, maximum deflection, saturation and repeatable take-up |
| Delayed input and cancellation | Effective latency, revoked requests, retained compliant state and consistent fallback |
| Full episode reset | Fresh command, smoothing, stance, play and recurrent initial state with no old effects admitted |

Compare a firmer profile with a deliberately compliant one using the same scenario and command trace. Then test human responsiveness and policy performance separately on the chosen profile. Select tolerances and success criteria before comparing variants; no exact real-world calibration or zero-error joint target is required.

