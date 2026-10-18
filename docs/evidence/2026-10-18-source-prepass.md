---
title: Cybersand actor control source prepass — 2026-10-18
kind: evidence
summary: Pinned source audit separates existing sampled Thunderbowl control from the physical actor and training interfaces still required.
owns: []
verified: 2026-10-18 @ b31a897b485a1913a67a02a39fb271495bacf027
---

# Cybersand actor control source prepass

## Identity and evidence class

**Class: reviewed.** GitHub metadata, repository trees, canonical documentation and the source files linked below were read. No build, runtime mechanics check, benchmark, training, target-device execution or gameplay acceptance was performed in this prepass. Repository descriptions of earlier test runs are historical evidence, not new execution.

**Owner scope clarification for this prepass:** the target is plausible gameplay emulation. Mechanics are judged against an authored gameplay profile, which may include bounded gyro-like attitude assistance, emulated or compliant joints, and deliberate flex/play to reduce twitchiness and solver glitches. Exact prismatic locking, finite rotor-momentum bookkeeping, mechanical energy conservation and actuator-work accounting are not universal acceptance prerequisites. Source inventory below identifies available features and missing interfaces; it does not impose first-principles fidelity. Record the chosen approximations and validate their actual response, stability, control limits and reset behaviour.

- Cybersand main: [`b31a897b485a1913a67a02a39fb271495bacf027`](https://github.com/techrote/cybersand/commit/b31a897b485a1913a67a02a39fb271495bacf027), merged 2026-10-18 18:07:33 UTC.
- [ACT-004 / issue #178](https://github.com/techrote/cybersand/issues/178) remains open. Its initial baseline `0d915ff096e7478acee2adfd5d11d6b9883954c2` and description of overlapping #177 as active are historical: [PR #177](https://github.com/techrote/cybersand/pull/177) is merged into the inspected main.
- [ACT-003 / PR #174](https://github.com/techrote/cybersand/pull/174) merged as `40a141ce2331cabc1c3b5e031a1fba7107720908`; reviewed head `138f9c7aaca4dffff0b5beb2e60deff40fe76a61`. Its sampled control laboratory is already in the inspected main.
- The open-PR list and all 105 branch entries were checked. No published ACT-004 implementation PR/branch appeared in that snapshot. This does not establish absence of unpublished or subsequent work.
- neural-cyber-automata's original [main `3a6820e16ea28395c0bb7585307504262dda535a`](https://github.com/techrote/neural-cyber-automata/tree/3a6820e16ea28395c0bb7585307504262dda535a) contains only its initial README. The substantive preliminary programme is in open [bootstrap PR #1](https://github.com/techrote/neural-cyber-automata/pull/1), head [`20d063423fda1e037f328fb32c40c00c73b121a5`](https://github.com/techrote/neural-cyber-automata/tree/20d063423fda1e037f328fb32c40c00c73b121a5): documentation/checker/workflow, no model, trainer or host implementation.
- This recharter preserves that preliminary programme as history. Neither old plans nor new design establish trained intelligent behaviour.

The Cybersand [agent guide](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/AGENTS.md) requires claim-by-claim status, source/runtime identity, native cell ownership, main-thread Rapier ownership, immutable copied publications and bounded capacity outcomes. Its source repository remains unchanged by this prepass.

## Pinned source map

Every Cybersand source link in this record identifies the inspected main SHA, rather than a moving branch.

| Area | Source and responsibility |
|---|---|
| Owner-approved conversion | [physical-actor-programme.md](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/docs/operations/physical-actor-programme.md): all 29 rigs physical by default; sampled references retained; assembly, finite actuation and usable obligations remain future work |
| Existing control taxonomy | [thunderbowl-control-reference.md](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/docs/systems/thunderbowl-control-reference.md): lane, decision placement, perception, coordination, model family and training privilege are separate axes |
| Actor descriptions | [actor_rig_catalog.gd](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/godot/scripts/actor_rig_catalog.gd): 29 identities, capabilities, provisional mechanisms and explicit limitations |
| Sampled session | [actor_prototype_session.gd](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/godot/scripts/actor_prototype_session.gd): eight slots, producer/lane assignment, epochs, held actions, press edges, expiry and copied state |
| Sampled motors | [actor_rig_motor.gd](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/godot/scripts/actor_rig_motor.gd): authored integration and sampled terrain collision; no Rapier articulation |
| Direct sampled actuation | [actor_rig_actuation.gd](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/godot/scripts/actor_rig_actuation.gd): three four-channel profiles, resource scaling and 32-value actuation receipts |
| Command validation | [thunderbowl_contract.gd](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/godot/scripts/thunderbowl_contract.gd): exact versioned fields, range/type checks and preset registry |
| Arena owner | [thunderbowl_arena.gd](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/godot/scripts/thunderbowl_arena.gd): exclusive native world, filtered observations, commands, virtual scoring, failure handling |
| Controller scheduling | [thunderbowl_driver.gd](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/godot/examples/thunderbowl/thunderbowl_driver.gd): due-view gathering and per-controller actor binding |
| Physical owner | [rapier_physics_bridge.gd](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/godot/scripts/rapier_physics_bridge.gd): live bodies/RIDs, manual stepping, hard terrain, body sampling and cellular response |
| Native coupling | [cyber_native_cell_world.cpp](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/godot/native_extension/cyber_native_cell_world.cpp) and [rigid_body_coupling.gd](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/godot/scripts/rigid_body_coupling.gd): rectangle masks and packed body/sample results |

## What already exists

Thunderbowl owns a native `CyberNativeCellWorld` and a sampled actor session through `configure(config)`, `observation(actor_id)`, `step(commands)`, `snapshot()` and `close()`. Configuration creates a candidate before swapping the old instance. It fixes native workers to one, disables simulation-window and cadence LOD, seeds the generated pit layout, and advances one sampled motor/material tick of 1/60 second per step. The inspected `configure` seeds layout generation; it does not expose a general independent native-RNG reset contract.

The current arena is a 256×144-cell pit at (96,96), with 1–8 actors and 1–36,000 ticks. Virtual tags use HP=3, bounded aim/range/LOS and cooldown; they are arena bookkeeping. Eliminating an actor neutralizes control but does not remove a rigid body. Sampled actors do not collide with each other. The scene's full evaluator display gives human pilots information beyond a local-policy view.

| Existing boundary | Source-reviewed behaviour | Extension needed for ACT-004/training |
|---|---|---|
| Wire lane | Movement, aim, effort, trim, reel, primary/secondary/auxiliary; authored assistance varies by rig | Preserve behavioural intent through the authored physical/emulated mechanism; publish assists and any gameplay resource costs |
| Metal lane | Ripper, Copter and Gemini only; four channels; bypasses wire dispatch | Versioned actuator manifest for heterogeneous physical joints/thrusters/grips, including active-channel masks |
| Actuation receipt | Requested/applied channels and wrench, passive wrench, actual post-collision motion, mass/inertia, energy and flags | Per-part/joint identity, requested/applied assists, clearly labelled constraint diagnostics and mount state; rotor momentum only if the profile models it |
| Observations | Own 40-value rig state and 32-value actuation state; 9×9 material stencil spaced four cells | Physical proprioception, declared contact/tension sensing, body/actuator masks, freshness and availability |
| Local perception | Alive neighbors within 64 cells and clear LOS; no neighbor HP, energy, actions or recurrence | Preserve filtering before encoders, recurrent policies, planners and image adapters |
| Oracle | All actors' current sampled states/HP and coarse 32×18 terrain samples | Keep explicitly privileged training/evaluation route; no claim of exact full native world or future knowledge |
| Messages | Typed four-float payload, same-team/radius/LOS restrictions, next-tick delivery, source-derived expiry | Preserve old-snapshot updates; separately version any bandwidth, delay, topology or message-round experiment |
| Execution | Humans sampled at 60 Hz; other registered controllers at 10 Hz; motors/materials at 60 Hz | Configurable physical servo/policy/strategic rates with recorded hold and timeout semantics |

The strict command envelope already includes `run_id`, `actor_id`, `epoch`, `sequence`, `observed_tick`, issued `observation_id`, lease, lane/action, tag, message and cancel. Leases are 1–18 ticks, normally 12, anchored to the observation tick. Late receipt does not renew old authority. Structural checks are followed by owner checks for issued observations, live slot, producer/lane, ordering and expiry. Press edges coalesce into one bounded action per tick; cancellation and expiry clear armed actions.

These are useful building blocks for the recharter. They are not yet a general multi-source arbitration system: each actor has a configured producer/lane, and the driver is a trusted in-process adapter. An arbitrary controller process requires its own transport identity and assignment enforcement; a guessed actor/observation field must not grant control.

## What is physical now, and what is not

The [single-body trial](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/godot/scripts/physical_actor_trial.gd) and [main controller](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/godot/scripts/main.gd) provide live Rapier barrel/Ripper rectangles, 8×14 and 18×8 respectively. Main applies the latest cellular result, bounded input impulses, one Rapier step and then publishes body samples.

A/D currently requests a bounded horizontal central impulse toward a target speed; Q/E applies capped angular impulse; Space's rising edge supplies an upward pulse capped at 24, with 0.6-second simulation-time cooldown. The angular impulse is capped at 0.2 per call. These trial controls are keyboard-only and intentionally permit the upward pulse airborne. They are not grounded locomotion or a reaction-wheel mechanism.

No multi-part actuator graph, usable grip/mount/recoil lifecycle or physical Thunderbowl session exists in the inspected main. Its trial has no persistent rotor-momentum model; that is a source observation, not a required defect under the gameplay-emulation scope. The sampled direct lane explicitly reports `rigid_body_force_feedback: false`; its numerical mass/inertia/force labels cannot be reused as evidence of physical Rapier actuation.

Rough family grouping below is a **design implication of catalog descriptions**, not a delivered morphology specification:

| Family and examples | Physical questions to settle before learned control |
|---|---|
| Ground/contact: Runner, Crawler, Scout, Scrabbler, Ripper | Contact, friction, stance transitions, slip and propulsion response, including declared grip or joint assists |
| Flight/attitude: Skimmer, Copter, Gemini, Groundeffect, Pouncer, Dropbot | Thrust allocation, force application points, mass/inertia, bounded gyro-like assistance and authored aerodynamic approximations |
| Attachment/traversal: Walljumper, Clamberer, Swinger, Lineshank, Winchbug | Admitted anchors, grip compliance, attachment lifetime, tension, obstruction, release and propulsion accounting |
| Elastic/gyro mechanisms: Springjack, Gyrocrate, Bellows | Authored spring response, compliant end stops, intentional flex/play, bounded attitude assistance and recovery; rotor state only where useful to the chosen profile |
| Terrain/usable interactions: Digger, Rocketjumper, Magpie and tool-bearing rigs | Bounded material effects, clearance, recoil and reachability; distinguish edits/explosions from actual contact forces |
| Scripted/persona roster | Trainable controllers consume the same declared mechanisms as humans; fictional labels are not claims about named real models |

The owner programme asks for one to eight physical parts per actor. It does not establish one universal action vector, one gait, or a need for every usable to consume a body.

## Native coupling and capacity gaps

Current native coupling parses at most **16** records with IDs 1–16. Input stride is 11 floats; output stride is 9. Input describes center, rotation, rectangular size, velocity, angular velocity, mass and sample serial. It has no actor/part incarnation, shape schema or mount identity. Output provides central impulse, correction and contact/displacement/unresolved counts; it has no cellular reaction torque or detailed contact manifold.

If a configuration admits eight actors with eight separate body parts each, it needs **64 body identities before attached items and other physical objects**. This conditional arithmetic is not a requirement that the current arena support that maximum combination, nor evidence that 64 bodies are affordable. A smaller explicitly supported total with clear refusal is valid. The assembly boundary needs a world total and reservation/refusal contract; existing parsing truncates excess complete records, so configurations exceeding the admitted total must be refused transactionally before creation. Emulated joints do not by themselves require additional body identities.

Native rasterization is ordered by body ID, with first-writer ownership where masks overlap. Jointed parts that overlap near a hinge can therefore alter per-part cell contact attribution. Distinguish self-overlap from body/body contact; do not infer a new occupancy/material owner from a neural controller.

Current sweeps treat movement over 32 cells between samples as endpoint-only teleport, use at most 24 intervals, and interpolate shortest rotation. Fast spinning parts, thin geometry, overlapping attachments and sparse samples require actual mechanism fixtures. Existing rectangle masks are not collider-exact arbitrary polygon support. A richer physical shape contract is a separately bounded extension, with matched rendering, occupancy and collision semantics.

Native cells remain authoritative. [Actor effects](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/godot/scripts/actor_rig_effects.gd) already bound local excavation/blasts to radius four, 64 reads and 64 writes through owner-local `emit_disc` calls, with protected-material and wall-occlusion checks. Reuse or extend that authority rather than allowing policies direct cell writes.

The physical bridge de-duplicates cellular sample results, rejects age above eight samples, suppresses correction above one and attenuates impulses by age. It can apply an authorized terrain correction to transform/velocity; therefore “no transform setters anywhere” would misdescribe the existing coupling. Control, emulated mechanisms and corrections should remain behind the common body owner with declared semantics, so controllers cannot silently bypass the authored gameplay profile.

## Pinned Godot–Rapier capability review

The inspected [dependency lock](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/godot/third_party/rapier2d.lock.json) pins Godot 4.7.x and Godot Rapier 2D release v0.35.2. Upstream tag resolution during this prepass identifies commit [`172c66f5a88bf1164c2b8cf2d775ed889edefd75`](https://github.com/appsinacup/godot-rapier-physics/commit/172c66f5a88bf1164c2b8cf2d775ed889edefd75). These source observations still require a fixture against the exact vendored runtime artifact.

| Feature | Observed support and qualification |
|---|---|
| Manual world step | Cybersand already calls `space_step`, active-body batch state reads and `space_flush_queries`; live objects stay on the main thread |
| Rotary position control | Vendored [RapierPinJoint2D](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/godot/addons/godot-rapier2d/custom_nodes/rapier_pin_joint_2d.gd) exposes target angle, stiffness, damping and enable, plus joint type and optional IK |
| Finite rotary motor | Upstream [revolute implementation](https://github.com/appsinacup/godot-rapier-physics/blob/172c66f5a88bf1164c2b8cf2d775ed889edefd75/src/joints/rapier_revolute_joint.rs) handles 2D `JointParam::MAX_FORCE` and forwards it to the motor. [Joint construction/update](https://github.com/appsinacup/godot-rapier-physics/blob/172c66f5a88bf1164c2b8cf2d775ed889edefd75/src/rapier_wrapper/joint.rs) uses `ForceBased` and `motor_max_force`. A finite cap is source-supported through the server; the custom node does not expose a separate cap field |
| Default motor authority | [Joint base](https://github.com/appsinacup/godot-rapier-physics/blob/172c66f5a88bf1164c2b8cf2d775ed889edefd75/src/joints/rapier_joint_base.rs) initializes max force to `f32::MAX`; configure the authored response/limits rather than assuming this default supplies a useful gameplay bound |
| Slide/end stops | [Groove construction](https://github.com/appsinacup/godot-rapier-physics/blob/172c66f5a88bf1164c2b8cf2d775ed889edefd75/src/joints/rapier_groove_joint_2d.rs) builds a bounded pin-slot. It does not automatically lock relative orientation. A compliant or emulated slide with deliberate rotation/play is valid when it matches the authored profile; exact prismatic fidelity is optional |
| Spring | [Spring construction/update](https://github.com/appsinacup/godot-rapier-physics/blob/172c66f5a88bf1164c2b8cf2d775ed889edefd75/src/rapier_wrapper/joint.rs) uses a ForceBased spring with rest length/stiffness/damping. That path alone does not add powered-actuator limits or end stops; composed or emulated response may supply the chosen gameplay behaviour |
| Reaction sensing | [Extra server API](https://github.com/appsinacup/godot-rapier-physics/blob/172c66f5a88bf1164c2b8cf2d775ed889edefd75/src/servers/rapier_physics_server_extra.rs) exposes linear and angular reaction impulse. Readback includes constraint, limit and motor contributions; divide by actual step duration for average reaction |
| Multibody qualification | The reaction-impulse implementation reads impulse joints; multibody joints report zero. Zero is unavailable telemetry, not evidence of zero physical load. Actuator-work accounting is optional unless a specific experiment claims it |
| IK | Source-exposed and usable as a declared emulation/controller aid. Validate the realised gameplay response and label it correctly; availability does not establish physical work or conservation |

A focused capability fixture should freeze coordinate/unit/sign conventions, gain/limit behaviour, anchors, intended flex/play, disabled controls, slip, damping, end-stop response and recoil before policies depend on those channels. Judge responsiveness, stability and bounded predictable assistance against the authored profile. Exact conservation, rotor bookkeeping and perfect geometric constraints are optional experiment-specific properties, not general acceptance gates. Deliberate compliance or emulated joints that avoid twitchiness and solver glitches are acceptable.

## Timing, determinism and causal reset

The [desktop threading contract](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/docs/architecture/simulation-tick-and-threading.md) has independently paced native and Rapier timelines. The cellular owner samples copied body state; main consumes delayed responses. Render updates and hard-terrain rebuilds have further clocks/budgets. Training through synchronous Thunderbowl and deploying through asynchronous desktop can therefore change the transition distribution.

A physical training host needs a declared causal cycle: completed state → admitted control → physical step/contact → copied part/cell coupling → material step → completed publication, with precisely specified prior-response use. The order is a proposal to test; current desktop/Web schedules are not interchangeable proof. Count completed physics/material ticks separately, timestamp observations and effects, and expose actual command consumption. If real-time deployment preserves lag, evaluate matching lag or train/evaluate controlled latency perturbations.

Synchronous actor policies must all read the same completed state. Existing Thunderbowl gathers due views before controller invocation and updates messages from prior delivered state. Preserve this under batching; in-place recurrent updates could give later actors extra information reach.

A reset must create a new episode identity and invalidate old observations, commands, model replies, pending effects/messages and recurrent state. Physical extension also needs fresh actor/part/joint/mount incarnations and invalidation of old cellular results and terrain publications. The current result serial alone does not identify a newly reused body slot. Failed native ticks quarantine the owner and retain the last completed publication; they are not legal episode transitions to retry or continue.

The dependency's deterministic build label does not establish end-to-end deterministic replay. Rapier body/joint order, solver settings, material scheduling, random streams, step sizes, stale-result decisions and policy state all matter. Crypto-generated run IDs differ between resets and belong to identity, not physical-state equality. Pair trials by frozen scenario/random streams and compare substantive state while retaining distinct identities.

[Level saves](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/docs/reference/level-saves-and-replay.md), [gameplay recordings](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/docs/operations/gameplay-recording.md) and the current final arena JSON do not serialize complete resumable simulator/controller state. Fresh deterministic reset plus an accepted action trace is an initial reproducibility target; arbitrary mid-episode restore is a separate feature.

## Existing neural/model links and minimal training bridge

The existing [graph candidate](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/godot/examples/thunderbowl/nca.gd) is an untrained 12→8→4 residual MLP with four recurrent values per actor and one old-message update round. It is not a terrain lattice, and changing a weight artifact's metadata to `trained_external` does not establish training quality.

The [smart link](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/godot/examples/thunderbowl/smart_link.gd) already separates a slow intention from fresh local reflex control. It allows one outstanding request, at least 30 ticks between requests, and source-anchored validity up to 120 ticks. Responses echo request/run/actor/epoch/observation identity. Missing/expired plans cancel actions. This two-second simulation-time validity needs deliberate reconsideration for slower frontier-model tasks, rather than renewed authority on arrival.

The example transport performs bounded synchronous filesystem reads/writes between steps. It does not wait for a model reply, but filesystem latency can delay the next step. The [headless CLI](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/godot/examples/thunderbowl/run_thunderbowl.gd) supports fast stepping and optional real-time pacing; these are different evaluation classes. It currently runs a whole match and writes final state, rather than supplying per-step learner transitions.

Minimal bridge design derived from this audit:

1. Reuse the sampled arena first as an explicitly labelled transport/observation reference; introduce the accepted physical actor session through the same owner boundary after ACT-004 qualification.
2. Add an external-policy registry/profile and a persistent bounded process protocol with `hello/capabilities`, `reset`, `step`, `observe` and `close`; enforce slot assignment and packet size at the transport owner.
3. Return only permitted policy observations, action masks, consumed-action receipts, per-step reward increments, episode IDs and separate task termination/time-limit truncation. Keep full evaluator state on a declared separate channel.
4. Specify action hold, servo rate, policy cadence, horizon and command expiry. Do not extend the old four-channel v1 packet by silently reinterpreting reserved channels.
5. Reset controller recurrence, normalizers and message/effect state with the episode; maintain per-environment state when batching. Save source/runtime hashes, simulator config, actor/shape/actuator manifests, policy artifact hash and seeds with every rollout.
6. Build CPU/mechanics and policy comparison fixtures before training throughput optimization. Test per-process worlds and vectorization explicitly; a main-thread-only Rapier API is not permission to call it from learner worker threads.
7. Provide a deployment parity test for the physical owner and policy artifact, then quantify asynchronous latency, terrain changes, payload/mount changes and sensor loss. Separate mechanism pass, trained checkpoint, held-out evaluation and owner feel acceptance.

No trainer, reward optimiser, external neural inference runtime, physical actor learner bridge, complete replay, or learned-control advantage was established by this prepass.

