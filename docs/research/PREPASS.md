---
title: ACT-004 neural and cybernetic control research prepass
kind: research
summary: Recommendations and discriminating experiments for expressive Cybersand actors under human, policy and model control.
owns: [research.prepass]
verified: 2026-10-18
---

# ACT-004 control research prepass

## Recommendation

[TENTATIVE] Build a layered controller around the authored game mechanics: fast response shaping and feedback, compact learned movement/skill policies, and interchangeable human or model decision sources. Begin with one useful skill on one accepted ACT-004 rig. Measure whether learning adds capability or better feel before expanding the framework.

The fidelity target is the owner's D-011 clarification: plausible emulation, purposeful flex/play and stable handling. A bounded gyro assist is sufficient; finite rotor-momentum bookkeeping and mechanically perfect connections are not required. Rapier supplies bodies and contacts while the game authors the response that makes them enjoyable.

This prepass is research and design. It contains no trained checkpoint, mechanics run or empirical controller ranking.

## What the live source already gives us

[SUPPORTED] The [pinned source audit](../evidence/2026-10-18-source-prepass.md) finds a substantial ACT003 Thunderbowl foundation on Cybersand main `b31a897b485a1913a67a02a39fb271495bacf027`: shared human/policy inputs, wire versus direct actuator lanes, permitted observations, strict command lifetime, headless sampled stepping, requested/applied/passive response records, local communication, an untrained recurrent graph and a mock slow-model/fresh-reflex adapter.

[Cybersand issue #178](https://github.com/techrote/cybersand/issues/178) requests the articulated Rapier migration. It remains open on the audited snapshot. Its old reference to #177 being active is stale: that selector PR has merged. The original NCA repository plan is also unmerged bootstrap documentation, not an existing trainer.

The key next task is to preserve and extend Thunderbowl's useful boundaries as actors become multi-part game bodies. A second control universe would lose existing distinctions and increase the cost of trustworthy comparison.

## What cybernetic control contributes

[TENTATIVE] Treat the actor as a feedback system with several useful forms of memory:

- Immediate response: damping, bounded motor correction, input shaping and contact-sensitive assistance.
- Skill state: charging, taking off, landing, gripping, recovering or releasing a tool.
- Short-term inference: remembering recent contacts, commands and observations to infer traction or hidden compliant state.
- Task context: where the actor is going, what it is using and when it should give up or replan.

A desired direction becomes a local movement objective; feedback adjusts the available actuators as contact and terrain change. High-level intent need not specify every limb or thruster. Low-level recovery should continue while a planner is delayed.

This hierarchy is useful even before neural learning. It isolates whether a failure comes from an impossible/unsupported command, bad response tuning, insufficient perception, an inadequate skill or a poor decision.

## A practical method comparison

| Candidate | Why it is worth considering | First limitation to test | Priority |
|---|---|---|---|
| Authored state machine with damped feedback | Cheap, inspectable baseline; easy personality and recovery tuning | May become brittle on unusual contacts and terrain | First |
| Behavioural cloning and corrective demonstrations | Starts from useful scripted/human motion | Errors compound in states absent from demonstrations | First learned comparison |
| Recurrent goal-conditioned policy | Uses recent history to respond under partial observations | Hidden-state reset, overfitting and inference budget | First learned comparison |
| Bounded residual policy | Adds learned correction to a competent baseline | Baseline may restrict useful strategies; clipped residuals complicate interpretation | First learned comparison |
| CPG/phase or gain modulation | Compact rhythmic control for articulated hoppers/crawlers | Inappropriate rhythm can restrict irregular or flying actors | Rig-specific |
| Shared modular/graph policy | Reuses a controller across different part layouts | Graph structure alone does not guarantee better learning | After a working per-rig skill |
| Actor graph communication | Learned coordination using actual permitted neighbors/messages | Benefit may come from information privilege or actor count | Later matched comparison |
| Spatial NCA or neural field | Local memory, hazard/repair/attractor signals, distributed behaviour | Coverage, update cost and action readout differ from an actor pilot | Conditional NCA track |
| Evolution strategies | Optimize response/gait parameters or small black-box controllers | Rollout cost grows; compare sample and wall-clock budgets | Bounded alternative |
| Learned world model / MPC | Predict consequences and compare short future plans | Destruction, contacts and authored assists cause model error | Only after a measured planning deficit |

These are candidates, not a universal winner ordering. [Control methods](CONTROL_METHODS.md) ties the comparisons to primary research. A small padded MLP/history baseline is necessary before attributing an improvement to recurrence or graph structure.

Residual RL combines conventional and learned control in published robot assembly work; that motivates the comparator, not a stability guarantee for the game. [Johannink et al.](https://arxiv.org/abs/1812.03201)

Shared Modular Policies demonstrated one shared modular controller across different planar agent structures. That is relevant to the short ACT-004 assemblies, but does not establish transfer to arbitrary damaged rigs or changed actuator meanings. [Huang et al.](https://proceedings.mlr.press/v119/huang20d.html)

## Emulation, compliance and personality

[TENTATIVE] The authored gameplay profile should specify response rather than mechanical perfection. Useful controls include gyro assist strength and ramp, position deadbands, transition hysteresis, soft end regions, connection play, damping, grip tolerance and recovery gains.

These should be purposeful and reproducible. Independent random jitter is a poor substitute for slack. Play can have a small internal state, such as which side of a clearance band is currently loaded. A policy can observe that state or infer it from a short history.

Excessive smoothing is also a failure: a stable actor that feels sluggish misses the intended vigour. Compare response onset and useful bursts alongside unwanted alternating correction. Keep rapid intentional jumps, recoil and scrambling distinguishable from chatter.

| Actor family or named example | Suggested useful learning target | Authored character to preserve |
|---|---|---|
| Ripper and grounded crawlers | Traversal, traction recovery and controlled drift | Scrappy lateral movement and commitment |
| Jolt and jump/charge actors | Directed launch, landing and quick recovery | Sudden purposeful bursts |
| Scrabbler and articulated leapers | Coordinated push, purchase and recovery after support loss | Active scrambling, imperfect but intelligible limb motion |
| Copter and multirotor variants | Orientation recovery, passage flight and payload compensation | Distinct lift/steering tradeoffs and recoverability |
| Pouncer | Launch/steer/land sequences under the eventual glide profile | Cybernetic squirrel-like commitment and correction |
| Digger | Progress through permitted material and adaptation after excavation | Aggressive burrowing through the native effect owner |
| Gyrocrate and Bellows | Recovery and movement using their authored response | Existing distinctive wobble/pulse behaviour |
| Lineshank and usable carriers | Aim, recoil recovery, grip/release and payload handling | Slim, directional handling rather than generic stabilisation |

This is a candidate task map from the issue and owner brief, not a claim that every listed physical mechanism is available. The rig manifest and ACT-004 implementation decide which experiment can actually run.

Style should be a visible condition or comparison: eager/hesitant, committed/cautious, punchy/soft response, willingness to drift, fast/settled recovery. Do not replace all identities with the same optimal-looking gait. Task and demonstration objectives can be combined, as shown by [DeepMimic](https://arxiv.org/abs/1804.02717); attainable Cybersand examples still need collection.

## Humans, decision models and frontier LLMs

[TENTATIVE] Share capabilities at explicit layers. A human may drive the wire/direct lane or issue a skill command. A compact decision model may select a skill. A frontier model can interpret a mission and choose among currently feasible skills. Each reaches the same game actuation owner.

The existing smart link already separates slow intent from fresh local reflex. Retain its source-anchored expiry. A late plan does not become fresh because it arrived now. Long-lived missions need a separate declared contract with local revalidation.

A useful model exchange describes what the actor can do, what its permitted sensors currently show, its active skill/result and available targets. The response selects a typed skill or bounded goal. Skill completion is observed, not inferred from the model saying it succeeded.

[SayCan](https://proceedings.mlr.press/v205/ichter23a.html) supports grounding language selection in executable, contextually feasible skills. [RT-H](https://rt-hierarchy.github.io/) supports intermediate motion descriptions and human corrective intervention. Neither shows that a remote model should directly update Cybersand's motors every physics tick.

A concrete sequence is “approach that object, grip it, carry it to the visible opening, release it.” The same approach/grip/carry/release machinery can execute a human shortcut or a model plan. The local controller deals with contacts and compliance; the game performs actual mounts/material effects. Failure or lost visibility becomes an explicit result that can trigger a new decision.

A human takeover should be immediate at the next simulation boundary. It clears the previous source's pending authority and armed requests as defined by the existing cancellation rules. This is internal control ownership, not an extra user approval flow.

## Where NCA fits now

[SUPPORTED] The existing actor NCA is a shared 12→8→4 recurrent graph with untrained weights. The original [Growing NCA](https://distill.pub/2020/growing-ca/) work concerns learned local updates for pattern growth, persistence and regeneration. Neither is evidence that NCA is already a strong actor controller.

[TENTATIVE] Preserve three experiments:

1. Part-local recurrent modules controlling a short assembly.
2. Actor-local recurrent modules sharing bounded peer messages.
3. A separate terrain lattice or active-material network shaping local behaviour.

A coarse field could encode attraction, hazards, construction/repair demand or useful terrain history. A human, script or model may place semantic goals into that field, while local policies interpret it through permitted sensors. Compare a hand-authored field first. A neural field should earn its complexity through memory, adaptation, coordination or robustness.

Repeated local update rounds increase communication reach. Global pooling or full-arena sensing changes the information budget even if the final readout is local. Hidden cellular state must follow declared movement/reset ownership.

The native material engine remains authoritative. A cellular network can request allowed deposition, conversion, movement or signals; it does not gain an unrestricted write path. The old generic conservation assumptions need adapting to the actual game rules, which may explicitly create/destroy material through usables.

## The initial research programme

[TENTATIVE] The [roadmap](../ROADMAP.md) provides issue-ready tasks, but the scientific path is simple:

- Inventory the actual accepted rig and freeze a small game/control profile.
- Establish a persistent host adapter and causal stepping/reset evidence.
- Tune a convincing authored baseline with the allowed compliance and assistance.
- Collect demonstrations and compare small learned control on one skill.
- Expand to morphology, coordination, tool sequences or model planning only when the first comparisons identify a useful next question.

A correct no-go is valuable: perhaps scripted control already meets the need, the observation profile hides a necessary state, the game host is too slow for the proposed budget, or a new policy is less fun despite a higher score. Record which explanation the evidence supports.

## Evaluation and practical limits

[TENTATIVE] Keep rig, sensors, assistance, action cadence and scenario sets matched when comparing controllers. Record both deployment and training privileges. A full-state teacher is a training advantage even if the final actor receives only local observations.

Use unseen scenarios and several independent training seeds; publish failures and uncertainty rather than one winning clip. [Agarwal et al.](https://arxiv.org/abs/2108.13264) explain why point estimates from few RL runs can mislead.

The [training design](../design/TRAINING_AND_EVALUATION.md) specifies termination/truncation, actual consumed-action logs, lifecycle faults, reward loopholes, morphology holdouts, latency sweeps and proposed compute ceilings. A million decisions at 10 decisions per simulated second is about 27.8 simulated hours; available game throughput should shape the budget before hardware or model size is prescribed.

No training speed, GPU compatibility, policy quality or final-game feel was measured in this prepass. The strongest present conclusion is an implementable research design aligned with the existing control lab and the owner's emulation target.
