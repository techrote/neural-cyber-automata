---
title: Control methods for Cybersand actors
kind: research
summary: Evidence, plausible gameplay emulation, candidate controllers and discriminating experiments for the planned articulated Rapier actors.
owns: [research.control-methods]
verified: 2026-10-18
---

# Control methods for Cybersand actors

## Evidence and scope

[SUPPORTED] This is a researched design prepass for [ACT-004, Cybersand issue #178](https://github.com/techrote/cybersand/issues/178). The [inspected ACT-003 control reference](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/docs/systems/thunderbowl-control-reference.md) and [source audit](../evidence/2026-10-18-source-prepass.md) record a sampled native headless lab, 60 Hz motor/material stepping, 10 Hz scripted/NCA/reflex decisions, 60 Hz human control, and requested/applied/passive/actual-motion receipts. Its existing 12→8→4 NCA graph is untrained. ACT-004's proposed articulated owner is not yet published. Neither this document nor a paper's benchmark constitutes implementation, training or acceptance evidence for these actors; current repository facts belong in [STATE](../STATE.md).

[TENTATIVE] Extend that lab with capability-specific controllers and comparisons. An actor's one to eight bodies, joint topology, actuator geometry and usable mounts define its gameplay capabilities. The research target is reliable, expressive behaviour on destructible cellular terrain. A universal neural controller is one candidate, not a prerequisite.

The owner's clarified target is plausible physical emulation: bounded gyro assistance, approximate connections, pseudo-flex and controlled play are desirable when they improve feel. Finite gyro momentum storage, perfect joint representations, strict passivity and physical energy accounting are not launch requirements. The plant to learn is the authored gameplay system, including these intentional accommodations.

## Feedback architecture and common abilities

[TENTATIVE] Keep motor regulation at the existing 60 Hz physics cadence. Initially evaluate learned decisions at the existing 10 Hz cadence, holding or interpolating documented targets between decisions. Fast contact handling can be a separate deterministic primitive; adding it changes the baseline and must be recorded. Human, scripted, decision-model and frontier-LLM commands use the same authored abilities. An asynchronous model response cannot be required to complete a physics tick.

Use three distinct quantities: desired motion, requested actuator command, and realized response. Implement desired motion through the actor's documented motor/assist primitives; do not let one command source gain an accidental transform-overwrite capability. Those primitives bound forces, assist torque, target ranges and rate changes sufficiently to avoid numeric explosions and uncontrolled motion. Effort or resource limits can be authored gameplay proxies rather than physical work integrals.

For a residual controller, a useful experimental form is:

\[
u_t=P_{\mathcal U(s_t)}\bigl(u_{\rm baseline}(s_t,c_t)+\alpha\Delta u_\theta(o_t,c_t,h_t)\bigr).
\]

Here projection means the actual runtime mixing/limiting operation, not necessarily Euclidean projection. The admissible set describes the actor's current abilities and configured gameplay limits. Bound the residual before combination and the combined request afterward. Assess the resulting motion in contact: bounded outputs alone do not eliminate chatter or instability.

[SUPPORTED] [Rapier's joint documentation](https://rapier.rs/docs/user_guides/javascript/joints/) describes motor PD targets and maximum force/torque. This establishes underlying concepts, not Godot API availability. The [pinned wrapper audit](../evidence/2026-10-18-source-prepass.md) found a revolute maximum-force path using `ForceBased`, with an effectively unbounded default. Its groove joint is a pin-slot mechanism, not automatically an orientation-locked prismatic joint. [TENTATIVE] Configure finite motor authority and calibrate useful response; choose a guided/compliant approximation when that produces the desired connection more reliably than an idealized construction.

## Motor mixing, gyro assistance and compliant play

[TENTATIVE] Begin with transparent actuator mixing: distribute move/turn requests across available motors and thrusters, clamp outputs and ramp changes. Explicit prioritization can preserve balance or aiming when a movement request exceeds the designed authority. For analysis, a 2D thruster with direction `d` and offset `r` contributes a force/torque column `(dx, dy, rx*dy - ry*dx)`. This helps reveal why a layout struggles without requiring an optimizer in the first implementation.

[SUPPORTED] [Härkegård's original allocation paper](https://www.diva-portal.org/smash/get/diva2%3A316757/FULLTEXT01.pdf) provides a constrained optimization approach with actuator-position/rate limits and redistribution under saturation. [TENTATIVE] Try it only if mixed or damaged actuator layouts expose a concrete weakness in simple mixing. It does not solve articulated contact planning or establish which motion feels best.

[TENTATIVE] Treat the gyro as a bounded assist torque with an authored response curve, damping and optional cooldown/effort cost. It can help upright recovery or expressive turns without simulating rotor momentum storage or desaturation. Rate limits and gain scheduling should prevent the assist from becoming a twitchy pose clamp. Its visual justification and consistent limits matter more here than a spacecraft-equivalent internal model.

Connections can deliberately include slack, compliance and modest angular play. Endstops can be softened with damping or a short deceleration zone. Approximate slide/spring assemblies are acceptable if their behaviour is consistent under load and remains within an authored motion envelope. Predictable flexibility is useful; frame-dependent glitches are not a personality feature.

[TENTATIVE] PD/impedance regulation should provide posture, landing compression, bracing and tool alignment. Use measured velocity for damping; introduce integral action only for a demonstrated steady error with anti-windup. Add deadbands near settled targets, hysteresis for contact/skill transitions, command slew limits, contact filtering and recovery-specific gains. Tune them together against input latency and responsiveness: excessive smoothing can remove the requested vigour. The foundational [impedance-control paper](https://summerschool.stiff-project.org/fileadmin/pdf/Hog1985.pdf) motivates regulating interaction as well as position; a formal passivity proof is unnecessary for this gameplay target.

Changing spring targets or gains can create abrupt kicks, so bound/ramp those parameters and test worst-case loads. The audited wrapper does not automatically cap its spring path. Total joint-reaction impulse includes constraint/limit effects; use it as a response/load diagnostic where available, not an exact motor-work meter. Unknown telemetry must remain labelled unknown. An authored effort score is sufficient when the intended mechanic needs a cost.

## Candidate comparison

All recommendations in this table are [TENTATIVE]; the linked papers support the stated method, with validation limited to their own experiments.

| Candidate | Useful role here | Discriminating limitation |
|---|---|---|
| PD/impedance plus finite-state skills | First posture, aim, brake, brace and recovery baseline; inspectable gains, soft targets and transitions | Contact transitions and support collapse can defeat a locally effective controller. High gains can create chatter; integral action can wind up. |
| Central pattern generators | Periodic coordination for articulated walkers, hoppers or pumps; learn phase, amplitude or frequency modulation | Unsuitable as the sole controller for aiming or arbitrary recovery. [CPG-RL](https://arxiv.org/html/2211.00458v1) learns oscillator parameters, but used 100 Hz policy/1 kHz motor loops and reports simulator-exploiting joint-space gaits. Its cadence is not ours. |
| Residual reinforcement learning | Improve traction, landing, recoil recovery or contact sequencing around the baseline | Residual cancellation, saturation and limited residual authority can obscure improvement. [Johannink et al.](https://arxiv.org/html/1812.03201v2) superpose conventional and learned control for contact assembly; they do not establish arbitrary locomotion stability. |
| PPO | First on-policy learning comparator with current rollouts and minibatch updates | Native simulation throughput may dominate training cost. [The original PPO paper](https://arxiv.org/abs/1707.06347) includes locomotion benchmarks, not this material/physics coupling. |
| SAC | Off-policy continuous-control comparator when reusing expensive rollouts is valuable | Replay must remain compatible with the authored dynamics and action schema; stale mechanics invalidate a casual mixture. [The original SAC paper](https://proceedings.mlr.press/v80/haarnoja18b.html) combines replay-compatible actor-critic learning with an entropy objective. |
| Frame history or recurrent policy | Infer missing contact history, assist state, material response or sensor delay | Extra state needs reset, snapshot and ownership rules. [Ni et al.](https://proceedings.mlr.press/v162/ni22a.html) show well-configured recurrence can be a strong POMDP baseline; it does not replace necessary observations. |
| Behaviour cloning, then DAgger | Bootstrap from human/scripted skills and collect corrections on learner-visited states | Cloning only successful demonstrations misses recovery states. [DAgger](https://proceedings.mlr.press/v15/ross11a.html) addresses the changing observation distribution, but needs a competent queried teacher. |
| Shared modular/graph policy | Share control modules across joints and morphology families | Compare with a small padded/masked MLP before accepting extra complexity. [Shared Modular Policies](https://proceedings.mlr.press/v119/huang20d.html) demonstrated planar morphology transfer; no arbitrary damaged-topology guarantee follows. |

Policy architecture and training algorithm are separate axes: PPO or SAC can train an MLP, a recurrent network, a residual policy or an oscillator modulator. Do not compare several changed axes at once and attribute the result to one algorithm. Behaviour cloning can initialize later RL; a scripted teacher sets an initial example, not an ultimate capability ceiling.

## Observations, action records and policy lifecycle

[TENTATIVE] Start with body-local proprioception: relative poses and velocities, gravity direction; joint positions, speeds, limits and masks; contacts; actuator/assist saturation and relevant gameplay resources; mount orientation and pending recoil; command age; and a bounded local terrain description. Supply morphology parameters such as mass, anchors, joint type and assistance strengths when they vary. Use exposed measurements; privileged training-only state must be identified and excluded from deployed actor inputs.

A masked MLP is a useful first reference for a maximum of eight bodies. Graph nodes can represent bodies and edges joints, with global command/tool context shared explicitly. Message-passing depth limits instantaneous communication through long chains. Hold out whole topologies and actuator layouts when testing transfer, not only new random dimensions of a familiar skeleton.

Extend ACT-003 receipts with schema/morphology revisions, timestamps and explicit limiting causes where needed. Store both requested action and realized actuator output or available response proxy. If the environment action is the requested vector, train the critic using that convention and treat mixing as part of the transition. Switching to an applied-action critic requires a consistent formulation; do not silently relabel data. Contact-driven actual motion is an outcome, not the action that caused it.

Specify hidden-state treatment for episode reset, despawn/respawn, damage, possession transfer and snapshot restore. Evaluate stale commands and delayed observations deliberately. Training must reproduce the target gameplay emulator's causal ordering, command hold, material rules, assists and compliant play. Fidelity to that authored system matters; fidelity to every real-world mechanism does not.

## Energetic personality within controllability

[TENTATIVE] Personality should be expressed through attainable gameplay preferences: acceleration response, stance width, hop cadence, permitted overshoot, bracing style, gaze/aim motion and recovery strategy. A cautious carrier, excitable hopper and forceful rammer can share the same command vocabulary while choosing different trajectories. Compliant settling and small purposeful anticipation can make motion lively without high-frequency twitching.

Some weak behaviour is an ability/design problem: insufficient thrust, assist torque, contact leverage or suspension travel cannot be repaired by a better reward. Establish the intended manoeuvre envelope, changing authored assistance where that improves the actor. Then optimize task success with separate style measures; an effort penalty alone can encourage immobility, while rewarding angular activity can encourage purposeless spinning. Recoil and recovery should be legible and consistent, including deliberate stabilization assistance.

[SUPPORTED] [DeepMimic](https://arxiv.org/abs/1804.02717) combines motion imitation with task objectives to obtain physically simulated character skills. [TENTATIVE] Its useful design lesson is explicit style examples: short, achievable reference motions plus task goals. Retargeting flashy trajectories without checking this actor's limits would give the trainer contradictory objectives. Compare replay clips at matched tasks and seeds, alongside numeric metrics.

## Experiments that decide the design

[TENTATIVE] Each comparison uses the same authored gameplay plant, motor/assist limits, observations, command timing and scenarios. Record training seeds, rollout count, simulated time, wall time and inference cost separately. Use multiple training seeds and held-out scenarios before selecting a default; assess engaging movement and player responsiveness alongside success and robustness.

| Experiment | Comparison and decisive measurements |
|---|---|
| Authored ability contract | Motor disabled, bounded drive/gyro assist, softened endstop, spring release and thruster offset. Check sign/range, defined play, numeric stability and comparable abilities for every command source. |
| Control representation | Deterministic baseline; direct motor-target RL; bounded residual; oscillator/gain modulation. Compare success, recovery time, tracking, effort proxies and limiting under matched abilities. |
| Support changes | Rigid floor, granular bed, slopes and terrain removed during stance. Measure falls, immobilization, recovery and collateral material disturbance. |
| Cadence and feel | Existing 10/60 Hz decisions/physics with defined holds; vary deadbands, hysteresis, filtering and ramps. Inject delay/jitter; measure response lag, unwanted chatter, endstop impacts and recovery. |
| Recoil and payload | Move a usable mount, vary payload and fire during stance/flight. Measure target error, recovery time and use of finite authority. |
| Memory value | Current observation versus frame history versus recurrence; hide only specified information. Measure benefit under sensor gaps and verify reset/restore behaviour. |
| Morphology transfer | Matched MLP versus modular policy; hold out topology, actuator direction and damage pattern. Report each family, including impossible manoeuvres separately. |
| Style and exploitation | Task-matched energetic/cautious variants; allow useful exploitation of intended assists and compliance. Flag unwanted solver chatter, teleport artifacts, runaway kicks and reward loops. Compare progress, effort and replay quality. |

Keep an independent failure suite and do not tune repeatedly against its seeds. If learning adds no reliable benefit over the baseline, retain the baseline for that family. A successful controller for one morphology does not qualify all actors.

## Conditional model prediction and planning

[TENTATIVE] Short-horizon MPC becomes useful when anticipatory bracing, landing or recoil handling beats local feedback enough to justify planning cost. First test local prediction from the gameplay emulator or a documented reduced model. Approximation is acceptable; measure whether its predictions improve decisions around the authored soft contacts and assists.

[SUPPORTED] [TD-MPC2](https://arxiv.org/abs/2310.16828) performs trajectory optimization in a learned latent world model across continuous-control tasks. [TENTATIVE] Consider this later if rollout cost is demonstrably limiting. Destruction changes contact topology; inaccurate imagined transitions can be actively exploited by a planner. Require held-out multi-step prediction and decision-quality tests around those events, including planning latency and fallback behaviour. Success on other simulators does not establish transfer here.

The first useful learned result is a bounded improvement to one demonstrated skill on the actual actor plant. That evidence can then justify broader recurrence, morphology sharing, imitation datasets or predictive planning.
