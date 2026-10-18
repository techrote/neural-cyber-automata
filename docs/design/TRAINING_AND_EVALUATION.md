---
title: Training environment and evaluation design
kind: guide
summary: A bounded route from authored actor mechanics to reproducible learning comparisons and deployment evidence.
owns: [design.training-evaluation]
verified: 2026-10-18
---

# Training environment and evaluation

This is a proposed experiment specification under D-010–D-013. Nothing here reports a training result. The target is the actual Cybersand gameplay profile, including simplified gyro assistance, compliant connections and deliberate flex/play.

## What the first experiment should answer

Can a small learned controller make one actor more responsive, capable or robust than a reasonably tuned authored controller, without losing its character or gaining extra abilities?

Start with a single rig and one interpretable skill. Suggested choices are Ripper traversing irregular ground, Jolt performing directed hops with recovery, or Copter recovering from a disturbance. Select from whichever ACT-004 rig and observation surface actually land first. Vector is a useful personality reference; do not assume that its sampled mechanism is already an articulated physics implementation.

Compare a tuned authored baseline, a behavioural-cloning policy and a small recurrent residual policy. An untrained network and neutral/no-drive actor are sanity controls. A direct learned-action comparator helps determine whether the baseline is helping or restricting learning. Do not start with all rigs, free-form missions and competitive multi-agent training at once.

## Environment route

Use three clearly labelled lanes:

| Lane | Purpose | What it cannot establish |
|---|---|---|
| Recorded/dummy protocol fixture | Shapes, schemas, lifecycle and transport tests | Useful control or physics behaviour |
| Existing sampled Thunderbowl | Prototype adapters, logs and command-source comparisons | ACT-004 contact, joint or payload performance |
| Accepted ACT-004 game host | Train and qualify the authored Rapier actor mechanics | First-principles mechanical fidelity, which is not the goal |

The recommended deployment-relevant host is a persistent headless Godot process using the same pinned Cybersand/Rapier build and relevant configuration as the game. Python handles learner/evaluation orchestration. A transport should be bounded and request/response driven; use one host per environment initially, then profile isolation and batching.

[Godot RL Agents](https://github.com/edbeeching/godot_rl_agents) is a useful existing integration reference: Godot/Python interfaces, multiple learner wrappers and recurrent agents already exist. Audit and adapt a suitable piece only if it fits Thunderbowl's ownership, clock and schema contracts. Its documented experimental ONNX route is not proof that it works with this project's Godot build or export targets.

A Python Rapier copy, MuJoCo scene or differentiable toy is a possible development surrogate. It must have a separate identity, and its policy needs re-evaluation against the game. Reimplementing the material engine solely to obtain GPU batches is not the first milestone.

## Step semantics

A proposed adapter follows [Gymnasium's reset/step API](https://gymnasium.farama.org/api/env/): reset returns (observation, info) with an owned/copied observation; step returns observation, reward, termination, truncation and diagnostic info.

The initial learning step spans six base ticks when comparing the existing 10 Hz policy against the 60 Hz game. The manifest records the actual cadence and any interpolation, slew limiter or local controller that runs between decisions.

For each transition:

1. Observe a completed, identified host state and issue its identity.
2. Submit an action against that observation and the actor's current lane/profile.
3. Advance the host's accepted schedule for the declared ticks; record when the request was consumed.
4. Copy the next permitted observation, interval reward and terminal result.
5. Store requested action, admission, applied response and passive motion separately.

Current sampled Thunderbowl reports cumulative rewards: derive interval reward from successive totals, reset the previous total at episode start, and never award the frozen terminal snapshot twice. The physical adapter must declare whether its source fields are cumulative or already incremental. The six-tick learning repeat is separate from v1's 12-tick helper lease and 1–18 tick expiry range.

Stop the repeated-action interval at its first terminal tick; never advance a replacement/reset episode under the old action. Return the last valid final observation, actual consumed tick count and terminal reason. State whether reward/discount use base-tick or decision-step units; variable-duration intervals use the corresponding elapsed-time discount. Time-limit bootstrapping uses a valid final observation, while recurrence/advantage recursion still resets across episodes. A transport/desynchronization error without a valid next state cannot bootstrap from a stale observation or a newly reset episode.

Do not pair an action with a later observation merely because the action was queued. Multi-rate material/physics publication needs recorded timestamps and a coherent causal interpretation. A headless loop that runs Rapier faster while leaving material effects on unrelated wall time changes the task. It requires correction or an explicit different environment label.

Separate task termination from administrative truncation. A run stopped by an external budget or transport error must not automatically become task failure with zero future value. Task-intrinsic deadlines are terminal only when defined as part of the task and made observable where needed. [Gymnasium's time-limit guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/) explains why the distinction affects value targets.

A physics/contract integrity error is recorded as an invalid experiment transition or explicit environment failure, not a silently successful/ordinary terminal episode. Exclude unusable transitions from learner updates as necessary, but count every policy-triggered integrity failure and censored rollout in evaluation; discarding them must not improve the apparent success rate.

Stepped training deliberately pauses for a learner request; deployment qualification also includes a continuously advancing real-time lane. In that lane inject measured inference/transport delays while local control and the game keep running. A policy trained with paused decision time is not evidence of real-time model control. Declare the lane on every result.

## Reset and replay

Reset must invalidate old commands and model replies, actor/part/mount identities, recurrent state, skill progress, input edges/armed charge, messages and queued effects according to the host's lifecycle. Reset the actual world state, timers and random streams. Restoring visible positions alone is insufficient.

Cancellation or source takeover is not a world reset: clear/revoke pending control and armed input at the next owner boundary, while retaining actual body velocity, compliant loading/play and existing physical consequences unless a named authored recovery action changes them. Reset alone reconstructs the episode's mechanical and controller initial state.

Record actual consumed actions and event timing. Reconstruct episodes through deterministic creation order when supported, or use a complete host-supported snapshot. Do not assume a serializable evaluator summary or a generic Rapier API gives a complete Godot/native-material snapshot.

Test repeatability on one pinned build first, then compare gameplay metrics and trajectories with declared tolerances across relevant deployment builds. [Rapier's determinism guidance](https://rapier.rs/docs/user_guides/rust/determinism/) has conditions on build configuration, ordering and initial values; it cannot establish determinism of the surrounding host. Network quantisation is a separate question.

## Observation design

Start with compact structured observations instead of rendered pixels. The profile should define:

| Group | Candidate values | Important distinction |
|---|---|---|
| Own motion | Local-frame velocity, angular velocity, orientation as sine/cosine, target displacement | Normalize by documented game scales; no assumed SI units |
| Parts and connections | Relative pose/velocity, connection target/error, limit/soft-region state, masks | Missing slot differs from a present stationary part |
| Authored mechanics | Assist mode/ramp, charge, cooldown, grip state, optional effort/fuel proxy | Expose influential hidden state or test memory explicitly |
| Local terrain | Fixed rays or small semantic/material patches with range/visibility masks | No full-world nearest-target preprocessing for a local policy |
| Local objects | Permitted visible target/mount candidates and their ages | Remembered objects do not receive unseen fresh state |
| Command history | Recent requested/applied action, source age, intervention flag | Gives recurrence a causal history, not hidden judge state |
| Goal/style | Skill target, pace, aggression, damping/recovery preference within bounds | Style is conditional behaviour, not extra actuator authority |

For 1–8 parts, a padded vector with masks is an honest first baseline. Stable part ordering matters; an arbitrary index permutation is a distinct test. A graph policy uses topology and shared module weights later. Do not let a fixed actor-ID embedding memorize every topology and then claim morphology transfer.

Keep evaluator state in a separate channel. Gymnasium info may contain privileged diagnostics; neither a policy nor its preprocessing helpers should receive all of it by default. Centralized training with a privileged critic or teacher is permitted as a declared training privilege, not hidden deployment information.

## Learning ladder

| Candidate | First use | Promotion question |
|---|---|---|
| Authored state machine plus PD/response shaping | Establish controllable movement and recovery | Is the task and actor capability sensible? |
| Behavioural cloning | Learn from scripted/human examples on the same profile | Does it reproduce useful behaviour on unseen initial states? |
| DAgger-style corrective collection | Label states the learner actually reaches | Do teacher corrections reduce compounding errors? |
| PPO with small MLP/history/GRU | General contact/terrain skill learning | Does it improve a meaningful outcome within the budget? |
| Residual PPO around baseline | Improve traction, landing, recovery or aim compensation | Is improvement due to learning rather than extra assist or tuning? |
| SAC | Reuse expensive environment interactions | Does saved sampling outweigh replay/recurrence complexity? |
| CPG or bounded gait-parameter policy | Rhythmic articulated movement | Is the constrained action representation helpful for this rig? |
| Shared modular/graph policy | Transfer across assemblies and partial damage | Does it beat a masked MLP on held-out structures? |

Detailed evidence and method limitations are in [CONTROL_METHODS.md](../research/CONTROL_METHODS.md). PPO can differentiate its policy objective without differentiating Rapier. Ordinary Growing NCA backpropagation through a differentiable grid does not make a native game differentiable.

For residual control define the policy's output space before training. Store the sampled residual/request and its likelihood consistently, along with the bounded combined action and applied receipts. Contacts and passive motion are consequences, not replacement action labels. Changing the baseline changes the residual policy's environment and artifact compatibility.

Recurrent PPO stores the old policy log-probability for the sampled request under its recorded observation/history. Train ordered sequences with declared unroll length, initial hidden-state reconstruction or burn-in, padding/loss masks and episode-boundary reset masks. Human/model takeovers, replaced actions and mixed-source intervals are labelled; exclude them from the on-policy actor loss unless an explicit mathematically valid collection method handles them. Keep intervention demonstrations as a separate dataset rather than assigning them the autonomous policy's likelihood.

Begin with fixed recurrent weights during gameplay. Hidden-state adaptation is allowed; unrestricted online weight updates would create different reproducibility and evaluation questions.

## Tasks, rewards and feel

Train an outcome rather than a visual adjective alone. A traversal task can reward progress toward a goal and successful arrival while penalizing a task-defined, observable deadline miss, unwanted flips or wasteful action. Administrative budget or transport cutoffs are not that deadline. A hop task can combine target landing, recovery and time. A usable task can require correct approach, attachment, aim, actuation and release.

Treat reward as a design proposal. Publish terms, normalization and caps before the run. Use signed progress or one-time checkpoints so oscillating back and forth cannot farm reward. Correct progress references after terrain changes explicitly. Exclude evaluator metrics from policy input even if they supply a training reward.

Energy can be an authored effort proxy; do not demand a physically exact work ledger. If fuel/heat/charge are absent from the game, do not invent them as mandatory resource mechanics for learning.

Keep motion style visible in evaluation: response onset, useful burst speed, time stuck, landing recovery, drift, pauses, wobble and usable success. Measure unwanted high-frequency alternating correction separately from purposeful gait, recoil and impacts. “Low jitter” must not reward standing still, and “vigour” must not reward aimless thrashing.

Offer matched clips of scripted, cloned and learned control with the same camera, initial state and assist profile. Owner judgement of feel remains distinct from numeric success.

## Scenario and data splits

Split by scenario family, topology and event pattern, not only random number seeds. Keep separate training, tuning and sealed final evaluation sets. Repeated tuning on a holdout converts it into a tuning set and requires a new final split.

Candidate tests include firm/granular support, slopes and gaps, support removed mid-stance, sudden recoil, offset payloads, temporary sensor loss, changed compliant-play setting, altered grip state and visible-target disappearance. Use feasible authored profiles; a motor-disabled actor is a robustness/failure-response test, not always a solvable locomotion task.

For morphology claims, hold out whole joint/actuator arrangements as well as lengths/masses. For coordination, fix count/area/actuation and compare independent actors, designed neighbor steering, messages and learned coordination separately. Shared weights do not imply shared memory or team information.

Use at least three independently trained seeds for a screening comparison and preferably five for a promotion comparison; these are proposed minimums, not a statistical-power guarantee. Use paired evaluation scenarios across candidates. Report individual runs, failures and interval uncertainty. Resample at the trained-policy level, not by pretending thousands of episodes from one checkpoint are independent training runs. [Agarwal et al.](https://arxiv.org/abs/2108.13264) motivate uncertainty-aware RL evaluation and robust aggregate summaries.

Matched algorithm comparisons hold observation/action timing, information and assistance fixed. Human-versus-agent system comparisons may retain the native 60 Hz human and 10 Hz policy interfaces, but are labelled asymmetric system comparisons with those differences reported. A changed decision layer is not a pure algorithm comparison.

## Negative controls and decision gates

| Check | Reject or revise when |
|---|---|
| Contract and profile parity | An undeclared advantage gives one source fresher sensors, stronger assistance, extra channels or ignored invalid actions |
| Idle/untrained controls | Rewards rise without performing the task |
| Reset/replay | Old replies act on replacements, charge releases during cancel, or recurrence leaks between episodes |
| Play/damping sweep | A policy depends on uncontrolled numerical chatter or one brittle solver setting |
| Timing sweep | Success disappears under the measured deployment schedule or inference delays |
| Reward audit | Loops, intentional reset abuse or needless actions increase score without task progress |
| Repeated-trial evaluation | A selected lucky checkpoint is the only evidence of improvement |
| Runtime export | Normalization, masks, cadence, recurrence or fallback differs from the trained bundle |

Deliberate gyro assistance and compliant movement are valid mechanics. Creative use of them is not a defect. Review unintended solver explosions, teleport-like corrections and distracting chatter against the authored profile before rejecting an unexpected strategy.

Promotion requires predeclared task improvement, acceptable feel, no critical lifecycle/profile regression and a measured runtime cost. If numeric improvement harms feel, retain alternatives and report the tradeoff. If the budget is exhausted without evidence, record inconclusive/no-go at that budget; do not generalize that the method cannot work.

## Compute and run budgets

Measure aggregate host base ticks/s, policy decisions/s, environment count, reset/serialization overhead, learner-update time, inference latency distribution, peak memory and achieved cadence. A small network can be cheap while the game is the bottleneck.

Illustrative arithmetic only: at six base ticks per decision, one million decisions require six million base ticks. At aggregate throughput 60, 600 or 6,000 base ticks/s, that is about 27.8 hours, 2.78 hours or 16.7 minutes, excluding optimizer/evaluation time. None of these rates is measured here.

Proposed initial ceilings:

- Contract smoke: 2,000 decisions or 2 minutes, whichever ends first.
- Throughput pilot: 10 minutes per configuration with no weight training.
- First learning screen: 200,000 decisions or 2 hours per seed, whichever ends first; publish partial/inconclusive status when capped.
- Candidate comparison: begin with three seeds and two learned configurations after a baseline. No automatic expanding sweep.
- Promotion: a separately budgeted evaluation with sealed scenarios; no paid frontier-model calls until an explicit service budget is part of its task.

These are editable experiment limits, not promises of sufficient learning. A run supports cancellation, checkpoint/resume and an immutable resolved manifest. Do not train in routine PR CI; use tiny conformance tests and an explicit manual experiment lane.

Initially resume the learner, optimizer and data collection at a fresh episode boundary, recording the resumed run's lineage. Exact mid-episode continuation is a separate capability requiring a complete validated host snapshot; do not claim it from saved weights or visible body poses alone.

## Artifact admission

Record learner/host SHAs, artifact hashes, observation/action/profile versions, normalization, recurrence/reset rules, training privileges, scenario splits, budget, dependency/runtime/device identity and complete results. Keep large data and checkpoints outside Git under C-005. The ledger records failed runs and corrected evidence without rewriting history.

A policy can be trained yet unevaluated, evaluated yet too slow, or mechanically successful yet poor to play. Publish those states separately.
