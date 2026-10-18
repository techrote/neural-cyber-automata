---
title: Primary sources and claim limits
kind: reference
summary: Research and source-code evidence used in the ACT-004 prepass, with explicit transfer and implementation limits.
owns: [research.sources]
verified: 2026-10-18
---

# Primary sources

These sources support ideas and factual descriptions, not measured Cybersand performance. No third-party controller implementation was copied or ported in this documentation prepass. Research papers are references; any later code reuse needs source/licence review under C-001. The original repository documentation/checker is retained from bootstrap revision 20d0634.

## Host authority

| Source | What it establishes |
|---|---|
| [ACT-004 issue #178](https://github.com/techrote/cybersand/issues/178) | Requested articulated actor scope; re-read current owner comments when implementing |
| [Pinned Thunderbowl contract](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/docs/systems/thunderbowl-control-reference.md) | Existing sampled inputs, observations, lifetime, schedules, controller examples and evidence boundaries |
| [Source audit](../evidence/2026-10-18-source-prepass.md) | Exact inspected actor, native-coupling, Godot and Rapier files; observed API facts and gaps |
| [Original bootstrap](https://github.com/techrote/neural-cyber-automata/tree/20d063423fda1e037f328fb32c40c00c73b121a5) | Historical NCA plan, decisions and tooling; no training implementation |

## Feedback and learning

| Primary source | Supported use | Limit |
|---|---|---|
| [Rapier joints guide](https://rapier.rs/docs/user_guides/javascript/joints/) | Motor target, stiffness/damping and effort-cap concepts | Generic API; pinned Godot wrapper evidence is in the audit |
| [Hogan, Impedance Control, 1985](https://summerschool.stiff-project.org/fileadmin/pdf/Hog1985.pdf) | Shape response during mechanical interaction rather than track pose alone | Does not require exact mechanics or passivity certification for a game |
| [Härkegård, Dynamic Control Allocation](https://www.diva-portal.org/smash/get/diva2%3A316757/FULLTEXT01.pdf) | Redistribute demand under actuator magnitude/rate constraints | Optional after simple mixing has a demonstrated limitation |
| [Johannink et al., Residual RL, 2018](https://arxiv.org/abs/1812.03201) | Superpose conventional feedback and learned residuals in contact tasks | No universal stability/transfer guarantee |
| [Schulman et al., PPO, 2017](https://arxiv.org/abs/1707.06347) | Policy-gradient surrogate optimization with rollout minibatch reuse | Does not prove the best method for these game mechanics |
| [Haarnoja et al., SAC, 2018](https://proceedings.mlr.press/v80/haarnoja18b.html) | Off-policy entropy-regularized continuous control | Sample reuse needs comparison with implementation/runtime complexity |
| [Ross et al., DAgger, 2011](https://proceedings.mlr.press/v15/ross11a.html) | Collect expert labels on states induced by the learner | Requires a suitable teacher/correction source |
| [Ni et al., Recurrent model-free RL, 2022](https://proceedings.mlr.press/v162/ni22a.html) | Careful recurrent baselines can handle partial observability well | History/reset implementation remains crucial |
| [Huang et al., Shared Modular Policies, 2020](https://proceedings.mlr.press/v119/huang20d.html) | Shared modules and messages across planar agent structures | Arbitrary new joint semantics/damage still need tests |
| [CPG-RL, 2022](https://arxiv.org/html/2211.00458v1) | Learned oscillator modulation for locomotion | Published 100 Hz/1 kHz rates do not validate the game's 10/60 Hz cadence |
| [Peng et al., DeepMimic, 2018](https://arxiv.org/abs/1804.02717) | Combine motion examples with task goals | Requires attainable examples for each actor |
| [Hansen et al., TD-MPC2, 2023](https://arxiv.org/abs/2310.16828) | Latent model learning and short-horizon planning | Destruction, contacts and emulated assists need independent validation |
| [Salimans et al., ES, 2017](https://arxiv.org/abs/1703.03864) | Black-box policy optimization using parallel perturbations | A candidate for bounded small-controller/parameter sweeps, not a cheap universal solution |

## Intent, skills, human control and slow models

| Primary source | Supported use | Limit |
|---|---|---|
| [Sutton, Precup and Singh, Options, 1999](https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf) | Skill initiation, policy and termination as temporal abstraction | Skill contracts still need game-specific outcomes |
| [Ichter et al., SayCan, CoRL 2022 / PMLR 2023](https://proceedings.mlr.press/v205/ichter23a.html) | Ground language choice in contextual skill feasibility | Does not validate per-tick remote motor control |
| [Belkhale et al., RT-H, 2024](https://rt-hierarchy.github.io/) | Intermediate motion descriptions and corrective human intervention | Demonstration/domain requirements remain |
| [Huang et al., Inner Monologue, CoRL 2022 / PMLR 2023](https://proceedings.mlr.press/v205/huang23c.html) | Replanning from environment feedback | Model narration is not an authoritative task result |
| [ROS 2 actions design](https://design.ros2.org/articles/actions.html) | Acceptance, execution, cancellation and terminal results are distinct | Concept reference only; no middleware adoption is proposed |
| [ROS 2 lifespan/deadline/liveliness](https://design.ros2.org/articles/qos_deadline_liveliness_lifespan.html) | Separate freshness and source-liveness concepts | Simulation and wall clocks still need explicit binding |
| [Javdani et al., Shared autonomy, 2015](https://roboticsproceedings.org/rss11/p32.html) | Assistance under uncertain human goals | Faster completion can conflict with perceived control |
| [Real-Time Execution of Action Chunking Flow Policies, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/300ccb2187dedd4edcc07f7e76d8e553-Abstract-Conference.html) | Asynchronous action continuation | Only relevant if chunked policies are introduced |
| [Real-Time Robot Execution with Masked Action Chunking, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c35834443b7881e782e52b3519fe27c7-Abstract-Conference.html) | Keep delayed planned actions consistent with executed actions and current observations | Not a replacement for local cancellation/lifetime checks |

## Cellular and distributed behaviour

| Primary source | Supported use | Limit |
|---|---|---|
| [Growing Neural Cellular Automata, 2020](https://distill.pub/2020/growing-ca/) | Local recurrent updates for growth, persistence and regeneration | Pattern growth is not actor-control evidence |
| [Self-classifying MNIST Digits, 2020](https://distill.pub/2020/selforg/mnist/) | Shared local processing can produce collective classifications | Different objective/topology from rigid actors |
| [Self-Organising Textures, 2021](https://distill.pub/selforg/2021/textures/) | Train local rules for continuing patterns | Historical/background candidate, not gameplay acceptance |
| [Adversarial Reprogramming of NCA, 2021](https://distill.pub/selforg/2021/adversarial/) | Local injected signals can steer a trained field | Motivates robustness tests, not arbitrary reliable command control |
| [Tolstaya et al., Local multi-robot controllers, 2020](https://proceedings.mlr.press/v100/tolstaya20a.html) | Learned local exchanges and multi-hop information | Effective communication reach and cost must be declared |

## Integration and evaluation

| Primary source | Supported use | Limit |
|---|---|---|
| [Godot RL Agents](https://github.com/edbeeching/godot_rl_agents) | Existing Godot/Python training interfaces and learner wrappers | No compatibility/ONNX/runtime qualification for this repo; audit licence before reuse |
| [Gymnasium Env](https://gymnasium.farama.org/api/env/) | Reset/step tuples, owned observations, termination and truncation | Adapter must map host semantics correctly |
| [Gymnasium time limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/) | Administrative truncation versus task termination and bootstrap | Transport corruption does not provide a valid bootstrap observation |
| [Agarwal et al., Statistical Precipice, 2021](https://arxiv.org/abs/2108.13264) | Interval estimates and careful aggregate RL evaluation | A handful of seeds is not a universal power guarantee |
| [rliable](https://github.com/google-research/rliable) | Available implementation of evaluation methods | Optional dependency; do not hide per-run failures behind an aggregate |
| [Rapier determinism](https://rapier.rs/docs/user_guides/rust/determinism/) | Build, initial-state and ordering conditions | Generic current guide, not proof of whole-host determinism at the pinned wrapper |
| [Godot physics interpolation](https://docs.godotengine.org/en/stable/tutorials/physics/interpolation/physics_interpolation_introduction.html) | Render interpolation and fixed physics are separate clocks | Rendered positions are not necessarily fresh control observations |

All online sources were used as primary research or official project documentation during this prepass. Pinned code is the authority for the inspected runtime; current general documentation can describe newer APIs and must not silently replace it.
