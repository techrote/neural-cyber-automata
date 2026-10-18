---
title: Control and learning glossary
kind: reference
summary: Plain meanings of the terms used in the Cybersand research programme.
owns: [glossary]
verified: 2026-10-18
---

# Glossary

| Term | Meaning here |
|---|---|
| Gameplay plant | The authored actor and world dynamics that a controller acts on, including deliberate assistance and compliance |
| Policy | A rule or learned model mapping permitted observations/history and goals to requests |
| Feedback | Adjusting requests using the actor's observed response |
| Gyro assist | Authored bounded attitude torque; no stored-momentum simulation is required |
| Compliance/play | Deliberate give, clearance or delayed take-up in a connection |
| Deadband/hysteresis | Avoid tiny corrections; use different enter/leave thresholds so states do not chatter |
| Wire / direct metal | High-level rig behaviour requests / supported direct actuator requests |
| Skill | A bounded continuing behaviour with initiation, progress and terminal result |
| Recurrent state | Memory belonging to one actor incarnation and policy/profile version |
| Residual policy | Learned correction combined with a baseline controller before shared action limits |
| CPG | Oscillator-based rhythmic primitive for movement where rhythm is useful |
| Behavioural cloning | Learn action choices from demonstrations |
| DAgger | Add teacher labels for states visited by the learner |
| PPO / SAC | On-policy surrogate policy optimization / off-policy entropy-regularized control |
| Morphology | Part layout, connections and actuator arrangement |
| NCA | A shared local recurrent update rule; grid, actor-graph and part-graph uses remain distinct |
| Perception privilege | Which state a deployed decision source is allowed to see |
| Training privilege | Extra information used by a teacher or critic during training |
| Episode / incarnation | One run segment / one lifetime of a particular actor or object |
| Termination / truncation | Task terminal state / an external stopping condition |
| Action receipt | What was requested, admitted, applied and observed, with identities and times |
| Profile | Versioned actor capabilities, sensors, assistance, response and action interpretation |
| Surrogate | A different, usually simpler training environment whose results require transfer evaluation |
| Quantisation | Lower-precision network arithmetic; it does not establish world determinism |
