---
title: 4LY-004 prospective preregistration and evidence custody
kind: reference
summary: Source-bound Trainer-only behavior-cloning design, split custody, exact work ceilings and fail-closed admission checklist.
owns: [flight.learning.preregistration, flight.learning.custody]
verified: 2026-10-25
---

# 4LY-004 prospective preregistration and evidence custody

**DRAFT / NOT AN EXECUTION ADMISSION.** This document freezes the intended
experiment shape for issue #74 without issuing an allocation, opening a final
payload, collecting teacher data, training a model or approving execution.

The accepted physical host is Cybersand PR #249 merge
`9935d5696721d2c72a8273ea70454ab0c5b11361`; the qualified
Trainer/controller/frozen-teacher lineage remains PR #229 merge
`b5a8570e5123e1f633766e9390761a1c8134cfa2`. The source fixes 60 Hz physics,
six-tick decisions, 54 flat observation values, two desired-velocity outputs,
profile `f1f9286a27b929c96caca5aaf2be70cbd82e0fd933f29968c1aaaa43ac8f1704`,
controller `fb86190c68c6a87e7d2ef7e845b1fb97305f188723a326f67f07caa6971cc8ae`
and teacher `0fb571cfd314a53fc86b3517fb5a8258a77d7215fcbcb43fc7f53519e7e29761`.
The one-use teacher-final suite remains teacher evidence only; any contract
change requires new preregistration review.

## Frozen learner configuration

Exactly one learned configuration is allowed:

- balanced **Trainer only**;
- direct feed-forward CPU policy, 54 -> 64 -> 64 -> 2 with tanh hidden/output activations;
- behavior cloning only, Adam `lr=0.001`, normalized desired-velocity MSE, batch capacity 128, twenty epochs;
- policy/training seeds exactly `74011`, `74013`, `74017`;
- checkpoint identities at epoch 0 and after every epoch; only epochs 5/10/15/20 are eligible for tuning selection;
- one persistent physical host, no parallel physical hosts;
- no PPO, architecture/hyperparameter sweep, recurrence, morphology expansion, domain randomization, CUDA path or inference-time navigation teacher.

Host-defined normalization, masks, axis conventions and action saturation are authoritative.
Additional fitted statistics require a new configuration and preregistration.

## Family membership and split separation

The source-defined course maps to these learner families:

| Learner family | Scenario IDs | Role |
|---|---|---|
| `hover` | `unsupported_hover` | Training auxiliary only; no navigation/promotion credit |
| `free_space` | `horizontal_reach`, `vertical_reach`, `diagonal_reach` | Core |
| `reversal` | `reverse` | Core |
| `arrival` | `arrival_braking` | Core |
| `obstacles` | `wide_corridor`, `offset_gates`, `corner`, `slalom` | Core |
| `recovery` | `wall_recovery` | Core |

The source `neutral_control`, `constant_control` and `power_off` scenario
IDs remain upstream host/teacher qualification controls. They are not learner
training/tuning/final families. The learner's **neutral-input comparator** is
instead the zero desired-velocity policy executed on the same learner-final
core cases as every other comparator.

Training capture uses only #229's public `development` geometry and the
registered development seeds `101,103,107,109`. It takes one episode for
every scenario/seed pair in the six learner families: 44 episodes total.
Tuning uses only #229's distinct `tuning` geometry and seeds
`211,223,227,229`, over the five core families: 40 cases per eligible
checkpoint (12 free-space, 4 reversal, 4 arrival, 16 obstacles, 4 recovery).

The **teacher qualification holdout is never a learner holdout**. The scored
teacher suite `4bd18d728581f64fa770c4dc29f765eff956f61999b96d03d8e378d54efc3401`,
its geometry, episode traces and payload stay outside the learner workflow.
#74 may consume only the accepted teacher identity and summary evidence. The
teacher-final generator source was visible in transient CI and its reviewer
label was automation-supplied, so the result does not establish independent
blind custody and the suite must not be transferred to learner evaluation.
Once any suite has been opened, disclosed or used to tune a teacher or learner,
it is development evidence and cannot later be represented as an untouched
learner final.

The learner final is a separate externally held suite. Its public
preregistration reveals only composition: exactly 60 core cases, with 12 cases
per core family. Free-space contains four cases for each of its three scenario
IDs; obstacles contains three cases for each of its four scenario IDs;
reversal, arrival and recovery contain twelve cases each. The custodian must
privately verify that final starts, goals and obstacle geometry are non-identical
to public development/tuning geometry and to the teacher-qualification
holdout, retain the registered minimum 60-cell nontrivial start/goal
separation, and preserve the source scenario deadlines and arrival/dwell rules.
The final payload, final case seeds and exact geometry remain opaque until
opening; only their cryptographic commitment and custody metadata may be
published before then.

## Capture provenance, cloning and checkpoint selection

There is one immutable teacher capture shared by all three policy seeds.
A supervised row is eligible only when it binds the exact accepted host,
Trainer/controller/teacher identities, an authoritative owner observation,
the admitted teacher action, the observation/action schema identities, a
confirmed physical tick transition and the terminal lineage for its episode.

Task-failed or deadline-truncated episodes remain in evidence and their
otherwise valid transitions remain eligible; behavior-cloning data is not
post-hoc filtered for teacher success. Refused actions/resets, stale or
mismatched identities, malformed/nonfinite observations/actions, uncertain
consumption, lost replies after potentially consumed work and physics/authority
integrity faults are retained as failures but do **not** become ordinary
supervised rows. No such work is replayed automatically.

Training loss gives each of the six training families weight `1/6`.
Within a family, each captured episode has equal total weight and each eligible
row within an episode shares that episode weight. This prevents the longer
obstacle/recovery deadlines from dominating the clone merely because they
produce more decisions.

For each seed, every admitted row is presented once per epoch in a
deterministic seed-bound shuffle. With at most 11,400 eligible labels, twenty
epochs permit at most 228,000 sample presentations and 1,800 optimizer updates
per seed (final partial batches count as updates). Across all three seeds the
ceilings are 684,000 presentations and 5,400 updates. There are at most 21
checkpoint artifacts per seed, 63 total.

Each seed is selected independently from epochs 5/10/15/20 on the same frozen
40-case tuning suite. Selection order is: highest equal-weight five-core-family
macro success, then lowest aggregate contact-tick rate, then earliest epoch.
Hover is reported during development but has no checkpoint-selection or
promotion weight. Teacher, neutral and untrained comparator rollouts are
**not** run during tuning.

## Prospective physical-work counts

The earlier PR #85 draft used a universal 960-tick episode ceiling. That is
incompatible with #229's source deadlines (for example, 2,400 ticks for
`slalom` and 2,100 for several obstacle/recovery cases), so this
preregistration replaces it with the authoritative per-scenario ceilings.

| Stage | Episodes | Max physical ticks | Max decisions |
|---|---:|---:|---:|
| Teacher capture: 44 development scenario/seed pairs | 44 | 68,400 | 11,400 |
| Tuning: one 40-case suite at 4 checkpoints x 3 policy seeds | 480 | 806,400 | 134,400 |
| Learner final: 60 cases x 6 comparators | 360 | 600,480 | 100,080 |
| **Whole #74 prospective maximum** | **884** | **1,475,280** | **245,880** |

The six final comparators are exactly: frozen qualified scripted teacher;
zero desired-velocity neutral input; one canonical untrained policy
(`seed=74011`, epoch 0); and the three independently selected trained
checkpoints for seeds 74011/74013/74017. Therefore #74 contains 104 teacher
episodes (44 capture + 60 learner-final comparison), 60 neutral episodes,
60 untrained episodes, 180 trained-final episodes and 480 learned tuning
episodes. The separate upstream #229 teacher-qualification campaign is not
charged or counted as #74 work.

Per final comparator, the source deadlines bound 60 cases to 100,080 physical
ticks and 16,680 decisions. The persistent-host design permits at most one host start and 884 reset
attempts. The qualified NCA bridge proves the source topology used by the
prospective experiment: every planned episode consumes one reset, every planned
decision consumes one step RPC, and one final close consumes the single cleanup
call. Therefore the exact prospective topology is now
`host_calls=884+245880+1=246765`, `startup_operations=1` and
`cleanup_operations=1`; no cancel/takeover/null-continuation call is part of
the planned experiment workload.

The prospective accounting declaration therefore freezes
`host_starts=1`, `resets=884`, `host_calls=246765`,
`physical_ticks=1475280`, `decisions=245880`,
`capture_labels=11400`, `optimizer_updates=5400`,
`sample_presentations=684000`, `tuning_episodes=480`,
`checkpoint_publications=63`, `startup_operations=1` and
`cleanup_operations=1`. These are work ceilings, **not an issued allocation**.

### Qualified host/bridge and whole-attempt resource envelope

Run `40211483325` qualified the exact host/bridge path and measured startup,
latency, throughput, serialization, shutdown/cleanup, RSS and evidence growth.
Exact host-path values remain in
[the bridge evidence](../evidence/2026-10-25-4ly004-bridge-conformance.md).

The separately versioned
[experiment-admission package](evidence/2026-10-25-4ly004-admission-package.md)
now supplies finite **whole-attempt hard ceilings**, rather than pretending the
host measurements are learner measurements: 18,000 s wall time, 4 GiB
process-tree RSS, 8 GiB retained attempt storage and 4 GiB publication bytes.
These are allocation caps, not expected consumption. A live 0.25 s process-tree
resource guard is required before a production accounting ledger can open or
resume; exceeding a cap makes the one attempt fail-stop/inconclusive and does
not enlarge its allocation.

## Final scoring, gates and failure disposition

For trained seed `s` and core family `f`, let `S[s,f]` be the proportion
of the 12 learner-final cases in that family that satisfy the authoritative
low-speed success/recovery rules. Define each seed score as the arithmetic mean
of its five family rates; define each family score as the arithmetic mean of
the three trained-seed rates for that family; and define mean core success as
the arithmetic mean of the three seed scores. No episode-count weighting is
allowed across families.

Promotion requires all of the following, unchanged from #74:

- mean core success >= 90%;
- every independently trained seed score >= 80%;
- every core-family score >= 80%;
- zero critical learned-policy authority/physics-integrity violations;
- demonstrated learned contribution rather than teacher substitution.

Learned contribution is prospectively defined as: trained mean core success is
strictly greater than both neutral and untrained macro success; every core
family's trained family score is strictly greater than both corresponding
control family scores; and every core family contains at least one paired case
where a trained policy succeeds while both controls fail. No teacher action may
replace, correct or mask a learned action during scored learned inference.
Teacher performance is reported as a gap, not a required superiority target.

Ordinary task failures, deadline truncations, collisions, overshoot and stalls
count against the declared metrics when evidence integrity is intact. A
malformed/unauthorized learned action is a promotion failure. A host,
transport, receipt or physics-integrity ambiguity that prevents trustworthy
attribution makes the experiment **inconclusive**, preserves consumed work and
stops the attempt. Missing evidence is never converted to failure or success.
There is no automatic retry, replay, extension or replacement suite.

## Independent custody and opening prerequisites

The learner-final custodian must be independent of learner implementation,
training, checkpoint selection and experiment execution. Teacher qualification
and learner final are distinct custody objects with distinct suite hashes and
admission records. The learner repository/executor may possess a commitment,
family/scenario counts and reviewer attestation before opening, but not final
geometry, starts, goals, hidden seeds or episode payload.

Before the first learner-final byte is made available to the executor, all of
these must already be immutable and independently verified:

- accepted #249 host source/build/runtime plus qualified #229
  Trainer/controller/teacher lineage;
- clean learner source and exact policy/configuration/preprocessing/export
  bundle;
- immutable teacher capture plus its complete provenance/readback;
- all three selected checkpoint hashes and their tuning evidence;
- the prospective declaration and complete finite work/resource allocation;
- a learner-final custody record binding suite SHA-256, suite schema,
  60-case composition, private disjointness/feasibility review, exact host
  source and evaluator contract;
- an independent execution/evidence review accepting those exact bytes and
  confirming that final access is still unopened.

Opening must durably record a final-open barrier containing the declaration,
suite and custody digests **before** the first final parse/reset. If the run
fails after opening, that suite is opened evidence; it cannot be recycled as an
untouched final in a retry or later experiment.

## Remaining external and post-admission state

The accepted #249 host/runtime, NCA bridge conformance, Trainer/controller/
teacher identities, public development/tuning manifests, CPU runtime contract,
finite work topology, whole-attempt resource ceilings and enforcement surface
are now resolved by source.

Two gates remain intentionally external before the first experimental capture
or optimizer work:

- a separate 60-case learner-final commitment/custody record from an independent
  custodian/private reviewer, with the payload still outside Git and unopened by
  the executor; and
- an independently authored execution review bound to the exact final learner
  source, public split hashes, custody digest, runtime/resource policy and
  `final_access=false`.

The exact valid-label count remains naturally unknown until admitted development
capture occurs; the fixed ceiling remains 11,400 and cannot be increased.
Tuning-only checkpoint selection and realtime/export qualification are
post-admission experiment/deployment gates, not facts that may be fabricated
before training. Learner-final bytes remain unavailable until the later
final-open barrier.

## Admission checklist

- [x] #249 is accepted at host merge
  `9935d5696721d2c72a8273ea70454ab0c5b11361`; the qualified Trainer,
  controller and frozen-teacher lineage remains bound to accepted #229
  `b5a8570e5123e1f633766e9390761a1c8134cfa2`.
- [x] The accepted source matches the 54-channel, two-output, 60 Hz, six-tick,
  Trainer/controller contracts above.
- [x] The exact NCA execution Godot/native/Rapier build is registered and the
  external bridge passes source-bound actual-host conformance with independent
  post-exit receipt readback (run `40211483325`, evidence
  `193c150e83d7e83217692be75407ccc5acb4038e1a60183fbb489bd0b78bece9`).
- [x] The CPU learner contract is frozen to 54-64-64-2 tanh BC, seeds
  74011/74013/74017, Linux x86_64, Python 3.12.10, Torch 2.8.0+cpu and one
  Torch thread; PPO/sweeps/morphology/CUDA remain absent. Exact learner revision
  and source-closure SHA-256 are generated from the clean final head.
- [x] Development and tuning manifests exactly freeze the public family/split
  membership: development SHA-256
  `8b68faa6d3bf6172d0ba1dcf6d2752aae6ff9e83d521cfe33cf537cbbad9e5e7`
  and tuning SHA-256
  `0295a669549bc7cf852fd12934d85506ef1f55ceec07340531f801084e08b3df`.
- [x] Capture validation excludes uncertain/invalid transitions without hiding
  failed or refused episodes.
- [x] Durable accounting reserves every host/optimizer/publication operation
  before work and carries failures/unknown consumption forward.
- [x] Host-call/startup/cleanup work ceilings are populated from the qualified
  persistent-host topology.
- [x] Whole-attempt wall/RSS/storage/publication hard ceilings are finite and
  enforced by the source-bound live resource guard; host-only projections are
  retained only as rationale, not substituted as learner measurements.
- [x] Teacher qualification holdout is referenced only by accepted identities
  and summary evidence; no teacher-final payload is imported as learner
  tuning/final data and its one-use suite is explicitly ineligible for reuse.
- [ ] A separate 60-case learner-final suite is committed under independent
  custody and privately verified disjoint/feasible without being opened.
- [ ] An independent source-bound execution review accepts the exact
  final-head review binding and external custody commitment before experimental
  capture/training begins.
- [ ] Exact-head admission/software CI passes and publishes the review candidate.
- [x] Final-access state remains false/unopened throughout this admission
  package.

After admission, all three trained checkpoints must still be selected and
hashed using tuning only before the final-open barrier, and realtime/export
qualification remains separate from training admission. Until the two external
pre-execution gates and exact-head CI pass, `training_admission()` and final
access remain refused. This document itself grants neither execution nor final
access.

© 2026 techrote. All rights reserved.
