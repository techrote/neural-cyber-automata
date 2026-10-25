---
title: Trainer flight-learning preparation
kind: guide
summary: Qualified Cybersand bridge, bounded Trainer learner design, and fail-closed experiment-admission package with external final custody and review still gated.
owns: [flight.learning.preparation]
verified: 2026-10-25
---

# Trainer flight learning

[PREPARATION ONLY] Cybersand PR #249 is accepted at lease-expiry repair merge
`9935d5696721d2c72a8273ea70454ab0c5b11361`. The qualified source lineage
for the Trainer/controller/frozen teacher remains accepted PR #229
`b5a8570e5123e1f633766e9390761a1c8134cfa2`. This reconciliation imports the
qualified Trainer profile `f1f9286a27b929c96caca5aaf2be70cbd82e0fd933f29968c1aaaa43ac8f1704`,
controller `fb86190c68c6a87e7d2ef7e845b1fb97305f188723a326f67f07caa6971cc8ae`
and frozen teacher `0fb571cfd314a53fc86b3517fb5a8258a77d7215fcbcb43fc7f53519e7e29761`.
It still launches no host, trains no policy, issues no allocation and opens no
learner final. The teacher qualification used checkout
`2bc2b6cfa9ad62512266df5804c5662b5ba8bd04` and one-use suite SHA-256
`4bd18d728581f64fa770c4dc29f765eff956f61999b96d03d8e378d54efc3401`;
that suite is evidence for the frozen teacher only and is explicitly ineligible
for learner-final reuse.

The separately registered `4ly-preparation@1` adapter now validates the exact
accepted Cybersand #249 source contract: real profile
`4ly_trainer`, actor slot `0`, `4ly-episode/v1`,
`4ly-observation/v1`, the host-produced
`4ly-flat-observation/v1` object with exactly 54 finite values, and the
two-component `4ly_velocity_v1` desired-velocity request at six physical ticks
per decision with a twelve-tick owner lease. Learner `training` maps only to
the host's public `development` geometry; `tuning` maps to `tuning`.
Every final/sealed variant remains refused before transport.

`src/nca/adapters/flight_host_bridge.gd` is an external launcher owned by this
repository. It wraps Cybersand's real `CyberPhysicalFlightEpisode`; it does
not add a Cybersand endpoint, patch the plant, reproduce Rapier/material physics
in Python, or relabel a fixture as host evidence. The loopback
`nca-4ly-flight/1` transport is one-request-in-flight and authenticates the
parent-owned connection. Reset, step, policy cancellation, explicit human
takeover and close are source-prepared. Replies bind request sequence, run,
actor, epoch, control generation, observation ID and physical tick. Fresh decisions remain six-tick-boundary admitted. The bridge additionally
exposes the owner's source-faithful 1-6-tick partial surface and a null
continuation that is valid only for an already-live lease, never renews
authority and may not cross the next decision boundary. Step uses the owner's
`ticks_consumed` and `physical_ticks_total`, stops on the first terminal tick
and preserves the owner's final copied observation.

The bridge adds only `source_observation_id` beside the host flat-observation
object as transport correlation metadata; it does not change the Cybersand
observation schema. Existing ACT004 `CaptureEvent`/`TickRecord` shared
fields are unchanged. `FlightSession.last_exchange` instead exposes a
separate `4ly-flight-exchange/v1` join record containing the exact sequence,
authority identities and confirmed physical interval for future capture and
accounting integration.

`nca.train.flight.build_policy` constructs a CPU 54-64-64-2 tanh direct policy.
It performs no optimization; optional Torch is imported only when called.
Outputs request desired velocity through the authored stabilizer, not rotor
balancing. No navigation teacher is used inside scored learned inference.
The preprocessing helper preserves host fixed scaling and validity channels;
there is no tuning/final-fitted normalization. Any change to preprocessing or
fitted statistics is a new policy configuration and requires a new
preregistration. The current model is feedforward, with no hidden state to
carry across episodes. Transport failure poisons the session; there is no
automatic host restart or replay of possibly consumed work.

## Actual-host bridge qualification

The bridge is qualified against accepted Cybersand #249 on one persistent
real native/Rapier host. Run `40211483325`, NCA source head
`7e3fd442e111d91e47239c5a2457943d96323737`, bridge SHA-256
`23f115d5947eb84858262208fc3411a481c5219f7fd0213901b0bf08e59d0bf6`
and evidence SHA-256
`193c150e83d7e83217692be75407ccc5acb4038e1a60183fbb489bd0b78bece9`
prove reset, full/partial stepping, null continuation, real tick-18 lease
expiry, cancellation, human takeover, re-reset, graceful shutdown and a
deliberately lost close reply. Independent post-exit readback reconciled all
20 confirmed physical ticks against owner receipts; the close remains unknown
and was not replayed. Earlier source-digest, non-finite-RPC and duplicate
cleanup-reservation attempts remain retained negative evidence.

This qualification resolves the exact host build, physical bridge semantics,
host-call topology and host-side performance measurements. The later
[experiment-admission package](evidence/2026-10-25-4ly004-admission-package.md)
adds a frozen CPU runtime contract, exact public development/tuning manifests
and finite whole-attempt resource enforcement. Neither step collects teacher
capture or grants training admission; independent learner-final custody and an
independently authored execution review remain external prerequisites.

## Trainer-only capture, cloning and export preparation

`nca.train.flight_learning` is an offline preparation surface over already
recorded values; it has no host launcher, capture command or execution-admission
override. Each capture episode is canonicalized and content-hashed against the
accounting declaration, capture identity and exact host/build/profile/controller/
teacher source tuple. Raw owner observations are paired with the host-produced
`4ly-flat-observation/v1` view, admitted teacher actions, confirmed physical
tick consumption and explicit terminal outcomes. Failed, refused, canceled and
uncertain work remains in the immutable episode record. Unknown consumption,
refused/unknown teacher actions, failed transitions and invalid identity/schema
rows never become ordinary supervised samples.

Training capture is restricted to the preregistered eleven development
scenario IDs and seeds `101/103/107/109`, with at most 44 episodes and 11,400
eligible labels. The dataset identity includes every retained episode and
eligible sample digest. Experimental cloning refuses an incomplete development
case set: all 11 scenarios x four registered seeds must be present exactly once,
even when an episode retained no eligible labels. The fixed loss assigns each of
the six training families nominal weight `1/6`; within a family, every retained
capture episode owns an equal nominal share and its admitted rows divide that
share. An episode with no eligible rows contributes zero rather than transferring
its share to successful episodes. The three fixed policy seeds use deterministic
seed-bound shuffles for twenty epochs. Every optimizer batch must first reserve one optimizer update and its
sample presentations in the source-bound `FlightAccountingLedger`; missing or
mismatched accounting refuses optimization.

Checkpoint preparation retains the exact compact-model state, Adam state,
deterministic shuffle RNG, the seed-bound Torch CPU initialization RNG,
learner source/revision, policy seed, capture/declaration/source identities,
the optimizer accounting attempt ID and hash-chain tip, schedule identity,
preprocessing/action identities and parent-checkpoint lineage. The exact export bundle contains the
54-channel host preprocessing contract, two-output action contract and numeric
CPU float32 weights. Its scored inference object has no teacher/oracle
reference and consumes only the raw owner identity needed to bind the separate
54-channel learner view; evaluator-only values are never policy inputs. Export
qualification flags remain explicitly false for trained quality, actual-host
parity and realtime operation, and final access remains false.

The focused learning tests run only tiny synthetic capture records and a tiny
synthetic CPU BC fixture behind a synthetic admitted accounting ledger. That
optimizer exercise is software conformance, not the declared #74 experiment:
it collects no teacher data, starts no host, initializes no CUDA path, reads no
teacher/learner/IK final payload and establishes no learned flight quality.

## Transport failure, accounting and capture handoff

A request that fails before dispatch may be an ordinary local/refusal failure.
Once a request has been sent, a missing, malformed, duplicated or reordered
reply cannot prove whether the owner consumed work. The transport closes,
records **unknown consumption** and refuses automatic retry/replay. A matching
host reply must classify consumption as `none`, `confirmed` or `unknown`;
successful work must be `confirmed`. This preserves the accounting rule that
uncertain work is charged and stops the attempt rather than being silently
reissued.

The durable accounting ledger remains authoritative and separate from the wire
protocol. When a ledger is attached, the transport itself calls
`reserve_reset()` before reset, `reserve_step(maximum_ticks=6, ...)` before
each decision, `reserve_host_call(...)` before cancellation/takeover, and
`reserve_cleanup(host_call=true)` before an orderly host close. A confirmed
reply completes the reservation with the owner's exact `ticks_consumed`;
`UnknownConsumptionError` maps to `mark_unknown(...)`. No accounting field is
added to `nca-4ly-flight/1`. Only after that durable outcome may a capture
stream admit a normal supervised row. `4ly-flight-exchange/v1` provides the
join keys without modifying the existing ACT004 capture record schema. The accepted bridge measurements resolve the exact planned topology:
246,765 host calls, one startup operation and one cleanup operation across the
884-reset / 245,880-decision work ceiling. Capture-label charging remains a
post-validation local accounting step and cannot turn refused or uncertain
physical work into supervision.

The accepted source and Linux runtime are bound in
`flight_transport.ACCEPTED_HOST` and `ACCEPTED_RUNTIME_BUILD`: Cybersand
#249 merge `9935d5696721d2c72a8273ea70454ab0c5b11361`, accepted tree
`607158a7b938471022edee62d150988db769bb78`, Godot 4.7, exact native/Rapier
artifacts and reviewed host-source hashes. `FlightGodotTransport` still
requires durable accounting before launch; source/runtime acceptance does not
grant experiment admission. The independent actual-host conformance has already
passed and is retained as immutable bridge evidence. No teacher or learner final
payload was opened during that qualification.

## Bounded prospective experiment

The source-bound design and custody rules are frozen in the
[4LY-004 prospective preregistration](FLIGHT_PREREGISTRATION.md). The experiment
shape is one balanced-Trainer CPU behavior-cloned direct policy configuration,
seeds `74011/74013/74017`, one persistent physical host, and no PPO, sweep,
morphology expansion, CUDA path or inference-time navigation teacher.

The six training families are hover (auxiliary), free-space, reversal, arrival,
obstacles and recovery. Capture uses the public #229 development split only:
one episode for every learner-family scenario and each registered development
seed, 44 episodes total. The authoritative source deadlines bound capture to
68,400 physical ticks and 11,400 decision labels. This supersedes the earlier
universal 960-tick draft ceiling, which was too short for #229's 2,100-2,400
tick obstacle/recovery deadlines.

Each seed receives at most twenty passes over the admitted immutable capture,
batch capacity 128, Adam `lr=0.001` and normalized desired-velocity MSE:
at most 228,000 sample presentations and 1,800 optimizer updates per seed.
Checkpoints 5/10/15/20 alone are tuning candidates. Tuning uses all registered
#229 tuning scenario/seed combinations in the five core families: 40 physical
episodes per checkpoint, 480 across all three seeds and four checkpoints.
Checkpoint selection is highest equal-weight core-family macro success, then
lowest contact-tick rate, then earliest epoch. No teacher/control rollout is
performed for tuning.

A future learner final is a separately held external 60-case suite, twelve
cases per core family, never the teacher-qualification holdout. Six comparators
run exactly once per final case: frozen qualified teacher, zero-input neutral,
canonical untrained seed-74011 epoch-0 policy and the three selected trained
seeds. That is 360 final episodes, bounded by source deadlines to 600,480
physical ticks and 100,080 decisions. Final access allowance remains ZERO until
the separate custody, declaration, resource and independent-admission gates
pass. Any opened/disclosed suite is thereafter development evidence and cannot
be relabeled an untouched final.

Promotion remains at least 90% equal-family mean core success, 80% for every
trained seed, 80% for every core family, zero critical learned authority/physics
violations and demonstrated learned contribution versus both neutral and
untrained controls. Family weighting and learned-contribution rules are
specified exactly in the preregistration. Arrival requires the accepted host's
low-speed dwell, not a high-speed crossing. Report teacher gap, collisions,
overshoot, settling/stalls, chatter, saturation and inference/cadence metrics.
Final failure or inconclusiveness stops the attempt; no automatic retry,
replacement suite, replay or budget extension exists.

## Durable accounting and remaining admission gates

`nca.train.flight_accounting` now implements the preparation-side immutable
declaration, fail-closed execution gate and hash-linked attempt accounting. The
declaration binds learner/host/build/profile/controller/teacher identities,
both owner-observation and 54-float learner schemas, action schema and cadence,
scenario/split commitments, CPU runtime, the three learner seeds, all work
ceilings, all resource ceilings, dependency evidence and the one-attempt/no-
replay policy. Its canonical JSON SHA-256 is the declaration identity.

Each future host call or optimizer operation must consume an fsynced reservation
before work. Related counters are reserved atomically and never refunded.
Confirmed physical ticks are recorded separately from reserved tick capacity.
The accepted-host launcher cannot pass its source-review barrier without a
ledger, and its RPC transport reserves reset/step/revocation/close work before
dispatch; capture-label counting is a local accounting phase and does not add a
wire-protocol field. Behavior cloning likewise refuses to enter Torch work
without a ledger whose declaration hash owns the immutable capture dataset.
Original failures remain sticky. A lost reply or crash after reservation but
before a durable outcome becomes explicit unknown consumption and ordinary
execution stops; it is never automatically replayed. Duplicate invocation,
resume of a closed consumed attempt and truncated accounting tails are refused.
Checkpoint publication has its own reservation and retains immutable parent
lineage across clean interruption.

The declaration now has complete finite work ceilings: one host start, 884
resets, 246,765 host calls, 1,475,280 physical ticks, 245,880 decisions, 11,400
capture labels, 5,400 optimizer updates, 684,000 sample presentations, 480
tuning episodes, 63 checkpoint publications, one startup operation and one
cleanup operation.

The admission package additionally freezes Linux x86_64 / Python 3.12.10 /
Torch 2.8.0+cpu / one-thread execution and hard whole-attempt caps of 18,000 s
wall time, 4 GiB process-tree RSS, 8 GiB retained attempt storage and 4 GiB
publication bytes. These are fail-stop allocation ceilings, not claims about
future learner consumption. A continuously running resource guard is mandatory
before a production accounting ledger can open or resume, and each subsequent
reservation is checked against it.

`training_admission()` intentionally continues to evaluate the incomplete
repository preparation declaration and remains fail-closed. The separately
versioned admission builder can produce an executable prospective declaration
only after it receives (1) an external 60-case learner-final custody record,
without receiving the payload, and (2) an independently authored execution
review bound to the exact final-head source/split/runtime/resource/custody
candidate. Missing or stale evidence refuses admission and `final_access`
remains false even after training admission.

Development capture determines the exact valid-label count below the fixed
11,400 ceiling. Tuning-only checkpoint selection occurs only after admitted
training, and continuously advancing export qualification is a later deployment
gate. None is fabricated as pre-execution evidence.

Windows CPU-only software checks from repository root:

```powershell
python -m unittest discover -s tests -p test_flight_preparation.py
python -m unittest discover -s tests -p test_flight_transport.py
python -m unittest discover -s tests -p test_flight_accounting.py
python -m unittest discover -s tests -p test_flight_learning_pipeline.py
python -m unittest discover -s tests -p test_flight_admission.py
python tools/check_docs.py --strict
```

The tests use scripted/synthetic fixtures only. They do not run Godot/native,
collect teacher data, execute the declared learning experiment, initialize CUDA
or open a final payload. The learning-pipeline test performs only tiny synthetic
CPU optimization to verify deterministic cloning, accounting, checkpoint and
export mechanics. Host/teacher source qualification, the public split manifests, CPU runtime
contract and fail-stop resource-enforcement surface are now bound. Trained
quality, realtime export qualification, external learner-final custody and the
independent execution review remain outstanding; exact-head admission CI must
also pass before this package is accepted.

© 2026 techrote. All rights reserved.
