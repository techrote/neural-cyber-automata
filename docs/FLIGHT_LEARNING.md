---
title: Trainer flight-learning preparation
kind: guide
summary: Versioned protocol fixtures and bounded design with physical execution blocked pending qualified flight host.
owns: [flight.learning.preparation]
verified: 2026-10-23
---

# Trainer flight learning

[PREPARATION ONLY] Issue #74 depends on accepted Cybersand #205/#206/#207.
PR #229 remains in progress. Nothing here launches a host, trains a policy,
issues an allocation or reads any final suite. IK #54, Jolt and morphology
evidence and gates remain unchanged. Both IK final suites are excluded, as are
the flight teacher's qualification final and every learner final payload.

The separately registered `4ly-preparation@1` adapter implements copied protocol
observations, numbered request/reply fencing, owner identity/tick validation,
explicit terminal/truncation handling, bounded unit-disk velocity actions and
fail-stop sessions. Its injected transport is a preparation protocol, NOT an
existing Cybersand RPC endpoint. Physical bridge serialization, native receipt
validation, actual-host normalization parity, lease/cancellation/takeover and
continuously advancing export qualification remain unmeasured and blocked.
The `flat` field is a learner-side bridge field, not a claimed host dictionary
field. The future bridge must use the host's exact 54-channel flatten adapter;
it must not reconstruct privileged observations or emulate physics in Python.

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

## Remaining admission gates and Windows checks

No executable training command is admitted. `training_admission()` remains
fail-closed. Before execution can exist, bind a clean learner source to the
accepted #229 host/build, Trainer/controller/teacher identities, exact
schemas/cadence, immutable capture/checkpoint identities and unopened
learner-final custody commitment.

Durable accounting must reserve work before host or optimizer operations,
retain failures and unknown consumption, and bind the complete declaration.
The preregistration now supplies non-resource maxima for host starts, resets,
physical ticks, decisions, capture labels, optimizer updates, sample
presentations, tuning episodes and checkpoint publications. Exact host-call
topology, startup/cleanup operation counts, wall time, resident memory, storage
and publication ceilings remain null until qualified #229/bridge measurements
exist. Missing measurements refuse admission rather than inventing resources.

Obtain independent source-bound admission only after focused software tests,
actual-host conformance, accepted teacher qualification and exact-head CI.
Neither fixtures nor this design can qualify flight or #74.

Windows CPU-only software checks from repository root:

```powershell
python -m unittest discover -s tests -p test_flight_preparation.py
python tools/check_docs.py --strict
```

The preparation tests use scripted/synthetic fixtures only; they do not run
Godot/native, collect teacher data, train a model, initialize CUDA or open a
final payload. Actual-host transport/export/realtime qualification and teacher
acceptance remain outstanding.

© 2026 techrote. All rights reserved.
