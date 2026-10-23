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
there is no tuning/final-fitted normalization. Any additional learned scaling
must be versioned, fitted only on authenticated training captures and exported
with the exact bundle. The current model is feedforward, with no hidden state
to carry across episodes. Transport failure poisons the session; there is no
automatic host restart or replay of possibly consumed work.

## Bounded prospective experiment

One configuration, balanced Trainer only, behavior cloning only, three seeds
74011 / 74013 / 74017. No PPO, sweep, morphology changes or parallel physical
hosts. One persistent actual host; optional learner device must be separately
qualified (CPU preferred; CUDA, if justified, TITAN RTX only, never V100).
Prepare point reaches, arrival/braking/dwell and simple obstacle families;
freeze exact family membership against the accepted course before admission.
Development precedes tuning and sealed evaluation by disjoint geometry/start/
goal families. Teacher qualification holdouts cannot silently become learner
holdouts, and any opened suite is development evidence only.

Proposed maximum capture: 120 training episodes, 20 per family for six frozen
families, at most 960 physical ticks/160 decisions each; 115200 physical ticks,
19200 action labels. Failed/invalid episodes retained with original outcomes;
invalid transitions excluded from ordinary supervised rows, never relabeled.
One shared immutable capture for all seeds. No capture now.
Per seed: at most 20 passes over admitted valid labels, batch capacity128,
at most 3000 optimizer updates and384000 presentations, whichever ceiling
first. Adam LR0.001, mean squared normalized desired-velocity imitation loss;
at most21 checkpoints (initial plus20). No recurrence or extra diagnostics.
Stop on nonfinite/lifecycle/accounting errors. One attempt; no automatic retry,
resume or budget extension. These are proposed ceilings, not issued budgets.

Tuning at epochs5/10/15/20: 30 paired episodes each, five per family, at most
115200 physical ticks and19200 decisions per seed. Select highest macro family
success, then lowest collision rate, then earliest epoch; never final scores.
Future final: 60 paired episodes (ten/family) for each of teacher, neutral,
untrained and three separately frozen trained seeds: at most345600 ticks and
57600 decisions across six comparators. These final counts are design only:
final access allowance remains ZERO until separately declared and admitted.
Here pairing means shared frozen scenario/seed identities, not additional
teacher rollouts. Tuning runs only the candidate; any teacher references are
preexisting frozen evidence and cause no extra physical calls. Final counts
include exactly one rollout per comparator/case. Fresh teacher tuning runs
would require a separately declared additional allocation and are excluded.

Require at least90% mean core-task success,80% every seed and80% every core
family, zero critical physics/authority violations, and demonstrated learned
contribution versus neutral/untrained controls. Freeze family weights and
uncertainty reporting before any capture. Arrival requires accepted host dwell
and speed criteria, not a high-speed crossing. Report teacher gap, collisions,
overshoot, settling/stalls, chatter, saturation and inference/cadence metrics.
Final failure/inconclusiveness stops; no automatic new experiment.

## Remaining admission gates and Windows checks

No executable training command exists. `training_admission()` always refuses.
Before adding execution, bind clean learner and accepted host SHAs/builds,
Trainer/controller/teacher hashes, schemas/cadence, scenario/split commitments,
runtime and immutable candidate/capture hashes. Implement durable per-attempt
owner receipts and reservations BEFORE physical work/optimizer work, charging
failures and preserving unknown consumption. Measure complete capture, host
starts/resets/calls, tuning, training, startup/cleanup and publication cost.
Derive actual wall/memory/storage ceilings from qualified #229 throughput;
missing measurements refuse admission rather than inventing a time allocation.
Obtain independent source-bound admissions after focused and actual-host tests
and exact-head CI. Neither fixtures nor this design can qualify flight or #74.

Windows CPU-only software checks from repository root:

```powershell
python -m unittest discover -s tests -p test_flight_preparation.py
python tools/check_docs.py --strict
```

The tests use a tiny scripted transport fixture, no Godot/native, optional
Torch, CUDA or training. Actual host transport/export/realtime qualification
and teacher acceptance are outstanding; the preparation is intentionally
incapable of issuing training or final actions.

© 2026 techrote. All rights reserved.
