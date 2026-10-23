---
title: Trainer flight-learning preparation
kind: guide
summary: Source-bound persistent flight-owner bridge preparation, fail-stop transport, and bounded design with actual-host execution blocked pending accepted #229.
owns: [flight.learning.preparation]
verified: 2026-10-23
---

# Trainer flight learning

[PREPARATION ONLY] Issue #74 depends on accepted Cybersand #205/#206/#207.
PR #229 remains in progress. Nothing here launches a host, trains a policy,
issues an allocation or reads any final suite. IK #54, Jolt and morphology
evidence and gates remain unchanged. Both IK final suites are excluded, as are
the flight teacher's qualification final and every learner final payload.

The separately registered `4ly-preparation@1` adapter now validates the exact
source contract reviewed at Cybersand PR #229 head
`2e20731de0559ed59280d7a11f84f596d1cba985`: real profile
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
actor, epoch, control generation, observation ID and physical tick. Step uses
the owner's `ticks_consumed` and `physical_ticks_total`, stops on the first
terminal tick and preserves the owner's final copied observation.

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

## Transport failure, accounting and capture handoff

A request that fails before dispatch may be an ordinary local/refusal failure.
Once a request has been sent, a missing, malformed, duplicated or reordered
reply cannot prove whether the owner consumed work. The transport closes,
records **unknown consumption** and refuses automatic retry/replay. A matching
host reply must classify consumption as `none`, `confirmed` or `unknown`;
successful work must be `confirmed`. This preserves the accounting rule that
uncertain work is charged and stops the attempt rather than being silently
reissued.

The durable accounting stream remains authoritative and is deliberately not
embedded in the transport. Before future execution, the driver must reserve
`reserve_reset()` before reset and `reserve_step(maximum_ticks=6, ...)`
before each decision. A confirmed reply completes the reservation with the
owner's exact `ticks_consumed`; `UnknownConsumptionError` maps to
`mark_unknown(...)`. Only after that durable outcome may a capture stream
admit a normal supervised row. `4ly-flight-exchange/v1` provides the join
keys without modifying the existing ACT004 capture record schema. Cancellation,
takeover and close host-call accounting must be finalized only after the
accepted bridge topology is measured; the declaration's `host_calls`,
startup and cleanup ceilings therefore remain null and execution stays blocked.

Actual-host launch is additionally hard-gated in
`flight_transport.ACCEPTED_HOST`. It is intentionally `None` now.
`FlightGodotTransport` therefore fails before inspecting a checkout or
spawning Godot. After #207/#229 is accepted, an independent handoff must bind
the exact accepted Git revision and tree, Godot version/binary SHA-256,
native artifact path/SHA-256, Rapier artifact path/SHA-256, and SHA-256 values
for the reviewed episode/observation/profile/controller/arena source files.
Then, and only then, run actual-host conformance for reset, full/partial
terminal step, lease expiry, cancel, human takeover, repeated reset, shutdown
and induced reply loss. Verify manifest/profile/controller hashes and all
sequence/run/actor/epoch/control-generation/observation/physical-tick receipts,
measure serialization/call/startup/cleanup/resource bounds, and publish an
independent readback. Do not open any teacher or learner final payload during
that handoff.

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

The preregistration supplies the non-resource maxima now represented by the
declaration: one host start, 884 resets, 1,475,280 physical ticks, 245,880
decisions, 11,400 capture labels, 5,400 optimizer updates, 684,000 sample
presentations, 480 tuning episodes and 63 checkpoint publications. Exact
`host_calls`, startup and cleanup operation ceilings remain null. Wall time,
peak resident memory, storage bytes and publication bytes also remain null.
No value is inferred from the source-specific #229 pilots while #207/#229
remains unqualified.

No executable training command is admitted. `training_admission()` evaluates
the incomplete preparation declaration and therefore remains fail-closed with
`final_access=false`. Execution additionally requires accepted current
dependency digests; missing or stale evidence refuses admission. A future
resource declaration must also bind validated wall/RSS/storage enforcement,
not merely measured numbers.

Before prospective execution can exist, record the accepted #229 host/build
and teacher evidence, final bridge actual-host conformance, frozen split
commitments, selected CPU runtime identity, source-bound throughput and
host-call/reset latency, whole-process startup/cleanup duration, process-tree
peak RSS, evidence/checkpoint/publication byte growth, durable storage
throughput, exact host-call and capture-label phase topology, finite
startup/cleanup operation counts and an independent source-bound review of the
complete declaration. Neither these fixtures nor
the accounting implementation qualify flight or issue an experiment admission.

Windows CPU-only software checks from repository root:

```powershell
python -m unittest discover -s tests -p test_flight_preparation.py
python -m unittest discover -s tests -p test_flight_accounting.py
python tools/check_docs.py --strict
```

The tests use scripted/synthetic fixtures only; they do not run Godot/native,
collect teacher data, train a model, initialize CUDA or open a final payload.
Actual-host transport/export/realtime qualification and teacher acceptance
remain outstanding.

© 2026 techrote. All rights reserved.
