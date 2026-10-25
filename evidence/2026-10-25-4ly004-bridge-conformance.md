---
title: 4LY-004 NCA actual-host bridge qualification
kind: evidence
summary: Source-bound Cybersand #249 bridge conformance, independent receipt readback, retained failures and measured host resources.
owns: [flight.learning.actual_host_conformance]
verified: 2026-10-25
---

# 4LY-004 NCA actual-host bridge qualification

This record qualifies only the NCA ↔ Cybersand physical bridge. It does not
authorize teacher capture, learner training, CUDA, a training allocation or
access to any teacher, learner or IK final suite.

## Accepted source and runtime

- Cybersand accepted repair merge:
  `9935d5696721d2c72a8273ea70454ab0c5b11361`
- accepted tree: `607158a7b938471022edee62d150988db769bb78`
- qualified teacher lineage merge:
  `b5a8570e5123e1f633766e9390761a1c8134cfa2`
- Godot 4.7 executable SHA-256:
  `f85bbc6b15e22416c7d797cd60b63286dd67b9cb13498847056c18520ae55a75`
- native GDExtension SHA-256:
  `ff3fb29e2260a587ab0f65b902abdd3f3e132885a31ed3a3b3a85cb80ab102cc`
- Rapier SHA-256:
  `566e68ae53bfb2eb7018c8e7d0dd3727121bf696462039b780a33d6b8369c66a`
- final repaired `physical_flight_episode.gd` SHA-256:
  `b8b42195c97dcf3b5292f841ff53043db8b7fb486e8b46a35707838a7e75f8d0`

Cybersand #249 exact-head qualification had already passed documentation,
native validation and all Godot/native/Rapier shards, including an independent
36-episode / 17,361-tick readback. Its earlier failed expiry-assistance
qualification remains historical negative evidence.

## Reviewed finite invocation

Before execution the PR recorded a finite one-host plan. The final reviewed
topology was one host start, four resets, fourteen host calls, seven physical
step calls, twenty reserved physical ticks, four fresh policy decisions, three
null continuations, one cancellation, one takeover and one close/cleanup.
The final close reply was deliberately discarded to test unknown-consumption
custody; replay and host substitution were prohibited.

## Retained failed NCA attempts

Failures were preserved rather than repaired into evidence:

1. An exact-host setup attempt stopped before physical conformance because the
   initially registered episode digest came from stale native provenance rather
   than the final #249 text. The accepted merge bytes were then hashed directly.
2. Run `40211118923`, artifact `12605507343`, sent the first real reset.
   A live unbounded metric serialized as non-finite `inf`; the strict NCA
   protocol rejected the reply, retained reset consumption as unknown and
   stopped without replay. Whole live metrics were removed from the wire.
3. Run `40211243751`, artifact `12605512523`, reached the final deliberately
   lost close reply, then the qualification harness attempted to reserve a
   second cleanup operation. The finite ceiling correctly refused that
   duplicate. No complete evidence/readback artifact existed, so the run was
   not accepted. The harness was corrected to preserve close as the single
   cleanup reservation.

## Accepted NCA qualification

Exact NCA head:
`7e3fd442e111d91e47239c5a2457943d96323737`

- workflow run: `40211483325`
- job: `121223450720`
- artifact: `12605461828`
- artifact ZIP SHA-256:
  `6578a638cf3f70a59974b152474b908ab5836b1baf29dc51f80b7d20695059ea`
- bridge SHA-256:
  `23f115d5947eb84858262208fc3411a481c5219f7fd0213901b0bf08e59d0bf6`
- evidence SHA-256:
  `193c150e83d7e83217692be75407ccc5acb4038e1a60183fbb489bd0b78bece9`
- independent readback: PASS

The sequence confirmed all 20 expected physical ticks. The tick-18 owner
receipt independently proves lease retirement with generation 0→1,
`command_live=false`, lease 0, `actuator_mode=passive_no_lease`, fallback
`expired` and assistance invalid. Cancellation advanced policy authority.
Takeover advanced authority and changed producer to `human`. Re-reset retained
a fresh final observation before shutdown. The final close was dispatched, its
reply deliberately not accepted, the host exited cleanly, and durable
accounting retained close consumption as unknown with no pending reservation.
No retry or replay occurred.

## Measurements and prospective resource basis

Measured on that exact one-host path:

| Measurement | Value |
|---|---:|
| Startup | 0.899644376 s |
| Reset max / median | 0.047778348 / 0.039766714 s |
| Step max / median | 0.018060448 / 0.010230353 s |
| Confirmed tick throughput | 282.8249101 ticks/s |
| Owner serialization max | 6,164 us |
| Shutdown | 0.163780543 s |
| Containment cleanup | 0.000515674 s |
| Process-tree peak RSS | 254,550,016 bytes |
| Owner receipt bytes | 36,038 |
| Event payload bytes | 63,519 |
| Evidence file bytes | 67,363 |
| Accounting journal bytes | 5,363 |

The source-derived full prospective topology is now exact:
`host_calls=246765`, `startup_operations=1`,
`cleanup_operations=1`.

For auditability, simple measurement projections are recorded but are **not**
execution ceilings: applying the observed max reset/step latencies to all
planned resets/decisions gives 4,484.003 seconds of host-call time; scaling
observed evidence density to 1,475,280 physical ticks gives 4,968,964,332
bytes; scaling observed journal density to 246,765 calls gives about
94,528,621 bytes. These projections do not include Torch learner memory/time,
full capture duplication or checkpoint retention and therefore cannot be used
to open the execution gate.

## Remaining admission boundary

The physical bridge dependency and throughput-measurement dependency are
accepted. Training remains refused until the exact learner source/revision,
development/tuning manifests, separately committed learner-final custody,
CPU platform/Python/Torch/thread identity, learner-inclusive whole-process
wall/RSS/storage/publication ceilings, stage-accounting/resource-enforcement
evidence, realtime/export qualification and independent combined execution
review are immutable and current. Final access remains false.

© 2026 techrote. All rights reserved.
