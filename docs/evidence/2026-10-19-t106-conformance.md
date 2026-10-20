---
title: Morphology interface conformance and bounded startup recovery
kind: evidence
summary: Passed fixed four-profile scripted interface diagnostic, with the first zero-request startup failure retained and both attempts charged to the original aggregate allocation.
owns: []
verified: 2026-10-19
---

# Morphology interface conformance

This record supports [T106 morphology preparation](2026-10-19-t106.md).

## Scripted profile conformance

The separate [conformance runner](../../evals/morphology/conformance.py) uses a fixed, frozen-source
actual-host schedule. Its first host bootstrap failed before any game request; the bounded continuation
below **passed all eight profile episodes**. This diagnostic is not a transfer score or learned-controller
trajectory.

Its declaration permits exactly two episodes on each of the four profiles, seed 6106, four six-tick
requests per episode: 32 requests and 192 completed ticks, with a 60-second wall ceiling. The two explicit
diagnostic targets are 40 cells right and left of the observed root. Requests are fixed in the JSON:
bounded drive/trim, neutral, bounded drive plus semantic impulse/brake/rebound, then neutral. Unsupported
channels retain masks. There is no model loading, policy inference, task-success score, reward fitting or
optimizer call.

The real host lifetime remains 240 ticks. Each diagnostic stops by owner cancellation at tick 24 with a
valid continuing observation. Cancellation must preserve the exact tick, part state, measured joint
state, pose and velocity while incrementing control generation and clearing every requested joint motor
target. The receipt retains the before/after targets separately; the target is the only joint field
excluded from the physical-state equality check. The second real reset must clear the recurrence clock; the
previous actual incarnation is checked as retired in the adapter without submitting an old packet to the
game.

The runner saves its prepared manifest before opening Godot, charges each attempt before its potentially
uncertain RPC, records every original packet and six per-tick admission/consumption receipts, and
preserves failed or partial episodes. It verifies profile hashes, original input attribution, exact
cadence, finite masked frames and source bytes before/after the screen. Reports retain exact host, Godot,
native, Rapier, bridge and imported source identities. The final summary must distinguish attempted,
confirmed and unconfirmed requests; no failed attempt can disappear into a retry.

```sh
python evals/morphology/conformance.py \
  --host ../cybersand-host \
  --godot ../godot-runtime/Godot_v4.7-stable_linux.x86_64 \
  --output morphology-conformance
```

## Preserved startup failure and bounded continuation

The first actual-host interface attempt ran from public source
`487fa04e4afea90051ef6a42667273a1ac83db45` on 19 October at 19:07 UTC, using Python
3.12.14 on Linux x86_64. It failed during host construction after **18.202958 seconds**, before any game
request or profile episode. The [immutable failure](t106-conformance-startup-failed.json) retains
`HostError`, zero attempted/completed requests, zero ticks and zero policy/optimizer calls. Its summary
content hash is `8d5b488123bb4ea1fa19cf32262abecb1859d86c7917c461260c7b45db5e4b60`.
This is failed startup evidence, not profile conformance. The original serializer omitted the underlying
non-contract exception detail, so the exact cause remains unknown. The preserved record is not rewritten.

The Windows unit run at public source `557799ce` exposed a separate checkout issue: automatic CRLF
conversion changed the immutable receipt's file hash. A narrow Git attribute now checks out this one
receipt with its original LF bytes on every operating system. The raw-byte admission check and declared
SHA-256 remain unchanged. A checkout with `core.autocrlf=true` reproduced the original
`a6b1f7286f9ad38638c1241033a1d4deb30eee3da9dfae8d7485e379077c48fd` after that correction. This is
a repository checkout correction; it does not establish the cause of the actual-host startup failure.

A separately declared [single startup continuation](../../configs/morphology/conformance-recovery-v1.json)
retains that failure and charges its time to the original ceiling. The new attempt may use at most
41.797 seconds, 32 requests and 192 ticks; the two attempts together remain below 60 seconds and at most
two host starts. It uses the ordinary 30-second startup transport timeout within that remaining wall
allocation. Mechanics, profile cases, fixed requests, admission checks and conformance predicates remain
unchanged. The runner writes the linked prior receipt and remaining allocation before opening the host,
refuses changed or incomplete prior evidence, and permits no automatic retry. New failures retain a
portable exception-cause class and known startup category without copying machine paths.

The public-source admission at the same `487fa04e` freeze remains `training_blocked`, with content hash
`071c5899dd4c2617cbb370ef2aeb399d58d8a8bc2f1f4aca54cf44324419876d` and all work counts zero.

## Executed continuation

The [compact result index](t106-conformance-result.json) and [run ledger](../RUNS.md) retain the exact
runtime, source, configuration, episode hashes and aggregate accounting. The continuation ran from
public source `557799ce4cf95d9d3ebcf862549cb363b5783a93`, with source-file aggregate
`3292659256bb24c2319c1071d00a74ccd9ccde5303f0c6f6859f893cd4962608`, on Python 3.12.14/Linux
x86_64 and the pinned Godot 4.7, native and Rapier artifacts at accepted host `f2e34cf9`.

| Receipt | First attempt | Continued attempt | Aggregate |
|---|---:|---:|---:|
| Host starts | 1 | 1 | 2 |
| Attempted/completed game requests | 0 / 0 | 32 / 32 | 32 / 32 |
| Completed physical ticks | 0 | 192 | 192 |
| Wall seconds | 18.202958 | 36.551695 | **54.754654 / 60 maximum** |
| Unconfirmed requests | 0 | 0 | 0 |

Jolt, Vector, Monowheel and Ripper each completed two episodes, eight requests and 48 ticks. All original
admissions and available consumed inputs matched the fixed requests. Unsupported channels retained
their explicit masks. Every same-tick owner cancellation preserved pose, velocity, part state and
measured joint state while revoking authority and clearing motor targets. Jolt and Vector had nonzero
requested joint targets before cancellation, so this exercised the corrected distinction in the actual
host. Each second reset retired the previous incarnation without submitting an old packet.

The passing summary content hash is
`3311fdfa3080659c8e49f0b5e64f528a2f4d2c8e3eace5979bd458e21af2d70f`. The immutable archive
`t106-conformance-evidence-v1.tar.gz` contains 20 portable members, including both attempts, all eight
episode traces, both public-source admissions and the frozen input declarations. Its SHA-256 is
`379b338682791a7e2ac5765eae9dbab79f67fad020804b4eb55025bae0e62275`; the index and exact 329,544-byte
size are recorded in the compact result. The archive was reopened and every member compared with its
recorded original bytes.

The admission at `557799ce` is still `training_blocked`, with content hash
`d66d7eb5a46fb32be91b5636a6a673ae88ed5d7baf9d8ec5590858abba51d120`. Both admissions have zero host
starts, policy calls, training decisions and optimizer updates. The interface diagnostic made zero
learned-policy calls, optimizer updates or task-success evaluations. Learned zero-shot transfer,
adaptation, independent geometry sweeps, arbitrary damage/assist perturbations, Windows qualification
and owner feel remain unmeasured.


© 2026 techrote. All rights reserved.
