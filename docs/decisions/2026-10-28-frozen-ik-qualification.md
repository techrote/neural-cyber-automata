---
title: D-037 fresh frozen-inference qualification
kind: decisions
summary: Owner authorizes one new sealed-suite qualification of the fixed frozen CPU inference path and the same three checkpoints.
owns: [decisions.frozen_ik_qualification]
verified: 2026-10-28
---

# D-037 — Fresh frozen-inference qualification

[ACCEPTED SCOPE] On 28 October 2026 the owner instructed: “Prepare a fresh
qualification of the frozen inference path using the same three checkpoints
and a new sealed final suite. Get two independent admissions and run it if all
checks pass, with unchanged thresholds.” This authorizes model-free new-suite
preparation/generation, prospective qualification, independent admissions and
conditional execution. It does not reopen or replace any historical allocation.

Use the fixed `runner-polar-cpu-frozen/v1` path on the same three trained v3
checkpoints and their actual epoch-zero controls. No optimizer, reselection,
runtime-route search, final timing retry or threshold change is authorized.
Use a new versioned recipe/custodian and a new original custody directory;
all consumed suites, payloads, predictions, timings and historical evidence
remain excluded. Existing deterministic recipe conformance tests may verify
historical software hashes without reading a consumed suite.

Freeze candidate-independent new geometry, family namespace, seed and full
population before teacher calls. Require independently reviewed train/tuning
identity and continuous root/joint-path guards, finite preparation/generation
budgets, a complete supervised seal and a separate frozen qualification packet.
Two new independent non-author admissions must bind exact source, proof,
candidates, suite provenance/isolation, resource feasibility and green CI.
Owner authority is conditional execution under those exact requirements.

Retain the unchanged accuracy, learned-contribution and <= 0.8 CPU p95 cost
gates. The three-seed outcome must be reported completely, including no-go or
failure. The earlier final no-go remains historical evidence. Export and
physical-host acceptance remain separately gated. The V100 prohibition and
targeted 20 GiB TITAN watchdog remain; this evaluation performs CPU inference.

See the [prospective contract](../design/IK_FROZEN_QUALIFICATION.md).

© 2026 techrote. All rights reserved.
