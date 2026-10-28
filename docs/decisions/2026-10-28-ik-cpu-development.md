---
title: D-036 IK CPU inference development
kind: decisions
summary: Owner commissions CPU utility diagnosis and inference improvements using training and tuning evidence only.
owns: [decisions.ik_cpu_development]
verified: 2026-10-28
---

# D-036 — CPU inference investigation

[ACCEPTED SCOPE] On 28 October 2026 the owner instructed: “Investigate why seed
43019 failed the CPU utility gate and work on improving inference performance
without relaxing the threshold or reusing the consumed final suite.” This
commissions a development CPU campaign under D-033, without training or a new
qualification attempt. Existing final results and all historical code/evidence
remain unchanged. No consumed final directory, payload, saved prediction or raw
timing sample is an input to this campaign; the published result summary supplies
the failure observation.

A new inference adapter may freeze existing v3 model arithmetic while retaining
all masks, bounds, polar features and refusal semantics. Validate output parity
on CPU fixtures and every original tuning row before measuring cost. Use the
authenticated train/tuning reader and the same three selected checkpoint hashes.
The complete decision protocol retains one thread, batch one, independent
bounds/FK checks, 100 warmups, 10,000 measured calls and the 0.8 numerical-cost
ratio. Development timing cannot replace the consumed qualification outcome.

The first campaign has two reversed seed/variant rounds, eager and frozen graph
variants, all three original seeds, at most one million predictor calls, zero
optimizer/final calls, a two-hour whole-process ceiling and 6 GiB Windows process
commit cap. Its declaration is written outside Git before corpus loading or
measurement. Keep every failure and full latency sample. CPU execution disables
CUDA visibility; the V100 prohibition remains in force.

© 2026 techrote. All rights reserved.
