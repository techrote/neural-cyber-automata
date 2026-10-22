---
title: Geometric IK dataset configurations
kind: guide
summary: Finite Runner corpus and tiny development recipes with unchanged accepted numerical populations.
owns: [ik.dataset_configuration_files]
verified: 2026-10-22
---

# IK dataset configurations

The [owning dataset reference](../../../docs/IK_DATASETS.md) defines identities,
family/region isolation, teacher failures, feature privileges and sealed access.
These records describe geometric tasks, not ACT004 episode captures.

| File | Intended population | Allocation |
|---|---|---|
| [runner_v1.json](runner_v1.json) | 10,220 tasks; 100 families; 2,000 final points plus forty 20-step trajectories and seven 20-case stress populations | One 900-second invocation; 512 MiB; 654,080 reserved iterations |
| [tiny_v1.json](tiny_v1.json) | 254 opened development/conformance tasks; not final qualification data | At most sixty seconds; 512 MiB; 16,256 reserved iterations |

Neither file starts an experiment. The tool emits an exact source/environment/
configuration/partition declaration for publication before an assigned measured
invocation. Keep completed failures and consumed attempts; changing a configuration
or budget creates a different identity and does not reopen an old allocation.

Both retain split seed 61003 and the source-bound Runner descriptor. The
[accepted criteria](../runner_right_sole_acceptance_v1.json) keep their original
identity. Corpus configuration changes cannot silently lower those minima.

The tiny recipe uses a one-second per-call teacher ceiling to reduce noisy CI
timeouts. The prospective corpus keeps its declared 0.05-second ceiling; timing
failures remain retained, so this difference cannot be substituted into a run.

Large shards and the durable opening registry stay outside Git. Only recipe,
small evidence indexes and independently calculable test fixtures belong here.
