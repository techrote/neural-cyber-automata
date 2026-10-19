---
title: T109 interrupted prerequisite and fixed candidate evaluation declaration
kind: evidence
summary: Preserved incomplete T105 prefix, unknown original termination, and one separately bounded evaluation of the prospectively selected BC1103 checkpoint.
owns: []
verified: 2026-10-19
---

# Interrupted prerequisite and fixed candidate

The [faithful prerequisite attempt](2026-10-19-t109-reproduction.md) began
after its declaration and implementation were public at
`33a4aa2ead8e2137c933225b8f9727deef40d45a`. It executed the unchanged learner
`745578947a6f1ac2159830c4ae4de5de5593f32f`. The launch receipt records
19:12:46.523729 UTC; the original learner manifest records
19:13:01.158257 UTC. One separate zero-policy, zero-tick host startup
preflight succeeded in 4.449 seconds before that attempt.

## Retained interruption evidence

The execution service disconnected during the run. The original tool handle
could no longer be resumed. After the declared wall deadline, an attempt to
interrupt that same owned session returned exactly
`write_stdin failed: Unknown process id 32967`. This result does not establish
an original exit code or process termination. **Original termination remains
unconfirmed.** No restart, resumed optimization or replacement training run
is authorized by this record.

The retained original collection has no `summary.json`, no
`evaluation-opened.json` and no wrapper termination receipt. The latest
retained file timestamp is 19:38:15.973724 UTC. An unchanged snapshot contains
all 536 retained original files. Each of its 236 compressed episode records
was parsed and checked against its original metric and artifact identities.
The retained receipts account for 6,160 decisions and 36,960 consumed ticks,
32 checkpoint files, and 21 completed PPO update ledgers. These are counts
from the retained prefix. They are not original-run totals; an in-flight
request, further attempted work and final wall cost remain unknown.

All 24 original teacher episodes and 680 imitation samples are retained. The
already selected BC seed 1103 completed its 40 epochs with a recorded
45.723878296-second learner cost. Its policy content is
`8196e77d5ecc6ea0ef3dff597b89e5f80e1b32804cc17eb7f305900c2c4d9a1b`.
This is a new policy instance with a different identity from the historical
T105 policy. The selected seed and checkpoint are unchanged. The original
residual seed 1103 update-seven optimizer/RNG boundary is also retained.
No original holdout evaluation or completed three-seed efficacy comparison
is claimed.

The separately named `interruption-index.json` records actual retained
members, metric rows, checkpoint metadata, completed update ledgers and the
unknown values explicitly. It is a post-interruption inventory; it is not a
learner-produced final summary. Original files were hashed before copying,
checked against the snapshot, and checked again after indexing and packing.
The new index and fencing receipt exist only in the snapshot.

| Retained identity | SHA-256 |
| --- | --- |
| Complete raw prefix archive, 29,733,124 bytes | `6f2a6f5beafaf0be629f63ca3162edc4858b8f44ae7d350e9167862e3fa06a5e` |
| Raw archive member index | `c80063cf88c7aac12485fd0fda45ddc514870498f183e3c388c08f782dddc30c` |
| Audited snapshot archive, 29,724,642 bytes | `f796b533eacce85b661162349f67641d6127b73545152d75fed75e1772ac934c` |
| Interruption index file bytes | `9b8dd6bd7a142c70ce0a47d0cec73091ebbbe702f5a4862fb948916de771d10e` |
| Interruption index self-hash | `5b10bb6881dea3dfd395e85b3d83e7ec6865e5e8d5198fcdb494c26f630c0d32` |

## Separate prospective candidate evaluation

The continued programme instruction authorizes one distinct evaluation of
the prospectively fixed trained BC1103 checkpoint. T109 requires a trained
and evaluated candidate. This distinct contract can satisfy that prerequisite
without declaring the interrupted three-seed experiment complete. The
complete-T105 `convert` command keeps its original completed-screen gate.

[bc1103-interrupted-evaluation-v1.json](../../configs/export/bc1103-interrupted-evaluation-v1.json)
binds all 538 snapshot files, including the original source/configuration,
dataset, selected checkpoint, numeric weights and interruption receipts.
Its canonical content SHA-256 is
`bd220b3f6699bbe283b5c50f0073a3282df01123cd5e09a0bd0e2af6399a969c`.
The declaration and all evaluator implementation bytes must be committed and
published before the first evaluation request.

The new evaluator uses the original eight holdout families, seeds, geometry,
goals, 240-tick task deadlines and six-tick cadence. It executes the original
learner/controller/capture source in a separate worker. The new output is a
distinct child collection of the frozen snapshot. It cannot overwrite the
snapshot inputs or the original interrupted collection.

The limit is **one attempt, 320 decisions, 1,920 ticks and 300 wall seconds**,
with no optimization, tuning, alternative checkpoint, automatic retry or
threshold change. Input admission and execution share the wall budget. A
parent watchdog signals only its own newly created worker group if the
deadline expires. Censored or missing outcomes remain failures and retain
confirmed receipt counts separately from unknown pending work.

Admission checks the actual teacher episode membership and sample count,
source/configuration/profile/host identities, completed BC training ledger,
checkpoint tensor equality with its numeric export, and the selected policy
content. A completed evaluation must retain one valid uncensored record for
each of the eight original families, including unsuccessful outcomes. Every
record must retain the exact source, policy, goal, seed, timing and admission
provenance. Original files and evaluator source bytes are checked again.

Before publication, read-only admission of the actual snapshot passed in
51.97 seconds. It verified the 24 teacher records, 680 samples, BC40 ledger,
checkpoint/numeric tensor equality and complete interrupted input identities.
The 24 original teacher metric records also passed the exact finite metric
schema. The export suite collected 37 tests: 33 passed and four actual-host
cases skipped because no evaluated export bundle exists yet. An independent
review reproduced and checked corrections for malformed metric admission and
reserved requests whose physical outcome remains unknown, then ran all five
staged contract tests successfully. These checks establish implementation and
input admission evidence; they perform no supplemental host evaluation.

## Export gate after the separate evaluation

The `convert-candidate` command requires the strict separate evaluation
contract and a second committed, published declaration binding its actual
manifest, summary and all eight episode file identities. It preserves
`training_status: interrupted_no_summary` and
`original_termination_confirmed: false` in the bundle. Its summary hash names
the separately completed candidate evaluation; its explicitly named training
receipt hash identifies the interruption index.

Conversion and qualification retain the byte-identical
[original CPU export criteria](../../configs/export/cpu-bc-v1.json), canonical
SHA-256 `f499cb94ec537f5c1cc1b601cfcfa37fdaf9f5e81abe365aecb23d72ad0f1df0`.
The formal deployment screen revalidates the full staged evidence. Its
optimizer/RNG smoke uses the actual retained update-seven boundary; its
learner cost record labels retained completed update costs and leaves
interrupted-run totals unknown. The screen still makes zero optimizer updates.

There is one declared export attempt. Any failed fixed criterion remains a
no-go result. The [offline parity auditor](../RUNTIME_EXPORT.md#diagnose-a-preserved-parity-failure)
may examine preserved inputs without new host steps; it cannot promote a
failed screen. No supplemental evaluation, conversion or new deployment
qualification result is recorded by this prospective declaration.
