---
title: T109 fixed candidate evaluation and export declaration
kind: evidence
summary: Complete separately bounded BC1103 evaluation, preserved interrupted training origin, and unchanged prospective deployment qualification.
owns: []
verified: 2026-10-19
---

# Fixed BC1103 candidate evaluation

The single candidate evaluation completed on 2026-10-19 at 20:58:43 UTC.
It used the public source revision
`4510347e02d08093f1b86bef4583f0282b7cf903`, tree
`4beafe78443e17878a30e6eb6df9e868cbe5c2ec`, and the unchanged
[prospective evaluation declaration](../../configs/export/bc1103-interrupted-evaluation-v1.json).
The declaration and complete implementation were published before the launch.
The 51-file scoped source map has canonical SHA-256
`5208d94da57d6b8fc1efe552be7ce5089032277dfe1fd280cc49aedeb47dc287`.
The new worker executed the original learner/controller/capture revision
`745578947a6f1ac2159830c4ae4de5de5593f32f` in a separate process.

The selected checkpoint remained BC seed 1103 final, policy content
`8196e77d5ecc6ea0ef3dff597b89e5f80e1b32804cc17eb7f305900c2c4d9a1b`.
No weights, seeds, task criteria or candidate choice changed. The original
[training attempt](2026-10-19-t109-staged-candidate.md) remains
`interrupted_no_summary`, with original termination unconfirmed. This complete
single-candidate evaluation does not complete that original three-seed screen
or establish a method comparison.

## Actual bounded evaluation

| Receipt | Observed result |
| --- | --- |
| Original evaluation families | 8 of 8 |
| Attempted / confirmed decisions | 271 / 271, within the 320-decision ceiling |
| Confirmed native ticks | 1,626, within the 1,920-tick ceiling |
| Evaluation wall time, including input admission | 90.273026981 seconds, within the 300-second ceiling |
| Outer invocation wall time | 94.449614226 seconds |
| New host starts / worker invocations | 1 / 1 |
| New worker and wrapper exit codes | 0 / 0, termination confirmed |
| Errors / unresolved reservations | 0 / 0 |
| New optimizer updates / automatic retries | 0 / 0 |

The unchanged task judged three cases successful and five unsuccessful.
Unsuccessful task outcomes are retained as complete evaluations; none was
replaced, excluded or retried. Every episode was uncensored and had zero
failed admissions or integrity failures.

| Original family | Decisions | Native ticks | Task result |
| --- | ---: | ---: | --- |
| floor-right-30 | 23 | 138 | Arrival |
| floor-left-30 | 40 | 240 | Task deadline |
| floor-right-66 | 21 | 126 | Arrival |
| floor-left-66 | 40 | 240 | Task deadline |
| drop16-right-48 | 27 | 162 | Arrival |
| drop16-left-48 | 40 | 240 | Task deadline |
| drop32-right-60 | 40 | 240 | Task deadline |
| drop32-left-60 | 40 | 240 | Task deadline |

No CPU deployment latency or throughput conclusion is drawn from these
evaluation timings. The separately declared export screen measures those
quantities. Owner feel remains unmeasured.

## Evidence admission and preservation

After the worker exited, the strict staged verifier re-admitted the original
538 input files and the complete new evaluation. It checked all eight typed
episode records, exact source, configuration, policy, goal, seed and host
identities, finite complete task metrics, full request/tick accounting and
unchanged checkpoint/numeric tensor equality. The read-only verification
passed in 49.687909706 seconds and used zero host ticks or optimizer updates.

The archive contains all 20 fresh evaluation files, the complete scoped public
source, the input declarations, the outer launch/termination receipts and the
readmission record. Its 80 members were checked byte-for-byte after packing;
the fresh source files remained unchanged. The earlier audited input archive
is a separate preserved dependency, SHA-256
`f796b533eacce85b661162349f67641d6127b73545152d75fed75e1772ac934c`.
These archives remain outside Git; their digests are not public download URLs.

| Identity | SHA-256 |
| --- | --- |
| Candidate evaluation manifest content | `2e932fcb3a0dee3cff8b3587606b23866f7008571b97e123f04b341824e69b66` |
| Candidate evaluation manifest file | `2bc377ba249d072584e85d324f968d5950bb5a5350412926b14e954c38d16711` |
| Candidate evaluation summary content | `247c19765f3db722df7d79847d55b3697efcc1a925c96b7fc0460eebfc6bb470` |
| Candidate evaluation summary file | `71f9ec2a48c501166bdca77d2643fdeb55acb92bf62cdab9ef0a8b9a8aca6cfd` |
| Strict readmission record content | `aa9de4f82c746eb47ee662ee5e2ed2b9db5c53912b7906cbebb4d1856dc39f90` |
| Complete candidate archive, 1,308,701 bytes | `f6f087913cadf4487c44c95b2ed1ac53961536ad00da3edba1ff6b24deead691` |
| Complete candidate archive member index file | `4e2ea74a7c2bfebcf26123479e55fa3cd13de9546b8d8417ffaae77287680145` |

## Prospective conversion and deployment qualification

[bc1103-staged-export-v1.json](../../configs/export/bc1103-staged-export-v1.json)
is the second declaration, canonical SHA-256
`f0b3344cdeada006123aca70f4c18602dbe9b13d4caed88d7445c4dfa1219c35`.
It binds the actual completed evaluation manifest,
summary, worker result and every original episode file, together with the
fixed checkpoint and interrupted training receipt. It must be committed and
published before conversion and the single deployment qualification attempt.
The scoped Python source map remains byte-identical to the candidate
evaluation freeze through conversion and qualification.

The criteria remain the original
[cpu-bc-v1.json](../../configs/export/cpu-bc-v1.json), canonical SHA-256
`f499cb94ec537f5c1cc1b601cfcfa37fdaf9f5e81abe365aecb23d72ad0f1df0`.
The arithmetic and trajectory tolerances, thirteen refusal cases, live expiry
and takeover check, two 500-decision throughput configurations, 1,000-call
NumPy-only cost measurement, matched pose clips and retained optimizer/RNG
boundary smoke are unchanged. The throughput phase retains its 600-second
ceiling and requires a separately reserved measurement interval.

This declaration permits one export attempt, zero new optimizer updates and
no automatic retry. The historical reported trajectory failure remains a
failure with unavailable original raw evidence. A new failed criterion must
remain a no-go under its own immutable receipt. At this declaration, conversion
and deployment qualification have not run.
