---
title: T109 fixed candidate evaluation and export declaration
kind: evidence
summary: Complete separately bounded BC1103 evaluation and conversion, preserved interrupted training origin, and linked failed deployment qualification.
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
The complete candidate archive and its member index have verified delivery
receipts, as does the earlier audited interrupted-input archive.

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

## Conversion under the published second declaration

The second declaration was published at
`eb9536152743db7477ba3aa4ce7b8b4ab7db00f2`, tree
`4e8e80b6b3e376f286955d0ae1c9fb4cbc53d019`. All 51 scoped source files
remained byte-identical to the earlier evaluation freeze. The sole offline
conversion completed with exit code zero at 21:08:44 UTC, in
61.870140223 seconds including strict input re-admission. It used no host
actions or optimizer updates and bound all 549 admitted training/evaluation
input files.

The resulting immutable bundle contains 224,501 bytes and preserves the exact
fixed policy content. Its training status remains `interrupted_no_summary`
with original termination unconfirmed; its separate candidate evaluation is
complete. The conversion receipt records `runtime_qualified: false`.

A fresh NumPy-only process reloaded the actual bundle under its exact runtime
context. All 18 runtime source modules matched, Torch was absent, and the
numeric policy identity matched the evaluated checkpoint. Its numeric weights
use 29,736 bytes; its 32-element FP32 recurrent state uses 128 bytes. This
admission check made zero inference calls and zero host starts. It supplies no
trajectory or throughput qualification.

An independent read-only receipt review recomputed all eleven declared
evaluation file hashes and the manifest, summary and worker self-hashes.
Decompressed episode event/tick totals matched the complete 271-decision,
1,626-tick accounting, with no unresolved reservations. The reviewer also
checked the public second declaration's timestamp before conversion launch,
the bundle's file and content hashes, the conversion receipt, and the
launch/termination/admission self-hashes. No blocker was found. That review
performed no host work, inference or optimization.

| Conversion identity | SHA-256 |
| --- | --- |
| Bundle file bytes | `30b40a2459eafc7bdcc271b6ddce55ec1f8442113b5edc0b21c4380e2793f3bc` |
| Unsigned bundle content | `1a9c30d267cbc9eeb6c6e44645c8ba56b25939fa6833dfcca6194a9e9f55a4e7` |
| Conversion receipt file | `d253553558dbcfb76c9facf888b63db81284cabd3453b62741a4a3fcc8b61d0b` |
| NumPy-only admission record content | `87ab027eb7b6bfddfb382c485e5aacafce3c3de709f02fe4cba15c636e615047` |
| Conversion archive, 305,185 bytes | `e20bab16f69317652c560030d8b8e526563e507453c12df9db8295f3dcdf773c` |
| Conversion archive member index file | `f7edc5eb7dac46beb5c4f64bd314e7b35892ee14d6e197495735218065888bfe` |

The conversion archive preserves the artifact, complete receipt set, source,
declarations and invocation programs in 65 byte-verified members. It has a
confirmed delivery receipt and remains immutable and separate from the later
qualification archive. The [single physical deployment screen](2026-10-19-t109-deployment-no-go.md)
subsequently completed with a fixed no-go: both trajectory pairs exceeded the
unchanged tolerances. Its original realtime reporting mismatch, prospective
checker correction and separate passing engineering tests are recorded there.
The count-based throughput interval lasted 79.033537896 seconds; a full
ten-minute observation remains outstanding. The bundle remains unqualified.
