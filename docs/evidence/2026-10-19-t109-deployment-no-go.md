---
title: T109 staged CPU deployment no-go and retained-receipt diagnosis
kind: evidence
summary: Complete fixed-candidate deployment failure, unchanged trajectory limits, measured short throughput interval, proven realtime checker correction and separate passing engineering tests.
owns: []
verified: 2026-10-19
---

# Fixed CPU deployment: no-go

The single staged deployment screen completed on 2026-10-19 at 21:28:03 UTC
with **`status: no_go` and `runtime_qualified: false`**. Both physical trajectory
pairs exceeded the unchanged tolerances. The realtime checker also reported a
failure; a separate audit subsequently proved that it had omitted the pinned
host's reason for an observation older than its retained history window.
The original failed summary and bundle remain unchanged.

The implementation, complete evaluated bundle and engineering conformance are
available. Deployment qualification remains open under [T109 / issue
19](https://github.com/techrote/neural-cyber-automata/issues/19): the closed-loop
trajectory limits were not met, and the requested full ten-minute throughput
observation was not completed. The actual count-based throughput phase lasted
79.033537896 seconds. It must not be described as a ten-minute observation.

The [compact evidence index](t109-staged-export-no-go-index.json) retains exact
source and artifact identities, every phase result, latency/memory measurements,
clip hashes, both offline diagnoses and the separate engineering gate. Its
canonical unsigned content SHA-256 is
`a14a4d6806eead6372b1284c45facc6d87919e4cfeb989e34c9d71367ee3f22b`.

## Source, candidate and execution identity

The measured source was the clean published revision
`eb9536152743db7477ba3aa4ce7b8b4ab7db00f2`, tree
`4e8e80b6b3e376f286955d0ae1c9fb4cbc53d019`. All 51 scoped source files
remained byte-identical to the earlier candidate-evaluation freeze; their
canonical map SHA-256 is
`5208d94da57d6b8fc1efe552be7ce5089032277dfe1fd280cc49aedeb47dc287`.
The [second prospective declaration](../../configs/export/bc1103-staged-export-v1.json)
was published before conversion and qualification. The original criteria
[cpu-bc-v1.json](../../configs/export/cpu-bc-v1.json), canonical SHA-256
`f499cb94ec537f5c1cc1b601cfcfa37fdaf9f5e81abe365aecb23d72ad0f1df0`,
were not changed after the results.

The fixed new policy is BC seed 1103 final, content SHA-256
`8196e77d5ecc6ea0ef3dff597b89e5f80e1b32804cc17eb7f305900c2c4d9a1b`.
Its [separate candidate evaluation](2026-10-19-t109-candidate-evaluation.md)
completed all eight original cases, with three successes and five retained
unsuccessful outcomes. Its [training origin](2026-10-19-t109-staged-candidate.md)
remains `interrupted_no_summary`; original training termination and total
original work are unconfirmed. Neither the completed candidate evaluation nor
this export screen relabels that interrupted three-seed experiment complete.

The converted artifact is 224,501 bytes, file SHA-256
`30b40a2459eafc7bdcc271b6ddce55ec1f8442113b5edc0b21c4380e2793f3bc`.
The actual lane was CPython 3.12.14, NumPy 2.3.5, Torch 2.8.0+cpu reference,
Linux x86_64 CPU, Godot 4.7 and the unchanged pinned Cybersand host
`f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497`. The exact native, Rapier,
Godot, bridge, physical scripts and profile identities remain in the manifest
and archived receipts.

The invocation launched at 21:25:06.161864 UTC and returned code 1 at
21:28:03.374169 UTC. New invocation termination is confirmed. Its wall time
was 177.203741869 seconds, including pre-screen input re-admission; the screen's
own measured interval was 128.912039387 seconds. All planned phases and five
clips completed. The summary contains no execution errors. There was one
formal invocation and zero new optimizer updates or automatic retries.

## Original fixed-gate results

| Gate | Recorded outcome | Evidence |
| --- | --- | --- |
| Sequential numeric arithmetic | Pass | 128 comparisons, four declared recurrent resets, exact feature and boolean agreement |
| Rightward 240-tick trajectory | Fail | Position 6.939272166 cells, angle 0.219019132 radians, continuous request difference 0.772696137; binary requests also diverged |
| Leftward 240-tick trajectory | Fail | Position 0.911179705 cells, angle 0.063469768 radians, continuous request difference 0.047280463; binary requests remained exact |
| Negative contracts and lifetimes | Pass | All thirteen declared cases |
| Changed decision cadence and physical profile | Pass | Twelve-tick cadence refused as `runtime_context_mismatch`; Astra profile refused as `profile_mismatch` |
| Realtime expiry, cancellation and takeover | Fail in original checker | Five checks passed; the exact old packet was refused with `lease`, which the original reason filter did not count |
| Fixed-count throughput | Pass within its declared ceiling | 500 decisions in each configuration, ten reset samples and 100 observe samples; 79.033537896 seconds total |
| Full-call NumPy-only timing | Complete | 1,000 actual-observation replay calls in a separate process with Torch absent |
| Retained optimizer/RNG boundary | Pass | All state and RNG checks, two fresh 24-tick stochastic episodes, zero optimizer updates |

The trajectory limits remain **0.05 cells**, **0.05 radians**, **0.00002**
continuous request difference and exact binary actions. Receipt clocks,
admission, cancellation and detachment comparisons passed in both pairs.
Those passing lifecycle comparisons do not override the trajectory failures.

The original 128-input arithmetic maxima were zero for features,
`6.258487701416016e-7` for hidden state, `9.5367431640625e-7` for logits,
`2.384185791015625e-7` for value and `2.1606683731079102e-7` for
outputs/continuous wire requests. All remain below the frozen arithmetic
limits. Network arithmetic and physical closed-loop parity are separate gates.

## Offline trajectory diagnosis

The original committed parity auditor ran against the unchanged checkpoint,
bundle, manifest, failed summary and four complete trajectory files. It made
no host requests or optimizer updates. Each stored observation, feature vector
and recurrent input was replayed through both independent implementations.

All **160 recorded input pairs** passed common-input arithmetic comparison.
The largest hidden difference was `9.5367431640625e-7`, largest logit difference
`1.9073486328125e-6`, and largest continuous wire difference
`1.6391277313232422e-7`; boolean outputs were exact. Reconstructing the original
recorded implementation produced zero differences in features, hidden state,
value and requested action. Each recorded recurrent output exactly matched
the next stored recurrent input. This rules out a reconstruction or recorded
recurrence mismatch in these files.

| First event | Rightward case | Leftward case |
| --- | --- | --- |
| Initial pose, feature vector and hidden input | Exact match | Exact match |
| Nonzero continuous request difference | Tick 0; `3.725290298461914e-8` | Tick 0; `1.1175870895385742e-7` |
| Nonzero physical pose difference | Tick 1; `1.1641532182693481e-10` radians, zero position difference | Same |
| Continuous request exceeds `0.00002` | Tick 108; `0.00009357929229736328` | Tick 30; `0.025554895401000977` |
| Physical pose first exceeds a fixed tolerance | Tick 132; `0.15222302950938` cells | Tick 29; `0.07278298590526844` cells |
| First binary request difference | Tick 156; `secondary` | None |

The retained evidence establishes a closed-loop parity failure despite small
arithmetic differences on common inputs. It does **not** establish a unique
cause, contact sensitivity or intrinsic host nondeterminism: no repeated-host
or counterfactual action replay experiment was performed. No arithmetic
implementation defect was identified by this bounded audit. The measured
failure remains a deployment limit for this exact policy/source/runtime.

## Proven realtime checker defect and prospective correction

The raw realtime receipt binds the original packet to source tick 7, sequence
1, epoch 1, actor 0 and a twelve-tick motor lease. That exact packet was
submitted at tick 31, giving an age of 24 ticks and a lease deadline of tick
19. The host recorded `accepted: false, reason: lease` on the tick 31–32
receipt. The after-delay view was tick 37. The original controller also
returned `source_expired`; cancellation preserved existing motion, and a
distinct source takeover retired the fresh invocation without sending an old
owner packet.

The pinned `godot/scripts/physical_actor_session.gd`, file SHA-256
`ea6bee1fa84b03b6a1d03d4e98cd88f238321d981b4217db99347181f3113fb3`,
sets `MAX_LEASE = 18`. It first rejects an observation older than
`tick - 18` as `lease`; only after that check does it compare the individual
packet's deadline and return `expired_observation`. The original benchmark
selected only the latter reason. With age 24, the former reason is the expected
branch. This is a proven checker mismatch, with a retained host refusal; it
is not evidence that the stale action was admitted.

The prospective correction is source commit
`38eec4add5badc20b74bb82340d6dbf9f77b53d7`. The helper in
[deployment.py](../../src/nca/bench/deployment.py) requires a complete valid
original twelve-tick packet, issued-source identity, fresh sequence, finite
bounded action, matching receipt and a causal submission tick. Ages twelve
through eighteen require `expired_observation`; ages above eighteen require
`lease`. An arbitrary lease refusal, unexpired packet, mismatched sequence,
invalid lease or accepted request cannot pass. The `1e-12` wire correlation
bound only accommodates the original Godot JSON echo; no arithmetic or
trajectory acceptance tolerance changed.

A separate immutable audit applies this corrected predicate to the **same
original receipt**. It finds the one valid expired-packet refusal; all six
realtime predicates then agree with the recorded evidence. It performs zero
policy calls, host ticks or optimizer updates and does not replace the original
failed summary or claim a new physical screen. Independent source and receipt
reviews found no blocker in that correction or its narrow interpretation.

## Actual throughput and runtime costs

| Actual configuration | Decisions / host ticks | Episodes | Measured seconds | Decisions/s | Host ticks/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| One learned Jev actor | 500 / 3,000 | 13 | 45.459353849 | 10.998836492 | 65.993018950 |
| One learned Jev plus one authored Monowheel demo | 500 / 3,000 | 13 | 32.036004447 | 15.607439462 | 93.644636770 |

These sequential intervals include reset, capture and cancellation overhead.
Within those actual host-loop intervals, median/p95 controller-call latency
was 2.240602/9.438829 ms for the one-actor configuration and
1.675351/7.634434 ms with the authored companion. These timing conditions
are separate from the NumPy-only replay measurements below.
The complete throughput phase lasted 79.033537896 seconds, including the
separate ten reset and 100 observe samples. Median/p95 reset RPC latency was
67.949597/88.101642 ms; median/p95 observe RPC latency was
7.3184015/11.311884 ms. The two configurations each have one learned controller;
multiple learned actors and long-duration stability remain unmeasured.

The committed criterion defines `maximum_wall_seconds: 600` and
`decisions_per_configuration: 500`; the implementation stops at the fixed count
or the ceiling. Its owning guide says “at most 600 seconds.” The execution
instruction additionally requested a **full measured ten-minute interval**.
That duration was not reached. The original count-based gate result remains
true, but the longer-duration instruction is outstanding. No idle time,
additional samples or new invocation has been added retroactively.

The independent NumPy-only subprocess measured the full validation,
preprocessing, recurrence, action and source-stamp call on 1,000 actual local
observations. Its median/p95/p99/maximum latencies were
**1.0164885 / 2.111485 / 3.845009 / 8.225355 ms**, with total call time
1.160199316 seconds. Twenty-five episode initializations were measured
separately. Transport and weight loading are excluded from this timing; the
actual host-loop numbers above include transport and host work.

Numeric weights occupied 29,736 bytes and recurrent state 128 bytes. The
NumPy-only process reported current and peak RSS of 35,708,928 bytes. The
combined reference/controller process reported 253,980,672 bytes. Host current
RSS was 194,600,960 and 195,039,232 bytes for the two throughput configurations;
host peaks were 210,694,144 and 211,300,352 bytes over that process's lifetime.
These are measured process totals, not isolated model allocations. The actual
NumPy wheel used OpenBLAS 0.3.30 with unset thread environment variables;
a single-thread NumPy pool is not asserted. The Torch reference was explicitly
limited to one thread.

The retained seed-1103 update-7 checkpoint restored all eleven optimizer
parameter states, policy content, action/global/scenario RNGs and the next
episode boundary. Both 24-tick fresh episodes passed the unchanged comparison.
No optimizer step was taken. Reported BC/update costs refer to retained
completed original work; total interrupted-run cost remains unknown.

## Separate evaluated-bundle engineering gate

The corrected source passed all **eleven** focused benchmark tests, including
three new expiry regression groups. The existing four tests in
[test_export_host.py](../../tests/test_export_host.py) then ran once against the
actual evaluated bundle and verified pinned host. All four passed with zero
skips, failures or errors in 8.013181232 seconds.

The separate engineering receipt records one host start, five resets, eight
controller calls, seven numeric forwards, five admitted wire packets, six
steps and **36 confirmed host ticks**, plus eight cancellations. The tests
cover sequential owner requests, refusal to refresh an old result with a newer
view, takeover retirement and old-actor refusal after reset. Host termination
is confirmed; the normal adapter teardown terminated its owned process with
return code -15. The test wrapper returned zero. There were zero optimizer
updates and zero new formal deployment attempts. Source, bundle and original
failed summary bytes were rechecked unchanged.

## Matched clips and preservation

The original screen produced five matched pose GIFs: `floor-right-24.gif` and
`floor-left-24.gif` each contain 41 frames over 240 simulation ticks;
`decision-12-refused.gif`, `astra-profile-refused.gif` and
`fresh-episode-resume.gif` each contain five frames over 24 ticks. They use the
actual recorded own-part geometry, fixed camera and labelled comparison
panels. They are pose reconstructions, not native engine-rendered footage or
owner feel acceptance. Full paths within the screen are `clips/<filename>`;
the index binds each clip to its exact original trajectory bytes.

The complete original qualification archive has 93 byte-verified members and
contains the bundle, launch/termination, all original screen files, clips,
declarations and frozen measured source. The separate diagnosis/conformance
archive has 67 byte-verified members, including both audits, complete engineering
exchanges, source/test programs and the pinned host proof. Both archives and
their companion indexes have confirmed durable delivery receipts. Earlier
interrupted-input, candidate-evaluation and conversion archives remain
separate dependencies. Large artifacts remain outside Git; hashes do not
imply a public download URL.

| Preserved identity | SHA-256 |
| --- | --- |
| Original screen manifest file | `8bd30bbd6ceea1a670a951771ff449d8973e1c03672e010a4b8d8f69924f7a15` |
| Original failed summary file | `f2625739e7d05e86d2de6d955209452173ff2d0c72acbd64dae4cee2add2f62d` |
| Original failed summary unsigned content | `4f7c36979a11a37304c20d3bd95332c3529b9ff79ed991f45cd63033843b0553` |
| Qualification archive, 3,765,383 bytes | `aa5439c0fb4171a25fe28a46651f8fea396eb1598bcb7d765e0768f3320467cb` |
| Qualification archive member index file | `fa744841efbf9f8da50835254a6bcfc08cee45269b65a3131bab97b2b34eebaa` |
| Original-source offline parity audit file | `fae90c46ddedb448e0cfe850b424d5601d76ab83669066feabd478a24aded292` |
| Corrected same-receipt diagnosis file | `9146f121e6801b45878abcfbbd514444b0c12ddce5d2f27b05095e9eb1d62fba` |
| Separate engineering summary file | `bdf9c28cddeca47b089f5e217a103a558c2123bf43f8e3c6da6bdd488833971b` |
| Diagnosis/conformance archive, 204,649 bytes | `902bcb7aa4a3f0ec4aede185147ccaaa64a28ac7f3503042a383ada1d89e69a4` |
| Diagnosis/conformance archive member index file | `79411a2979eb4ba1113552376213742cafd86dbac279ed27e6fa3a18b556e381` |

The historical earlier T109 trajectory failure also remains a failure with
unavailable original raw evidence. This new policy instance and its complete
receipts do not replace or explain that historical result. Further physical
qualification needs a separate prospective scope and source/input freeze;
none is implied by publishing these diagnostics and engineering corrections.
