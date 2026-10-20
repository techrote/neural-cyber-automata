---
title: Current state
kind: state
summary: CPU contract implementation and explicit evidence limits for the current Cybersand control programme.
owns: [state.summary]
verified: 2026-10-20
---

# Current state

Nine of ten programme tasks have accepted results. T-106 ([#16](https://github.com/techrote/neural-cyber-automata/issues/16)) remains open:
its designated primary fails the fixed 75% mean / 50% every-seed reliability prerequisite.
[The roadmap](ROADMAP.md) maps all ten programme tasks and their remaining gates.

[Export authority #45](https://github.com/techrote/neural-cyber-automata/pull/45) and [native CI #47](https://github.com/techrote/neural-cyber-automata/pull/47) are accepted.
The [Jolt arrival reference](evidence/2026-10-20-jolt-arrival-reference.md) has 27/32 descriptive arrivals versus 17/32.
Its historical complete-journal audit remains incomplete. Durable evidence software is accepted in [PR #58](https://github.com/techrote/neural-cyber-automata/pull/58).
The [recovery v1 diagnostic](evidence/2026-10-20-jolt-recovery-v1.md) stopped at 24/96 rows; corrective [#60](https://github.com/techrote/neural-cyber-automata/issues/60), #43 and morphology #16 remain open.

## T-100 — Accepted research baseline

The T-100 research programme landed in PR #21 at
`95d8788686980974d4f849e892c33f531d992fe7`.
## Accepted CPU foundation

T-101 merged through [PR #22](https://github.com/techrote/neural-cyber-automata/pull/22)
at `9e6da91ff520e27bc78d11dd8196844a9fa37ff3`, adding the installable CPU package,
strict gameplay/observation/action contracts, backward-compatible ACT003 mappings,
a versioned extension registry and immutable reproducible run records.

| Area | Evidence |
|---|---|
| CPU package and command line | Editable installation, environment report and recorded example exercised locally |
| Contracts, registry and run store | 27 focused tests; corruption, concurrency, lifecycle and privilege negative controls |
| Command-line conformance | Three additional tests for immutable records, optional-import isolation and invalid input |
| Linux/Windows package workflow | Both operating systems and the documentation check passed on the exact final T-101 head |
| Physical host adapter | T-102 implementation and actual Linux conformance described below |
| Authored gameplay controllers | Jolt/Vector feedback skills and matched actual-host screen below |
| Trained weights or scored learning runs | No accepted training result in T-101 |
| Runtime export, model service and device qualification | No accepted result in T-101 |

The [contract evidence](evidence/2026-10-19-t101-contracts.md) records source digests,
observed checks and corrected failures. The [package guide](PACKAGE.md) contains
commands; [progress checkpoints](PROGRESS.md) record ongoing implementation.

## Persistent actual host

The [host adapter](HOST_ADAPTER.md) implements owned reset/step over one persistent
Godot process, preserving the accepted native material/Rapier causal schedule.
Recorded playback, actual ACT003 sampled wire/metal and actual ACT004 physical
lanes have distinct identities. Six-tick decisions, twelve-tick helper leases,
exclusive observation expiry, cancellation and terminal durations are tested.

The [T-102 evidence](evidence/2026-10-19-t102.md) reports 23 focused passing cases,
including fifteen real integrations, and bounded 1,200-tick pilots for Monowheel
and Ripper. Realtime mode advances while a client is delayed and returns the
completed terminal state once. Linux host CI verifies pinned source and runtime
before executing the integration tests; final-head results belong to the PR.

These tests qualify the adapter on the selected host and CPU runtime. The actual
host has no separate passive-wrench decomposition, arbitrary body replacement,
support-removal or assist-mutation API. Those capabilities remain unavailable.
Windows physical runtime, desktop pacing and owner feel remain unmeasured.

## Authored skill reference

The [authored controllers](CONTROLLERS.md) implement Jolt directed hops, Vector
burst traversal and their declared recovery mappings. The accepted-adapter
screen ran 18 matched episodes across firm/balanced/compliant tuning and
step/pulse/reversal fixtures. All six step variants reached their target;
compliant Jolt missed the reversal deadline, and Vector drifted after completing
arrival. These limitations and all exact source/artifact hashes are in the
[T-103 evidence](evidence/2026-10-19-t103.md).

Thirty-three focused controller/measurement checks pass. Forced perturbation
fixtures are published but explicitly refuse unsupported host operations;
arbitrary disturbance, support removal, payload replacement and joint override
remain physically unmeasured. Owner feel approval is not recorded.

## Demonstration and replay boundary

The [data API](../configs/datasets/README.md) captures every consumed base tick,
the original source observation, dispatch-time view, source/policy identity,
requested/admitted action, actual impulse/motion, intervention and lifecycle
boundary. Unknown, refused, mixed or corrected consumption cannot acquire an
ordinary on-policy likelihood. Per-episode files publish immutably and family
splits reject contamination.

The [T-104 evidence](evidence/2026-10-19-t104.md) contains three actual capture
pipelines with 296 base receipts, compatible-host replay and separately recorded
skill outcomes. The +48-cell Jolt invocation succeeded by its local estimate;
the +20-cell invocation remained unsuccessful. No optimization occurred. The
recorded script transfers are not human or live-model demonstrations, and replay
qualifies copied local trajectories on the same build, not whole-world identity.

## Authored usable skills and composition

The [usable runtime](USABLE_SKILLS.md) provides typed approach, aim/use, recovery,
grip, carry and release over the accepted Vector launcher and Hammerhead cutter.
Pouncer charge-hold is the separate disarming example. Twenty-nine focused tests
pass, including thirteen actual-host integrations. The five-case
[T-107 pilot](evidence/2026-10-19-t107.md) exercised native clearance refusal with
recoil, declared recovery, physical pickup/carry/use/release, a retained failed
late pickup, persistent applied effects and charge cancellation without launch.

Owner control and observed attachment generations are independently fenced.
Local skill predicates remain separate from evaluator success; the scripted
human/model source labels do not claim an actual input session or live service.
Independent arbitrary object replacement and owner feel remain unmeasured.

## First learned Jolt skill

The optional [CPU learning lane](LEARNING.md) implements behavioural cloning and
recurrent residual PPO with original latent-action likelihoods, ordered sequence
updates, valid duration/continuation masks and immutable checkpoints. Its
[T-105 screen](evidence/2026-10-19-t105.md) used three policy seeds, six frozen
final checkpoints and eight unseen paired scenarios. It completed 11,755 game
decisions in 685.6 seconds under the predeclared 13,000-decision / 30-minute cap.

Mean success was 45.83% for cloning and 50.00% for residual PPO, versus 25.00%
for the authored reference. Both passed the declared improvement, integrity,
character and inference gates. Every failed scenario and weaker seed remains
recorded. The complete artifact audit found no held-out data in optimization.
An original deterministic-evaluation likelihood label defect is preserved in
historical records and corrected in the final source; it did not alter trained
weights or wire outcomes. Owner feel remains unjudged.

These are bounded exploratory improvements. Neither family met the independently
predeclared 75% mean / 50% minimum-seed morphology entry gate. Transfer learning
remains blocked by that result. The separately evaluated export candidate is described below.

## Gated morphology preparation

The [morphology preparation](evidence/2026-10-19-t106.md) pins accepted Jolt,
Vector, Monowheel and Ripper contracts with eight part slots, seven connection
slots and separate presence, measurement and activity masks. Optional padded
recurrent and two-round shared graph candidates have 13,048 and 13,192 parameters.
The initial forty-one focused checks and independent review cover locality, padding,
source-age limits, recurrence and confirmed owner-revocation boundaries.

The original immutable admission runner retains the failed predeclared T105 reliability
prerequisite and emits zero training, candidate-game and optimizer work. Its
fixed conditional comparison budget is preparatory; no transfer result exists.
A separate [32-request / 192-tick scripted profile diagnostic](evidence/2026-10-19-t106-conformance.md)
passed on all four profiles at public source `557799ce`. It retains the first
zero-request startup failure; both attempts total 54.754654 seconds within the
original 60-second allocation. This establishes interface conformance; learned
transfer, adaptation, damage recovery and owner feel remain unmeasured.
The research question remains blocked by the stated prerequisite.

The separately commissioned [Jolt direction-frame BC follow-up](evidence/2026-10-19-t106-direction-frame.md)
retains two separately identified executions. The designated primary at public
`1222d2e8` completed all 136 episodes and twelve initial/final checkpoints:
4,693 requests / 28,158 ticks / 11,760 optimizer updates in 943.605670498
internal seconds. Direction-frame BC reached 25% mean success and 12.5%
minimum-seed success; world-frame BC reached 29.17% and 0%. Its original
character/runtime failures remain in the [primary index](evidence/t106-direction-frame-result.json).
The later concurrent GitHub alternate completed 4,660 requests / 27,960 ticks /
11,760 updates in 385.969869986 internal seconds, with direction-frame success
33.33% / 25% and world-frame success 25% / 0%. Both fail the unchanged 75% mean /
50% every-seed prerequisite. Neither result is substituted for the other.

The combined 9,353 requests / 56,118 ticks / 23,520 updates exceed the original
6,000-request declaration. Full combined wall time is unknown because the internal
timers exclude outer/startup overhead. [PR #36](https://github.com/techrote/neural-cyber-automata/pull/36)
merged the corrected provenance and [transport recovery](evidence/t106-primary-archive-recovery-20261020.json).
All 438 original primary members match the unchanged index. A separately named
440-member transport preserves those bytes, the original index and a recovery
receipt after the available original TAR copy was found truncated. The repair
starts no experiment and changes no result or admission gate.

[PR #38](https://github.com/techrote/neural-cyber-automata/pull/38) merged the
[conditional capture and learning implementation](evidence/2026-10-19-t106-learning.md),
including the guarded orchestrator, semantic waypoint reference and complete
148-row comparison summary. Ninety-nine focused software checks pass; the
148-row physical comparison remains unexecuted. The original 12,704 comparison
requests plus 160 prospective teacher-qualification requests total 12,864 within
the unchanged 13,000-request ceiling. The immutable primary designation prevents
a concurrent repeat from replacing the failed prerequisite.

The complete public-source admission at `eb48867` finished in 217.304178604
seconds, separately from the earlier 282.603-second development admission. It
returned CLI exit 2, `training_blocked` and `admitted=false`, with no continuation
declaration and zero host starts, model constructions, checkpoint loads, policy
calls, game requests or optimizer updates. Torch and the optional morphology
capture, learner and checkpoint modules remained absent; generic evidence/capture
and host adapter definitions were imported. The receipt keeps its original source
and tree identity after integration. A separately commissioned passing prerequisite
and a qualified semantic waypoint reference remain required before morphology
learning or transfer; no family is selected and issue #16 remains open.

## Asynchronous source ownership

The [asynchronous adapter](ASYNC_CONTROL.md) connects bounded mock/local-worker
providers and typed human/script intent to the same usable skill runtime.
Original observation/plan age, actual actor control generation and attachment
changes are fenced independently. Local feedback continues while retired model
work remains busy; an unacknowledged call cannot accumulate replacement calls.

The [T-108 evidence](evidence/2026-10-19-t108.md) distinguishes proposal admission,
local skill feedback and actual owner receipts. Model smart plans expire at the
original observation tick plus 120. Each ordinary twelve-tick motor lease is capped
by that same plan deadline; fresh feedback cannot renew the plan. Longer explicit
typed local missions are separate. The original frozen actual pilot passed ten cases, recording 338 owner
operations and 790 completed native ticks in 19.169 seconds with no skips or source
drift. It remains evidence from its original source.

[PR #40](https://github.com/techrote/neural-cyber-automata/pull/40) corrects later
conformance fixtures that assumed native leases would remain valid while a realtime
client was delayed. Exact delay/jitter, duplicate/reordered reply, outage, takeover,
reset and attachment sequences now use the stepped actual host. Three realtime
cases verify independent native advancement, terminal readback and human charge
followed by receipt-proven lease expiry. The focused 37-test suite passed without
skips, including nine stepped and three realtime actual-host cases; all four final
workflows passed before the reviewed tree was merged. Production ownership and
expiry rules are unchanged. The revised two-mode pilot recorder is implemented;
the original ten-case pilot is not relabelled as an execution of that revision.
No live service or owner feel verdict is claimed.

## Qualified Torch CPU export and preserved no-go runs

The optional [runtime export lane](RUNTIME_EXPORT.md) binds the fixed BC1103 policy,
preprocessing, recurrence, action mapping and exact runtime context.
Its NumPy and Torch CPU backends preserve source, ownership, deadline and profile
fences. The ordinary converter requires complete T105 evidence; the staged
converter retains interrupted training and requires separate candidate evaluation.

The [dispatch authority follow-up](evidence/2026-10-20-t109-dispatch-authority.md) binds
unused decisions to the producing controller, confirmed native cleanup and capped deadlines.
Six native cases have passing coverage; original artifact qualification keeps its original source.

The current fixed BC1103 artifact came from the interrupted reproduction of the
accepted learner implementation. Its [separate evaluation](evidence/2026-10-19-t109-candidate-evaluation.md)
at public source `4510347e` completed all eight fixed cases with three successes
and five retained failures. This candidate result is distinct from the completed
historical three-seed method comparison above. Its training origin remains
`interrupted_no_summary`, with original termination and total work unconfirmed.

The [original NumPy screen](evidence/2026-10-19-t109-deployment-no-go.md) at `eb953615`
remains no-go: both physical pairs exceeded unchanged tolerances.
Its 79.033537896-second throughput record did not supply the requested ten minutes.

[PR #41](https://github.com/techrote/neural-cyber-automata/pull/41) accepts the
[separate composite qualification](evidence/2026-10-20-t109-torch-cpu.md) for the 227,186-byte artifact, SHA-256
`1345c35582bbd16f7f6474996976dc3b0fac3dd775a2f83f916347b43e5fb96c`.
This satisfies T-109/#19 for the declared Linux external Python lane with
Torch 2.8.0+cpu, one Torch thread, the pinned ACT004 host and unchanged policy.
The original physical invocation at `f3bf96fa` passed arithmetic, both paired
trajectories, contract/refusal/realtime gates and 600.520737906 active seconds.
Throughput was 95.163 ticks/s for one learned Jolt and 77.673 ticks/s with an
additional authored Monowheel; both configurations had only one learned actor.

That original summary remains `no_go/runtime_qualified=false` because variant
profile metadata was omitted before rendering. A separate offline completion at
`78cc377b` produced all four pose GIFs, with both nominal GIFs byte-identical,
in 24.190651844997774 complete supervisor seconds and zero new host, policy or
optimizer work. Combined with the original invocation, 666.2482367169978 seconds
remained within 900. Qualification requires the distinct composite receipt;
the artifact envelope and both original no-go results remain unchanged.

Standalone full-decision cost was median 1.494447 ms / p99 4.120924 ms over
1,000 calls. The full cost tables retain a 10.368-second maximum host/transport
call. Average stepped throughput does not qualify desktop pacing, continuous
realtime deadlines, multiple learned actors, other devices or owner feel.

## Conditional per-part graph result

The [fixed Jolt graph experiment](evidence/2026-10-19-t110-screen.md) completed 119 episodes with six preselected
checkpoints, normal/delayed/lost/partitioned/reset communication, and untrained controls. Its three-part star has
private recurrence, one neighbor round and root-only readout; the equal-size no-message candidate shares its features
and action limits. Original declarations, failures and audit corrections remain in the [evidence index](evidence/2026-10-19-t110.md).

The predeclared result is no-go: graph and no-message each succeeded in one of
six normal holdout trials, with zero paired success gain; the graph was slightly
slower under the capped-completion metric. Each untrained method succeeded in
three of six. The trained successes did not require a purposeful hop, a retained
limitation of the frozen quiet-arrival task. Other declared gates passed, but
these observations provide no reason to promote communication in this screen.
Aggregate work stayed within 4,000 decisions / 900 seconds: 3,315 decisions and
227.1865 seconds through final ledger publication. Owner feel remains unjudged.

The independent local reproduction stopped producing records after 112 episodes,
3,096 confirmed decisions and 18,576 ticks; its completion and termination are
unconfirmed. Its audited archive is saved. A concurrently launched GitHub repeat
later completed 119 rows / 3,297 decisions and reported the same no-go. These are
separate executions. Their combined confirmed decisions are at least **6,393**,
above the original aggregate **4,000** ceiling; the GitHub receipt's reported
142.008-second total omits the interrupted local work. This deviation is retained
in [the reconciliation and evidence](evidence/2026-10-19-t110-reproduction.md),
with both immutable artifact identities. It does not qualify graph utility.

The graph runner records native cancellation and clears private memory/messages on early arrival; its host regression passed.
Historical source/records are unchanged. Owner feel and unsupported topology remain unmeasured.

## Host authority

The ACT-004 host accepted through Cybersand PR #181 is pinned at
`f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497`; T-102 qualifies its configure/reset,
observation, submit/cancel, step and snapshot API, with no host changes by T-101.
Package/recorded tests do not measure gameplay, learned-policy value, owner feel or GPU qualification.
