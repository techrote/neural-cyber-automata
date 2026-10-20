---
title: Jolt recovery v4 — offline correction and prospective diagnostic
kind: evidence
summary: A separately commissioned interior-settling correction preserves earlier failed runs and fixes its single-attempt review and evidence requirements before execution.
owns: [evidence.jolt-recovery-v4]
verified: 2026-10-20
---

# Jolt recovery v4

The owner commissioned completion of
[#64 at the JOLT-RECOVERY-004 recommendation](https://github.com/techrote/neural-cyber-automata/issues/64#issuecomment-6300675643).
[D-017](../DECISIONS.md#d-017--complete-the-separately-commissioned-fourth-jolt-recovery-diagnostic)
records that fresh scope: an offline correction and independently reviewed
source/declaration, followed by one prospectively published matched diagnostic.
The recommendation's original proposal-only status does not consume or extend
an earlier allocation. Every earlier source, artifact, failure and cost stays
separately identified.

This page initially records the **offline pre-freeze checkpoint**. No new v4
physical host, owner request, native tick, optimizer update, checkpoint execution
or learned-policy call has occurred at this checkpoint. A new teacher outcome
is unmeasured. The exact execution commit/tree, complete declaration and final
independent admission must be published before the single host start.

## Preserved baseline and remaining problems

Accepted main at task entry is `d7d4abe49c537426ef6c6aed0be60de4cdd8b70e`, tree
`12656145fe9f2a2c25d52dddee2eb70a19bc8da8`, integrating
[PR #65](https://github.com/techrote/neural-cyber-automata/pull/65). The earlier
physical source remains `c6f558bbf6f185b31d7ea2c9b57336b3c7fee1ac`; it is not
reattributed to that later merge or to this new candidate.

[V3's original result](2026-10-20-jolt-recovery-v3.md) records 28/32 arrivals
versus 27/32 for the arrival reference and 17/32 for the original controller.
All 32 cases retain purposeful airborne motion, and all fourteen initial
right-floor pulses advance from sampled tick 42 to tick 18. Its two historical
rescues are `demo-floor-left-60-2` and `reference-drop12-right-45`; the unchanged
minimum is three. `demo-floor-right-36-1` is its one regression against arrival.

V3's original supervisor is `completed=false`: final verification found the
extra `owner-journal.jsonl.gz.pending` outside the closed 196-artifact inventory.
The extra gzip matches the final journal byte-for-byte, with SHA-256
`f4930c47ec91ead8c7def33d1606bca756df4ae12c84e451b75bf0f780014452`.
Its historical origin remains unestablished. No original extra file, inventory,
receipt or failure is removed or resealed. New durability software cannot
retroactively qualify that failed run.

The independently reviewed [remaining failures](jolt-recovery-v3-remaining-failures-20261020.json)
identify the correction's concrete scope:

| Retained failure | Observed v3 limitation |
| --- | --- |
| `demo-floor-right-36-1` | Repeated tolerance entries, longest continuous quiet span nine ticks; final distance 3.765681 cells and speed 6.843073 cells/s |
| `reference-drop28-left-57` | Repeated outer-boundary crossings, longest quiet span ten ticks and final span five; final distance 3.898402 cells |
| `demo-floor-left-60-1` | Second pulse loses about 0.453 horizontal cells by landing; later supported traction remains weak; final distance 5.055274 cells |
| `reference-floor-left-63` | Second pulse gains about 0.600 horizontal cells by landing; later supported correction remains insufficient; final distance 8.131480 cells |

The [independent history audit](jolt-recovery-v4-historical-costs-20261020.json)
replays all 96 v3 rows and 2,303 controller decisions, including 754 candidate
decisions, 37 candidate pulses and five second edges. That exact replay verifies
the retained result; it does not predict the result of a new action.

## Versioned ordinary movement correction

`authored-jolt-recovery@4` inherits the exact v3 pulse-vector and fresh initial
contact-transition methods. It retains v1/v2/v3, original/arrival references,
shared shaping and minimal supervisor source bytes. No host mechanics, profile,
gyro limit, task predicate, two-edge cap or original deadline changes.

The pinned [host traction implementation](https://github.com/techrote/cybersand/blob/f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497/godot/scripts/physical_actor_rig.gd)
applies contact traction only with nonzero smoothed horizontal demand. Neutral
therefore withholds active traction; it is not an active zero-speed command.
The retained drop28-left57 trajectory alternates around the four-cell Euclidean
boundary after v3 switches a sampled quiet state to neutral. This source rule
and measured sequence motivate continued slow inward correction; they do not
identify passive/contact-force contributions.

The new controller enters capture on current raw support inside the caller's
tolerance. It requests bounded 0.10 inward traction toward an interior region;
inside that region, excessive horizontal speed requests bounded counter-traction.
Neutral requires Euclidean distance at most 0.75 times tolerance, total speed
at most two cells/s and the unchanged 0.32-radian posture bound. Capture persists
through small outward disturbances, releasing beyond 1.25 times tolerance or
on support uncertainty/loss, gaps or authority boundaries. These are local
policy choices. The evaluator still requires its original four-cell/six-cells/s
continuous twelve-tick success predicate.

The second change concerns the immediately observed post-pulse state. Exactly
six ticks after the second pulse, a current consecutive observation with
positive cooldown acknowledgment, raw support and upward speed at least
six cells/s may request one six-tick traction coast. Its original-source action
lease is narrowed to six ticks; a later, delayed, nonrising or uncertain sample
cannot extend the witness. This may preserve useful motion while a rising
post-pulse state develops, but only native evidence can establish its effect.
There is no inferred support flag, extra primary edge or supervisor movement.

Sampled arrival stays advisory and `status="running"`. The caller validates
its independent continuous native outcome and current ownership before canceling
local and native intent. The tick-223 aliasing regression remains intact.
[The controller reference](../CONTROLLERS.md#interior-settling-in-recovery-v4)
documents the public state and action limits.

## Offline software evidence

The [retained prepass](jolt-recovery-v4-retained-prepass-20261020.json) verifies
all 32 v3 candidate episode hashes and record identities, and exactly reproduces
their 754 original decisions. The v4 controller consumes the same source
observations for 754 offline arithmetic probes: 123 requests differ, 112 probe
decisions carry capture state and five carry the bounded post-pulse coast.
All first-pulse source ticks remain identical to v3.

After a changed request, later retained observations are **not a counterfactual
physical rollout**. The prepass claims no new arrival, rescue, airborne result,
continuous quiet span, native closure or teacher acceptance.

The eleven new [controller tests](../../tests/test_jolt_recovery_v4.py) pass,
and all 86 recovery tests pass at the controller checkpoint. They cover retained
contact/settling samples, Euclidean margin, braking, hysteresis, lifecycle/gap/
emergency boundaries, acknowledged second-pulse coast and its six-tick lease,
the two-edge cap, goal-name independence and the tick-223 aliasing failure.
The final offline suite runs 682 tests: 521 pass and 161 optional/physical cases
are expected skips with no game host configured. All 57 new tests pass, including
strict artifact inventory, durable publication, complete-request timeouts,
supervisor deadline failures and fourteen external-executor cases. A harmless
Linux Python process test confirms cleanup across separate sessions in the
execution context. The default sandbox's inconsistent process namespace is
refused before launch. Strict documentation validation passes 72 files.

The [independent source review](jolt-recovery-v4-source-review-20261020.json)
binds every package source, Jolt test and execution tool. It reproduces all
32 first-launch prefixes, comprising 605 unchanged requests before their first
divergence, and finds no remaining source or admission-rule blocker. Exact
published commit/tree and a clean detached checkout still require prospective
admission before the host starts; final-head CI remains a separate merge gate.

| Offline identity | SHA-256 |
| --- | --- |
| New controller source | `d7903561e37bc0845b3993d3b608ff83200021084e4d41e6f8ab19dff38edeec` |
| New candidate specification | `2a04924885e3147881fef5ed3421c7c0e43457faa83e5c13b8e3c351c0384038` |
| Preserved v3 controller | `367447b4dd732a8e999f1d172795685fa4591028026ebbdd2697d12dc0c2aab3` |

Independent byte verification also checks the unchanged host revision
`f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497`, tree
`445a185e9f9824b16953fec27bf0157dfeedca77`, eleven host scripts, 37 native source
inputs and the pinned Godot/native/Rapier artifacts. It does not start an engine
or exercise a physical host. The complete hashes are in the history audit.

## Single prospective allocation

The [commissioned recommendation](https://github.com/techrote/neural-cyber-automata/issues/64#issuecomment-6300675643)
sets these hard ceilings. They are limits, not expected consumption estimates.

| Resource | Ceiling |
| --- | ---: |
| Diagnostic attempts / persistent host starts | 1 / 1 |
| Opened cases / controller families / fresh episodes | 32 / 3 / 96 |
| Global controller decisions / requested ticks | 4,000 / 24,000 |
| Per-episode decisions / requested ticks | 40 / 240 |
| Effective total decisions / requested ticks | 3,840 / 23,040 |
| Owner calls including lifecycle and cleanup | 4,500 |
| Whole-process seconds | 900 |
| Shutdown, sealing and independent-verification reserve within the 900 | 120 |
| Optimizer updates / checkpoint executions / learned-policy calls | 0 / 0 / 0 |
| Automatic retries / extensions | 0 / 0 |

The original controller, arrival reference and new candidate each receive a
fresh episode for every one of the same 32 cases in their existing order. No
old episode is reused or resumed. All remain previously opened development
diagnostics, including the eight evaluation-origin cases; none becomes sealed
learned evaluation. No tuning against interim physical outcomes is admitted.

Before the host starts, complete synthetic durability/cleanup/strict-inventory
tests, independently review the integrated source and unchanged acceptance,
publish the complete versioned declaration and exact commit/tree, verify a
clean execution checkout, and publish an independent admission decision and
historical ledger. Stop admitting new episodes before the 120-second reserve
is endangered. Startup, native termination, raw/compressed sealing and original
independent post-exit verification all fit inside the single 900-second ceiling.

The sole external executor passes one original monotonic clock through the
supervisor and worker. Normal transport admission reserves fifteen seconds for
the complete exchange, ending normal work by elapsed 780 seconds. The worker
must exit by 840 seconds, leaving sixty seconds for independent replay and final
accounting; six of those seconds remain for finalization. The executor measures
through the original supervisor's actual exit, binds its immutable receipt and
finalization marker, and requires the complete owned process tree to be gone.
Its own stream synchronization and receipt publication are subsequent offline
accounting. They cannot extend the physical allocation or turn a timeout into a
completed diagnostic.

A timeout, uncertain request, surplus artifact, invalid lease or integrity
failure consumes an incomplete/failed attempt. No retry or budget extension
follows. Independent forensic subchecks do not replace an unsuccessful original
final-inventory verification.

## Unchanged teacher gates

Acceptance remains the original [teacher rule](../../src/nca/evaluation/jolt_recovery_acceptance.py).
Require strict improvement over the contemporaneous arrival reference, at least
three of the five named historical rescues including the required
`reference-drop12-right-45` and a supported failure, and at most one regression.
Restore at least one named missing-airborne case and retain overall purposeful
airborne count against the contemporaneous reference. Report any loss against
v2/v3's 32/32 separately. Mean ordinary trim-reversal increase stays at most
0.5/s, mean absolute-angle increase at most 0.1 rad, vigor ratio 0.7–1.5, and
every-case inference p95 at most five milliseconds. Every second primary edge
needs its fresh physical reason and unchanged resource/authority admission.

Complete receipt-bound evidence, original final-inventory acceptance, native
closure and zero lifecycle/integrity/admission defects remain independently
necessary. A 30/32 arrival result is an aspiration, not a substitute for any
gate. A passing teacher releases only #62's teacher dependency; learned
collection, training and morphology retain their separate authority and gates.

## Historical actual costs before v4

The independent historical ledger preserves each invocation's original source,
summary hash, supervisor hash, failure and actual wall interval. No v4 physical
cost has been incurred at this offline checkpoint.

| Invocation | Hosts | Rows | Decisions | Episode-confirmed ticks | Owner calls | Whole-process seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original arrival | 1 | 64 | 1,549 | 9,294 | 1,722 reported | 221.630556 |
| Recovery v1 | 1 | 24 | 479 | 2,874 | 549 | 61.764330 |
| Recovery v2 | 1 | 96 | 2,449 | 14,694 | 2,711 | 444.465661 |
| Recovery v3 | 1 | 96 | 2,303 | 13,818 | 2,568 | 294.951525 |
| All four | 4 | 280 | 6,780 | 40,680 | 7,550 reported | 1,022.812072 |

The original arrival journal ends at unanswered request 843; its full 1,722
calls and the cumulative 7,550 count are reported, not wholly journal-verified.
Cumulative ticks are episode-confirmed. V1's 24 rows include an incomplete
episode. V2's complete verification retains its failed numeric teacher result;
v3 retains its failed original inventory and insufficient historical rescues.
Offline replay, source review and publication do not repair those qualifications
or count as a new physical invocation.
