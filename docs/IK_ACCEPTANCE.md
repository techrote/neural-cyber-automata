---
title: Runner sole IK acceptance declaration
kind: reference
summary: Source-derived numerical criteria and prospective learned, runtime and physical gates for the first IK pilot.
owns: [ik.acceptance]
verified: 2026-10-20
---

# Runner sole IK acceptance

The [versioned declaration](../configs/ik/runner_right_sole_acceptance_v1.json)
is `runner-right-sole-acceptance@1`, schema `nca-ik-acceptance/v1`.
It records **prospective, unmeasured criteria** for the
[IK programme / #49](https://github.com/techrote/neural-cyber-automata/issues/49).
[Issue #50](https://github.com/techrote/neural-cyber-automata/issues/50)
checks the profile, contract and declaration in software. It performs no training,
CUDA qualification or physical-host execution. Passing those checks cannot make
the later learned or physical qualification flags true.

## Identity and ownership

The declaration binds the packaged
[Runner descriptor](../src/nca/ik/runner_right_sole_v1.json) by two different hashes:
`profile_semantic_sha256` is the validated `IKProfile` Record's canonical content
hash; `descriptor_file_sha256` hashes exact file bytes. A future run preregistration
also records the exact acceptance file's SHA-256 externally. There is no circular
self-hash. Preserve LF checkouts for the two byte-bound JSON files. Changed bytes,
geometry, limits or conventions need a new admitted identity.

| Issue | Evidence owned by that task |
|---|---|
| [#51](https://github.com/techrote/neural-cyber-automata/issues/51) | Actual CUDA preflight, forward/backward/optimizer smoke and synchronized measurements |
| [#52](https://github.com/techrote/neural-cyber-automata/issues/52) | Independent FK/Jacobian/numerical IK and geometric status semantics |
| [#53](https://github.com/techrote/neural-cyber-automata/issues/53) | Dataset families, partitions, held-out regions and immutable manifests |
| [#54](https://github.com/techrote/neural-cyber-automata/issues/54) | Three-seed training, raw/corrected/fallback results and learned contribution |
| [#55](https://github.com/techrote/neural-cyber-automata/issues/55) | Separately tracked host joint-target capability and owner conformance |
| [#56](https://github.com/techrote/neural-cyber-automata/issues/56) | Exact CPU export, common-input parity, lifecycle and runtime cost |
| [#57](https://github.com/techrote/neural-cyber-automata/issues/57) | Exact-bundle physical tracking, obstruction/realtime evidence and handoff |

## Geometry behind the tolerances

Source is Cybersand `f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497`:
the [catalogue](https://github.com/techrote/cybersand/blob/f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497/godot/scripts/physical_actor_catalog.gd)
defines Runner's five parts and rotary limits;
the [rig](https://github.com/techrote/cybersand/blob/f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497/godot/scripts/physical_actor_rig.gd)
defines rest-relative angles, inner target margins and the bounded motor response.
The source's semantic wire controls do not provide independent joint targets.
The host task supplies that capability before physical scoring.

The effector is the **sole point `(0, 1)` on `foot_1`**, in cells. It is not the
foot centre. With `h = (3.6, 2.8)`, `a = (0, 4.95)`, `b = (1, 1.25)` and
rest-relative hip/ankle coordinates `qh`, `qa`, nominal geometry is:

```text
p = h + R(qh) a + R(qh + qa) b
R(q)(x,y) = (cos(q)x - sin(q)y, sin(q)x + cos(q)y)
p(0,0) = (4.6,9.0)
L = norm(a) + norm(b) = 6.550781059358212 cells
```

Positive rotation is clockwise on the screen in the down-Y game frame. Initial
part angles and rotary zeros are already represented by the anchors; adding the
right leg's `0.18 rad` again changes the chain. Root-local conversion uses the
root pose from the same issued observation. The root is never an IK variable.

Only the two point coordinates are objectives. Foot orientation is diagnostic.
Hard limits are hip `[-0.75, 0.75]` and ankle `[-0.3, 0.3] rad`; the source's
`0.03 rad` inner margin gives command ranges `[-0.72, 0.72]` and `[-0.27, 0.27]`.
The nominal dataset workspace is the smaller hip `[-0.6, 0.6]`, ankle
`[-0.2, 0.2]` rectangle mapped through FK. This is geometric reachability,
not a claim that the foot can pass through terrain or other parts.

Runner inherits **`0.055 rad` play**. `0.05` is a missing-field fallback in the
rig, not Runner's value. The first-order allowance from both deadbands is
`0.055 * (norm(a) + 2*norm(b)) = 0.4483359165294033 cells`.
It explains why physical settling allows `0.75 cell`, while equation/serialization
checks use much smaller errors. It does not predict every contact or soft-pin error.

Within the inner command range, the Jacobian determinant magnitude is at least
`4.95 * (cos(0.27) - 1.25*sin(0.27)) = 3.1202651724990154 cells^2`.
Radius is monotone in the admitted ankle interval: there is no singular or second
admitted branch in this pilot. A `0.10-cell` point change bounds the ankle change
by `L*0.10/3.1202651724990154 < 0.210 rad`; the hip bound is smaller.
The declared `0.25-rad` continuity ceiling retains margin. Singular/redundant
synthetic reference fixtures have a separate identity and cannot be advertised as
Runner holdout coverage.

## Numeric and learned gates

Every endpoint error is Euclidean distance in cells. Normalized error is that
distance divided by `L`; **both** limits apply wherever a row has both. Joint
differences use the validated rest-relative coordinates; wrapping never bypasses
the hard or command bounds. Percentiles use nearest rank `ceil(p*N)`, without
interpolation or averaging percentiles from different episodes.

FK scoring uses the resolved desired position. Add a bounded-delta result to its
original measured configuration before raw scoring; do not score it after a
hidden projection or solve. An intermediate best-effort step is not a solved full
target. Target lifetime (at most eighteen ticks), original observation freshness
and helper actuation (at most twelve ticks) are distinct; current deadlines can
shorten all permitted dispatch intervals.

| Metric | Prospective limit |
|---|---|
| Independent float64 transform fixtures | Endpoint `1e-8 cell`; angle `1e-9 rad` |
| Reference IK on certified feasible cases | Endpoint `1e-6 cell` and `1e-6 L` |
| Jacobian element comparison | Absolute `1e-5 cell/rad` plus relative `1e-5` times the reference; central difference step `1e-6 rad` |
| Raw learned endpoint, every seed | p95 `0.06 cell` and `0.01 L`; worst `0.18 cell` and `0.03 L` |
| Raw learned feasible success | At least 95% at the raw p95 endpoint limits, counting every nominal query |
| Hybrid corrected endpoint, before fallback | p95 `0.006 cell` and `0.001 L`; worst `0.018 cell` and `0.003 L` |
| Raw hard-bound or nonfinite outputs | Zero |
| Nominal full-reference fallback | At most 1% of all scheduled nominal calls, per seed |
| Declared numerical refinement | At most two iterations per call; no hidden convergence loop |
| Soft-margin projection | At most `0.03 rad`; at most 5% of nominal calls change by more than `2e-5 rad` |
| Continuous same-branch solutions | At most `0.25 rad` per joint for target changes at most `0.10 cell` in the inner domain |
| Runner admitted-branch switches | Zero |

Choose direct output or learned initialization plus two numerical iterations
before training. Direct mode has zero numerical-refinement calls. Hybrid mode
may refine every call, but must retain raw error and the cost of each iteration.
Compare the same two-iteration solver with its normal initializer, plus the
independent analytic/reference result. Projection, refinement, full fallback and
emergency intervention are four separate counters.

Each trained seed's raw p95 error must be at most half its matched untrained
initialization's p95. Before training, select one primary practical-utility gate:
CPU p95 decision cost at most `0.8` of the matched numerical pipeline at the same
acceptance quality, or at least `0.05` absolute feasible-success gain over the
same two-iteration solver with its normal initializer. Report the analytic
reference's cost in either case. Accuracy, useful contribution and practical
utility have separate verdicts; a geometric pass cannot conceal a utility no-go.

## Splits, samples and denominators

Training seeds are `41011`, `42013`, `43019`; each has a separate initialization
and optimization history. No best-seed substitution is allowed. Split seed
`61003` fixes 70/15/15 training/tuning/sealed-evaluation family assignments over
at least 100 generation families. Whole trajectories, derived starting-pose
variants and augmentations inherit their family partition. The dataset task
publishes exact assignments, eight withheld target regions and exclusion bands
before learning; held-out targets and their derived copies cannot enter tuning.

Per seed, sealed nominal evaluation has at least 2,000 point queries over a
10-by-8 hip/ankle coverage grid, at least 25 per bin, including at least 200 queries
from withheld target regions. Forty complete trajectories have at least twenty
decisions each. Interior, rest neighbourhood, sweeps, reversals and root-frame
transforms remain separately tagged. Boundary, invalid and unsupported cases have
their own populations, at least twenty cases per class; they do not increase the
ordinary reach-success count. Apply raw endpoint gates to the withheld-region
subset as well as the complete nominal population.

The primary denominator is **every prospectively scheduled nominal query or
episode**, including refusals, failed outputs, fallback and interrupted attempts.
An absent/nonfinite raw output fails completeness; calculating a percentile only
over the survivors cannot qualify the model. Full-fallback fractions use every
scheduled nominal call, not only successful calls. Exact attempt counts and
finite allocations must be published; sample minima do not authorize extra runs.

Invalid, lifecycle and obstruction classes retain all scheduled attempts and
separate expected outcomes. Missing receipts or setup failures cannot be dropped
from those populations. Missing measurements are `null` with a reason, never zero.
An incomplete ledger is inconclusive or no-go, never qualified. These rules do not
turn a deliberate invalid-input refusal into an ordinary geometric failure.

## CPU export and owner boundary

Use at least 1,000 identical input/state comparisons across CUDA, CPU reference
and serialized runtime, with at least twenty per declared class. Maximum joint
output difference is `2e-5 rad`; FK endpoint difference is `2e-4 cell`. Status,
mask and admission semantics agree exactly. This is arithmetic conformance,
not physical trajectory or whole-game determinism evidence.

On one prospectively identified deployment CPU, use one thread, batch one,
100 warmup calls and 10,000 measured calls. Full-decision latency is at most
`2 ms` p95, `5 ms` p99 and `10 ms` maximum, including features, model, FK, guards
and actual correction/fallback work. Report startup, loading, transport, host
steps, memory and raw model cost separately. This timing gate does not run as a
generic shared-CI test.

At least three cases per lifecycle class exercise stale observations, expiry
before/at/after its boundary, cancellation/replay, takeover, consumed decisions,
actor/generation changes, profile/topology changes, unknown host capability and
ambiguous-receipt cleanup. Every case must match its stated expectation.
Geometric result status cannot confer dispatch authority.

## Physical reach and tracking gates

These apply to the exact exported bundle through the accepted host lane.
The whole actor remains physical, with unchanged assistance and servo limits.
Non-commanded joints retain explicitly declared host behaviour. No root locking,
pose assignment, hidden reset or stronger motor is part of the comparison.

| Nominal physical metric | Prospective limit |
|---|---|
| Episodes per evaluated bundle | At least 10 static and 10 smooth-sweep episodes; success at least 90% separately in each family |
| Static point acquisition | Error within `0.75 cell` and `0.13 L`, with the entire qualifying dwell completed by admission tick + 90 / 1.5 s |
| Static dwell | At least 18 consecutive completed tick intervals / 0.3 s within the endpoint limits, point speed at most `1 cell/s`; merely entering before the deadline does not pass |
| Initial static target displacement | Between 1 and 2 cells inclusive at admission; every trial starts outside the settling envelope |
| Nominal moving target | Speed at most `2 cells/s`; acceleration at most `8 cells/s^2` |
| Required sweep motion | Target diameter at least 3 cells, travelled path at least 6 cells, speed at least `0.5 cell/s` for at least half the scored intervals, and at least one reversal on a predeclared unit sweep axis |
| Sweep scoring | At least 240 scored ticks after at most 90 declared acquisition ticks; p95 `1 cell` and `0.16 L`, worst `1.5 cells` and `0.24 L` |
| Fresh target admission | At most 6 base ticks / 100 ms to reflected owner reference/motor receipt |
| Directed-response trials | At least ten predeclared nominal static trials per bundle; at least one initial requested joint-reference change of `0.1 rad` |
| Meaningful free-motion response | In every directed-response trial, at most 12 ticks / 200 ms from admission to directed `0.01 rad` joint change or `0.05-cell` error improvement; an absent request/response fails |
| Final static chatter window | Fixed final 60 ticks / 1 s; at most two reference-direction reversals per joint, ignoring increments at most `0.01 rad` |
| Static reference / actual-joint excursion | Peak-to-peak at most `0.11 rad` / `0.165 rad`, twice / three times authored play |

Actual endpoint error uses the sole and root from the same completed owner frame.
Target-admission latency and physical-motion latency are different metrics.
Directed response needs requested/admitted/applied evidence; passive drift alone
cannot establish controller response. Every scored tick is retained; do not choose
the quietest hold segment. Mechanical play and imperfect settling are allowed.

The minimum static displacement and mandatory response trials prevent already-at-
target cases from supplying the whole result. Sweep diameter/path/moving-time
criteria exclude constant targets and brief excursions embedded in long holds.
Measure target motion from the prescribed base-tick trajectory, independently of
the actor. Ignore projected speeds below `0.5 cell/s` when counting its reversal.

Every nominal fixture also runs a paired stationary-reference control: use the
same initial state, targets, assistance and servo, while refreshing the two initial
measured joint references at the same cadence. Separately for static and sweep
families, that control must succeed on at most 50% of scheduled episodes and the
learned system must improve success by at least 0.4 absolute fraction. Keep all
paired outcomes. Score the control using the same endpoint/dwell or sweep-tracking
and common integrity limits. Its lack of a new reference or attributed response
cannot by itself count as task failure; report those fields separately. If holding
the initial configuration already passes the suite,
the suite cannot establish useful physical IK, even if the learned scores pass.

Use at least four separately declared obstruction cases per bundle. An obstructed
target need not be reached; it must produce finite bounded requests, truthful
best-effort/blocked evidence and successful replacement/cancellation. Recovery
uses a predeclared free target. Free-space geometry never certifies an obstacle-free
physical path. Unsupported perturbations remain unmeasured.

The physical task chooses all three seeds or one named representative before
physical holdouts. It retains every attempted candidate and never selects the
best physical result afterward. A reference-only capability check determines
prospective nominal fixtures; failed nominal attempts cannot later be relabelled
as obstruction to improve the denominator.

## Failure semantics and versioning

Solved, best effort, certified geometric infeasibility and numerical budget
exhaustion remain distinct. Failed convergence cannot certify infeasibility.
Invalid, stale, unsupported, cancelled or already-consumed results have no
actionable payload. Presence, measurement, solver participation and actuator
authority remain separate masks. Missing either selected joint measurement or the
issued root pose rejects the nominal solve; nominal sole geometry is derived from
q and the descriptor. Physical scoring separately needs actual sole/root telemetry
from the same completed owner frame. Missing physical telemetry remains incomplete
or no-go. Absent state cannot be replaced by an observed zero.

Critical admitted nonfinite/bounds/authority/lifecycle/privilege defects have a
zero threshold. Nominal quality gates cannot compensate for one. The current
Jolt and morphology gates remain independent and unchanged.

Reference-only calibration can publish a new immutable version before any learned
training or sealed scoring. Retain this declaration, the change rationale and all
earlier evidence. Once learned or final physical outcomes are visible, changing
this identity's thresholds is forbidden by its admission rule. A new experiment
requires its own declaration and fresh evaluation discipline.

© 2026 techrote. All rights reserved.
