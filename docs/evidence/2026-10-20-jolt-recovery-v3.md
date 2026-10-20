---
title: Jolt recovery v3 — movement improvement and failed acceptance
kind: evidence
summary: One frozen v3 diagnostic records 28 of 32 arrivals but fails final inventory verification and the unchanged three-rescue teacher gate.
owns: [evidence.jolt-recovery-v3]
verified: 2026-10-20
---

# Jolt recovery v3

[JOLT-RECOVERY-003 / #64](https://github.com/techrote/neural-cyber-automata/issues/64)
adds a separately versioned movement correction after the complete negative
[v2 diagnostic](2026-10-20-jolt-recovery-v2.md). Its sole prospective allocation
has ended. V3 records **28/32 arrivals**, versus **27/32** for the contemporaneous
arrival reference and **17/32** for the original controller. It retains purposeful
airborne activity in all 32 cases and rescues the required drop case, but only
two of the five historical failures succeed; the unchanged minimum is three.

The original supervisor also reports **`completed=false`**: post-exit verification
found an extra compressed-journal staging file outside the worker's closed
inventory. The worker's completed flag, the failed supervisor receipt and every
original byte remain preserved. These separate failures prevent teacher
acceptance. This is a bounded negative result, not permission to collect or
train. Issue #64 and teacher #48 remain open; #62 and morphology #16 retain
their separate prerequisites. No retry or outcome-driven tuning occurred.

## Retained observation replay

Independent replay uses the original frozen v2 source
`8da294ae4a263dd95c34b181abb795b906060c2d`. It verifies all 93 source inputs,
32 candidate episode records and 900 decisions, including all 42 pulses and ten
second pulses. The fourteen right-floor initial anticipations lose raw support
between ticks 12 and 18 after only about 0.022405 cells of displacement. The
observed speed at the latter sample is about 0.967788 cells/s and the measured
floor distance about 2.005573 cells. V2 abandons that launch and waits for a
new stationary unsupported window. These observations establish a contact/timing
sequence; they do not measure its passive force cause.

The three right-60 regressions have quiet native intervals 231, 235–237 and
239–240, with longest/final spans of three/two ticks. Their near-target forward
commands remain positive at samples 228 and 234 despite speed above six cells/s.
The retained left-60 stall also demonstrates little useful movement after both
pulses. All historical artifacts and outcomes remain unchanged.

## Versioned correction

`authored-jolt-recovery@3` preserves the original references, v1/v2 source bytes
and minimal physical supervisor. It changes ordinary controller behavior only:

- A pending first launch can use its exact next six-tick support-loss observation
  when current floor, displacement, speed, resources, room and original lifetime
  still admit it. Gaps, unknown support, changed motion or authority discard the
  witness. No retained support flag authorizes a stale launch.
- A fresh justified second pulse uses a shorter room-bound arc with explicit
  signed horizontal velocity and vertical/time limits. At most two primary
  edges remain possible; energy, cooldown acknowledgment and freshness remain
  necessary.
- Near-target supported motion receives bounded braking and an immediate neutral
  quiet hold. Its status stays running; only caller-validated completion retires
  native authority. The tick-223 aliasing regression remains covered.
- After both pulses, a fresh supported no-progress window can request up to
  twelve ticks of ordinary traction correction. A new physical window is needed
  to restart it. The emergency supervisor does not supply movement policy.

No host mechanics, physical profile, task success threshold, gyro bound, neural
architecture or learning prerequisite changes. The [controller contract](../CONTROLLERS.md)
describes the local heuristics and their limits.

The [retained replay](jolt-recovery-v3-retained-replay-20261020.json) and
[independent source review](jolt-recovery-v3-source-review-20261020.json) preserve
the pre-run evidence. The latter checks 20,736 analytic state combinations;
all 4,076 admitted vectors satisfy the declared direction, effort, room,
ceiling and time envelope. This arithmetic check starts no simulator and does
not establish contact behavior or task outcomes.

The integrated offline suite ran 625 tests in 32.340 seconds: 464 passed and
161 expected optional-runtime/physical-fixture cases skipped. The 25 new v3
controller/declaration tests pass; pre-run strict documentation validation passes 71
files without errors or warnings. The unchanged synthetic process-cleanup test
fails under inconsistent sandbox process visibility and passes with consistent
visibility, matching the retained v2 environment finding. The new invocation
uses that passing execution context. Pinned host/runtime identities were checked
without starting a physical session. No supervisor or test was weakened.

## New prospective allocation

The [v3 declaration](../../configs/baselines/jolt-recovery-diagnostic-v3.json)
has canonical SHA-256
`03c58b439b808581a63541a8dbb005a4b73486acd9afd2a0fd635f1604abfb03`.
Its candidate specification SHA-256 is
`e180844789194c1a928c673599aed48a987db617e26be96e7043dc7ee6c72a2e`.
The exact source was prospectively published in
[issue checkpoint 6299485583](https://github.com/techrote/neural-cyber-automata/issues/64#issuecomment-6299485583)
before the single invocation:

- Source: `c6f558bbf6f185b31d7ea2c9b57336b3c7fee1ac`.
- Tree: `9ac1f17b16868319b88e32cbbf8df3e5bdbcce70`.
- Controller source: `367447b4dd732a8e999f1d172795685fa4591028026ebbdd2697d12dc0c2aab3`.
- Host: `f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497`, unchanged from v2.

The independent [frozen admission](jolt-recovery-v3-frozen-admission-20261020.json)
verified all 339 tracked blobs and a clean detached execution checkout.
All five [hosted source checks](jolt-recovery-v3-source-ci-20261020.json) passed
before execution. Final documentation integration does not reattribute the
physical run to a later commit.

| Account | Maximum |
| --- | ---: |
| Invocations / persistent hosts | 1 / 1 |
| Three-family matched episodes | 96 |
| Six-tick decisions | 3,840 |
| Requested native ticks | 23,040 |
| Owner calls | 4,226 |
| Whole-process seconds | 900 |
| Optimizer / checkpoint / learned-policy executions | 0 / 0 / 0 |

The wall ceiling includes startup, complete raw/sealed evidence, native cleanup
and independent post-exit verification. There is no automatic retry, budget
extension or tuning loop. Every episode retains its 40-decision/240-tick limit.
The same 32 already-open cases are used; their eight evaluation-origin cases
remain development diagnostics, never fresh sealed learning evaluation.

## Original verification failure

The [worker result](jolt-recovery-v3-result-20261020.json) records 96 complete
rows and `status=completed`. The unchanged [supervisor receipt](jolt-recovery-v3-supervisor-20261020.json)
records worker exit 0, no timeout, terminated worker/process group and
`completed=false` after **294.951525241 seconds**. Its independent verifier
raises `arrival_final_output: final artifact inventory is incomplete or changed`.
The supervisor wraps that verifier subprocess failure as `ValueError`; its
`error_code=null` does not erase the original contract error in the retained log.

All 196 declared artifacts exist with their original hashes, but the directory
contains one additional `owner-journal.jsonl.gz.pending`. It is 8,198,511 bytes,
identical to the final gzip with SHA-256
`f4930c47ec91ead8c7def33d1606bca756df4ae12c84e451b75bf0f780014452`.
The source sealer unlinks and directory-fsyncs staging before publishing its
seal and inventory. The retained evidence does not establish why this extra
path exists after exit. No file was removed to force a passing inventory, no
result was resealed and no later forensic check replaces the original failure.

The [original seal](jolt-recovery-v3-seal-20261020.json), raw journal, both
compressed files, all episodes, all logs and the failed receipt are retained.
Independent offline reviews distinguish valid declared bytes and native receipt
subchecks from whole-run qualification; they perform zero physical calls.
The [native evidence audit](jolt-recovery-v3-independent-evidence-audit-20261020.json)
verifies all 5,330 chain records, 96 episode receipts, 2,568 answered requests
and 13,818 requested-confirmed ticks, including source/action/lease binding,
native close and zero pending requests or failed admissions. It retains its
initial strict refusal and a later audit-script tuple/list assertion failure.
A separately identified offline continuation corrects that audit assertion and
binds the already completed native replay; no original run artifact is changed.

## Native movement and remaining failures

[Independent behavior replay](jolt-recovery-v3-independent-behavior-audit-20261020.json)
reproduces all 2,303 policy decisions, including
754 v3 decisions, and recomputes all 96 task/character results. All fourteen
initial right-floor transitions now request their first pulse at sampled tick
18, applied at native tick 19, versus v2 ticks 42/43. First toward motion on
those cases improves from 0.716667 to 0.316667 seconds. Mean first toward motion
over all cases is 0.321354 seconds, versus v2's 0.496354 and the arrival
reference's 0.227604. Every family's first active request is at zero and first
aggregate actuation at 1/60 second; motion response is distinct from inference.

| Measured quantity | Arrival reference | Recovery v3 |
| --- | ---: | ---: |
| Arrivals | 27/32 | 28/32 |
| Purposeful airborne cases | 25/32 | 32/32 |
| Confirmed primary pulses | 32 | 37 |
| Pulses with subsequent airborne confirmation | 25 | 35 |
| Mean airborne seconds | 0.578646 | 0.725000 |
| Mean peak toward speed, cells/s | 53.974276 | 42.979247 |
| Mean ordinary trim reversals/s | 0.490190 | 0.226210 |
| Mean absolute angle, radians | 0.026368 | 0.026414 |
| Worst individual-case inference p95, ms | 2.438675 | 1.022418 |

All 37 requested pulses have native upward impulse confirmation, including five
freshly justified second edges. Only three of those five second pulses have
subsequent airborne confirmation. No v3 case exceeds two primary edges. The
unchanged emergency supervisor intervenes on 19 decisions in five events; that
assistance is reported separately from ordinary movement. Native receipts
measure aggregate actuator impulses; passive/contact forces and separate assist
impulse contributions remain unmeasured.

The new phases record 33 braking decisions, 54 quiet-hold decisions and five
ground-correction decisions. All braking requests precede lower six-tick-end
speed; 32 end at or below six cells/s. Ground correction forms three bursts in
two cases, requesting 12/6/12 ticks and observing only 0.05684/0.02087/0.03107
cells of distance reduction during those requests. Neither case is rescued.
These are descriptive sequences, not causal recovery experiments. Sixteen
quiet-hold decisions have a nonquiet native tick in the next six, and six have
no quiet native tick in that interval. A sampled quiet state remains insufficient
to establish the caller's continuous twelve-tick completion predicate.

The separate [remaining-failure review](jolt-recovery-v3-remaining-failures-20261020.json)
matches all 160 decisions and task metrics over the four failed cases' 960
native ticks without changing the controller or requesting new physical work.

| Remaining failure | Native outcome and limit |
| --- | --- |
| `demo-floor-right-36-1` | New regression against arrival and v2; reaches tolerance but longest quiet span is nine ticks. Final distance 3.765681 cells; speed remains 6.8431 cells/s. |
| `demo-floor-left-60-1` | Second pulse loses about 0.453 horizontal cells by landing; subsequent bounded traction produces little progress. Final distance 5.055274 cells. |
| `reference-floor-left-63` | Second pulse gains about 0.600 horizontal cells by landing, but supported movement remains insufficient. Final distance 8.131480 cells. |
| `reference-drop28-left-57` | V2's previous rescue is lost. It enters tolerance repeatedly but longest quiet span is ten ticks, final span five; final distance 3.898402 cells. |

All four end at the original task deadline, with no deadline extension. Relative
to v2, five failures now arrive: right48-0, all three right60 cases and the
required drop12-right45 case. Two v2 successes are lost: right36-1 and
drop28-left57. No v2 purposeful-airborne case is lost. The new response and
landing outcomes do not establish the unmeasured passive-force causes.

## Unchanged teacher acceptance

The [negative teacher review](jolt-recovery-v3-review-20261020.json) retains
the original failed evidence gate and independently recomputed numeric checks.

| Gate | Result |
| --- | --- |
| Complete final evidence and zero integrity defects | **Fail:** original inventory verification failed |
| Strict arrival improvement | Pass: 28 > 27 |
| At least three named historical rescues | **Fail:** only left60-2 and drop12-right45 succeed |
| Required unsupported/drop and supported rescue | Pass: both represented |
| At most one regression against arrival | Pass: right36-1 only |
| Restored and overall airborne character | Pass: seven restored cases; 32/32 overall |
| Trim increase at most 0.5/s; angle increase at most 0.1 rad | Pass: -0.263980/s and +0.000045 rad |
| Vigor ratio 0.7–1.5 | Pass: 0.796291 |
| Every-case inference p95 at most 5 ms | Pass: worst case 1.022418 ms |
| At most two primary edges with fresh physical authority | Pass: all 37 pulses replayed |

The better arrival total does not admit teacher collection. The failed final
inventory and the insufficient historical rescues are independent blockers.
There is no new learning, morphology qualification or owner-feel acceptance.
Future physical work requires a separately commissioned, prospectively bounded
allocation; the unused portion of this allocation cannot fund a continuation.

## New and cumulative actual costs

| Invocation | Hosts | Rows | Decisions | Episode-confirmed ticks | Owner calls | Whole-process seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original arrival | 1 | 64 | 1,549 | 9,294 | 1,722 reported | 221.630556 |
| Recovery v1 | 1 | 24 | 479 | 2,874 | 549 | 61.764330 |
| Recovery v2 | 1 | 96 | 2,449 | 14,694 | 2,711 | 444.465661 |
| Recovery v3 | 1 | 96 | 2,303 | 13,818 | 2,568 | 294.951525 |
| Recovery v1–v3 | 3 | 216 | 5,231 | 31,386 | 5,828 | 801.181516 |
| All four | 4 | 280 | 6,780 | 40,680 | 7,550 reported | 1,022.812072 |

V3 remains within every numeric ceiling, with zero pending owner requests and
zero optimizer, checkpoint or learned-policy executions. Its worker accounting
interval is 183.302180 seconds; worker time through seal is 294.532832 seconds.
Only the 294.951525-second supervisor interval is used for whole-process totals.
Offline review and archive transfer are separately labelled, zero-physics work.

The original arrival journal still ends at pending request 843: 842 responses,
739 confirmed step requests and 4,434 raw-journal-confirmed ticks. Its 1,722 full
owner calls are reported, not fully journal verified. The 40,680 cumulative
tick total is episode-confirmed and does not repair that earlier incomplete
journal. V1's 24 rows still include its one incomplete episode. No historical
failure, truncated transport or outcome is replaced by v3.

## Preserved archive and publication

The [complete index](jolt-recovery-v3-index-20261020.json) identifies every
original output, supervisor stream and separately labelled derived review.
The external archive `nca-jolt-recovery-v3-diagnostic-20261020.tar.gz` contains
**229 files / 34,469,879 bytes**, SHA-256
`398e81ffc465150b9407cba64b43ed1ba0070aeb9d61180e96650aa1bedef7e8`.
It preserves the undeclared pending gzip, failed receipts and every audit
attempt. No original file is omitted or repaired to improve a verdict.

The [delivery audit](jolt-recovery-v3-delivery-audit-20261020.json) verifies a
fresh persisted readback against the complete archive identity, all 229 member
hashes/sizes and the gzip trailer. Archive integrity does not repair the
original final-output inventory or release a teacher gate. All packaging and
review are offline, with zero host, owner, simulation or learning work.

[PR #65](https://github.com/techrote/neural-cyber-automata/pull/65) integrates
the versioned experimental controller and this negative evidence. Required
final-head checks and accepted-main verification are recorded in its public
checkpoint. The physical source remains the prospectively frozen commit above.
