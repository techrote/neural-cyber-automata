---
title: Jolt recovery v3 — bounded contact and arrival correction
kind: evidence
summary: Prospective source, unchanged acceptance and separately bounded diagnostic for the movement correction in issue 64.
owns: [evidence.jolt-recovery-v3]
verified: 2026-10-20
---

# Jolt recovery v3

[JOLT-RECOVERY-003 / #64](https://github.com/techrote/neural-cyber-automata/issues/64)
adds a separately versioned movement correction after the complete negative
[v2 diagnostic](2026-10-20-jolt-recovery-v2.md). At this checkpoint no new physical
invocation has started and no teacher improvement is established.

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
controller/declaration tests pass; strict documentation validation passes 70
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
The exact public source/tree will be recorded in the issue and PR before the
single invocation; the detached executor must match those identities.

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

## Unchanged acceptance and cost boundary

Require complete source/receipt-bound evidence and zero integrity, lifecycle or
admission defects. Teacher acceptance still needs strict improvement over the
contemporaneous arrival reference, at least three historical rescues including
the required unsupported/drop case and a supported case, at most one regression,
restored/overall purposeful airborne activity, trim increase at most 0.5/s,
angle increase at most 0.1 rad, vigor ratio 0.7–1.5, every-case p95 at most 5 ms
and fresh authority for every second pulse.

The earlier original arrival diagnostic, incomplete v1 and complete negative v2
retain separate actual ledgers. V1/v2 already consumed two invocations, 2,928
decisions, 17,568 requested/confirmed ticks, 3,260 answered owner calls and
506.229990789001 whole-process seconds. The original truncated journal's
reported full counts remain distinguished from its verified prefix.

A failed bounded result stays failed. Evidence completion cannot admit teacher
collection. An accepted teacher releases only #62's teacher dependency; #16
still requires its independent learned gate.
