---
title: T109 controller-bound dispatch and native cleanup
kind: evidence
summary: Issue 42 fixes retained-decision replay, absolute deadlines and uncertain cleanup, with software regressions and bounded synthetic-policy native-owner tests.
owns: []
verified: 2026-10-20
---

# Controller-bound dispatch and native cleanup

[Issue #42](https://github.com/techrote/neural-cyber-automata/issues/42) corrects
the export owner boundary on base
`3db02bcb0bbe7703162d8ff597f965af3f1bf50f`. The work changes neither the NumPy
nor original Torch numerical operators. It introduces no training, checkpoint
selection, new host source or long deployment qualification.

## Confirmed defects and corrected contract

A retained accepted decision could previously be dispatched after its controller
cancelled or replaced the invocation. Dispatch checked that the supplied view
matched the decision's original source, but had no producing-controller
authority to consult. Reusing that view could therefore mint a fresh packet
sequence from a retired decision. A separate six-tick goal reproduction passed
`current_tick=6` with the original tick-zero observation: inference still
accepted it and the uncapped twelve-tick helper lease outlasted the goal.

Current dispatch requires `dispatch(env, controller, decision, issued)`. The
producing controller holds exactly one unused pending decision, and dispatch
consumes it before packet construction or transport. Cancellation, local reset,
replacement, source changes and ambiguous failures invalidate retained results.
Copied, foreign-controller and previously used decisions cannot acquire a new
packet sequence. A live helper must be revoked before a new goal, source or
part/mount boundary becomes eligible.

The controller retains a native-cleanup uncertainty latch. Successful local
cancellation only clears local memory; `confirm_owner_cancel(env.cancel())`
requires a complete same-owner native observation with a strictly advanced
control generation and no observed clock regression. An unsuccessful cancel
leaves future inference fenced. A validated new host reset can clear that
uncertainty; a same-run epoch must advance, so a previously unseen older epoch
cannot masquerade as a reset. Independent review identified and closed that
older-epoch case.

Absolute goal deadlines use the actual supplied current clock at inference and
dispatch. A packet keeps its original source identity and a lease capped by the
goal deadline. The final repeat can be shorter than six ticks. Native receipts
must prove admission and complete consumption throughout an unchanged actor,
part and mount boundary, with the entire interval inside both the original
source lease and goal deadline. Delayed observations remain admissible when
that complete interval fits; timestamps are never refreshed to make them fit.
Terminal revocation preserves the host's unavailable final action instead of
inventing an actuation value.

The prospective deployment/screen callers use the new API and explicitly pass
the expected capped lease to screen validation. Its default remains the original
twelve-tick contract, and a regression rejects interpreting a six-tick receipt
as an original twelve-tick receipt. No frozen declaration, old result or
qualification index was rewritten. The [runtime guide](../RUNTIME_EXPORT.md)
documents current use; the [Torch guide](../TORCH_RUNTIME_EXPORT.md) keeps its
historical three-argument example tied to the original qualifying source.

## Source and software checks

The source below was frozen for the final software checks. These hashes identify
this corrected boundary, independently of the historical qualified closure.

| File | SHA-256 |
| --- | --- |
| `src/nca/export/runtime.py` | `02dfcd25707762128886d6750ebac59b27abb5329bd60c8312b2106215290c59` |
| `src/nca/bench/deployment.py` | `cf2c2411563ce3520561a5bd7e87fe47a7d7615dcf8abecf2789cfd77d9f9e4c` |
| `src/nca/bench/torch_screen.py` | `1353b1e4d7e5c48bdd39ddd8512d4f83e6ebddd5d72b38fcdbde4be2df15367f` |
| `tests/test_export_runtime.py` | `75101bd3e7e75096d45b2c40a4c55c25b4eb109880a07281301b772106195aa7` |
| `tests/test_export_host.py` | `8fe55bcd7505dff44bc6938f3c603706c043d94ac3b1bc84a0cd40444bbf992c` |
| `tests/test_export_torch_screen.py` | `99f2843843f1e96fe9f5b6ca84ab82b4eacfecfb0c53ffb532d2de44b2c43d2d` |

Using Python 3.12.14, NumPy 2.3.5 and Torch 2.8.0+cpu:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_export*.py' -q
```

The final invocation completed **86 tests in 21.135 seconds: 76 passed and
10 expected integration checks were skipped**. Four require an independently
evaluated export artifact; six require a configured actual host. The six owner
checks were exercised separately below. The software invocation made no host
request and performed no optimizer work. It includes exact sequential Torch
reference parity, autocast refusal, source checks, RNG preservation and no-learner
import coverage. `encode`, `map_action`, `NumpyGRU` and the Torch operator path
remain unchanged.

Focused fixtures also reject canceled/reset/replaced/copied/foreign decisions,
ambiguous transport with failed cleanup, stale/foreign/incomplete cancel
receipts, backward actor epochs, late current clocks, receipt intervals crossing
the source lease, missing or mixed native traces, topology drift and different
consumed actions. Independent review found and corrected the full-interval
source-lease and post-interval topology cases before source freeze.

An earlier 77-test invocation had one stale assertion expecting dispatch for a
rejected bundle before a controller existed; that caller now directly cancels
the owner, and the test checks that cancellation. A subsequent 85-test software
invocation passed with ten expected skips before the final echo-roundoff
regression was added. These were implementation checks, not policy evaluations.
Strict documentation validation checked 57 files with zero errors or warnings;
`git diff --check` passed.

## Bounded actual-owner checks

The six `ExportOwnerBoundaryTests` use a fixed synthetic NumPy policy fixture
with real ACT004 native receipts. They require only `NCA_HOST_CHECKOUT` and
`NCA_GODOT`, and are separate from the four evaluated-artifact integration
checks. No learned-skill result is inferred from these synthetic weights.

| Environment boundary | Identity |
| --- | --- |
| Host source | `f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497` |
| Host tree | `445a185e9f9824b16953fec27bf0157dfeedca77` |
| Godot | 4.7 stable, `5b4e0cb0f`, Linux x86_64 |
| Godot binary SHA-256 | `f85bbc6b15e22416c7d797cd60b63286dd67b9cb13498847056c18520ae55a75` |
| Native artifact SHA-256 | `5bdc55e7f355004fc8a6468234b0d018d3919339ebfe47faca9997de3af17847` |
| Rapier artifact SHA-256 | `147ab5673b9687ccb7a3b12fec07ea55cc674ad8a75f529d1085bf7d61fbd831` |
| Actor/configuration | Jolt `jev`, original profile and assists, seed 73, 60 Hz stepped host, at most six ticks per dispatch |

Every case below has a passing local invocation after its relevant correction.
Passing coverage came from the original suite and focused reruns; it is not
represented as a single clean final six-case invocation.

| Actual-owner case | Passing observed boundary | Native ticks in its passing execution |
| --- | --- | ---: |
| Absolute deadline and final repeat | Seven-tick lease; six ticks followed by one; next passive tick advances control generation and exposes empty native action | 8 |
| Delayed source | Two passive ticks then six consumed ticks; packet still records original tick zero | 8 |
| Repeat dispatch | Original six ticks succeed; reuse sends no second packet and cancellation advances control generation | 6 |
| Source transfer | Revocation is confirmed before a fresh invocation; recurrence starts at zero | 12 |
| Dispatch at deadline | Six passive ticks; late decision sends no packet and actual owner cancellation advances generation | 6 |
| Terminal readback | Four-tick native horizon; final action remains unavailable and local cleanup is fenced | 4 |

The command for the complete boundary class is:

```bash
PYTHONPATH=src:tests python -m unittest test_export_host.ExportOwnerBoundaryTests -v
```

### Failed attempts and focused corrections

Two initial setup attempts stopped because the copied Godot executable lacked
execute permission; a third stopped at sandbox denial of the local control
socket after the version probe. None started a game host or advanced simulation
ticks. The actual boundary test then ran in an environment permitting that
socket. Host source and binary contents were unchanged.

The first actual six-case suite completed in **5.096 seconds** with one passing
case and five errors, after **36 native ticks**. Full equality of the echoed
packet rejected tiny Godot JSON floating-point roundtrip differences. The fix
permits absolute error at most `1e-12` only in echoed action numbers; packet
source, sequence, lease and all other fields remain exact. The existing
`2e-5` native FP32 consumption tolerance is a separate comparison, and boolean
channels remain exact in both. A synthetic regression admits the tiny echo
roundoff and rejects a substantive action change.

The five failed cases were rerun in **3.586 seconds**, with four passing and
one test error after **38 native ticks**. The deadline case reached tick seven,
then correctly observed expiry at tick eight, but its assertion looked for an
`effort` field in the host's empty expired action. The assertion was corrected
to require the actual empty action object; runtime code did not change.
That single deadline case then passed in **3.396 seconds**, using **8 ticks**.

Across those three actual-host unittest invocations, the executed bounded test
paths total **82 native ticks**. Each class closed its host through teardown;
no host test remained running. These are observed unittest paths and results,
not an independently metered deployment journal. Console output was reviewed
during implementation; complete request/response journals and a durable raw-log
archive were not captured. The early errors remain part of this record.

## Qualification and remaining limits

The historical Torch bundle with SHA-256
`1345c35582bbd16f7f6474996976dc3b0fac3dd775a2f83f916347b43e5fb96c` retains its
[original composite qualification](2026-10-20-t109-torch-cpu.md), tied to the
original source closure and receipts. Current runtime bytes deliberately do
not match that closure. This fix does not relabel its embedded qualification
flag, replace its source hashes or carry qualification to a newly converted
bundle. A changed bundle requires a new conversion and a separately declared
physical qualification on its actual runtime source.

The native tests establish bounded owner-boundary behavior with synthetic
weights. They do not measure learned goal success, a new paired trajectory,
sustained deployment cost, continuous realtime deadline reliability, multiple
learned actors, in-engine execution or owner feel. Their wall times are test
runner observations, not deployment latency or throughput measurements. The
existing learned reliability gate and morphology block remain unchanged.
