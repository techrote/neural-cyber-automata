---
title: Authored usable skills and local task composition
kind: reference
summary: Selected real mount capabilities, typed skill lifecycle, bounded recovery and explicit local-success evidence.
owns: [skills.usables, skills.lifecycle, skills.composition]
verified: 2026-10-19
---

# Authored usable skills

The usable runtime issues bounded ACT004 wire requests to the persistent
[actual host adapter](HOST_ADAPTER.md). Cybersand owns the body/joint graph,
attachment clearance, resource spending, recoil and native material effects.
The runtime never edits those mechanisms or changes the authored profile.

The selected capabilities come from accepted Cybersand
[`f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497`](https://github.com/techrote/cybersand/commit/f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497).
`UsableCapability.from_profile` requires the exact digest of the copied host
catalogue definition. A familiar model name with modified joints, mount offsets
or assistance values is refused.

## Selected examples

| Host model | Equipment | Available skills |
|---|---|---|
| `astra` / Vector | Slot-zero recoiling launcher on the chassis, with two articulated thrust pods | `approach`, `aim_use`, `recover` |
| `hammerhead` | Slot-zero cutter on the held hammer body, connected through the articulated arm | `approach`, `grip`, `aim_use`, `carry`, `recover`, `release` |
| `pouncer` | Existing charge-jump control, used solely to qualify disarming | `charge_hold` |

These are two usable examples and one cancellation example. No unpublished
32-item usable catalogue, arbitrary pickup object or independent equipment
replacement API is integrated. The package supplies the versioned
`authored-usables` skill extension and its `register_skills` registration function.

Vector's approach reuses the T103 authored feedback controller. Hammerhead
requests horizontal movement through its existing gait and bounded trim through
its shared attitude channels. Recovery uses Vector's brake and Hammerhead's
declared self-right input. Vector's auxiliary mode toggle is never interpreted
as a self-right button. Those controller choices are identical for human,
script and model-intent labels.

## Initiation, progress and outcomes

`SkillRuntime(profile).start(skill, parameters, view, ...)` returns a typed
`SkillInitiation`. Acceptance means the selected plan passed local admission;
it neither advances the host nor proves that a trigger or attachment succeeded.

`progress(view, observations=...)` consumes fresh permitted feedback and returns
a `SkillProgress` containing a strict `WireAction`, state/reason, local success
estimate and copied effect receipts. `evaluator_success` is always null. A
privileged task evaluator must publish its outcome separately.

| Skill | Parameters and local completion predicate |
|---|---|
| `approach` | Arena point `target`, tolerance, effort and bounded timers; local position reaches tolerance |
| `grip` | Existing own held-part reference and bounded timers; detached flag becomes false after the owner attachment request |
| `aim_use` | Nonzero bounded direction `aim`, effort, settling interval, timers and `require_change`; a new owner effect receipt is consumed without refusal |
| `carry` | Point/tolerance/effort/timers; reaches the local point while the selected usable remains attached |
| `recover` | Angle/angular-speed tolerances, settling interval, effort/timers; the observed attitude remains within the requested thresholds |
| `release` | Timers; the selected held body becomes detached |
| `charge_hold` | Target charge, effort/timers; holds existing primary input until explicitly cancelled or timed out, without a release-to-launch transition |

An effect can be admitted yet refused by the native material owner. Launcher
recoil is also possible on such an attempt. `aim_use` therefore waits for actual
local effect feedback and reports reasons such as `effect_blocked`,
`effect_source_identity`, `effect_no_change` or `no_effect_feedback`.
With `require_change: false`, `local_effect_consumed_no_change` means the owner
consumed the usable effect without changing cells. It is not a claim of target
damage or successful excavation. With `require_change: true`, zero changed cells
fails the skill's explicit local predicate.

The six-tick repeat can contain an effect earlier than the returned observation.
Pass `interval_observations(info)` to retain the already-permitted local views
from every recorded base tick. The helper refuses a known missing interval;
the runtime will not invent a lost effect receipt from a cooldown or a narrative.
The effect receipt retains its original event tick. A completed effect at or
before the skill deadline can be recognized when its interval is reported at the
deadline; a late effect remains recorded without on-time success.

## Source and equipment lifetime

Every request carries its invocation ID, source label/generation, original source
observation ID/tick, actor lifetime and mount reference. A plan is too old to
start when its source is 120 or more base ticks old. Admitting a delayed plan
does not restamp its source observation. Fresh local control then creates fresh
wire packets through the ordinary six-tick/lease contract.

The runtime fences run, actor slot, epoch, model/profile and control generation
on every progress call. Equipment references additionally contain the exact
profile digest, mount slot/name, held-part name and observed attachment generation.
Detaching/regripping changes that connection generation while preserving the same
physical body. Control-only cancellation preserves the connection generation and
advances the separate control-generation fence. Full reset changes the actor/object
lifetime and resets the observed connection count. A stale reference
cannot silently select the current object or another part.

The accepted host does not expose arbitrary independent mount replacement.
Derived attachment generations describe observable connection changes, not an
invented host replacement ID. Real reset-based replacement and stale-reference
refusal are qualified; arbitrary replacement remains unavailable.

One actor resource is owned by one active skill. A competing start returns
`resource_busy`; it cannot silently seize control. A repeated invocation ID with
the same intent returns its original initiation receipt and does not start again.
The same ID with a different payload is refused. Terminal results are retained
for that invocation; receiving the same result twice supplies no additional reward.
The bounded runtime retains at most 512 initiation receipts and refuses further
new invocations at capacity. Create a new runtime for a new long-running session.

## Cancellation and failure

`cancel(reason)` revokes the runtime's future requests. The owner must also call
`env.cancel()` at the next loop boundary, including when `cancel_host` is true on
a terminal result. Handle that terminal flag once per request ID; it is a retained
receipt rather than a request to repeatedly advance the control generation.

Actual owner cancellation clears queued input, armed charge and issued views.
It preserves body velocity, the current attachment and already-applied material
changes. `charge_hold` must use this path: replacing primary with a neutral/release
step would invoke different gameplay semantics. Interrupted aim creates no delayed
trigger. Cancellation after firing cannot erase native Fire/material evolution
or recoil that already happened.

No-progress and duration limits end a skill with a specific failure/timeout and
request owner cancellation. A released object outside the eight-cell owner pickup
range can remain unreachable; the runtime neither teleports it nor asserts grip.
The task runner attempts cleanup after transport, recording or unexpected errors,
preserves the original exception and never issues a neutral/release step to clean
up charge. If the connection itself is unavailable, real cancellation is a
best-effort attempt; the adapter's existing transport/lifetime fences remain in force.

## Declared compound tasks

`TaskPlan` contains one to twelve steps, at most two explicit recovery skills per
step, and a bounded total decision count. A recovery branch is part of the task's
declared meaning; failed primary attempts remain recorded when the allowed branch
succeeds. The runner performs no task search, model call or implicit trigger retry.

For example, with a reset Vector at the configured local point:

```python
from nca.skills import SkillRuntime
from nca.skills.tasks import SkillSpec, TaskPlan, TaskStep, run_task

runtime = SkillRuntime(host.request("profile")["profile"])
plan = TaskPlan("launcher-task", (
    TaskStep(SkillSpec("approach", {"target": [304, 300], "tolerance": 2.5})),
    TaskStep(
        SkillSpec("aim_use", {"aim": [1, 0]}),
        (SkillSpec("recover"),
         SkillSpec("aim_use", {"aim": [0, -1], "require_change": True})),
    ),
))
result, observation, trace = run_task(env, runtime, plan, observation, source="script")
```

For an asynchronous source, use the same `start` and `progress` methods from the
continuously advancing owner loop. Preserve the proposal's `source_observation_id`,
`source_tick` and source generation; transfer/revoke must cancel both pending host
input and the active skill. A source label is not a grant of priority or additional
perception. No live model service is called by this implementation.

## Reproduction and evidence

After [host setup](HOST_ADAPTER.md):

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_skills*.py' -v
python tools/skill_pilot.py --seed 73 --max-seconds 120 --output usable-pilot.json
```

The immutable pilot records actual launcher recovery, articulated pickup/carry/use,
an expected failed late pickup, native changes surviving cancellation and charge
disarming. It records exact host/runtime/profile/source hashes, resolved scenarios,
typed outcomes and separate requested/admitted/applied/passive evidence. The
[T107 evidence record](evidence/2026-10-19-t107.md) supplies measured results and
failed attempts. These fixtures establish no learned replacement, exhaustive
usable integration, Windows/desktop parity or owner feel acceptance.
