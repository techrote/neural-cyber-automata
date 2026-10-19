---
title: Human takeover and asynchronous skill intent
kind: reference
summary: One bounded provider call, source-anchored plans and continuously advancing local skill control.
owns: [control.async]
verified: 2026-10-19
---

# Asynchronous skill control

`SkillControlSession` joins a typed decision provider, the
[authored usable runtime](USABLE_SKILLS.md) and the accepted
[game adapter](HOST_ADAPTER.md). It is registered as `asynchronous-skills@1`.
All session/router methods run on one serialized owner thread. A worker may
produce a proposal through the provider queue; it never receives the environment
or calls host/router methods directly.

The caller keeps invoking `step()` while a provider is pending. Realtime Godot
continues its native/Rapier cycle independently. An active local skill continues
feedback control; with no active skill, fallback issues fresh neutral requests.
Cancellation uses the owner's explicit cancellation operation to disarm prior
charge and pending input. It does not undo motion or already applied effects.

## Sources and ownership

| Entry | Meaning |
|---|---|
| `transfer("model")` | Explicitly grant a provider permission to propose the selected profile's declared skills |
| `submit_intent("human", skill, parameters)` | Apply a typed human shortcut at this owner boundary, revoking previous source authority first |
| `submit_intent("script", skill, parameters)` | Use the identical skill/capability boundary for an explicit scripted mission |
| `revoke(reason)` | Retire logical authority and the backend identity before attempting real owner cancellation |
| `reset(seed=...)` | Revoke, reset the actual actor, clear local recurrence and admit no automatic new source |

The human entry is an integration API. The conformance tests inject that priority
programmatically; they do not claim a live human input session or UI.

Actor run, slot, epoch, model/profile, control generation and observed attachment
generation are fenced independently where the owner can change them. Expected
grip/release feedback publishes its local result before outstanding old-equipment
plans are retired. A successful or otherwise terminal skill revokes its grant
once; a later invocation needs an explicit new grant.

If cancellation raises, the old grant is already gone. The session sends no
neutral/release or new command after unconfirmed cancellation; a successful
actual reset is required before it can resume. The current bounded host lease
still governs any previously admitted input.

## Source age and plan semantics

The provider receives a copied whitelist of own actor state, own articulated
parts/joints, rig rays and the bounded native cell patch. It receives no reward,
global evaluator state, arbitrary code execution or physical mutation handle.
Replies must retain their original request ID, observation ID/tick and source
grant generation. Typed parameters reject unknown fields and invalid values.

Model smart plans expire exclusively at original observation tick plus 120.
That bound applies both while awaiting a reply and while an admitted model skill
runs. Every subsequent motor lease is capped to the same original expiry; fresh
feedback cannot renew the plan. The ordinary helper lease remains twelve ticks
and the host's legal one-to-eighteen range is unchanged. A zero-delay provider
gets one immediate nonblocking poll at its original owner boundary.

An explicitly submitted human/script mission uses the typed skill parameters
and their declared finite timeout (at most 600 ticks). This separate mission
entry is the supported longer-lived semantic. A model reply cannot silently
turn its old smart plan into a fresh mission. ACT003 wire/metal packet meanings
and its source-anchored expiry contract remain unchanged.

## Bounded providers

`DelayedMockProvider` supplies deterministic delay, jitter, outage and cancellation
acknowledgement controls. `CallableWorkerProvider` wraps one local callable on a
daemon thread with a one-result queue. It is an adapter primitive, not a network
service integration or evidence of a live LLM.

At most one active or retired call occupies the backend slot. Timeout/reset may
retire its identity immediately, but only completion or an exact boolean `True`
cancellation acknowledgement frees the slot. An unresponsive backend therefore
cannot accumulate abandoned threads or replacement calls. Duplicate, reordered,
malformed, expired and unsolicited results are refused. A wall timeout also
handles a paused simulation clock. New requests are at least thirty ticks apart.

## Example

With a reset ACT004 environment and its exact profile:

```python
from nca.adapters.decisions import DelayedMockProvider, mock_proposal
from nca.adapters.skill_control import SkillControlSession

provider = DelayedMockProvider(
    lambda request: mock_proposal(request, "aim_use", aim=[0, -1]),
    delay_ticks=30,
)
session = SkillControlSession(env, profile, view, provider)
session.transfer("model")
for _ in range(30):
    result = session.step()
    if result.terminated or result.truncated:
        break
session.revoke("example_complete")
```

`OwnerStep` separates decision acceptance, skill initiation, inferred local
progress and the actual owner observation/receipts. None is an evaluator verdict.
The new `observe_with_info()` surface copies current idle receipts without
draining them; the session deduplicates completed ticks and filters old control
lifetimes from new skill feedback while preserving the complete audit receipts.
Lost feedback causes explicit cancellation rather than inferred success.

If the host terminates during a repeat or client pause, an active skill whose
final receipts have not reached local adjudication reports
`host_terminal_censored` with `local_success=None`. The actual final effect and
motion receipts remain in `OwnerStep.info`; the session does not convert them
into an inferred success or failure. A previously established local terminal
result is retained. Repeated terminal calls return the same result and advance
zero additional ticks.
A new source grant is refused on a completed host until an explicit reset.

## Verification and limits

Run the focused router tests and actual realtime tests, or capture an immutable
full conformance pilot:

```sh
python -m unittest discover -s tests -p "test_decisions*.py" -v
python tools/async_pilot.py --checkout HOST_CHECKOUT --godot GODOT_EXECUTABLE --output NEW_PILOT.json.gz
```

The pilot has a 120-second/2,000-step hard ceiling and records the executed source,
runtime, every owner operation and failed cases. Complete compressed receipts
stay outside Git; their compact index belongs with the
[T-108 evidence](evidence/2026-10-19-t108.md). Missing host runtimes are explicit
skips. Zero paid calls, zero spend and unmeasured token usage remain explicit.
Full reset and existing-part attachment changes are exercised; arbitrary
independent equipment replacement, other physical platforms, desktop feel and
live services are separately unqualified.

© 2026 techrote. All rights reserved.
