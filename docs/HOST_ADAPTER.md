---
title: Persistent Cybersand host adapter
kind: reference
summary: Pinned actual-host setup, copied reset/step semantics, causal traces and measured limits.
owns: [adapter.host, adapter.protocol, adapter.setup]
verified: 2026-10-19
---

# Persistent Cybersand host adapter

The Python adapter starts one headless Godot process and retains its arena across
reset/step requests. Its external GDScript launcher belongs to this repository.
It calls Cybersand's accepted public arena API; it does not modify Cybersand
source or reproduce its physics in Python.

## Environment identities

| Lane | Implementation and evidence |
|---|---|
| `recorded/playback/v1` | Exact fixed action/transition playback through `RecordedEnv`; no new physics or control result |
| `cybersand-sampled/stepped/v1` | Actual ACT003 sampled motor/native-material host, with original wire or four-channel metal packets |
| `cybersand-act004/stepped/v1` | Actual ACT004 articulated Rapier/native-material host; learner decides when to advance |
| `cybersand-act004/realtime/v1` | Same synchronous causal cycle advances at a wall-clock 60 Hz target while the client thinks |

Realtime mode qualifies delay against the synchronous physical arena. It does
not reproduce the independently paced desktop worker schedule.

## Accepted host and runtime

The audited host is Cybersand
[`f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497`](https://github.com/techrote/cybersand/commit/f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497),
which merged [ACT004 PR #181](https://github.com/techrote/cybersand/pull/181).
The adapter rejects a different host revision/tree, changed tracked host inputs,
native source provenance mismatch, wrong native/Rapier bytes or different
Godot version. Linux also verifies the exact Godot executable digest.

The [public physical arena](https://github.com/techrote/cybersand/blob/f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497/godot/scripts/physical_actor_arena.gd)
already exposes configure, reset, observation, step, cancellation and close.
The stock headless runner runs a whole prerecorded match and exits. The external
launcher adds persistent transport around the accepted owner.

The native library is the source-matched published ACT004 artifact; the
Godot and Rapier releases are pinned by Cybersand's dependency records. This
repository copies no third-party engine implementation or binary into Git.
Downloaded tools retain their upstream notices in the host checkout/archive.

## Linux setup

Prerequisites: Python 3.11 or later, Git, Git LFS, and Linux x86_64 capable of
running the pinned Godot 4.7 and native extension. Network access is required
for the first setup; no model service or GPU is involved.

Run from this repository:

```bash
python tools/host_setup.py --destination ../nca-host-runtime
export NCA_HOST_CHECKOUT="$(cd ../nca-host-runtime/cybersand && pwd)"
export NCA_GODOT="$(cd ../nca-host-runtime/godot && pwd)/Godot_v4.7-stable_linux.x86_64"
PYTHONPATH=src python -m unittest discover -s tests -p 'test_host*.py' -v
```

The installer fetches the exact accepted commit, materializes only the relevant
native and Rapier libraries, verifies the Godot archive size/hash, imports the
host resources, and rechecks source identity. Runtime directories and imports
are outside this repository. It refuses an unrelated existing checkout.

In GitHub Actions, use
`python tools/host_setup.py --destination "$RUNNER_TEMP/nca-host" --github-env "$GITHUB_ENV"`.
The next step receives the two required environment variables. A skipped
integration class is explicitly an unavailable runtime, never an ACT004 pass.

## Windows preparation

Prerequisites: Python 3.11 or later, Git with Git LFS, and the official Godot 4.7
Windows x86_64 executable. Windows is a prepared, separately unmeasured lane.
The installer requires the explicit executable instead of inventing an unpinned
Windows download identity. In PowerShell, replace the executable placeholder:

```powershell
python tools/host_setup.py --destination ../nca-host-runtime --godot '<Godot 4.7 executable>'
$hostConfig = Get-Content ../nca-host-runtime/host-environment.json | ConvertFrom-Json
$env:NCA_HOST_CHECKOUT = $hostConfig.NCA_HOST_CHECKOUT
$env:NCA_GODOT = $hostConfig.NCA_GODOT
$env:PYTHONPATH = 'src'
python -m unittest discover -s tests -p 'test_host*.py' -v
```

The setup manifest is local operational data and is not a public run record.
Source/runtime identities in public evidence contain hashes and relative source
names, not machine paths.

## Using an environment

```python
import os
from nca.envs import CybersandEnv, GodotHost, physical_config

with GodotHost(os.environ["NCA_HOST_CHECKOUT"], os.environ["NCA_GODOT"]) as host:
    env = CybersandEnv(host, physical_config("jev", ticks=600, seed=73))
    observation, info = env.reset()
    while True:
        action = {"move": [1.0, 0.0], "aim": [0.5, -0.8], "effort": 0.7}
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
```

This is a raw wire example, not a tuned Jolt policy. The local observation
includes pose, velocity, parts, joints, authored resources, contact/ray sensors
and a bounded material/body-ID patch. Evaluator snapshots containing every
actor, terrain totals and native feedback are not policy observations.

`host.request("profile")["profile"]` returns the current actor's copied,
authored catalogue definition. It does not alter joint or assist settings.

## Step and lifetime semantics

Reset returns owned `(observation, info)`. Step returns owned
`(observation, reward, terminated, truncated, info)`. The default decision repeat
is **six** completed base ticks, while the helper requests a **twelve**-tick
lease. Legal leases are **one through eighteen** ticks. Admission is valid only
while `current_tick < observed_tick + lease_ticks`; network receipt does not
renew the source observation.

The physical causal schedule remains
`synchronous-mask-actuation-effects-material-feedback-terrain-rapier/v1`:
current-pose mask, input/actuation, copied effects, native material tick,
cellular response, terrain publication, Rapier step and completed observation.
Traces record tick identities, native tick and monotonic receipt/completion
timestamps. Physics and materials never run through separate adapter loops.

A repeat stops at the first terminal/truncated tick. A thirteen-tick episode
therefore returns durations six, six and one. Calling step again raises
`EpisodeFinished`; there is no automatic reset or duplicate terminal reward.
Host failure raises `HostIntegrityError` and supplies no stale bootstrap state.
If a realtime episode ends while inference runs, the next step returns its valid
final observation with zero additional repeat ticks and the elapsed background
interval. The late command is recorded as unconsumed, and subsequent steps are
fenced by the same terminal rule.

ACT003's cumulative virtual tag scores are differenced between successive
completed state ticks. ACT004 defines no task score; its explicit
`no_task_reward` result is `0.0`. Learning needs a separately declared task
reward and success evaluator. Zero here does not mean a failed or successful
skill.

ACT003 terminal suspension increments its input epoch while retaining the same
run. The adapter permits that documented one-generation terminal revocation
only; other unexpected episode/profile changes fail. Full physical reset
reconstructs the native world, body/joint graph and actor epoch. Cancellation
revokes issued views and armed charge while preserving existing body motion.

## Delays, receipts and capability limits

Transport is bounded newline JSON over an authenticated parent-owned loopback
connection. Only the configured policy actor can be addressed; one request may
be in flight. Stale commands remain stale, malformed schema is refused by the
host, and correlation/transport errors invalidate the client connection.
The simulation thread uses partial, nonblocking writes with a one-packet output
bound and a two-second stalled-output deadline. An unresponsive reader cannot
accumulate an unbounded response queue; the bridge closes that failed session.

In realtime mode, the next request can refer to an observation older than the
current simulation tick. `actual_ticks` counts the requested repeat;
`elapsed_ticks` includes time advanced since the preceding transition, and
`delay_ticks_before_repeat` identifies the gap. Up to thirty-six unattended
base-tick traces are retained. Longer client delays report the exact dropped
trace count and `consumed_trace_complete: false`; such incomplete traces must
not be presented as complete demonstration data.

Receipts keep the original requested packet, owner admission, actual actuation
snapshot and passive/motion availability separate. ACT003 metal exposes its
separate passive force/torque. ACT004 exposes aggregate actor impulse/torque and
actual part motion, but no separate passive wrench decomposition. That record
is explicitly unavailable. Terminal suspension clears current wire action
state, so terminal action availability is also explicit; retained impulse/motion
snapshots are not relabelled as complete per-part force receipts.

The accepted API provides reset-based body replacement, initial position and
layout seed. It has no arbitrary single-body replacement, injected torque,
support removal, or assist/profile mutation operation. No such facility is
manufactured here. Native usables remain available through the accepted wire
buttons on profiles that actually declare them.

## Evidence and pilot

The [T102 evidence record](evidence/2026-10-19-t102.md) separates protocol tests,
actual sampled reward tests, physical lifetime tests, realtime delay tests and
bounded CPU throughput. Reproduce a pilot with:

```bash
python tools/host_pilot.py --model ripper --seed 73 --decisions 200 \
  --max-seconds 120 --output pilot-ripper.json
```

The pilot refuses to overwrite evidence, records adapter and host hashes, and
has a hard configuration ceiling of 2,000 decisions and 600 seconds. It measures
this process, CPU, schedule, copied observations and transport. It trains no
weights and establishes no owner feel or desktop/Windows performance claim.
