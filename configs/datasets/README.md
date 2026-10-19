---
title: Demonstration datasets and consumed-input replay
kind: guide
summary: Capture API, policy feature boundary, original observation age, dataset splits and external artifact verification.
owns: [datasets.capture, datasets.replay]
verified: 2026-10-19
---

# Demonstration capture and replay

The [T-104 manifest](t104-capture-replay.json) indexes three bounded, actual
ACT-004 captures and their independent replay reports. Complete trajectories
remain in an external `t104-artifacts` collection under C-005. Relative artifact
paths, compressed byte hashes, canonical episode hashes, profiles, source
identities and splits are recorded in the manifest. The [run receipts](runs/)
retain the completed capture pipelines and the first failed development attempt.
The [evidence record](../../docs/evidence/2026-10-19-t104.md) states the measured
scope and the separate skill outcomes.

The [final acceptance manifest](t104-final-acceptance.json) repeats all three
pipelines on one isolated committed candidate and indexes the `t104-final`
artifact collection. Both candidate and final evidence remain identifiable.

All published scenario families are in the **evaluation** partition.
Training and tuning partitions are empty. These artifacts exercise capture,
replay and data preparation; they do not constitute a trained policy or a broad
skill evaluation. The only trajectory stored in Git is the deliberately
[synthetic schema fixture](../../tests/fixtures/data/synthetic-episode.json).
Its fabricated values and `test_fixture` marker supply no physical evidence.

## Record boundaries

`nca.data.EpisodeRecorder` wraps the actual `CybersandEnv`. An episode stores the
resolved host configuration, authored profile, runtime and build hashes, source
file hashes, seed and scenario family.

An installed package locates its matching source checkout from `source_root`,
`NCA_SOURCE_ROOT`, or the current Git checkout. It verifies the complete installed
Python/Godot source file set and bytes against that checkout before attributing
the revision. An absent checkout or mismatched installation fails explicitly.
A source-tree import continues to use its own checkout by default.

Each step stores the following separately:

| Record | Meaning |
|---|---|
| `issued_observation` | Copied owner view at dispatch, before the world advances |
| `source_observation` | The original issued view named by the submitted packet, even when older than dispatch |
| `policy_observation` | Whitelisted own-state fields from the source view; unavailable values stay null |
| `sample` | Optional original policy request or latent residual, with policy hash and optional distribution likelihood |
| `requested_packet` | Original envelope, sequence, observation ID, age, lease and action |
| Per-tick `requested` and `admission` | Attempts seen by the owner and their individual accepted/refused reasons |
| `consumed_request` and `source_id` | Which admitted request remained active for this base tick, including a held request |
| `consumed_action` and `applied` | Owner action state and aggregate actual impulse/energy receipt |
| `passive` and `actual_motion` | Explicit passive-wrench availability and independently observed motion |
| Next observation and terminal fields | Final local view, terminated/truncated flags and reason |

The physical owner does not expose a separately decomposed passive/contact
wrench. `passive.available` therefore remains false with an explanation;
unavailable is not a numerical zero. Movement continuing after lease expiry or
cancellation is retained in actual motion without inventing a new action source.

Six-tick action repeats retain all six base receipts. A delayed request does not
receive a new source observation or lease. Unknown source views, missing receipts,
mixed consumption, refused requests and replaced actions cannot silently become
ordinary on-policy samples. Recorder errors close further stepping and retain a
failure event without claiming a fresh final state.

## Learning views and interventions

`policy_view` defines `act004-local-proprioception-rays/v1`. It admits declared
own position, velocity, attitude, contact, local rays and available actuator state.
It excludes evaluator/reward/success dictionaries, other actors, full parts/joints
and the local cell patch. Full source observations remain available to audit the
record, but `Dataset.imitation_samples()` returns only the declared policy view,
explicit caller-supplied goal context, requested action and provenance labels.
Any broader policy feature view requires a separately declared schema.

`ActionSource` identifies scripted, human, learned, model or neutral control and
records any privileged teacher or critic status. These captures are scripted;
the source-transfer fixture is not relabelled as a human demonstration.
Applications can label a supplied correction with `human_correction` or a model
replacement with `model_override`. Corrected actions remain separate imitation
examples when their consumption is valid. `replaced=True` retains the original
sample and actual replacement for audit but excludes the interval from the
ordinary imitation iterator.

`CaptureEvent.require_on_policy()` accepts only a learned source with a recorded
likelihood for the unchanged sampled request/latent, no intervention, known
source view and complete matching consumption. The likelihood is never copied
onto an applied impulse or a clipped/combined action. A rejected duplicate cannot
inherit the previous held command's consumption and pretend to be a fresh PPO
sample. These checks label admissible data; T-104 performs no optimizer update.

Cancellation and takeover call the actual owner and mark a recurrence boundary.
Cancellation clears control targets and pending edges while retaining body
velocity, current part poses and loaded joint angle/error/speed. Full reset
starts a new recorder and fresh episode. Observed actor, profile, morphology,
mount generation or part topology changes mark a boundary. Attach/release
requests conservatively mark a mount-dependency boundary where the current owner
does not expose every mount identity. Hidden physical topology mutation is not
claimed. Valid terminal suspension is distinguished from invalid final sensors.

## Capture and replay

Set up the pinned host and native artifacts using the
[host integration guide](../../docs/HOST_ADAPTER.md). From the repository root,
with `NCA_HOST_CHECKOUT` and `NCA_GODOT` set to the chosen local installations:

```sh
PYTHONPATH=src python tools/capture_replay.py --suite control-boundaries --ticks 32 --seed 7 --output artifacts/boundaries.json.gz
PYTHONPATH=src python tools/capture_replay.py --suite jolt-demo --ticks 132 --seed 1703 --target-offset 48 --pace .8 --output artifacts/jolt.json.gz
```

In PowerShell, after the same host setup:

```powershell
$env:PYTHONPATH = "src"
python tools/capture_replay.py --suite control-boundaries --ticks 32 --seed 7 --output artifacts/boundaries.json.gz
python tools/capture_replay.py --suite jolt-demo --ticks 132 --seed 1703 --target-offset 48 --pace .8 --output artifacts/jolt.json.gz
```

The tool refuses an existing destination, bounds each run to 100 decisions and
120 seconds, writes an immutable compressed episode, replays it on the same
host, and writes a separate report and run receipt. Its exit status describes
the capture/replay pipeline. Skill quality remains in `episode.evaluation`.

To replay an existing artifact, first verify its compressed byte hash against
the manifest, then use the Python API:

```python
import os
from nca.adapters.godot import GodotHost
from nca.data import read_episode, replay_episode

episode = read_episode("artifacts/boundaries.json.gz")
with GodotHost(os.environ["NCA_HOST_CHECKOUT"], os.environ["NCA_GODOT"]) as host:
    report = replay_episode(episode, host)
print(report.to_json())
```

The loader checks a bounded compressed payload, strict record fields and the
canonical episode digest. The writer publishes a unique file atomically and
refuses replacement. Replay checks the source/runtime/build, profile, schedule
and stepped mode before interpreting the evidence. Only accepted original
requests are reissued, at their original input ticks. Holding input is reproduced
without resubmission; denied attempts are counted separately. Run, epoch and
observation IDs map to newly issued peers while sequence, observed tick, lease
and action remain unchanged.

Replay compares declared local body/part trajectories, local cell patches,
consumed actions and aggregate applied receipts within the stored tolerance.
The report separately records every input identity mapping. It does not promise
whole-world reconstruction, equal wall time, real-time causal replay or exact
cross-platform trajectories. A different binary/build correctly refuses these
fixed-build artifacts; a fresh capture on that platform can establish its own
compatible replay evidence.

© 2026 techrote. All rights reserved.
