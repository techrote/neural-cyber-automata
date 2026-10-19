---
title: Authored actor controllers and response fixtures
kind: reference
summary: Jolt and Vector feedback skills, shared request shaping, repeatable matched fixtures and explicit physical limits.
owns: [controllers.authored, controllers.fixtures]
verified: 2026-10-19
---

# Authored actor controllers

`AuthoredController` provides Jolt (`jev`) `directed_hop`, Vector (`astra`)
`burst_traverse`, and their declared `recover_level` mapping. The balanced Jolt
controller is the initial learning reference. Firm and compliant variants remain
available in [the resolved configs](../configs/baselines/); no variant changes
the game's authored gyro, joints, grip, resource costs or physics owner.

The controller reads only copied local feedback, keeps a goal invocation and
returns a bounded `WireAction`. It does not have access to evaluator snapshots.
Unknown actor/profile mappings require an explicit implementation. An arrival
result ends an invocation; it does not promise sustained holding or hovering.

## Use an authored skill

After [host setup](HOST_ADAPTER.md), with a reset Jolt `CybersandEnv`:

```python
from nca.controllers import AuthoredController, FeedbackObservation, SkillGoal

view = env.observe()
controller = AuthoredController("jev")
goal = SkillGoal("hop-1", "directed_hop",
                 target=(view["position"][0] + 48, view["position"][1]))
while True:
    frame = controller.step(FeedbackObservation.from_host(view), goal)
    if frame.status in {"succeeded", "failed", "canceled"}:
        env.cancel()
        break
    view, reward, terminated, truncated, info = env.step(frame.action.to_dict())
    if terminated or truncated:
        break
```

On takeover, call `controller.cancel()` and `env.cancel()` together. The first
clears controller intent; the second revokes pending owner input and issued
views. Neither operation removes existing motion or material effects. A canceled
goal cannot silently restart; create a fresh invocation ID. On episode reset,
reset controller memory as well. The registered policy ID is
`authored-feedback@1`, constructed with `actor="jev"` or `actor="astra"`.

## Repeat the matched screen

Prerequisites: the CPU package and pinned host from the host guide. Matched GIF
generation additionally uses the optional Pillow dependency. In a configured
Python environment, these commands work in Bash or PowerShell:

```text
python -m pip install --no-build-isolation ".[evidence]"
python tools/control_baseline.py --output artifacts/baseline-screen-1
```

Set `NCA_HOST_CHECKOUT` and `NCA_GODOT` using the host guide. Add `--skip-clips`
to run without Pillow. The output directory must be new. The fixed screen has
18 episodes: two actors, three tunings and step/pulse/reversal fixtures, seed
1703, 1,080 decisions and 6,480 base ticks, capped at 600 seconds. It does not
train a policy. Failures retain completed traces and an explicit failure record.

The output includes compressed per-base-tick receipts, a summary with source,
host, profile, tuning and artifact hashes, and six matched pose-replay GIFs.
Each GIF shows recorded body poses and observed local cells with the same camera
across tunings. These are pose replays; actual shaded game footage and owner
feel review remain separate evidence.

## Fixtures and measurement

The [scenario definitions](../configs/fixtures/) also declare neutral disturbance,
support loss/return, soft stops, recoil/payload changes and recovery. Request
shaping tests cover hysteresis, deadbands, damping, time-scaled slew, bounded play,
cancellation, mapped soft stops and recovery feedback. Response metrics separate
onset, arrival, overshoot, useful progress, settling and corrective chatter.
Intentional launch, flight, recoil and recovery motion is excluded from the
corrective-trim chatter count.

The actual public arena has no arbitrary force/torque injection, support-removal,
payload-replacement or joint-profile override call. Those requested fixture
operations explicitly refuse. The software fixtures and ordinary launch/contact
observations establish no controlled physical perturbation result for these
missing facilities. Native usable recoil is a separate actual capability used
through its declared mounted-tool interface.

The [evidence report](evidence/2026-10-19-t103.md) preserves tuning alternatives,
the compliant Jolt reversal failure, Vector's drift after arrival, original and
corrected metric results, and the distinction between measured behavior and
owner feel. [Run records](RUNS.md) identify the executed screens.
