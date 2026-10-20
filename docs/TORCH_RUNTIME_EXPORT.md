---
title: Qualified original-operator Torch CPU deployment
kind: reference
summary: Exact evaluated Torch CPU bundle, composite physical and pose-evidence qualification, original no-go records and measured cost limits.
owns: [export.torch_runtime]
verified: 2026-10-20
---

# Original-operator Torch CPU deployment

The `torch-operators-cpu/v1` backend has a **composite qualification for the
exact evaluated BC1103 bundle** in an external Linux Python CPU loop. It runs
the original batch-one Torch Linear/GRUCell operators using the unchanged
evaluated numeric weights. The [measured record](evidence/2026-10-20-t109-torch-cpu.md),
[physical artifact index](evidence/t109-torch-cpu-qualification-20261020.json) and
[completion index](evidence/t109-torch-cpu-clips-completion-20261020.json) bind
the physical run, separate pose-evidence completion, costs and source identities.

The physical invocation at `f3bf96fa402fc3131ce871b5bd23e65855722ade` passed
the arithmetic, trajectory, ownership, active-duration and standalone gates,
then failed while rendering the refusal clips because their saved envelopes
omitted the live part profile. Its immutable result remains **`no_go`,
`runtime_qualified: false`**. A separately declared zero-host completion at
`78cc377b5af9e62efeb73d93cc1e16bd36a013a4` recovered only those profiles from
the captured owner responses and completed all four original pose comparisons.
That separate result records **`composite_runtime_qualified: true`**.

The [original NumPy deployment](RUNTIME_EXPORT.md) remains unqualified with
its failed trajectory results unchanged. The Torch artifact retains its
conversion-time `runtime_qualified: false` marker; the external receipts own
qualification. Keep those receipts with the verified bundle instead of editing
its embedded flag.

## Supported bundle and runtime

| Boundary | Qualified scope |
| --- | --- |
| Artifact | `bc-seed1103-torch-cpu.json`, `nca-torch-cpu-policy/v1`, 227,186 bytes |
| Exact file SHA-256 | `1345c35582bbd16f7f6474996976dc3b0fac3dd775a2f83f916347b43e5fb96c` |
| Numeric candidate | Fixed evaluated BC seed 1103; policy SHA-256 `8196e77d5ecc6ea0ef3dff597b89e5f80e1b32804cc17eb7f305900c2c4d9a1b` |
| Runtime | Linux x86_64, Python 3.12.14, NumPy 2.3.5, Torch 2.8.0+cpu, one Torch thread, batch one, CPU FP32 |
| Arithmetic settings | CPU autocast off, `float32_matmul_precision="highest"`, MKLDNN enabled |
| Owner | Pinned ACT004 host `f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497` and exact Godot/native/Rapier artifacts |
| Control | Jolt `jev`, original local profile/assists, 60 Hz base clock, six-tick decisions, twelve-tick motor lease |
| Measured actors | One learned Jolt; separately, one learned Jolt with one authored Monowheel |
| Source identity | Original 19-file runtime closure unchanged through the separate completion source; loading verifies its bytes |

The backend imports no learner or checkpoint loader, performs no optimization
and preserves the caller's Torch RNG state during construction. The existing
`ExportController` retains observation age, recurrence, actor/profile/mount and
source-generation boundaries. `dispatch` binds the original issued observation
to the packet and cancels the real owner on fallback. FP16/BF16 CPU autocast,
changed versions, thread count, arithmetic settings or source bytes are refused.

## Measured performance and limits

The original physical run completed **600.520737906 active seconds**, with no
idle padding. Both 240-tick paired trajectories had zero measured action,
recurrent-state, part-position and angle differences under the unchanged
tolerances. Sixteen negative contracts, both profile/cadence refusals and the
bounded realtime ownership checks passed. These are the declared cases and
ownership checks; the result does not establish broader learned task success.

| Active configuration | Seconds | Learned decisions | Native ticks | Decisions/s | Ticks/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| One learned Jolt | 300.494368451 | 4,766 | 28,596 | 15.860530 | 95.163181 |
| Learned Jolt + authored Monowheel | 300.026369455 | 3,884 | 23,304 | 12.945529 | 77.673173 |

Every counted decision has its admitted packet and complete native consumption
receipts. Timing includes the external stepped host, controller and evidence
capture on the recorded Xeon Platinum 8573C container with an eight-CPU quota
and 8 GiB memory limit. The short realtime fixture separately checks ownership.

| Cost scope | Median ms | p95 ms | p99 ms | Maximum ms |
| --- | ---: | ---: | ---: | ---: |
| Standalone full decision, 1,000 calls | 1.494447 | 2.712591 | 4.120924 | 10.463967 |
| One-actor live inference | 2.169334 | 5.750022 | 13.172491 | 721.849249 |
| Two-total-actor live inference | 2.323875 | 6.581416 | 14.433147 | 513.057780 |
| One-actor step, transport and host | 43.194744 | 74.627480 | 138.975392 | 1,151.637162 |
| Two-total-actor step, transport and host | 50.992877 | 88.683186 | 185.644755 | 10,367.889817 |

Standalone timing includes validation, preprocessing, recurrence, action mapping
and source stamping; weight loading and transport are separate. Its process
peak RSS was **224,108,544 bytes (213.73 MiB)**. The model's 29,736 numeric-weight
bytes and 128 hidden-state bytes are smaller accounting scopes. The sustained
host's peak RSS was 213,561,344 bytes (203.67 MiB). The observed long tails prevent
a guaranteed realtime deadline claim; a fast standalone median cannot provide
that guarantee. Reset/observe costs and full environment metadata are in the
[measured record](evidence/2026-10-20-t109-torch-cpu.md).

The realtime last-observation count is a lower bound on autonomous work. Its
full 600-tick episode horizon bounds the unobserved close tail. Report those
bounds separately from stepped-mode ticks. Multiple learned actors, in-engine
execution, Windows or GPU physical deployment, exact mid-episode host resume,
whole-game determinism and owner feel remain unmeasured.

## Load and dispatch the qualified artifact

Install the declared CPU environment using [learning setup](LEARNING.md),
prepare the pinned [host](HOST_ADAPTER.md), and obtain the unchanged bundle from
the verified evidence collection. The optional `.[torch-export]` dependency
group declares NumPy/Torch; select the CPU Torch wheel explicitly. Use source
whose installed runtime closure still matches the bundle. The following bounded
example uses the recorded rightward case and its original observation at every
dispatch:

```python
import os
import torch
from nca.controllers.baseline import SkillGoal
from nca.envs import CybersandEnv, GodotHost, physical_config
from nca.export import context_for, dispatch
from nca.export.torch_runtime import TorchRuntimeBundle, torch_export_controller

torch.set_num_threads(1)
bundle_file = "artifacts/t109-torch-cpu-followup-v1/bc-seed1103-torch-cpu.json"
bundle_sha256 = "1345c35582bbd16f7f6474996976dc3b0fac3dd775a2f83f916347b43e5fb96c"

with GodotHost(os.environ["NCA_HOST_CHECKOUT"], os.environ["NCA_GODOT"]) as host:
    config = physical_config("jev", ticks=240, seed=73)
    config["actors"][0]["position"] = [176.0, 346.5]
    env = CybersandEnv(host, config, decision_ticks=6, lease_ticks=12)
    view, _ = env.reset()
    profile = host.request("profile")["profile"]
    bundle = TorchRuntimeBundle.load(
        bundle_file, context_for(profile, host.identity),
        expected_sha256=bundle_sha256,
    )
    controller = torch_export_controller(bundle)
    goal = SkillGoal("floor-right-24", "directed_hop", (200.0, 346.5),
                     pace=0.8, tolerance=4.0, timeout_seconds=4.0)
    try:
        while True:
            issued = view
            decision = controller.step(issued, goal, source_generation=1)
            if decision.cancel_owner:
                dispatch(env, decision, issued)
                break
            view, _, terminated, truncated, info = dispatch(env, decision, issued)
            refused = any(not item["accepted"]
                          for row in info["trace"] for item in row["admission"])
            if terminated or truncated or refused or not info["consumed_trace_complete"]:
                break
    finally:
        controller.cancel()
        env.cancel()
```

The registry entry is `policy/jolt-torch-cpu-export/v1`. A realtime caller must
also account for elapsed host ticks and retain idle receipts. Cancellation
retires the old goal and clears recurrence; a fresh invocation requires a fresh
goal ID and current owner observation. The host's final admission is authoritative.

## Completed evidence path

The single physical invocation used
[torch-cpu-followup-v1.json](../configs/export/torch-cpu-followup-v1.json) and
`tools/export_torch_policy.py` from the exact `f3bf96fa` source. Its 900-second
wall ceiling, 25,000 requests, 22,000 step requests, 132,000 requested step ticks
and two host starts were respected. Its **642.057584872 external seconds**,
complete owner exchanges and original clips-stage no-go remain immutable.

The one offline completion used `tools/complete_export_clips.py` from
`78cc377b5af9e62efeb73d93cc1e16bd36a013a4` with
[torch-cpu-clips-completion-v1.json](../configs/export/torch-cpu-clips-completion-v1.json),
canonical declaration SHA-256
`035151338deb32ec69b7ec77be5b92942d2c2f64dd27e49928aaedfef65b6771`.
It ran in a separate directory, pinned all original files and the passing
fresh-episode resume/retained learner-cost receipts, and recovered each missing
profile only from its original reset/profile/actor-linked exchange. No saved
action, native tick, pose or original result was changed.

The unchanged renderer produced four clips under the original camera and
timebase, with 41, 41, 5 and 5 fully decoded frames. Both nominal GIFs are
byte-identical to their original files. These are actual own-part pose
reconstructions; terrain and native effects are outside their rendering scope.
The supervised completion took **24.190651845 seconds**, with zero host starts,
host requests, policy evaluations, new simulation ticks or optimizer updates.
The two invocations total **666.248236717 external seconds**, within the
900-second aggregate ceiling and the separate 60-second offline cap.

Both declared invocations are complete. Further measurements or postprocessing
require a new explicit scope and preserve these results. The future capture
fix saves profile provenance and durable child stdout/stderr; it does not
rewrite the original missing-envelope or missing-traceback record.
