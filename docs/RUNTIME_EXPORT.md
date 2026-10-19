---
title: Numeric CPU policy export and deployment screen
kind: reference
summary: Independent NumPy GRU, exact gameplay and timing admission, original-source dispatch, bounded qualification and fresh-episode checkpoint resume.
owns: [export.runtime, export.qualification]
verified: 2026-10-19
---

# Numeric CPU policy export

The optional `nca.export` package implements a small, independent NumPy FP32
runtime for the fixed T105 Jolt behavior-cloning policy. Its first target is an
**external Linux Python controller loop** over the accepted ACT004 owner. The
bundle carries the trained weights and their evaluated lineage; a separate
immutable screen receipt determines whether the declared deployment passed.
Conversion alone leaves `runtime_qualified` false.

The implementation and frozen deployment criteria are present. The final
qualification result is tracked in [T109 evidence](evidence/2026-10-19-t109.md).
Unmeasured runtime combinations are rejected rather than inheriting a measured
lane's status. This first export does not execute inside Godot and does not
qualify a residual policy, accelerator, Windows runtime or legacy device.

## Bundle contract

| Content | Meaning |
|---|---|
| Numeric weights | FP32 row-major matrices for a 26-to-32 encoder, 32-state GRU, six action heads and scalar critic; the learner's three unused BC log-standard-deviation values remain for exact policy content identity |
| Normalization | Explicit fixed divisors, clips, missing-value bits, elapsed-goal fraction and previous requested action; no inferred means or hidden evaluator state |
| Sensor identity | ACT004 local proprioception and rays, explicit caller goal and separate local part/ownership metadata; training privilege `none` |
| Action identity | `physical_wire_action/v2`; move X and trim use tanh, effort uses sigmoid, three binary channels use a 0.5 threshold, and aim uses the declared normalized goal displacement |
| Assistance identity | Full authored profile plus a separate digest of all physical tunes and joints; profile changes cannot retain an old qualification |
| Recurrence | Reset/update/new gates with the reset applied after the hidden candidate affine map; zero state on episode, actor, profile, mount, goal or source/control change |
| Timing | 60 base ticks per second, six-tick decisions and twelve-tick motor leases; expiry at `now >= observed_tick + lease_ticks` |
| Runtime | CPython 3.12.14, NumPy 2.3.5, Linux x86_64, CPU FP32; the Torch 2.8.0+cpu reference is confined to conversion and comparison |
| Evidence | Checkpoint, numeric source, configuration, dataset, original learner source, manifest, complete evaluation summary and selected episode hashes; accepted dependency commit remains separate from original training revision |

The fixed candidate is **BC seed 1103 final**, selected before reading T105
evaluation outcomes. The arithmetic, cases, budgets and tolerances are frozen in
[cpu-bc-v1.json](../configs/export/cpu-bc-v1.json), canonical SHA-256
`f499cb94ec537f5c1cc1b601cfcfa37fdaf9f5e81abe365aecb23d72ad0f1df0`.
The complete authored Jolt profile has SHA-256
`97559dcca7b1523ee622fd886615f05f5d7f85bacc0655ba9d642e909dc112bc`.

`RuntimeBundle.load` requires the externally supplied **file-byte SHA-256**.
Inside the artifact, `sha256` identifies canonical unsigned bundle content;
`policy_sha256` identifies the learner's ordered specification and FP32 tensor
bytes. These hash spaces are named separately throughout the evidence. Load
checks also pin the installed preprocessing and controller dependency bytes.
Changing those bytes requires another explicit conversion and screen.

## Ownership and fallback

`ExportController.step` accepts an issued ACT004 view and a typed `SkillGoal`.
Every accepted result carries an immutable `SourceStamp` with the original run,
actor, epoch, control generation, tick and observation ID, plus a digest of the
local inputs and part/mount boundary. `dispatch` verifies this stamp before
constructing a packet. Supplying a newer view for an old result is refused and
attempts real-owner cancellation; it cannot refresh the result's motor lease.

Malformed, stale, reordered, unavailable or incompatible input produces a
zero-effort fallback with `cancel_owner=True`. Dispatch sends the ACT004 cancel
operation, which revokes pending intent while preserving already applied motion
and effects. It does not send a neutral wire step that could release a charge.
Command-construction and transport exceptions also attempt owner cancellation
and preserve the original exception if cleanup fails.

Cancellation zeros recurrence and the previous-request history, clears the
old cadence cursor and retires the invocation ID. A fresh invocation can begin
from a fresh completed view; an old goal cannot silently resume. Physical actor
lifetimes remain retired separately. The highest observed tick and owner/source
generations survive cancellation, so a new goal ID cannot revive an older view
or a revoked grant. An actual new actor lifetime resets these counters. An ongoing invocation requires exact
six-tick feedback spacing. A source/control or mount change takes precedence
over a cadence error and retires the old invocation. Source grants belong to
the [command and skill ownership contract](design/COMMAND_AND_SKILL_CONTRACT.md); this runtime receives the
explicit source generation and preserves it across numeric decisions.

The host's final admission remains authoritative. Callers retain refusal
receipts and cancel their local controller before starting a new invocation.
An arithmetic match does not authorize a delayed or foreign request.

## Use the runtime

Prerequisites are the pinned [host installation](HOST_ADAPTER.md), the qualified
Python/NumPy environment, a complete numeric bundle and its external hash
receipt. Numeric inference does not require Torch. The following excerpt runs a
bounded stepped episode with an explicit caller goal:

```python
from nca.controllers.baseline import SkillGoal
from nca.envs import CybersandEnv, GodotHost, physical_config
from nca.export import ExportController, RuntimeBundle, context_for, dispatch

with GodotHost(host_checkout, godot_executable) as host:
    env = CybersandEnv(host, physical_config("jev", ticks=240, seed=73))
    view, _ = env.reset()
    profile = host.request("profile")["profile"]
    bundle = RuntimeBundle.load(
        bundle_file, context_for(profile, host.identity),
        expected_sha256=bundle_file_sha256,
    )
    controller = ExportController(bundle)
    goal = SkillGoal("hop-1", "directed_hop", (200, 346.5), timeout_seconds=4)
    try:
        while True:
            issued = view
            decision = controller.step(issued, goal, source_generation=1)
            if decision.cancel_owner:
                dispatch(env, decision, issued)
                break
            view, _, terminated, truncated, info = dispatch(env, decision, issued)
            refused = any(not item["accepted"] for row in info["trace"] for item in row["admission"])
            if terminated or truncated or refused:
                break
    finally:
        controller.cancel()
        env.cancel()
```

The example preserves `issued` through inference and dispatch. A realtime
caller must also account for elapsed host ticks and retain idle receipts; the
host independently rejects expired requests. The bounded realtime export
screen measures refusal/cancellation conformance, separately from learned
task performance or the 120-tick planner lifetime.

## Convert and reproduce the screen

Conversion requires the complete T105 artifact collection, its accepted merge
commit and the exact CPU Torch reference environment. The training checkpoint
and numeric source must match; all eight fixed evaluation families must be
present and identify the selected checkpoint. An evaluated unsuccessful
checkpoint is exportable with that unsuccessful outcome retained.

```bash
python tools/export_policy.py convert \
  --training-root artifacts/t105-screen-v1 \
  --accepted-t105-commit "$T105_ACCEPTED_COMMIT" \
  --output artifacts/t109/bc-seed1103-cpu.json

python tools/export_policy.py screen \
  --training-root artifacts/t105-screen-v1 \
  --accepted-t105-commit "$T105_ACCEPTED_COMMIT" \
  --bundle artifacts/t109/bc-seed1103-cpu.json \
  --sha256 "$BUNDLE_FILE_SHA256" \
  --render-python "$PILLOW_PYTHON" \
  --output artifacts/t109/screen-v1
```

`T105_ACCEPTED_COMMIT` is the verified dependency merge; `BUNDLE_FILE_SHA256`
comes from the conversion receipt. `PILLOW_PYTHON` names an interpreter with
NumPy and Pillow 12.3.0. Set `NCA_HOST_CHECKOUT` and `NCA_GODOT` as described in
the host guide. The actual screen requires an isolated source checkout whose
recorded dependencies match the accepted revisions. Existing outputs are
immutable and cause refusal; a correction receives a new artifact directory.

The screen freezes exact source and configuration hashes before execution and
checks the source bytes again afterwards. Its qualification gates are:

1. 128 sequential arithmetic comparisons with four explicit recurrence resets,
   maximum absolute error 0.00002 for hidden/logits/value/outputs, 0.0000001 for
   features, 0.00002 for continuous wire channels and exact binary channels.
2. Two fixed 240-tick Torch/NumPy trajectory pairs. Own-part position and angle
   differences must remain within 0.05 cells and 0.05 radians; receipt clocks,
   admission and cancellation are compared separately.
3. Version, profile, sensor, action, assistance, cadence, lease, lifetime,
   nonfinite and source-order refusals. Actual cadence/profile variants retain
   their passive outcomes and matched pose clips.
4. A live realtime delayed-packet expiry and an independent fresh-invocation
   source takeover, at most four learned calls and twelve wall seconds. An
   explicitly recorded authored pulse establishes nonzero existing motion for
   cancellation evidence; it is not credited as learned performance.
5. At most 600 seconds of throughput measurement: 500 decisions with one
   learned-controlled actor, then 500 with one learned actor and one authored
   Monowheel demo. Ten reset and 100 observe RPC samples are separate from
   inference/step costs. Multiple learned-controlled actors remain unmeasured.
6. A fresh NumPy-only subprocess replays 1,000 actual local observations and
   times the complete feature/recurrence/action/source-stamp call. Weight load,
   transport and episode initialization are measured or labelled separately.
7. The actual residual learner checkpoint restores model, all optimizer state
   and three RNG streams at a fresh episode boundary. Two 24-tick stochastic
   fresh game episodes compare outputs and poses without updating any weights.

Median, nearest-rank p95/p99 and maximum latency are recorded. Memory identifies
the NumPy-only process, the comparison process that also loads Torch, the
persistent host's current/peak RSS, numeric weight bytes and recurrent state
bytes. Host peak RSS covers that host process's lifetime, not an isolated model
allocation. Original T105 learner update and BC training costs are reused by
hash; the export screen performs zero optimizer updates.

Pose GIFs reconstruct actual own-part geometry under one fixed camera and
simulation timebase. They do not render the terrain or native effects and are
labelled accordingly. Neither those clips nor sequential floating-point parity
establish whole-game determinism or owner acceptance of game feel.

## Focused checks

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_export*.py' -v
python tools/check_docs.py --strict
```

Optional NumPy/Torch absences and missing actual-host artifacts are explicit
skips. Actual export integration additionally requires `NCA_EXPORT_BUNDLE` and
`NCA_EXPORT_BUNDLE_SHA256`; those checks use evaluated weights rather than a
synthetic qualification flag. The regular contract fixtures are portable;
their successful execution on another OS does not qualify the deployed bundle
for that OS.
