---
title: NumPy CPU policy export and deployment screen
kind: reference
summary: Independent NumPy GRU, exact gameplay and timing admission, original-source dispatch, bounded qualification and fresh-episode checkpoint resume.
owns: [export.runtime, export.qualification]
verified: 2026-10-20
---

# NumPy CPU policy export

The optional `nca.export` package implements a small, independent NumPy FP32
runtime for the fixed T105 Jolt behavior-cloning policy. Its first target is an
**external Linux Python controller loop** over the accepted ACT004 owner. The
bundle carries the trained weights and their evaluated lineage; a separate
immutable screen receipt determines whether the declared deployment passed.
Conversion alone leaves `runtime_qualified` false.

The [staged NumPy screen](evidence/2026-10-19-t109-deployment-no-go.md) remains
`runtime_qualified: false`: both trajectory comparisons failed, and its
79.033537896 seconds did not complete the full ten-minute timing gate.
Passing engineering checks do not change that result or its [lineage](evidence/2026-10-19-t109.md).
The [Torch CPU runtime](TORCH_RUNTIME_EXPORT.md) now has an exact-bundle composite
qualification from separately recorded physical work and completed pose evidence.
Its original `f3bf96fa` clips-stage no-go is preserved; this NumPy lane remains unqualified.
Both exports are external Python loops; other runtimes and devices need separate evidence.

The [issue #42 correction](evidence/2026-10-20-t109-dispatch-authority.md) changes
the owner boundary while preserving numerical operators. Current runtime bytes
refuse the older bundle's source hashes. New conversion and physical qualification
are required; historical qualification remains attached to its original source.

## Bundle contract

| Content | Meaning |
|---|---|
| Numeric weights | FP32 row-major matrices for a 26-to-32 encoder, 32-state GRU, six action heads and scalar critic; the learner's three unused BC log-standard-deviation values remain for exact policy content identity |
| Normalization | Explicit fixed divisors, clips, missing-value bits, elapsed-goal fraction and previous requested action; no inferred means or hidden evaluator state |
| Sensor identity | ACT004 local proprioception and rays, explicit caller goal and separate local part/ownership metadata; training privilege `none` |
| Action identity | `physical_wire_action/v2`; move X and trim use tanh, effort uses sigmoid, three binary channels use a 0.5 threshold, and aim uses the declared normalized goal displacement |
| Assistance identity | Full authored profile plus a separate digest of all physical tunes and joints; profile changes cannot retain an old qualification |
| Recurrence | Reset/update/new gates with the reset applied after the hidden candidate affine map; zero state on episode, actor, profile, mount, goal or source/control change |
| Timing | 60 base ticks per second, six-tick decisions, at most twelve-tick motor leases capped by the goal deadline; expiry at `now >= observed_tick + lease_ticks` |
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

`ExportController.step` accepts an issued ACT004 view and typed `SkillGoal`.
`dispatch(env, controller, decision, issued)` consumes that controller's single
pending decision before transport. Its immutable source stamp binds the original
run, actor, epoch, generation, tick, observation ID and local part/mount inputs.
A newer view, copied result or retired decision cannot refresh its authority.

Malformed, stale, reordered or incompatible input produces a zero-effort fallback
with `cancel_owner=True`. Dispatch revokes actual native intent while preserving
applied motion and effects. Command or transport failures attempt the same cleanup
and preserve the original exception. No neutral wire step releases a held charge.

Cancellation/reset clears recurrence, request history, cadence and pending state,
and retires the goal. Actor lifetimes and observed tick/control/source high-water
marks survive local cancellation. Ongoing feedback is spaced six ticks apart;
source or mount changes retire the invocation before cadence is considered.
Source grants come from the [asynchronous ownership layer](ASYNC_CONTROL.md).

A live helper requires native cancellation before a new source or invocation.
`controller.confirm_owner_cancel(env.cancel())` validates the revocation receipt;
missing, foreign, stale or unchanged-generation receipts retain the cleanup latch.
Local reset cannot clear it. A validated new host reset can retire the old world;
same-run epochs must advance. Dispatch confirms its own fallback cancellation.

Inference and dispatch check supplied `current_tick` against the absolute goal
deadline. The original-source lease is capped by that deadline, and a final repeat
can be shorter than six ticks. Native delay cannot re-stamp the source. The full
returned interval must fit the source lease and deadline, preserve part/mount
identity, and prove original admission and consumption before dispatch succeeds.

The host's admission remains authoritative. Refused, mixed, incomplete or expired
intervals cancel. A terminal result retains its unavailable final action, retires
local state and requires cleanup or a new host reset. See the linked correction
record for failure cases and the separate echo/readback numeric tolerances.

## Use the runtime

Prerequisites are the pinned [host installation](HOST_ADAPTER.md), the pinned
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
                dispatch(env, controller, decision, issued)
                break
            view, _, terminated, truncated, info = dispatch(env, controller, decision, issued)
            refused = any(not item["accepted"] for row in info["trace"] for item in row["admission"])
            if terminated or truncated or refused:
                break
    finally:
        controller.cancel()
        controller.confirm_owner_cancel(env.cancel())
```

The example preserves `issued` through inference and dispatch. A realtime
caller must also account for elapsed host ticks and retain idle receipts; the
host independently rejects expired requests. The bounded realtime export
screen measures refusal/cancellation conformance, separately from learned
task performance or the 120-tick planner lifetime.

## Convert and reproduce the screen

Conversion requires complete T105 artifacts, their accepted merge and the exact
CPU Torch reference environment. Checkpoint and numeric source must match; all
eight fixed evaluation families must identify it and retain unsuccessful outcomes.

The converter pins manifest configuration, original learner revision/source map,
dataset receipt and every selected episode. Missing, changed, duplicate or redirected
files are refused. The screen verifies frozen file-byte identities before and after execution.

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

## Separately evaluated candidate from an interrupted prerequisite

The [BC1103 continuation](evidence/2026-10-19-t109-staged-candidate.md) preserves
`interrupted_no_summary` and unconfirmed original termination. Ordinary `convert`
still requires a complete T105 experiment; the continuation uses a separate contract.

After the candidate declaration and evaluator source are published, its one
bounded evaluation can be invoked with:

```bash
python tools/export_policy.py evaluate-candidate \
  --training-root artifacts/t105-interrupted-snapshot \
  --learner-source "$ORIGINAL_T105_CHECKOUT" \
  --declaration configs/export/bc1103-interrupted-evaluation-v1.json \
  --host "$NCA_HOST_CHECKOUT" --godot "$NCA_GODOT" \
  --output artifacts/t105-interrupted-snapshot/candidate-evaluation-v1
```

The exact declared learner revision and immutable snapshot are required. Failed
or censored invocations cannot convert; complete unsuccessful cases remain valid.
Eight families establish a single-candidate evaluation, not a three-seed comparison
or a missing original summary.

Conversion requires a published `nca-fixed-candidate-export-declaration/v1`
binding the completed evaluation manifest, summary and artifact identities.
The converter checks actual receipts and committed source. The resulting bundle
uses `screen` with unchanged CPU criteria; a failed screen remains unqualified.

The declared continuation completed its eight-family evaluation from public
revision `4510347e02d08093f1b86bef4583f0282b7cf903`; the actual receipts and
the second declaration are linked in the
[candidate evaluation evidence](evidence/2026-10-19-t109-candidate-evaluation.md).
After publishing that second declaration, its conversion command is:

```bash
python tools/export_policy.py convert-candidate \
  --training-root artifacts/t105-interrupted-snapshot \
  --evaluation-root artifacts/t105-interrupted-snapshot/candidate-evaluation-v1 \
  --declaration configs/export/bc1103-staged-export-v1.json \
  --accepted-t105-commit "$T105_ACCEPTED_COMMIT" \
  --output artifacts/t109-staged/bc-seed1103-cpu.json
```

The scoped Python source map must still match the evaluation freeze. Keep the
interrupted training receipt and the completed candidate summary as distinct
inputs. This command alone never sets deployment qualification to true.

## Diagnose a preserved parity failure

Keep a failed screen and its original bundle unchanged. The read-only auditor
uses the original trajectory observations, feature vectors and hidden-state
inputs to compare the two arithmetic implementations:

```bash
python tools/export_policy.py audit \
  --training-root artifacts/t105-screen-v1 \
  --screen-root artifacts/t109/screen-v1 \
  --bundle artifacts/t109/bc-seed1103-cpu.json \
  --sha256 "$BUNDLE_FILE_SHA256" \
  --output artifacts/t109/screen-v1-audit
```

It records the first differing request, the first differing part pose and the
first crossing of the original pose tolerances. Replaying an identical stored
feature/hidden pair through Torch and NumPy isolates arithmetic differences
from the different observations generated by a diverging closed-loop run.
Reconstructed features and requests are checked against the original records.
Every consumed tick requires its original receipt and frame; empty, shortened,
nonfinite or malformed comparisons cannot pass.

The audit performs zero host steps and zero optimization. Its result cannot
promote a failed screen or establish that contact sensitivity caused a
divergence. A source correction needs a separately declared bounded screen,
with the original failed record and every original tolerance retained.

## Focused checks

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_export*.py' -v
python tools/check_docs.py --strict
```

Optional dependencies and unconfigured actual-host artifacts cause explicit skips.
Evaluated export integration needs `NCA_EXPORT_BUNDLE` and `NCA_EXPORT_BUNDLE_SHA256`.
Portable fixtures passing on another OS do not qualify deployment there.

`ExportOwnerBoundaryTests` needs just the pinned host and uses synthetic weights
for actual-owner receipts, single-use dispatch, takeover, delayed sources,
deadline-capped leases and terminal readback. That conformance does not establish
useful learned behaviour or qualify an evaluated deployment artifact.
