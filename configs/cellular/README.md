---
title: Conditional Jolt per-part graph experiment
kind: guide
summary: One preregistered three-part graph, explicit local sensors and memory ownership, matched no-message control, bounded CPU training and sealed physical evaluation.
owns: [cellular.jolt_experiment]
verified: 2026-10-19
---

# One per-part graph experiment

[jolt-three-part-hop-v1.json](jolt-three-part-hop-v1.json) commissions exactly
one T-110 experiment: Jolt's `directed_hop` with a fixed three-part star. It was
declared before implementation and before training or opening task outcomes.
There is no terrain lattice, swarm engine or morphology-transfer claim.

`impact_core` is node 0 and the only action readout. The left and right rebound
pads are nodes 1 and 2. Each node owns its own body motion, size and mass sensors.
The caller's explicit goal and previous actor request are declared common
context. Only the root gets the actor's support, energy, cooldown and ground/
ceiling rays; pad slots are unavailable with false validity masks. No node reads
an evaluator, other actor, full scene or hidden pooled state.

Each node owns 16 hidden floats. One synchronous neighbor-message round per
six-tick decision transmits eight floats per directed edge. A root action can
depend on adjacent pad information after one round. A pad-to-pad effect needs
two successive decisions through the root. Reset, profile/actor replacement,
source/control/mount generation and goal changes clear the corresponding
memory and queued messages. Physical motion stays in the host.

The learned graph and no-message model have identical 2,878 parameters and run
the same three-node population. A local self-message network remains active in
both. The no-message model disables inter-node edges; its root cannot secretly
read pad features. The authored controller receives the same permitted local
actor observation budget; its existing algorithm elects to use root feedback
only. The graph/no-message comparison supplies the strict matching of all local
part features and their locations. The authored result supplies the gameplay
reference gate; a difference against it alone cannot establish message utility.
All methods use the same six-channel wire mapping and
the same support/cooldown/edge gate on primary requests. The host's authored
assistance, compliant joints, Rapier and native material rules remain common.

Direct creation, destruction and conversion request budgets are zero for all
three methods. This Jolt profile's primary pulse invokes the native physical
jump, and no usable tool/mount channel is enabled. The policies have no cell
write API. Contacts, masks and subsequent material evolution remain in the
unchanged native owner and are recorded as host behaviour.

Three policy seeds each train a graph and no-message model for 30 fixed BC
epochs, using eight newly captured authored training episodes. Unsuccessful
teacher trajectories are retained. The tuning family is diagnostic only, and
the final epoch is selected in advance. All six checkpoint hashes must be
frozen before the two holdout families are opened.

The normal, one-decision delay, 25% edge-loss, right-pad communication partition
and tick-60 memory-reset conditions are separate. These affect controller
communication or memory, not physical joints or part topology. Unknown/detached
physical parts are refused. The reset condition leaves the actual body state,
caller goal and external task deadline intact. Source/control/mount lifecycle
reset is also checked independently at the software boundary.

The declared ceiling is 4,000 game decisions and 900 seconds on CPU with one
Torch thread; the worst planned task matrix uses 3,570 decisions. The screen
requires a paired benefit over the no-message model, no success regression
against the authored reference, and the fixed integrity, style, inference and
communication-stress bounds. A failed gate means no-go for advancing this graph
under this screen. It does not disprove graph policies generally.

The older Growing NCA, damage and evolution-strategies references remain
available in [scope migration](../../docs/SCOPE_AND_MIGRATION.md) and the
[implementation roadmap](../../docs/ROADMAP.md). Image-regrowth results are not
used to establish actor efficacy. Owner feel judgement remains separate.

## Reproduce the bounded screen

Use a clean committed learner checkout and the pinned host/runtime described in
[the host guide](../../docs/HOST_ADAPTER.md). The runner refuses edited or
untracked executed source files and a changed declaration. It accepts exactly
Torch `2.8.0+cpu` and NumPy `2.3.5`; core imports and registry discovery do not
need either optional dependency. Linux is the physical screening lane. Windows
commands are supplied for reproducibility; a Windows physical result must be
measured separately.

From the learner checkout, Linux setup and invocation are:

```sh
python3.12 -m venv .venv-learning
.venv-learning/bin/python -m pip install setuptools==84.0.0 wheel==0.45.1
.venv-learning/bin/python -m pip install --no-build-isolation -e . numpy==2.3.5
.venv-learning/bin/python -m pip install torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
.venv-learning/bin/python -m unittest discover -s tests -p test_cellular.py -v
.venv-learning/bin/python tools/cellular_screen.py --source-root . --config configs/cellular/jolt-three-part-hop-v1.json --output ../t110-screen-artifacts
```

Set `NCA_HOST_CHECKOUT` and `NCA_GODOT` as described in the host guide, or pass
`--host` and `--godot`. PowerShell equivalents use an explicit interpreter:

```powershell
py -3.12 -m venv .venv-learning
.\.venv-learning\Scripts\python.exe -m pip install setuptools==84.0.0 wheel==0.45.1
.\.venv-learning\Scripts\python.exe -m pip install --no-build-isolation -e . numpy==2.3.5
.\.venv-learning\Scripts\python.exe -m pip install torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv-learning\Scripts\python.exe -m unittest discover -s tests -p test_cellular.py -v
.\.venv-learning\Scripts\python.exe tools/cellular_screen.py --source-root . --config configs/cellular/jolt-three-part-hop-v1.json --output ../t110-screen-artifacts
```

The destination must be new. It holds immutable compressed episodes, requested/
admitted/consumed/applied/passive distinctions for every base tick, message
sources and delivery ticks, private hidden state before each update, six initial
and six final checkpoints, the pre-holdout freeze, training histories, outcome
rows, hashes and one run manifest per episode plus the whole screen. Actual
blobs and checkpoints stay outside Git. A failed or capped attempt remains a
separate artifact; the runner never extends its budget or resumes a partly
completed episode. There is no PPO likelihood claim for these BC demonstrations.

The runner publishes a validated `prepared` run manifest before opening the
host. Run-ledger identifiers use `jolt-per-part-proprioception-goal-v1`; the
unchanged feature schema `jolt-per-part-proprioception-goal/v1` stays in the
resolved declaration, dataset and model specification. A recording failure
after game work is a failed attempt; an escaped CLI error does not imply that
the host consumed zero decisions. Surviving artifacts establish actual progress.

When a corrected attempt belongs to the same commissioned screen, pass
`--prior-budget ../prior-budget.json`. This immutable JSON receipt has schema
`nca-cellular-prior-budget/v1` and a `receipts` array. Each entry supplies a unique
`id`, the supporting evidence's `content_sha256`, measured `game_decisions`, and
measured `wall_seconds`. The runner derives prior totals from these entries and
charges them against the original 4,000-decision / 900-second ceiling. It
refuses malformed, negative, nonfinite, duplicated or excessive usage and offers
no option to raise a ceiling. The copied receipt and its canonical hash are
retained with the attempt. Screen summaries separate attempt usage from the
aggregate, including earlier failed attempts and physical preflights. A missing
old ledger may be reconstructed only as a separately labelled postprocessing
receipt; the original run files remain untouched.

The [T-110 evidence record](../../docs/evidence/2026-10-19-t110.md) distinguishes
source/unit checks, actual game measurements and the fixed utility decision.

© 2026 techrote. All rights reserved.
