---
title: CPU package operations
kind: guide
summary: Install, inspect and exercise the versioned CPU contracts and immutable run store.
owns: [operations.package]
verified: 2026-10-19
---

# CPU package operations

The core package uses Python 3.11 or newer and no third-party runtime dependency.
Python 3.12.14 is the conformance version. NumPy and PyTorch are optional learning
dependencies; importing the core, validating a profile and running the recorded
example do not import them or create a GPU device.

## Windows

Prerequisites: Git and Python 3.12 with the `py` launcher. From the repository:

```powershell
powershell -ExecutionPolicy Bypass -File tools/setup_cpu.ps1
.\.venv\Scripts\nca.exe plugins
.\.venv\Scripts\nca.exe fixture --output artifacts/records --run-id first-recorded-example
.\.venv\Scripts\nca.exe runs verify artifacts/records
```

The setup script stops on a failed external command. The hosted CPU lane checks
installation and conformance on Windows Server 2022; host/game qualification is a
separate test lane.

## Linux

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install setuptools==84.0.0 wheel==0.45.1
.venv/bin/python -m pip install --no-build-isolation -e .
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/nca doctor
.venv/bin/nca fixture --output artifacts/records --run-id first-recorded-example
.venv/bin/nca runs verify artifacts/records
```

Every recorded-example invocation requires a unique run ID and publishes its
manifest atomically without replacing an existing record. It validates three
recorded transitions. It does not simulate game mechanics or claim gameplay
success. `nca doctor` reports dependency versions and a digest of installed source
bytes without emitting machine names or local paths. An installation outside a
Git checkout requires `--learner-revision` with the full source revision for a run.

## Contracts and extension points

`nca validate profile configs/profiles/recorded-rig-v1.json` checks the complete
versioned manifest. Other record kinds are `command`, `legacy-command`,
`observation`, and `wire-action`. Structural schemas are in `schemas/`; Python
constructors additionally enforce topology, bounds, masks and lifecycle rules.

`builtin_registry()` returns actual profile, recorded-adapter, idle-policy and
evaluator implementations. Register a versioned factory with explicit
capabilities before adding framework-wide dispatch. Unknown versions, duplicate
IDs and missing required capabilities produce named refusals. Legacy ACT003
wire and four-slot metal mappings remain distinct from ACT004 wire values.

## Provenance and artifacts

`RunManifest` includes learner/host/profile/config identity, environment, seed
families, scenario partitions, budget, status and artifacts. `RunStore` publishes
one immutable JSON envelope per run using a no-replace atomic operation. A
finished failure or capped run requires its reason; unmeasured metrics use null.
The Markdown index is derived with `nca runs index`, so workers do not race to
append to one ledger. Large datasets, checkpoints and clips remain outside Git.

The current gameplay and acceptance evidence is in [STATE.md](STATE.md) and
[RUNS.md](RUNS.md). Package conformance alone is not a host, policy or feel result.
