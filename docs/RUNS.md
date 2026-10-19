---
title: Run ledger
kind: evidence
summary: Immutable training and benchmark provenance ledger, with source identities and evidence limits.
owns: [runs.ledger]
verified: 2026-10-19
---

# Run ledger

Bounded host pilots have run. No trained result is accepted in this increment. Documentation checks and source review belong to the evidence directory and are not training records.

T-101 and T-104 specify immutable per-run manifests and a generated/merged ledger index. Workers do not concurrently append to a shared Markdown file. Corrections identify the prior record and retain it.

| Run ID | Date | Learner and host revisions | Device/environment | Profile/config hash | Seed set and split | Suite | Result and evidence class | Artifact/evidence |
|---|---|---|---|---|---|---|---|---|
| t102-monowheel-final-pilot | 2026-10-19 | NCA scoped sources `7ea78b97…`; host `f2e34cf9…`; exact hashes in evidence | CPU, Linux x86_64, Python 3.12.14, Godot 4.7 | `6da0f8ba…`; complete config in manifest | 73; conformance only | 200 decisions / 1,200 ticks | Actual ACT004 host, 326.884 ticks/s; no skill or owner-feel claim | [Manifest](evidence/t102-monowheel-final-pilot.json), [evidence](evidence/2026-10-19-t102.md) |
| t102-ripper-final-pilot | 2026-10-19 | Same exact scoped source and host as Monowheel | Same measured CPU environment | `be59daec…`; complete config in manifest | 73; conformance only | 200 decisions / 1,200 ticks | Actual ACT004 host, 267.686 ticks/s; no skill or owner-feel claim | [Manifest](evidence/t102-ripper-final-pilot.json), [evidence](evidence/2026-10-19-t102.md) |
| t102-needlejet-pilot | 2026-10-19 | Earlier adapter hashes preserved in manifest; same host | CPU, Linux x86_64, Python 3.12.14 | Complete profile/config hash in manifest | 73; conformance only | 200 decisions / 1,200 ticks | Historical pre-review pilot, 241.542 ticks/s; three-part rig | [Manifest](evidence/t102-needlejet-pilot.json), [limits](evidence/2026-10-19-t102.md) |
