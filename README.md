# neural-cyber-automata

Research, training and evaluation for intelligent behaviour in [Cybersand](https://github.com/techrote/cybersand): physical actor control, movement and tool skills, human and model command sources, coordination, and neural cellular automata where local distributed behaviour helps the game.

**Stage: implementation, 19 October 2026.** The CPU contract and run-record foundation is implemented. Host integration, authored controls and the bounded learning programme have separate acceptance records in [current state](docs/STATE.md). The programme uses the articulated ACT-004 host and preserves the existing Thunderbowl control contracts.

## Start here

- [Research conclusions and first experiments](docs/research/PREPASS.md)
- [Control methods and physical constraints](docs/research/CONTROL_METHODS.md)
- [Human, model and LLM command contract](docs/design/COMMAND_AND_SKILL_CONTRACT.md)
- [Training environment and evaluation design](docs/design/TRAINING_AND_EVALUATION.md)
- [Implementation programme](docs/ROADMAP.md)
- [Pinned source audit](docs/evidence/2026-10-18-source-prepass.md)
- [Full documentation map](docs/README.md)

The recommended first comparison is a tuned physical controller, an imitation policy, and a small recurrent residual policy on the same rig, sensors, actuator limits and scenarios. Graph policies, learned coordination, world models and cellular controllers enter when a specific experiment can show why they are useful.

The original NCA programme is preserved in history and retained as a Cybersand research track. It no longer makes pattern-growth reproduction, a toy sand engine or legacy GPU qualification prerequisites for actor control. [Scope and migration](docs/SCOPE_AND_MIGRATION.md) explains the change.

Agents: read [AGENTS.md](AGENTS.md), the current state and the task's dependency gates before starting.


## Run the CPU foundation

The installable `nca` package provides strict gameplay contracts, preserved ACT003
wire/direct mappings, a versioned plugin registry and immutable experiment records.
See [package setup](docs/PACKAGE.md) for Windows and Linux commands.

```sh
python -m pip install -e .
nca doctor
nca plugins
nca fixture --output artifacts/records --run-id first-recorded-example
```

The recorded example checks protocol semantics; it makes no game-physics or
training claim. Follow [current evidence](docs/STATE.md) and the
[implementation checkpoints](docs/PROGRESS.md) for programme progress.

© 2026 techrote. All rights reserved.
