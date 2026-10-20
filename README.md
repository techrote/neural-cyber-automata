# neural-cyber-automata

Research, training and evaluation for intelligent behaviour in [Cybersand](https://github.com/techrote/cybersand): physical actor control, movement and tool skills, human and model command sources, coordination, and neural cellular automata where local distributed behaviour helps the game.

**Stage: nine of ten programme tasks have accepted results, 20 October 2026.** The CPU package, physical host adapter, authored controls, capture/replay, learned Jolt skill, usable skills and asynchronous ownership are implemented. The bounded graph experiment is complete with a recorded no-go. The [exact evaluated Linux Torch CPU bundle](docs/evidence/2026-10-20-t109-torch-cpu.md) has separate composite deployment qualification, including two active five-minute host windows and all four matched pose clips; its original rendering failure and the earlier NumPy no-go remain recorded. Morphology transfer is the remaining programme task: the designated primary still fails the fixed 75% mean / 50% every-seed reliability prerequisite. [Current state](docs/STATE.md) records the results and limits, and [progress checkpoints](docs/PROGRESS.md) track the remaining work. The programme uses the articulated ACT-004 host and preserves the existing Thunderbowl contracts.

## Start here

- [Research conclusions and first experiments](docs/research/PREPASS.md)
- [Control methods and physical constraints](docs/research/CONTROL_METHODS.md)
- [Human, model and LLM command contract](docs/design/COMMAND_AND_SKILL_CONTRACT.md)
- [Training environment and evaluation design](docs/design/TRAINING_AND_EVALUATION.md)
- [Implementation programme](docs/ROADMAP.md)
- [Pinned source audit](docs/evidence/2026-10-18-source-prepass.md)
- [Full documentation map](docs/README.md)

The first learned-skill comparison evaluated an authored physical controller, an imitation policy and a small recurrent residual policy on matched Jolt scenarios. The bounded improvements and directional failures are retained in the [learning evidence](docs/evidence/2026-10-19-t105.md). Graph and cellular candidates have separate declared experiments; their structure alone does not establish useful behaviour.

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
