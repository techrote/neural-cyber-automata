# neural-cyber-automata

Research, training and evaluation for intelligent behaviour in [Cybersand](https://github.com/techrote/cybersand): physical actor control, movement and tool skills, human and model command sources, coordination, and neural cellular automata where local distributed behaviour helps the game.

**Stage: research and design prepass, 18 October 2026.** No trainer, trained checkpoint, physical learning bridge or runtime inference implementation exists in this repository. The programme is grounded in [ACT-004 / Cybersand #178](https://github.com/techrote/cybersand/issues/178) and the existing Thunderbowl control lab.

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

© 2026 techrote. All rights reserved.
