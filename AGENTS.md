---
title: Agent instructions — neural-cyber-automata
kind: navigation
summary: Where any agent starts in neural-cyber-automata, which documents decide what, and the few rules that always apply.
owns: [agents.entry]
verified: 2026-10-13
---

# AGENTS.md — neural-cyber-automata

Neural cellular automata (NCA) training, toy world and compact runtime for living materials (nanite swarms and similar) in the cybersand engine. Personal project of techrote.

## Read first, in this order

1. [docs/README.md](docs/README.md): the map; find the pages your task needs.
2. [docs/STATE.md](docs/STATE.md): what exists **now**. Untagged lines here are current; nothing else is.
3. [docs/CONSTRAINTS.md](docs/CONSTRAINTS.md): what is required, forbidden or preferred.
4. [docs/DECISIONS.md](docs/DECISIONS.md): what has been decided and why. Most design entries start as `[PROPOSED]`.
5. Your issue, all its comments, and its read-set.
6. Live state: default branch, open PRs touching your write-set, CI.

## How to read status

Every claim carries a status by its location and tag. This section and [tools/check_docs.py](tools/check_docs.py) are the in-repo copy of the documentation rules.

- `[CURRENT]` `[PARTIAL]` `[DEFECT]`: exists now.
- `[TARGET D-###]` `[PLANNED T-###]`: accepted or scheduled, **not built**.
- `[PROPOSED]` `[IDEA]` `[DEFERRED]` `[REJECTED]` `[ABANDONED]`: **not** planned or approved. Never implement these unless your issue says so.
- `[FORBIDDEN until: …]`: do not do this. Only the owner lifts it.
- Research findings: `[ESTABLISHED]` `[SUPPORTED]` `[TENTATIVE]` `[REFUTED]` `[OPEN]`. A finding never becomes a requirement by itself; only an accepted decision citing it can.

When sources disagree:

- code on the default branch decides what *is*;
- the owner's recorded decisions and constraints decide what *should be*;
- chat, issue comments and history decide nothing until written into these docs.

Record the conflict; never silently pick a side.

## Rules

- One issue → one branch → one PR. Merge only when checks pass on the final head; then verify the merge landed.
- The PR includes a **Docs impact** section. Behaviour changes update `docs/STATE.md` (or the owning page) in the same PR.
- Report evidence by class: reviewed / built / tested / ran-on-target / measured / owner-accepted. Never present a weaker one as a stronger one.
- Prefer new files behind named extension points ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)) over edits scattered through existing code.
- Run `python tools/check_docs.py` before pushing docs changes.
- Messages to the owner: plain language, direct links, one card at a time, pros/cons for any choice, evidence class stated.

## Project-specific rules

- **Proposed design stays proposed.** D-002 to D-008 are `[PROPOSED]`. A task that depends on one waits until the owner accepts it; say so on the issue and work on something independent.
- **Training evidence ladder** (finer than the general classes): `implemented` → `tested-on-CPU` → `ran-on-GPU <device>` → `trained` (checkpoint exists; config and seed recorded) → `evaluated` (scored on a versioned behaviour suite). A missing GPU is a blocker, never evidence of success.
- **Every training or benchmark run** gets a row in [docs/RUNS.md](docs/RUNS.md): commit, device, environment, config hash, seed, result. No row, no claim.
- **Public repo** (C-003): no personal context, machine nicknames, local paths or credentials in docs, issues or logs.
- **cybersand is read-only from here.** Read its docs and code freely; never edit it from a task in this repo (I-001, awaiting the owner).
- **Copyright line only** (C-001). If any third-party code is copied or ported, record its source and licence in [docs/research/SOURCES.md](docs/research/SOURCES.md) first.
