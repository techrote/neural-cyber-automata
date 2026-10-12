---
title: Ideas
kind: ideas
summary: Everything optional, undecided, postponed, rejected or abandoned, kept so it is neither forgotten nor mistaken for a plan.
owns: [ideas.register]
verified: 2026-10-13
---

# Ideas

> **Nothing on this page is planned, approved or built.** Do not implement anything here unless an issue explicitly says so. An idea that gets adopted becomes a task or decision with its own ID, and its entry here becomes `[SUPERSEDED by T-###]`.

## Open

### I-003 — A strategist character commands nanites through signal fields

[IDEA] Raised in discussion 2026-10-13. A small language model character in a host game issues commands such as "defend sector 4", which become signal-field writes the cells respond to (D-005). Would need: M4 complete and the strategist project itself.

### I-004 — Behaviour catalogue beyond the first swarm

[IDEA] Raised in discussion 2026-10-13: self-repair of a structure; build a bridge or plug a breach where a signal points; consume a target material; scatter from explosions; split into groups; quarantine a region. Would need: M2 working, then one roadmap task per behaviour.

### I-005 — GPU compute-shader runtime inside the game

[IDEA] Raised by agent 2026-10-13. Run nanite updates on the player's GPU instead of the CPU, to allow far larger swarms. Would need: M3 CPU cost numbers showing the CPU is the limit; a decision on how this keeps determinism (D-004).

### I-006 — Nanites in cybersand's Web build

[IDEA] Raised by agent 2026-10-13. The Web build runs cellular simulation on a single worker, so nanite counts would need a lower cap there. Would need: M3 cost numbers and an owner decision on Web support.

### I-008 — Use the trained NCAs for art and textures

[IDEA] Raised in discussion 2026-10-13. Self-organising textures and animated patterns as visual assets or effects, independent of gameplay. Would need: nothing beyond M1; purely optional.

### I-009 — Many parallel experiments on the K80 lanes

[IDEA] Raised in discussion 2026-10-13. Each GPU die runs a different seed, setting or behaviour overnight (parameter sweeps, population-based training). Would need: M1 lane benchmark numbers (T-007).

## Closed (kept so they are not re-proposed)

### I-001 — Rule: do not edit cybersand until M4 is approved

[SUPERSEDED by C-004] Accepted by the owner 2026-10-13.

### I-002 — Nanites react to macro world events

[SUPERSEDED by T-018] Raised by owner 2026-10-13 ("distributed pseudo-intelligences which react to macro world events maybe?"); planned 2026-10-13 ("all as guessed") on top of D-005.

### I-007 — Store trained weights outside Git

[SUPERSEDED by C-005] Accepted by the owner 2026-10-13.
