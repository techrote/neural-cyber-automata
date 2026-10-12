---
title: Research questions
kind: research
summary: The open questions this project's work is meant to answer, what would answer each one, and which roadmap task addresses it.
owns: [research.questions]
verified: 2026-10-13
---

# Research questions

A finding never becomes a requirement by itself; only an accepted decision citing it can. No findings exist yet.

### R-001 — Is the K80 a practical training lane for NCAs?

[OPEN] Why it matters: decides whether the K80s carry real work or stay spares (D-006, I-009). Answered by: the legacy environment reporting `sm_37` support (T-002) and the lane benchmark comparing K80 training speed with CPU and modern GPUs (T-007).

### R-002 — Can nanites that only output intents, under strict mass conservation, learn to swarm towards a signal?

[OPEN] Why it matters: D-002 makes physics authoritative, which removes the network's freedom to simply redraw the world. If learning fails under that restriction, the design needs revisiting before anything depends on it. Answered by: T-012 on behaviour suite v1.

### R-003 — Do learned behaviours survive host-engine-like disorder?

[OPEN] Why it matters: real engines (cybersand is the motivating example) update cells in their own orders with their own movement rules, not as a calm synchronous grid. Answered by: robustness scores under random update order, noise and damage in T-012, then a host test scene in M4.

### R-004 — What does a nanite update cost on a desktop CPU?

[OPEN] Why it matters: sets how many active cells a host scene can afford next to its own physics, which in cybersand is already the main CPU cost. Answered by: T-016 (updates per second per core, integer runtime).

### R-005 — How much behaviour quality is lost by integer quantisation?

[OPEN] Why it matters: D-004 trades exactness of arithmetic for determinism. Answered by: suite v1 scores before and after quantisation in T-014.

### R-006 — How far does a toy-world behaviour transfer into a real engine?

[OPEN] Why it matters: D-007 trains away from the real engine. Answered by: M4, comparing suite-style scenarios in the toy world and in a host test scene.
