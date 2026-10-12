---
title: Evidence records
kind: navigation
summary: Dated, unchangeable records of what was built, tested or measured, against which exact version. Each record is true only for the version it names.
owns: [evidence.index]
verified: 2026-10-13
---

# Evidence records

One file per event: `YYYY-MM-DD-short-slug.md`. Records are **never edited** after publication; a correction is a new record linking the old one. Training and benchmark runs are additionally listed in [RUNS.md](../RUNS.md).

## Record outline

```markdown
---
title: <What was checked> — <YYYY-MM-DD>
kind: evidence
summary: <one plain line: what this shows>
owns: []
verified: <YYYY-MM-DD> @ <commit>
---
Identity: commit / build / artifact hash / device (+ driver and toolkit versions)
Class: reviewed | built | tested | ran-on-target | measured | owner-accepted
Training ladder (if relevant): implemented | tested-on-CPU | ran-on-GPU <device> | trained | evaluated
Command(s) and settings: <exact>
Result: <numbers with units and conditions; "not measured" never 0>
Not covered: <what this does NOT show>
Related: T-### / D-### / R-###
```

## Records

None yet.
