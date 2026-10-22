---
title: LEARN-JOLT-003 time-limit removal and CI handoff
kind: evidence
summary: Owner-authorized removal of operational wall-clock limits after PR 73, with CI followed by a stop.
owns: [evidence.jolt003-time-limit-revision]
verified: 2026-10-22
---

# LEARN-JOLT-003 time-limit removal and CI handoff

[D-020](../DECISIONS.md#d-020--remove-operational-wall-clock-limits-and-rerun-ci)
records the owner's instruction after [PR #73](https://github.com/techrote/neural-cyber-automata/pull/73):
remove all time limits, rerun CI, then stop until the owner reports completion.
The implementation starts from merged source `95873260c2411da320cac40d28644df2a45e6738`.

The new [configuration v2](../../configs/learning/jolt-learn-003-v2.json) declares
explicit null timing limits and separately binds this revision's authority.
The original allocation commission, 39 operation counters, fixed cases, seeds,
learning quantities and qualification thresholds remain unchanged. Historical
finite configuration and original evidence remain byte-identical.

The policy reaches capture, native startup/transport, closure, audit,
publication, executor/verifier cleanup and independent observation. Elapsed time
remains finite, ordered telemetry; explicit source-bound configuration selects
its interpretation. Missing fields, mixed finite/null limits, malformed records,
output overflow and incomplete cleanup still fail. The seven non-time resource
gates remain required; the two time-only gates explicitly become null.

Repository CI job timeout overrides and the CPU/pipeline synthetic diagnostic
wall guards are removed. Fixed sample counts remain. GitHub still enforces its
[six-hour maximum for hosted jobs](https://docs.github.com/en/actions/reference/limits).
The requested handoff is fresh CI on the changed source, without waiting for
results, merging or opening the experiment.

Focused offline regression passed across admission, clocks, executor/observer,
capture, native transport, streaming output and independent readback. This
includes simulated elapsed times beyond every historical cutoff and retained
negative controls for malformed evidence, byte overflow and incomplete cleanup.
The documentation check passed with zero errors and warnings. An independent
read-only integration review found no remaining operational time cap reachable
through configuration v2 and no missing/null timing-field ambiguity.

Full-worker/native memory and final-source admission are unresolved.
`learn-jolt-003-20261022-a1` remains unopened and unspent; this change records no
new physical learning evidence, qualified family or morphology release.
