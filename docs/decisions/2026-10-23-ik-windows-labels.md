---
title: Windows IK label compatibility continuation
kind: decisions
summary: Owner-authorized narrow retained-label repair and one conditionally admitted TITAN batch.
owns: [decisions.ik_windows_labels]
verified: 2026-10-23
---

# Windows continuation — Narrow Windows retained-label compatibility and one gated TITAN batch

[ACCEPTED] 2026-10-23 by owner for #54 / PR #78: fix the Windows retained-label
compatibility refusal while preserving the original corpus and every admission
gate, rerun CPU preparation, obtain two independent TITAN-specific admissions,
then execute and independently verify exactly one seed-41011 batch.

Use the original byte-authenticated training/tuning rows. The compatibility
inventory binds exact original and Windows replay label digests; it supplies no
replacement labels or general numeric tolerance. The owner's continuation also commissions a narrow task-replay compatibility
inventory, resolution of PR #78's main conflict, CPU preparation and CI, then
two independent prospective TITAN admissions and the single verified batch.
Exact ordered-task identity is checked against either equal replay bytes or an
explicit original/Windows whole-task digest pair after original corpus byte
authentication. Original task inputs, registry, training-only normalization,
sealed-final protections, seeds and finite allocations remain unchanged. V100 initialization is prohibited. A failed CPU
gate blocks the declaration, prospective admissions and CUDA invocation; source
reviews cannot substitute for execution admission. No further owner authorization
for the first batch is needed once all gates genuinely pass.


© 2026 techrote. All rights reserved.
