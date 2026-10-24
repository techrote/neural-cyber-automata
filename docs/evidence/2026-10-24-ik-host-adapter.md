---
title: IK-CUDA-006 joint-target adapter conformance
kind: evidence
summary: Exact-head NCA software and actual-host conformance for the separate Cybersand physical_joint_target/v1 candidate.
owns: [ik.host-handoff]
verified: 2026-10-24
---

# IK-CUDA-006 joint-target adapter conformance

This record belongs to [IK-CUDA-006 / #55](https://github.com/techrote/neural-cyber-automata/issues/55)
and NCA [PR #87](https://github.com/techrote/neural-cyber-automata/pull/87).
It qualifies the NCA adapter boundary against one exact Cybersand ACT-IK-001
candidate. It does not qualify a learned IK model, CPU export, owner feel,
morphology transfer or arbitrary host perturbations.

## Bound source and runtime

The exact NCA implementation head exercised by the retained hosted run was
`77f94d40163dee549e26ad608fb5a162151d7da7`.

The joint-target host setup independently authenticated:

- Cybersand source `993a47044119dedd54db8f1327e957949c478bb3`;
- Cybersand tree `ab7ff12af22606926566b408c8b417753c074b6e`;
- Godot 4.7 binary SHA-256 `f85bbc6b15e22416c7d797cd60b63286dd67b9cb13498847056c18520ae55a75`;
- Cybersand native SHA-256 `3bfe1679c6d0201203fcf99b6f53d7f50875a530611776787131d7dc1d5098c2`;
- Rapier SHA-256 `566e68ae53bfb2eb7018c8e7d0dd3727121bf696462039b780a33d6b8369c66a`;
- composite joint-host runtime SHA-256 `f651c83eb77f185f96d2cd54552807601add1ab6ad8675cb91248c3f963bf21b`.

The historical ACT003/ACT004 host pin remains unchanged. The new joint-target
pin is separate and therefore does not re-label earlier experiments.

## Hosted conformance

Pinned-host workflow run `40138244106`, job `121004271420`, completed
successfully on the exact NCA head above.

The complete host suite reported **1,292 tests, OK**, with 293 pre-existing
platform/feature-gated skips. The new
`test_actual_host_authority_receipts_contact_expiry_and_takeover` executed and
passed; it was not skipped.

The actual-host path covers:

- capability/source/runtime/profile negotiation before grant issuance;
- numerical-reference and learned-source use of identical joint IDs, limits and
  motor semantics;
- source/grant/goal/observation/generation binding and single-use sequences;
- foreign-incarnation refusal without revoking the legitimate owner;
- duplicate lookup without second application;
- real Rapier support/contact and non-zero bounded tracking residual;
- one receipt per consumed base tick with requested/admitted/limited/motor/motion
  and integrity separation;
- explicit unavailable force telemetry rather than inferred torque;
- confirmed cancellation without momentum reset, old-owner refusal and source
  takeover under a fresh generation;
- exclusive lease/original-observation expiry.

On the same exact head, documentation, Jolt memory envelope, Linux/Windows CPU
conformance and optional CPU learning conformance also passed.

## Cross-repository gate

This evidence proves NCA-side compatibility with the named Cybersand source and
runtime. Final cross-repository acceptance still requires Cybersand #210 / PR
#224 to accept that source through its own exact-head hosted gates and merge
process. The authoritative final disposition is therefore the pair of issue/PR
records, not this intermediate evidence note.

© 2026 techrote. All rights reserved.
