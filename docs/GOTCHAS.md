---
title: Research and integration gotchas
kind: gotchas
summary: Concrete source and design traps identified by the ACT-004 prepass.
owns: [gotchas.register]
verified: 2026-10-18
---

# Gotchas

## G-001 — Published links can describe unmerged plans

The original default branch contains only a README, while its docs live in bootstrap PR #1. Several old issues describe prerequisites as merged despite no implementation. Inspect trees and PR state before relying on status prose. Scope: 18 October source audit.

## G-002 — A six-tick decision cadence is not a six-tick expiry

The current helper requests a 12-tick lease and the v1 maximum is 18, anchored to the issued observation. Record decision period, consumed duration and timeout separately. Source: [Thunderbowl contract](https://github.com/techrote/cybersand/blob/b31a897b485a1913a67a02a39fb271495bacf027/docs/systems/thunderbowl-control-reference.md).

## G-003 — Sampled and physical host steps are different environments

The existing native sampled arena does not establish Rapier joint/contact or payload performance. Accelerating only one host clock also changes coupling. Use the [audit](evidence/2026-10-18-source-prepass.md) and explicit environment labels.

## G-004 — Telemetry can be unavailable or composite

Pinned reaction impulses include more than motor action; some multibody queries return zero because the reading is not implemented. This is not a reason to require an exact energy ledger. Record available response or authored effort proxies and mark unknown telemetry. Source: audit.

## G-005 — Cancel is not reset

Cancel/revoke old input at the next owner boundary, disarm charge and follow fallback/grip rules. Do not erase compliant loading or body velocity merely because the controller changes. Full episode reset reconstructs mechanical and policy state.

## G-006 — Cumulative reward is not interval reward

Sampled Thunderbowl snapshots carry cumulative rewards and terminal snapshots freeze. Take differences with episode identity; never repeatedly reward the same final snapshot. Stop repeated actions on the first terminal tick.
