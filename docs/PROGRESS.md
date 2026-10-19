---
title: Implementation progress checkpoints
kind: evidence
summary: Timestamped checkpoints for the commissioned ten-issue implementation programme.
owns: [evidence.programme-progress]
verified: 2026-10-19
---

# Implementation progress

## 2026-10-19 00:09 UTC — Start

The owner commissioned all ten active issues, T-101 through T-110, with repository
checkpoints every 30 minutes during execution. Baseline:
`95d8788686980974d4f849e892c33f531d992fe7`. No overlapping open PRs.
Historical issues #2–#9 remain superseded, not retrospectively implemented.

Package/contracts, host transport and authored-controller work have separate
owners. Each issue uses a separate branch and PR with final-head checks.

## 2026-10-19 00:21 UTC — Host dependency refresh

Cybersand ACT-004 is present on its accepted main at
`f2e34cf9b6fbaa5d9dedf9aa2cbc721a87472497`. Its existing owner API supports a
persistent external adapter from this repository. The initial real Rapier smoke
advanced a five-part Ripper with repeats of 6, 6 and 1 ticks and stopped at the
declared horizon. This is an implementation smoke; broader conformance and
policy acceptance have separate evidence records.

The first programme checkpoint was also posted on issue #11. Contracts, immutable
run records and Jolt/Vector feedback controllers are under implementation and
review. No trained policy, full issue completion or owner feel acceptance is
claimed at this checkpoint.

## 2026-10-19 00:39 UTC — First 30-minute checkpoint

T-101 is merged through PR #22 at `9e6da91ff520e27bc78d11dd8196844a9fa37ff3`.
Its 30 CPU tests, strict documentation check and both hosted operating-system
checks passed. Windows uses the available Python 3.12.10 build.

T-102 has real host conformance evidence; independent review found malformed
reset, delayed final-state handling and blocking-output risks. Those fixes precede
acceptance. T-103 has 32 unit checks and 18 recorded Jolt/Vector pilots (1,080
decisions), with full per-tick receipts and matched pose replays. Final host freeze
and replay are separate gates; Vector's later drift and an incomplete compliant
Jolt reversal are retained.

T-104 capture/replay is under implementation. The T-108 asynchronous core passes
12 unit tests including stale replies, immediate takeover and bounded hung work.
Learning, morphology, export and NCA experiments have no result at this checkpoint.
Paid model calls: zero. The same checkpoint is recorded on issue #12.

## 2026-10-19 01:08 UTC — implementation checkpoint

T-101, T-102 and T-103 are accepted through PRs #22, #23 and #24. Their final
heads passed all applicable hosted checks, and merged trees were verified. The
accepted-adapter authored screen reproduced 18 earlier final poses and goal
arrival ticks over 1,080 decisions; the reversal failure and Vector drift remain
visible. Immutable traces and matched pose-replay clips are preserved outside Git.

T-104 is independently reviewed and staged: eleven synthetic and five actual
host checks, three compatible capture/replays with 296 base receipts, 49
imitation-eligible rows and no PPO rows. A frozen committed-candidate repeat is
next. T-105 has a predeclared three-seed, 13,000-decision/30-minute screen; tests
are correcting learned-controller lifecycle fences before optimization.

T-107 has actual mounted-effect, recoil, detach and compound-task probes under
test. T-108's independent review found owner-control-generation and malformed
provider-envelope paths being corrected before actual delay qualification.
T-110 is declaring one per-part graph experiment. T-106 and T-109 remain gated
on a useful trained/evaluated candidate. No trained result, paid model call,
purchased compute or Cybersand source change is claimed at this checkpoint.
