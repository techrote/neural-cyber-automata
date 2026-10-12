---
title: Current state
kind: state
summary: What exists and works today, and nothing else. Plans live in the roadmap; ideas live in the ideas page.
owns: [state.summary]
verified: 2026-10-13
---

# Current state

Only things on the default branch, checked against code or evidence, belong here. Untagged lines are `[CURRENT]`. No future tense: anything not built is in [ROADMAP.md](ROADMAP.md) or [IDEAS.md](IDEAS.md).

## Summary

Nothing is built yet. The repository holds planning documents, the documentation checker and its CI workflow only.

## By area

### Documentation tooling

- Docs checker — [tools/check_docs.py](../tools/check_docs.py), run by [.github/workflows/docs.yml](../.github/workflows/docs.yml) on every push and pull request.
- Issue and pull-request templates — [.github/](../.github/pull_request_template.md).

### Training, toy world, runtime, cybersand bridge

- No code exists for any of these.

## Not checked

- No GPU, driver or Python environment has been checked for this repository on any machine.
