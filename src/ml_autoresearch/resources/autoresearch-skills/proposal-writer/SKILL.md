---
name: proposal-writer
description: Draft Experiment Proposals before Candidate Experiment implementation.
---

# Proposal Writer

## Use

Use when a Research Loop needs an Experiment Proposal for the next Candidate Experiment. State the hypothesis, Comparison Target, expected effect, implementation sketch, constraints, budget, success criteria, and fallback decision.

## Read first

- `CONTEXT.md` for Research Problem, Candidate Experiment, Run, Result, and Experiment Proposal terms.
- `docs/candidate-experiment-contract.md` for required `PROPOSAL.md` sections.
- `research-notes/README.md` and recent Research Notes for prior Results.

## Instructions

Write a concise proposal that can be copied into `PROPOSAL.md`. Tie every proposed variation to Harness-owned contract features. If the hypothesis requires unavailable authority, stop and request a Capability Request rather than weakening safety.

## Guardrails

- No covert workarounds: if the Candidate Experiment Contract blocks an idea, create a Capability Request instead of bypassing it.
- No direct Harness changes during autonomous operation; changes require separate human-supervised work.
- No direct Research Ledger edits; use Harness-owned CLI/API commands.
- No arbitrary filesystem access; use only documented run, candidate, note, request, report, and artifact paths.
- No network access from Candidate Experiment code and no agent-driven runtime fetches for candidates.
- No runtime weight downloads; use Approved Weight Artifacts or a reviewed Pretrained Weight Request path.
