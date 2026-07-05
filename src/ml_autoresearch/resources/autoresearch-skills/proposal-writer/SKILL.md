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

For architecture-family proposals, explicitly classify the proposal as one of:

- **promotion candidate** — intended to beat or credibly tie the current best Result;
- **family scout** — first or early probe of a substantially new family, not expected to beat a heavily tuned incumbent immediately;
- **family-development continuation** — a controlled follow-up within a family after a scout.

For a family scout or family-development continuation, include continuation criteria that are not limited to beating the current best Result. Acceptable criteria can include metric distance from the incumbent, improvement over the family baseline, secondary-metric or failure-bucket strengths, learning-curve behavior, resource efficiency, or diagnostically useful differences. If proposing the first Run in a substantially new family, state the bounded follow-up budget or the hard stop conditions that would justify abandoning the family.

## Guardrails

- Do not require an untuned or lightly tuned family scout to beat a mature incumbent before any further family development is allowed.
- Compare mature candidates to the current best Result for promotion; compare scouts to their family baselines and other scouts when deciding research allocation.
- No covert workarounds: if the Candidate Experiment Contract blocks an idea, create a Capability Request instead of bypassing it.
- No direct Harness modifications during autonomous operation; changes require separate human-supervised work.
- No direct Research Ledger edits; use Harness-owned CLI/API commands.
- No arbitrary filesystem access; use only documented run, candidate, note, request, report, and artifact paths.
- No network access from Candidate Experiment code and no agent-driven runtime fetches for candidates.
- No runtime weight downloads; use Approved Weight Artifacts or a reviewed Pretrained Weight Request path.
