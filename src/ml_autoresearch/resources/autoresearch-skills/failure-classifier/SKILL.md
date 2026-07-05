---
name: failure-classifier
description: Classify unsuccessful Runs and decide repair, new proposal, request, or pause.
---

# Failure Classifier

## Use

Use when a Run fails, is rejected, times out, or produces a bad Result. Apply the Run Failure Classification vocabulary and decide whether to use a Repair Candidate, new Experiment Proposal, Capability Request, or pause.

## Read first

- `CONTEXT.md` for Run, Result, Candidate Experiment, and Harness terms.
- `docs/run-lifecycle.md` for Run Failure Classification and Repair Candidate policy.
- `docs/candidate-experiment-contract.md` for repair lineage.

## Instructions

Prefer the Harness-recorded failure_classification when present. Approved classes are candidate_bug, contract_violation, resource_failure, harness_failure, bad_research_result, and unknown. A Repair Candidate is valid only for candidate bugs or contract issues that preserve hypothesis and Comparison Target.

When a successful Run from a substantially new architecture family underperforms the current best Result, classify the scientific outcome precisely. A first or lightly tuned family Run is usually a negative or mixed scout, not proof that the family is exhausted. Recommend abandoning the family only if there is a hard stop reason: contract violation, parameter/resource infeasibility, catastrophic metric gap, unstable training, implementation defect, or diagnostics showing the family cannot address the campaign failure modes. Otherwise, decide whether the evidence justifies a bounded family-development continuation, a different family scout, or a campaign-level pause.

## Guardrails

- Do not turn "failed to beat the mature incumbent" into "this architecture family is disproven" unless the proposal's hard stop criteria are met.
- No covert workarounds: if the Candidate Experiment Contract blocks an idea, create a Capability Request instead of bypassing it.
- No direct Harness modifications during autonomous operation; changes require separate human-supervised work.
- No direct Research Ledger edits; use Harness-owned CLI/API commands.
- No arbitrary filesystem access; use only documented run, candidate, note, request, report, and artifact paths.
- No network access from Candidate Experiment code and no agent-driven runtime fetches for candidates.
- No runtime weight downloads; use Approved Weight Artifacts or a reviewed Pretrained Weight Request path.
