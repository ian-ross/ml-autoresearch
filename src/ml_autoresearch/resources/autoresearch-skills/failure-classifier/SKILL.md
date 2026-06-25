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

## Guardrails

- No covert workarounds: if the Candidate Experiment Contract blocks an idea, create a Capability Request instead of bypassing it.
- No direct Harness changes during autonomous operation; changes require separate human-supervised work.
- No direct Research Ledger edits; use Harness-owned CLI/API commands.
- No arbitrary filesystem access; use only documented run, candidate, note, request, report, and artifact paths.
- No network access from Candidate Experiment code and no agent-driven runtime fetches for candidates.
- No runtime weight downloads; use Approved Weight Artifacts or a reviewed Pretrained Weight Request path.
