---
name: candidate-implementer
description: Implement Candidate Experiments under the Candidate Experiment Contract.
---

# Candidate Implementer

## Use

Use after an Experiment Proposal is approved to create or repair a Candidate Experiment under the Candidate Experiment Contract, including `manifest.yaml`, `model.py`, optional helpers, README, and required autonomous-mode `PROPOSAL.md`.

## Read first

- `CONTEXT.md` for Candidate Experiment and Harness vocabulary.
- `docs/candidate-experiment-contract.md` for allowed files, manifest fields, repair lineage, and `PROPOSAL.md`.
- `docs/harness-capabilities.md` for implemented versus planned capability.
- ADR-0001, ADR-0002, and ADR-0003.

## Instructions

Keep Candidate Experiment code narrow: model architecture only plus allowed manifest choices. Do not add custom data loaders, training loops, shell scripts, Dockerfiles, dataset paths, MLflow writes, or weight-fetching code. For Repair Candidate work, preserve the original hypothesis and Comparison Target or return to proposal writing.

## Guardrails

- No covert workarounds: if the Candidate Experiment Contract blocks an idea, create a Capability Request instead of bypassing it.
- No direct Harness modifications during autonomous operation; Harness changes require separate human-supervised work.
- No direct Research Ledger edits; use Harness-owned CLI/API commands.
- No arbitrary filesystem access; use documented run, candidate, note, request, report, and artifact paths only.
- No network access from Candidate Experiment code and no agent-driven runtime fetches for candidates.
- No runtime weight downloads; use Approved Weight Artifacts or a reviewed Pretrained Weight Request path.
