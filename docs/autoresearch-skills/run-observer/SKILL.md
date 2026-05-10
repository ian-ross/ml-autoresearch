---
name: run-observer
description: Observe Runs and Results through Harness-owned artifacts and commands.
---

# Run Observer

## Use

Use after submission or execution to inspect a Run and summarize its Result without modifying artifacts.

## Read first

- `CONTEXT.md` for Run and Result terms.
- `docs/run-lifecycle.md` for statuses and artifact layout.
- `README.md` Inspecting local Runs section.

## Instructions

Use `list-runs`, `run-summary` / `get-run-summary`, and `get-best-runs` to inspect status, metrics, best-validation Result, prediction samples, and logs. Preserve the distinction between final metrics and best-validation metrics. If status is failed, rejected, smoke_failed, or scientifically poor, send observations to failure classification.

## Guardrails

- No covert workarounds: if the Candidate Experiment Contract blocks an idea, create a Capability Request instead of bypassing it.
- No direct Harness modifications during autonomous operation; Harness changes require separate human-supervised work.
- No direct Research Ledger edits; use Harness-owned CLI/API commands.
- No arbitrary filesystem access; use documented run, candidate, note, request, report, and artifact paths only.
- No network access from Candidate Experiment code and no agent-driven runtime fetches for candidates.
- No runtime weight downloads; use Approved Weight Artifacts or a reviewed Pretrained Weight Request path.
