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

Use `ml-autoresearch-agent list-runs`, `ml-autoresearch-agent run-summary` / `ml-autoresearch-agent get-run-summary`, and `ml-autoresearch-agent get-best-runs` to inspect status, metrics, best-validation Result, prediction samples, and logs. Omit `--runs-root` during normal autonomous operation; agent-safe observation defaults to `/history/runs`. Use `ml-autoresearch-agent list-batches` and `ml-autoresearch-agent batch-summary` for Experiment Batch history; these default to `/history/batches`. Preserve the distinction between final metrics and best-validation metrics. A host operator may use `ml-autoresearch run-status --run-id <id>` to observe Managed Run Execution and `ml-autoresearch reconcile-run --run-id <id>` to terminalize the same existing Run. After caller disconnection, never submit or launch the Candidate again; retain the stable Run ID and request status/reconciliation through the Harness. If status is failed, rejected, smoke_failed, or scientifically poor, send observations to failure classification.

## Guardrails

- No covert workarounds: if the Candidate Experiment Contract blocks an idea, create a Capability Request instead of bypassing it.
- No direct Harness modifications during autonomous operation; changes require separate human-supervised work.
- No direct Research Ledger edits; use Harness-owned CLI/API commands.
- No arbitrary filesystem access; use only documented run, candidate, note, request, report, and artifact paths.
- No network access from Candidate Experiment code and no agent-driven runtime fetches for candidates.
- No runtime weight downloads; use Approved Weight Artifacts or a reviewed Pretrained Weight Request path.
