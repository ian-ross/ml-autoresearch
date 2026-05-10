---
name: evaluation-request-writer
description: Create bounded Evaluation Requests for Post-Run Evaluations.
---

# Evaluation Request Writer

## Use

Use when a completed Run needs a bounded Harness-owned diagnostic before the next Research Loop decision. Write an Evaluation Request and run it with `run-post-run-evaluation` only after validation.

## Read first

- `CONTEXT.md` for Run, Result, and Research Loop terms.
- `docs/evaluation-request-format.md` for schema, modes, parameters, and artifact locations.
- `docs/design/autonomous-research-campaign-plan.md` for ledger events.

## Instructions

State the diagnostic question and expected decision impact. Keep parameters within documented bounds. Do not create ad hoc diagnostics outside approved modes such as threshold_sweep and failure_bucket_review.

## Guardrails

- No covert workarounds: if the Candidate Experiment Contract blocks an idea, create a Capability Request instead of bypassing it.
- No direct Harness modifications during autonomous operation; Harness changes require separate human-supervised work.
- No direct Research Ledger edits; use Harness-owned CLI/API commands.
- No arbitrary filesystem access; use documented run, candidate, note, request, report, and artifact paths only.
- No network access from Candidate Experiment code and no agent-driven runtime fetches for candidates.
- No runtime weight downloads; use Approved Weight Artifacts or a reviewed Pretrained Weight Request path.
