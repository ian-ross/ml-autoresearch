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
- `docs/campaign-autonomy-architecture.md` for ledger events.

## Instructions

State the diagnostic question and expected decision impact. Keep parameters within documented bounds. Treat Whole-Validation Failure Analysis as the umbrella Harness-owned diagnostic; request-gated modes such as `threshold_sweep` and `failure_bucket_review` select bounded parts. Do not create ad hoc diagnostics outside approved modes.

## Guardrails

- No covert workarounds: if the Candidate Experiment Contract blocks an idea, create a Capability Request instead of bypassing it.
- No direct Harness changes during autonomous operation; changes require separate human-supervised work.
- No direct Research Ledger edits; use Harness-owned CLI/API commands.
- No arbitrary filesystem access; use only documented run, candidate, note, request, report, and artifact paths.
- No network access from Candidate Experiment code and no agent-driven runtime fetches for candidates.
- No runtime weight downloads; use Approved Weight Artifacts or a reviewed Pretrained Weight Request path.
