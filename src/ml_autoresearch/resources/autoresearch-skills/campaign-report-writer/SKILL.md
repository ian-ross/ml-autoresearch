---
name: campaign-report-writer
description: Write Campaign Reports at review points.
---

# Campaign Report Writer

## Use

Use at scheduled checks, before pauses, or when humans need a compact campaign status. Write a Campaign Report and record it with `record-campaign-report`; if pausing, use `pause-campaign` after the report.

## Read first

- `CONTEXT.md` for Research Loop, Result, Run, and Capability Request terms.
- `docs/campaign-report-format.md` for required headings and Campaign Pause Conditions.
- `docs/design/autonomous-research-campaign-plan.md` for ledger recording.

## Instructions

Summarize current best Result, recent Runs, failures, pending Capability Requests, budget use, next hypothesis, and pause recommendation. Do not let the report replace per-Run Research Notes.

## Guardrails

- No covert workarounds: if the Candidate Experiment Contract blocks an idea, create a Capability Request instead of bypassing it.
- No direct Harness modifications during autonomous operation; Harness changes require separate human-supervised work.
- No direct Research Ledger edits; use Harness-owned CLI/API commands.
- No arbitrary filesystem access; use documented run, candidate, note, request, report, and artifact paths only.
- No network access from Candidate Experiment code and no agent-driven runtime fetches for candidates.
- No runtime weight downloads; use Approved Weight Artifacts or a reviewed Pretrained Weight Request path.
