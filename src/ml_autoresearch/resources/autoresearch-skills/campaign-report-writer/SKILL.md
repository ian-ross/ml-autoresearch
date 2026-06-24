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

A Campaign Report must use the exact headings from `docs/campaign-report-format.md`: `## Current best Result`, `## Recent Runs`, `## Failures`, `## Pending Capability Requests`, `## Budget use`, `## Next hypothesis`, and `## Pause recommendation`. Do not replace them with synonyms such as `## Evidence checked`, `## Current best retained`, or `## Decision`; put that content under the required headings.

The `## Pause recommendation` section must include one machine-readable line written exactly as either `- Pause condition: none` or `- Pause condition: <approved_value>`, where `<approved_value>` is one of `budget_exhausted`, `repeated_failures`, `repeated_resource_failures`, `stalled_research_progress`, `too_many_pending_capability_requests`, `storage_risk`, or `scheduled_check_in`. Do not add punctuation or prose to that line; put explanation on the `- Human decision needed:` line.

## Guardrails

- No covert workarounds: if the Candidate Experiment Contract blocks an idea, create a Capability Request instead of bypassing it.
- No direct Harness modifications during autonomous operation; Harness changes require separate human-supervised work.
- No direct Research Ledger edits; use Harness-owned CLI/API commands.
- No arbitrary filesystem access; use documented run, candidate, note, request, report, and artifact paths only.
- No network access from Candidate Experiment code and no agent-driven runtime fetches for candidates.
- No runtime weight downloads; use Approved Weight Artifacts or a reviewed Pretrained Weight Request path.
