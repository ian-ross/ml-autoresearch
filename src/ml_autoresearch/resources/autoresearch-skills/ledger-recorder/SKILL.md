---
name: ledger-recorder
description: Record Research Ledger events through Harness-owned commands.
---

# Ledger Recorder

## Use

Use whenever an autonomous step needs an auditable Research Ledger event.

## Read first

- `CONTEXT.md` for Research Ledger-adjacent vocabulary.
- `docs/campaign-autonomy-architecture.md` for event types and `record-research-event`.
- `docs/campaign-report-format.md` for report and pause commands.

## Instructions

The Research Ledger is append-only and Harness-owned. Use `ml-autoresearch record-research-event`, `record-campaign-report`, `pause-campaign`, or corresponding APIs. Do not edit `research-ledger.jsonl` directly. Include paths and IDs needed to trace proposals, candidates, Runs, Results, Research Notes, Evaluation Requests, Capability Requests, Campaign Reports, and pauses.

## Guardrails

- No covert workarounds: if the Candidate Experiment Contract blocks an idea, create a Capability Request instead of bypassing it.
- No direct Harness changes during autonomous operation; changes require separate human-supervised work.
- No direct Research Ledger edits; use Harness-owned CLI/API commands.
- No arbitrary filesystem access; use only documented run, candidate, note, request, report, and artifact paths.
- No network access from Candidate Experiment code and no agent-driven runtime fetches for candidates.
- No runtime weight downloads; use Approved Weight Artifacts or a reviewed Pretrained Weight Request path.
