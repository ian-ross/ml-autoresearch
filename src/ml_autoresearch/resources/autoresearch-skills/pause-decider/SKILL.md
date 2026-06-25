---
name: pause-decider
description: Decide whether an autonomous campaign must pause for human review.
---

# Pause Decider

## Use

Use before starting another Autonomous Research Iteration and whenever budget, failure, storage, capability, or progress risk appears. Apply Campaign Pause Conditions using the approved vocabulary.

## Read first

- `CONTEXT.md` for Research Loop and Human-Guided Research Iteration language.
- `docs/campaign-report-format.md` for Campaign Pause Conditions and approved vocabulary.
- `docs/campaign-autonomy-architecture.md` for `campaign_paused` events.

## Instructions

Pause for budget_exhausted, repeated_failures, repeated_resource_failures, stalled_research_progress, too_many_pending_capability_requests, storage_risk, or scheduled_check_in. Create or update a Campaign Report when useful, record with `pause-campaign`, and stop until human review.

## Guardrails

- No covert workarounds: if the Candidate Experiment Contract blocks an idea, create a Capability Request instead of bypassing it.
- No direct Harness changes during autonomous operation; changes require separate human-supervised work.
- No direct Research Ledger edits; use Harness-owned CLI/API commands.
- No arbitrary filesystem access; use only documented run, candidate, note, request, report, and artifact paths.
- No network access from Candidate Experiment code and no agent-driven runtime fetches for candidates.
- No runtime weight downloads; use Approved Weight Artifacts or a reviewed Pretrained Weight Request path.
