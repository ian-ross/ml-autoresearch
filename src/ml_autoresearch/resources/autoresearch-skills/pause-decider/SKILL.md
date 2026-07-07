---
name: pause-decider
description: Decide whether an autonomous campaign has a real blocker requiring operator review.
---

# Pause Decider

## Use

Use before starting another Autonomous Research Iteration and whenever budget, failure, storage, capability, or progress risk appears. Identify real blockers and recommend operator review using the approved Campaign Pause Conditions vocabulary; do not pause the campaign yourself.

## Read first

- `CONTEXT.md` for Research Loop and Human-Guided Research Iteration language.
- `docs/campaign-report-format.md` for Campaign Pause Conditions and approved vocabulary.
- `docs/campaign-autonomy-architecture.md` for `campaign_paused` events.

## Instructions

Recommend operator review for budget_exhausted, repeated_failures, repeated_resource_failures, stalled_research_progress, too_many_pending_capability_requests, storage_risk, or scheduled_check_in. Create or update a Campaign Report when useful. Do not run `pause-campaign`; actual campaign pausing is an operator-level control outside the Agent Control Boundary.

Do not treat local exhaustion of the current approach as a pause condition. If the current line looks stalled, broaden the search frontier by proposing work in a different architecture family, training policy, data policy, loss function, augmentation strategy, calibration/thresholding approach, hard-negative mining strategy, or preprocessing path. Use a Capability Request only when a real resource, contract, data, infrastructure, or policy blocker prevents that next work.

## Guardrails

- No covert workarounds: if the Candidate Experiment Contract blocks an idea, create a Capability Request instead of bypassing it.
- No direct Harness modifications during autonomous operation; changes require separate human-supervised work.
- No direct Research Ledger edits; use Harness-owned CLI/API commands.
- No arbitrary filesystem access; use only documented run, candidate, note, request, report, and artifact paths.
- No network access from Candidate Experiment code and no agent-driven runtime fetches for candidates.
- No runtime weight downloads; use Approved Weight Artifacts or a reviewed Pretrained Weight Request path.
