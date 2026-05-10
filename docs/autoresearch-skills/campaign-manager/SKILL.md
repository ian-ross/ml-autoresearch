---
name: campaign-manager
description: Orchestrate one Autonomous Research Iteration for ML Autoresearch campaigns.
---

# Campaign Manager

## Use

Use this top-level skill to run or review exactly one Autonomous Research Iteration. It coordinates focused skills and stops when a Campaign Pause Conditions decision requires human review.

## Read first

- `CONTEXT.md` for project vocabulary.
- `docs/design/autonomous-research-campaign-plan.md` for Research Ledger events and loop state.
- `docs/campaign-report-format.md` for Campaign Pause Conditions and Campaign Reports.

## Autonomous Research Iteration

1. Review current Research Ledger, recent Research Notes, Campaign Reports, pending Capability Requests, and current best Result.
2. If a pause condition is already met, delegate to `../pause-decider/SKILL.md` and stop.
3. If a new hypothesis is ready, delegate to `../proposal-writer/SKILL.md`.
4. Implement only the approved Candidate Experiment through `../candidate-implementer/SKILL.md`.
5. Submit/run through Harness-owned commands, then delegate observation to `../run-observer/SKILL.md`.
6. If the Run failed or regressed, delegate to `../failure-classifier/SKILL.md` before deciding repair, new proposal, or pause.
7. If bounded diagnostics are needed, delegate to `../evaluation-request-writer/SKILL.md`.
8. Capture outcomes with `../research-note-writer/SKILL.md`.
9. Record auditable events with `../ledger-recorder/SKILL.md`.
10. If the contract blocks a hypothesis, delegate to `../capability-request-writer/SKILL.md`.
11. At review intervals or before pausing, delegate to `../campaign-report-writer/SKILL.md` and then `../pause-decider/SKILL.md`.

Do not continue automatically after a pause decision. Require human review before unattended use or before installing this review-only skill set.

## Guardrails

- No covert workarounds: if the Candidate Experiment Contract blocks an idea, create a Capability Request instead of bypassing it.
- No direct Harness modifications during autonomous operation; Harness changes require separate human-supervised work.
- No direct Research Ledger edits; use Harness-owned CLI/API commands.
- No arbitrary filesystem access; use documented run, candidate, note, request, report, and artifact paths only.
- No network access from Candidate Experiment code and no agent-driven runtime fetches for candidates.
- No runtime weight downloads; use Approved Weight Artifacts or a reviewed Pretrained Weight Request path.
