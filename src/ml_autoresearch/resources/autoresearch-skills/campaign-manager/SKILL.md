---
name: campaign-manager
description: Orchestrate one Autonomous Research Iteration for ML Autoresearch campaigns.
---

# Campaign Manager

## Use

Use this top-level skill to run or review exactly one Autonomous Research Iteration. When invoked by an Autonomy Step, one Autonomy Step means one primary handoff outcome, then stop. It coordinates focused skills and escalates real Campaign Pause Conditions for operator review without directly pausing the campaign.

## Read first

- `CONTEXT.md` for project vocabulary.
- `docs/campaign-autonomy-architecture.md` for Research Ledger events and loop state.
- `docs/campaign-report-format.md` for Campaign Pause Conditions and Campaign Reports.

## Autonomous Research Iteration

1. Review current Research Ledger, recent Research Notes, Campaign Reports, pending Capability Requests, and current best Result.
2. If a real pause condition is already met, delegate to `../pause-decider/SKILL.md` and create the appropriate report or Capability Request for operator review.
3. If a new hypothesis is ready, decide whether it is a promotion candidate, a family scout, or a family-development continuation; delegate to `../proposal-writer/SKILL.md`.
4. Implement only the approved Candidate Experiment through `../candidate-implementer/SKILL.md`. When two to four related, resource-profiled variants form one controlled comparison, use `../experiment-batch-writer/SKILL.md` instead of separate Candidate handoffs.
5. Submit/run through Harness-owned commands, retain the returned stable Run ID, then delegate observation to `../run-observer/SKILL.md`. If the initiating caller disconnects or a Run remains non-terminal, request Harness status/reconciliation for that same Run; never relaunch the Candidate.
6. If the Run failed or regressed, delegate to `../failure-classifier/SKILL.md` before deciding repair, new proposal, broader frontier, or operator escalation.
7. If bounded diagnostics are needed, delegate to `../evaluation-request-writer/SKILL.md`.
8. Capture outcomes with `../research-note-writer/SKILL.md`.
9. Record auditable events with `../ledger-recorder/SKILL.md`.
10. If the contract blocks a hypothesis, delegate to `../capability-request-writer/SKILL.md`.
11. At review intervals or when a real blocker requires operator review, delegate to `../campaign-report-writer/SKILL.md` and then `../pause-decider/SKILL.md`.

The agent must not pause or terminate the campaign because it believes no promising experiments remain. If the current approach appears exhausted, broaden the search frontier by proposing experiments in a different architecture family, training policy, data policy, loss function, augmentation strategy, evaluation/calibration method, mining strategy, or preprocessing path. Actual `pause-campaign` authority belongs to the operator-facing CLI; use Campaign Reports or Capability Requests when human review is genuinely needed.

## Guardrails

- Use the current best Result for promotion decisions, not as the sole reason to abandon an immature architecture family after one scout.
- For substantially new architecture families, require either a bounded family-development plan or a documented hard stop reason before declaring the family exhausted.
- No covert workarounds: if the Candidate Experiment Contract blocks an idea, create a Capability Request instead of bypassing it.
- No direct Harness modifications during autonomous operation; changes require separate human-supervised work.
- No direct Research Ledger edits; use Harness-owned CLI/API commands.
- No arbitrary filesystem access; use only documented run, candidate, note, request, report, and artifact paths.
- No network access from Candidate Experiment code and no agent-driven runtime fetches for candidates.
- No runtime weight downloads; use Approved Weight Artifacts or a reviewed Pretrained Weight Request path.
