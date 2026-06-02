# Campaign Report format and Campaign Pause Conditions

A Campaign Report is a human- and agent-readable status artifact for an autonomous Research Loop campaign. It summarizes campaign state at a review point; it does not replace per-Run Research Notes.

Recommended path: `campaign-reports/YYYY-MM-DD-status.md`.

## Markdown structure

Use these headings so later automation can parse and extend the scaffold:

```markdown
# Campaign Report: <campaign or Research Problem name>

## Summary
<One-paragraph status and decision context.>

## Current best Result
- Run: <run_id or none>
- Candidate Experiment: <candidate_id or none>
- Best-validation metric: <metric/value or unknown>
- Result artifacts: <paths/links>
- Why it is current best: <short rationale>

## Recent Runs
| Run | Candidate Experiment | Status | Key Result | Note |
| --- | --- | --- | --- | --- |

## Failures
| Run | Failure classification | Symptom | Follow-up |
| --- | --- | --- | --- |

## Pending Capability Requests
| Request | Status | Why it matters | Blocking? |
| --- | --- | --- | --- |

## Budget use
- Wall-clock budget used: <amount/unknown>
- Compute budget used: <amount/unknown>
- Storage used/risk: <amount/unknown>
- Remaining budget: <amount/unknown>

## Next hypothesis
<The next Experiment Proposal direction or the reason no next hypothesis is ready.>

## Pause recommendation
- Pause condition: <approved value or none>
- Human decision needed: <yes/no and details>
```

The required summary areas are: current best Result, recent Runs, failures, pending Capability Requests, budget use, and next hypothesis.

## Recording a Campaign Report

After writing the artifact, record it in the Research Ledger:

```bash
ml-autoresearch record-campaign-report \
  --report-path campaign-reports/2026-05-10-status.md \
  --ledger-path research-ledger.jsonl
```

The command/API records a `campaign_report_written` event containing `report_path`.

## Campaign Pause Conditions

`campaign_paused` events must use this approved vocabulary in `reason`:

- `budget_exhausted` — the Wall-Clock Budget Policy or campaign-level compute budget is spent.
- `repeated_failures` — multiple recent Runs failed for non-resource reasons and need review.
- `repeated_resource_failures` — repeated Resource Failures suggest infrastructure or budget settings need review.
- `stalled_research_progress` — recent Results are not improving enough to justify automatic continuation.
- `too_many_pending_capability_requests` — the campaign is blocked or distorted by accumulated pending Capability Requests.
- `storage_risk` — artifacts or logs risk exceeding available storage or retention policy.
- `scheduled_check_in` — a planned human review point has been reached.

Record a pause with an optional report link:

```bash
ml-autoresearch pause-campaign \
  --reason scheduled_check_in \
  --report-path campaign-reports/2026-05-10-status.md \
  --ledger-path research-ledger.jsonl
```

The command/API records a `campaign_paused` event containing `reason` and, when available, `report_path`.

## Resuming after human review

After a human resolves the pause condition, record a resume event before starting another autonomous iteration:

```bash
ml-autoresearch resume-campaign \
  --reason human_review_complete \
  --report-path campaign-reports/2026-06-01-resume.md \
  --ledger-path research-ledger.jsonl
```

The command/API records a `campaign_resumed` event. New Autonomy Step prompts explicitly tell the agent not to treat earlier `scheduled_check_in` or resolved capability-request pause recommendations as active blockers when a newer resume event exists.
