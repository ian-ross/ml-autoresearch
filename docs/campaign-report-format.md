# Campaign Report format and Campaign Pause Conditions

A Campaign Report is a human- and agent-readable status artifact for an autonomous Research Loop campaign. It summarizes campaign state at review points; it does not replace per-Run Research Notes.

Recommended path: `campaign-reports/YYYY-MM-DD-status.md`.

Inside the Agent Control Boundary, agents must create Campaign Reports with `ml-autoresearch-agent create-campaign-report`. Agents must not hand-author reports with shell heredocs, direct file writes, the write tool, or ad-hoc edits; if the command is unavailable, they should stop instead of fabricating a report.

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
<The next Experiment Proposal direction or the reason no next hypothesis is ready. State whether the direction is a promotion candidate, a family scout, a family-development continuation, or no ready hypothesis. For architecture-family work, distinguish promotion criteria from continuation criteria.>

## Pause recommendation
- Pause condition: <approved value or none>
- Human decision needed: <yes/no and details>
```

The `Pause condition` line is machine-read. Write it exactly as `- Pause condition: none` or `- Pause condition: <approved_value>` using one approved value below. Do not add prose, punctuation, or explanation to that line; put explanation under `Human decision needed` or elsewhere in the section. A non-none value in an agent-authored Campaign Report is only an operator-review recommendation; it does not grant the agent authority to record `campaign_paused` or pause the campaign.

Required summary areas are: current best Result, recent Runs, failures, pending Capability Requests, budget use, and next hypothesis. The next hypothesis summary should not treat failure to beat the current best Result as sufficient by itself to abandon an immature architecture family; if the current line appears exhausted, it should identify a broader frontier such as another architecture family, training policy, data policy, loss function, augmentation strategy, calibration/evaluation method, mining strategy, or preprocessing path unless a real blocker requires operator review.

## Creating and recording a Campaign Report

Agents inside the Agent Control Boundary create the artifact with the scaffolded agent-safe command, passing Markdown content for each required section:

```bash
ml-autoresearch-agent create-campaign-report \
  --output campaign-reports/2026-05-10-status.md \
  --title "Ground-Camera Contrail Detection" \
  --summary "One-paragraph status." \
  --current-best-result "- Run: run_..." \
  --recent-runs "| Run | Candidate Experiment | Status | Key Result | Note |\n| --- | --- | --- | --- | --- |" \
  --failures "- none" \
  --pending-capability-requests "- none" \
  --budget-use "- unknown" \
  --next-hypothesis "Next direction or no hypothesis." \
  --human-decision-needed "no" \
  --pause-condition none
```

After ingesting or otherwise accepting the artifact, record it in the Research Ledger:

```bash
ml-autoresearch record-campaign-report \
  --report-path campaign-reports/2026-05-10-status.md \
  --ledger-path research-ledger.jsonl
```

The command/API records a `campaign_report_written` event with `report_path`.

## Campaign Pause Conditions

Campaign pausing is an operator-level control. The autonomous agent may recommend review in a Campaign Report or create a Capability Request for a real blocker, but `pause-campaign` is an operator-facing command.

`campaign_paused` events must use this approved vocabulary in `reason`:

- `budget_exhausted` — the Wall-Clock Budget Policy or campaign compute budget is spent.
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

The command/API records a `campaign_paused` event with `reason` and, when available, `report_path`.

## Resuming after human review

After a human resolves the pause condition, record a resume event before another autonomous iteration:

```bash
ml-autoresearch resume-campaign \
  --reason human_review_complete \
  --report-path campaign-reports/2026-06-01-resume.md \
  --ledger-path research-ledger.jsonl
```

The command/API records a `campaign_resumed` event. New Autonomy Step prompts tell the agent not to treat earlier `scheduled_check_in` or resolved capability-request pause recommendations as active blockers when a newer resume event exists.
