# Campaign Report: Ground-Camera Contrail Detection

## Summary

A new `campaign_resumed` ledger event with reason `human_review_complete` exists after the prior artifact-visibility report, so earlier `scheduled_check_in` and resolved pause recommendations were not treated as active blockers. I attempted to continue the Research Loop by observing prior Runs through the agent-safe interface, but the read-only Run history remains empty: `ml-autoresearch-agent list-runs --runs-root ../agent-history/runs --json` returns `[]`, and `../agent-history/runs` contains no Run directories. Because the unresolved latest completed Run `run_20260603_164830_9fcbc4` is still not observable, I did not submit a new Candidate Experiment or write a per-Run Research Note from unsupported metrics.

## Current best Result

- Run: `run_20260602_203450_c05550`
- Candidate Experiment: `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60`
- Best-validation metric: documented prior best `val/dice = 0.849036` at epoch 50; request-gated whole-validation evaluation documented `val/dice = 0.849040`, `val/iou = 0.737679`, `val/precision = 0.867067`, `val/recall = 0.831747` at threshold 0.5.
- Result artifacts: canonical `runs/run_20260602_203450_c05550/outputs/` artifacts are referenced by prior Research Notes, but no Run artifact tree is currently exposed in this Agent Control Boundary.
- Why it is current best: It remains the best documented in-contract Result. The later head-dropout Run may change the decision, but its metrics are still unavailable to the agent-safe observation path.

## Recent Runs

| Run | Candidate Experiment | Status | Key Result | Note |
| --- | --- | --- | --- | --- |
| `run_20260602_203450_c05550` | `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60` | completed | Current best documented Result; whole-validation `val/dice = 0.849040`. | Research Note and failure-bucket evaluation note written. |
| `run_20260603_094446_05dea3` | `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60_detail_fuse` | completed | Regressed to best `val/dice = 0.846567`; recall rose but precision fell. | Research Note written; do not promote. |
| `run_20260603_164830_9fcbc4` | `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60_head_dropout_p005` | completed per ledger | Result remains unknown in this boundary; Run artifacts are not visible through `ml-autoresearch-agent`. | Needs artifact visibility before Research Note and next Candidate decision. |

## Failures

| Run | Failure classification | Symptom | Follow-up |
| --- | --- | --- | --- |
| `run_20260601_121820_adbab7` | candidate bug / contract-surface mismatch | Boundary auxiliary target was present in the manifest but unsupported by training at that point. | Historical note and Capability Request filed; not the active blocker. |
| Multiple completed local variants after current best | `bad_research_result` | Boundary auxiliary, refinement, and detail-fusion variants regressed or failed to improve enough. | Avoid further local architecture tweaks until the latest head-dropout Result is observed. |
| Current Autonomy Step | observation blocker | Agent-safe Run history is empty after resume; latest completed Run cannot be summarized. | Pause for human review of Agent Control Boundary Run artifact exposure. |

## Pending Capability Requests

| Request | Status | Why it matters | Blocking? |
| --- | --- | --- | --- |
| `capability_2026_06_13_completed_run_artifact_exposure` | Created and ingested; outcome still ineffective in the refreshed boundary. | Requests read-only exposure of completed Run artifacts, especially `run_20260603_164830_9fcbc4`, so the agent can write a provenance-grounded Research Note and choose the next hypothesis. | Yes. This step confirms the blocker persists after the later resume event. |

## Budget use

- Wall-clock budget used: unknown from observable artifacts.
- Compute budget used: at least the Runs documented in the Research Ledger and prior Research Notes; exact total unknown.
- Storage used/risk: unknown; the immediate issue is missing read-only artifact exposure, not confirmed storage exhaustion.
- Remaining budget: unknown.

## Next hypothesis

No new Candidate Experiment is safe in this Autonomy Step. The next useful primary handoff after Run artifacts are visible should be a Research Note for `run_20260603_164830_9fcbc4`. If head dropout improves or nearly matches the current best with a better precision/recall tradeoff, follow head regularization. If it regresses, stop head-dropout tuning and consider a different in-contract architecture direction or a scheduler/early-stopping Capability Request.

## Pause recommendation

- Pause condition: `stalled_research_progress`
- Human decision needed: yes. Refresh or mount the read-only Research History `runs/` artifact tree so `ml-autoresearch-agent run-summary --runs-root ../agent-history/runs --run-id run_20260603_164830_9fcbc4 --json` can inspect completed Run artifacts before the autonomous loop resumes.
