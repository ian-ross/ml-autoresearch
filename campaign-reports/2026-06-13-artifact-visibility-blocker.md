# Campaign Report: Ground-Camera Contrail Detection

## Summary

The campaign has a newer `campaign_resumed` ledger event with reason `human_review_complete`, so the earlier scheduled check-in is not treated as an active blocker. However, this Agent Control Boundary still cannot observe any completed Run artifact trees through the agent-safe history view: `ml-autoresearch-agent list-runs --runs-root /home/iross/code/ml-autoresearch/agent-history/runs --json` returns `[]`, and the unresolved decision-critical Run remains `run_20260603_164830_9fcbc4` (`single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60_head_dropout_p005`). Because the latest completed Run cannot be summarized, I cannot safely write its per-Run Research Note or submit a new Candidate Experiment without breaking the observe-before-propose loop.

## Current best Result

- Run: `run_20260602_203450_c05550`
- Candidate Experiment: `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60`
- Best-validation metric: Run-level best `val/dice = 0.849036` at epoch 50; request-gated whole-validation evaluation reports `val/dice = 0.849040`, `val/iou = 0.737679`, `val/precision = 0.867067`, `val/recall = 0.831747` at threshold 0.5.
- Result artifacts: canonical paths recorded in prior notes under `runs/run_20260602_203450_c05550/outputs/`, but the current agent-safe runs artifact view is empty.
- Why it is current best: It remains the best documented in-contract Result, improving over the p=0.075 40-epoch parent and the p=0.05 40-epoch precision-biased contender while reducing missed-positive samples in whole-validation diagnostics.

## Recent Runs

| Run | Candidate Experiment | Status | Key Result | Note |
| --- | --- | --- | --- | --- |
| `run_20260602_203450_c05550` | `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60` | completed | Current best documented Result; whole-validation `val/dice = 0.849040`. | Research Note and failure-bucket evaluation note written. |
| `run_20260603_094446_05dea3` | `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60_detail_fuse` | completed | Regressed to best `val/dice = 0.846567`; recall rose but precision fell. | Research Note written; do not promote. |
| `run_20260603_164830_9fcbc4` | `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60_head_dropout_p005` | completed per ledger | Result unknown in this boundary; final metrics path is recorded, but artifact tree is not visible to agent-safe observation. | Needs artifact visibility before Research Note / next Candidate decision. |

## Failures

| Run | Failure classification | Symptom | Follow-up |
| --- | --- | --- | --- |
| `run_20260601_121820_adbab7` | candidate/contract-surface mismatch | Boundary auxiliary training rejected the documented target during training. | Historical Capability Request filed; later boundary variants regressed, so this is not the active blocker. |
| completed regressions after current best | `bad_research_result` | Boundary auxiliary variants, high-resolution refinement, and detail fusion did not beat `run_20260602_203450_c05550`. | Avoid ungrounded local architecture tweaks until the latest head-dropout Result is observed. |
| current Autonomy Step | observation blocker | Agent-safe runs history is empty and `run_20260603_164830_9fcbc4` artifacts are unavailable. | Do not submit another Candidate Experiment from incomplete evidence. |

## Pending Capability Requests

| Request | Status | Why it matters | Blocking? |
| --- | --- | --- | --- |
| `capability_2026_06_13_completed_run_artifact_exposure` | Created after the resume event; outcome not visible in this boundary. | Requests read-only exposure of completed Run artifacts, especially `run_20260603_164830_9fcbc4`, so the agent can write a provenance-grounded Research Note and choose the next hypothesis. | Yes in practice for this step, because the artifacts are still not observable. |

## Budget use

- Wall-clock budget used: unknown from current observable artifacts.
- Compute budget used: at least the Runs documented in the ledger and prior notes; exact total unknown.
- Storage used/risk: unknown; current risk is missing read-only artifact visibility rather than confirmed storage exhaustion.
- Remaining budget: unknown.

## Next hypothesis

No new Candidate Experiment is safe in this Autonomy Step. The next useful handoff after artifacts are visible should be a Research Note for `run_20260603_164830_9fcbc4`. If head dropout improves or nearly matches the current best with a better precision/recall or false-positive tradeoff, follow that signal; if it regresses, stop local head-dropout tuning and prefer a different in-contract architecture hypothesis or a scheduler/early-stopping Capability Request.

## Pause recommendation

- Pause condition: `stalled_research_progress`
- Human decision needed: yes. The prior scheduled check-in is resolved, but progress is newly stalled because decision-critical completed Run artifacts are not visible inside this Agent Control Boundary. Refresh/mount the read-only Research History `runs/` artifact tree, then resume with observation and a Research Note for `run_20260603_164830_9fcbc4` before any new Candidate Submission.
