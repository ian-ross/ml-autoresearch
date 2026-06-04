# 2026-06-03 detail-fusion architecture regression

## Hypothesis

`single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60_detail_fuse` tested whether a shallow high-resolution detail-fusion head on top of the current best 60-epoch p=0.075 extra-wide U-Net could recover tiny positives and improve large-mask completion. The expected effect was higher Dice or recall by feeding compact first-encoder features directly into the final prediction heads while keeping the proven Line Target auxiliary weight, dropout, data policy, optimizer, and 60-epoch budget unchanged.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60_detail_fuse`
- Candidate Experiment path: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60_detail_fuse`
- Relevant Experiment Proposal: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60_detail_fuse/PROPOSAL.md`
- Primary Comparison Target: `run_20260602_203450_c05550` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60`

## Run(s)

- Run ID: `run_20260603_094446_05dea3`
- Dataset mode/subset: GVCCS Working Validation Split
- Harness/backend notes: completed on CUDA with Docker backend and no Resource Failure retry. Best-validation model artifact persisted at `outputs/models/best_epoch_model.pt`. Model summary reports 7,896,706 parameters, still below the 10,000,000 parameter budget.

## Key metrics

Best-validation metrics, selected by max `val/dice`:

| Run | Candidate Experiment | Best epoch | val/dice | val/iou | val/precision | val/recall | val/loss | val/total_loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `run_20260602_203450_c05550` | `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60` | 50 | 0.849036 | 0.737673 | 0.867065 | 0.831741 | 0.490690 | 0.500491 |
| `run_20260603_094446_05dea3` | `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60_detail_fuse` | 48 | 0.846567 | 0.733954 | 0.852935 | 0.840293 | 0.494264 | 0.504197 |

The detail-fusion candidate regressed by about `-0.00247` best-validation Dice versus the current best comparison target. It raised recall by about `+0.00855`, but precision dropped by about `-0.01413`, so the extra high-resolution texture appears to trade too much empty/background cleanliness for sensitivity.

Final completed-epoch metrics for `run_20260603_094446_05dea3` were weaker: final `val/dice` 0.835487, `val/iou` 0.717456, `val/precision` 0.857915, `val/recall` 0.814201, `val/loss` 0.517499, and `val/total_loss` 0.529434. The best-to-final Dice gap is about `0.01108`, exceeding the proposal's approximate `0.005` stability guardrail and showing that this architecture became less stable late in training.

## Qualitative observations

The first-N prediction samples improved relative to the comparison target, but this did not generalize to aggregate validation Dice:

- `val/000000`: Dice improved from 0.8633 for the comparison target to 0.8828 for detail fusion.
- `val/000001`: Dice improved from 0.8571 for the comparison target to 0.8874 for detail fusion.

These first-N samples are therefore misleadingly favorable. The aggregate metrics show a precision-led regression and late-epoch instability, so a failure-bucket evaluation is not justified before another Candidate Experiment unless a human specifically wants diagnostics for this architecture family.

![detail-fusion sample 000 overlay](../runs/run_20260603_094446_05dea3/outputs/prediction_samples/sample_000_overlay.png)

![detail-fusion sample 001 overlay](../runs/run_20260603_094446_05dea3/outputs/prediction_samples/sample_001_overlay.png)

![detail-fusion sample 001 heatmap](../runs/run_20260603_094446_05dea3/outputs/prediction_samples/sample_001_probability_heatmap.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-detail-fusion-sample-000-overlay
    source_run_id: run_20260603_094446_05dea3
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: First saved validation overlay for the detail-fusion Run; selected because this first-N example improved despite aggregate validation Dice regressing.
  - figure_id: fig-detail-fusion-sample-001-overlay
    source_run_id: run_20260603_094446_05dea3
    source_artifact_path: outputs/prediction_samples/sample_001_overlay.png
    reason: Second saved validation overlay for the detail-fusion Run; selected to document the misleadingly positive first-N qualitative signal.
  - figure_id: fig-detail-fusion-sample-001-heatmap
    source_run_id: run_20260603_094446_05dea3
    source_artifact_path: outputs/prediction_samples/sample_001_probability_heatmap.png
    reason: Probability heatmap for the second saved validation sample, selected to inspect confidence after adding high-resolution detail fusion.
```

## Decision

Do not promote `run_20260603_094446_05dea3`. Treat the high-resolution detail-fusion head as a bad research result: it improved recall and first-N qualitative samples, but regressed best-validation Dice, materially reduced precision, and exceeded the final-vs-best stability guardrail.

Keep `run_20260602_203450_c05550` as the current best in-contract Result and preferred comparison target.

## Next proposed change

Abandon this detail-fusion path for now. The next in-contract hypothesis should preserve the stable 60-epoch p=0.075 xwide Line Target baseline while making a more precision-safe change, or should request Harness-owned scheduler/early-stopping support before further architectures with late-epoch instability.
