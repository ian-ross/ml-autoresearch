# 2026-06-02 xwide p=0.05 dropout 40-epoch precision-biased improvement

## Hypothesis

`single_frame_xwide_unet_line_aux_w010_dropout_p005_epoch40` tested whether reducing the current best extra-wide U-Net bottleneck dropout from `p=0.075` to `p=0.05`, while keeping the 40-epoch budget and Line Target auxiliary weight 0.10, would recover more faint or under-segmented contrails without losing the regularization benefit that made the p=0.075 model strong.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_xwide_unet_line_aux_w010_dropout_p005_epoch40`
- Candidate Experiment path: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p005_epoch40`
- Relevant Experiment Proposal: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p005_epoch40/PROPOSAL.md`
- Primary Comparison Target: `run_20260601_162117_66bf89` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40`
- Contract choices: Single-Frame RGB Input, mask logits output, Line Target auxiliary output with `weighted_bce` weight 0.10, deterministic shuffle Sampling Policy, no augmentation, `bce_dice`, AdamW, learning rate 0.001, requested/effective batch size 8, max epochs 40.

## Run(s)

- Run ID: `run_20260602_101047_294476`
- Dataset mode/subset: GVCCS Working Validation Split
- Harness/backend notes: completed on CUDA with no Resource Failure retry. Best-validation model artifact persisted at `outputs/models/best_epoch_model.pt`. Model summary reports 7,785,794 parameters, unchanged from the p=0.075 comparison target and below the 10,000,000 parameter budget.

## Key metrics

Best-validation metrics, selected by max `val/dice`:

| Run | Candidate Experiment | Best epoch | val/dice | val/iou | val/precision | val/recall | val/loss | val/total_loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `run_20260601_162117_66bf89` | `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40` | 40 | 0.843361 | 0.729148 | 0.855490 | 0.831572 | 0.500193 | 0.510662 |
| `run_20260602_101047_294476` | `single_frame_xwide_unet_line_aux_w010_dropout_p005_epoch40` | 39 | 0.844391 | 0.730689 | 0.866790 | 0.823120 | 0.499712 | 0.509931 |

The p=0.05 run improves best-validation Dice by about `+0.00103` and IoU by about `+0.00154` over the p=0.075 40-epoch comparison target. It also improves precision by about `+0.01130`, but recall drops by about `-0.00845`, violating the proposal guardrail that recall should improve or remain within 0.002 of the comparison target.

Final completed-epoch metrics for `run_20260602_101047_294476` are slightly below the selected best epoch: final `val/dice` 0.839533, `val/iou` 0.723445, `val/precision` 0.859870, `val/recall` 0.820137, `val/loss` 0.504820, and `val/total_loss` 0.515896. The best-to-final gap is small but indicates epoch 39, not epoch 40, was the strongest checkpoint.

## Qualitative observations

The saved first-N prediction samples are mixed relative to the p=0.075 40-epoch comparison target:

- `val/000000`: Dice 0.8428 and IoU 0.7283, below the comparison target's saved sample 000 Dice 0.8645 / IoU 0.7614.
- `val/000001`: Dice 0.9116 and IoU 0.8375, above the comparison target's saved sample 001 Dice 0.8533 / IoU 0.7442.

The aggregate metrics and first-N samples both suggest a precision-biased model rather than the expected recall recovery. The high precision and lower recall could still be useful if it reduces empty-mask false positives or broad spillover, but it should not be treated as a clean successor to the p=0.075 model without whole-validation failure-bucket diagnostics.

![p005 epoch40 sample 000 overlay](../runs/run_20260602_101047_294476/outputs/prediction_samples/sample_000_overlay.png)

![p005 epoch40 sample 001 overlay](../runs/run_20260602_101047_294476/outputs/prediction_samples/sample_001_overlay.png)

![p005 epoch40 sample 001 heatmap](../runs/run_20260602_101047_294476/outputs/prediction_samples/sample_001_probability_heatmap.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-xwide-dropout-p005-epoch40-sample-000-overlay
    source_run_id: run_20260602_101047_294476
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: First saved validation overlay for the p=0.05 dropout Run; selected because this sample regressed despite the run's aggregate best-validation Dice improvement.
  - figure_id: fig-xwide-dropout-p005-epoch40-sample-001-overlay
    source_run_id: run_20260602_101047_294476
    source_artifact_path: outputs/prediction_samples/sample_001_overlay.png
    reason: Second saved validation overlay for the p=0.05 dropout Run; selected because this sample improved strongly and illustrates the mixed qualitative behavior.
  - figure_id: fig-xwide-dropout-p005-epoch40-sample-001-heatmap
    source_run_id: run_20260602_101047_294476
    source_artifact_path: outputs/prediction_samples/sample_001_probability_heatmap.png
    reason: Probability heatmap for the stronger saved validation sample, selected to inspect the precision-biased p=0.05 output distribution.
```

## Decision

Treat `run_20260602_101047_294476` as the new top Result by best-validation Dice, but do not fully promote it as the next design baseline yet. It cleared the Dice and precision criteria, but failed the recall guardrail and shifted the precision/recall balance toward tighter masks. The p=0.075 40-epoch model remains the safer baseline for recall-sensitive follow-up until diagnostics show that p=0.05's higher Dice is not hiding missed-positive regressions.

## Next proposed change

Request a bounded failure-bucket Post-Run Evaluation for `run_20260602_101047_294476`, comparing it to `run_20260601_162117_66bf89` on threshold sweep, false-negative-heavy positives, missed-positive masks, false-positive-heavy positives, and empty-mask false positives. If the p=0.05 run mostly improves empty-mask/false-positive buckets while only mildly worsening missed positives, it may be worth promoting for high-precision exploration; otherwise keep p=0.075 as the preferred dropout setting.
