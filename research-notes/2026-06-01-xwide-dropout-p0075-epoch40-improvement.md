# 2026-06-01 xwide p=0.075 dropout 40-epoch improvement

## Hypothesis

The current best in-contract Result, `run_20260601_085755_25cd06`, reached its best validation Dice at the final epoch 30 with the extra-wide base-64 U-Net, Line Target auxiliary weight 0.10, and bottleneck `Dropout2d(p=0.075)`. The hypothesis for `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40` was that the same architecture might still be improving under the fixed 30-epoch policy, so extending only the Harness-owned `max_epochs` knob to 40 could improve best-validation Dice without a large precision or stability penalty.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40`
- Candidate Experiment path: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40`
- Relevant Experiment Proposal: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40/PROPOSAL.md`
- Primary Comparison Target: `run_20260601_085755_25cd06` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_rerun`
- Secondary context: `run_20260530_180658_0af8a8` / `single_frame_xwide_unet_line_aux_w010_dropout` and `run_20260601_122336_17faa3` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_refine`.
- Contract choices: Single-Frame RGB Input, mask logits output, Line Target auxiliary output with `weighted_bce` weight 0.10, deterministic shuffle Sampling Policy, no augmentation, `bce_dice`, AdamW, learning rate 0.001, requested/effective batch size 8, max epochs 40.

## Run(s)

- Run ID: `run_20260601_162117_66bf89`
- Dataset mode/subset: GVCCS Working Validation Split
- Harness/backend notes: completed on CUDA with no reported Resource Failure retry. Best-validation model artifact persisted at `outputs/models/best_epoch_model.pt`. Model summary reports 7,785,794 parameters, unchanged from the 30-epoch p=0.075 base and below the 10,000,000 parameter budget.

## Key metrics

Best-validation metrics, selected by max `val/dice`:

| Run | Candidate Experiment | Best epoch | val/dice | val/iou | val/precision | val/recall | val/loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `run_20260530_180658_0af8a8` | `single_frame_xwide_unet_line_aux_w010_dropout` | 29 | 0.833852 | 0.715049 | 0.856300 | 0.812552 | 0.510518 |
| `run_20260601_085755_25cd06` | `single_frame_xwide_unet_line_aux_w010_dropout_p0075_rerun` | 30 | 0.834603 | 0.716153 | 0.846543 | 0.822995 | 0.508752 |
| `run_20260601_122336_17faa3` | `single_frame_xwide_unet_line_aux_w010_dropout_p0075_refine` | 28 | 0.833781 | 0.714944 | 0.840078 | 0.827578 | 0.504853 |
| `run_20260601_162117_66bf89` | `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40` | 40 | 0.843361 | 0.729148 | 0.855490 | 0.831572 | 0.500193 |

The 40-epoch run improves best-validation Dice by about `+0.00876` over the 30-epoch p=0.075 comparison target and by about `+0.00951` over the earlier p=0.10 dropout reference. It also improves both sides of the precision/recall tradeoff relative to the 30-epoch base: precision rises by about `+0.00895` and recall rises by about `+0.00858`. This is stronger than the proposal's desired `+0.0005` Dice improvement and clears the precision and recall guardrails.

Final completed-epoch metrics for `run_20260601_162117_66bf89` are the same as best-validation metrics because the best epoch was the final epoch 40: `val/dice` 0.843361, `val/iou` 0.729148, `val/precision` 0.855490, `val/recall` 0.831572, `val/loss` 0.500193, and `val/total_loss` 0.510662. The absence of a best-to-final gap supports the training-budget hypothesis and suggests the 30-epoch p=0.075 result was not saturated.

## Qualitative observations

The saved first-N prediction samples are also stronger than the 30-epoch p=0.075 base and the rejected refinement variant on the same first two validation examples:

- `val/000000`: Dice 0.8645 and IoU 0.7614, stronger than the 30-epoch p=0.075 sample 000 Dice 0.8154 and the refinement sample 000 Dice 0.8533.
- `val/000001`: Dice 0.8533 and IoU 0.7442, stronger than the 30-epoch p=0.075 sample 001 Dice 0.8321 but below the refinement sample 001 Dice 0.8811.

Unlike the refinement Run, where first-N samples improved despite aggregate regression, the 40-epoch Run improves aggregate best-validation metrics and the saved examples together. This makes the qualitative evidence more consistent with the full validation Result, though first-N samples remain too narrow to rule out false-positive or missed-positive regressions elsewhere.

![epoch40 sample 000 overlay](../runs/run_20260601_162117_66bf89/outputs/prediction_samples/sample_000_overlay.png)

![epoch40 sample 001 overlay](../runs/run_20260601_162117_66bf89/outputs/prediction_samples/sample_001_overlay.png)

![epoch40 sample 001 heatmap](../runs/run_20260601_162117_66bf89/outputs/prediction_samples/sample_001_probability_heatmap.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-xwide-dropout-p0075-epoch40-sample-000-overlay
    source_run_id: run_20260601_162117_66bf89
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: First saved validation overlay for the 40-epoch p=0.075 dropout Run; selected because sample Dice improved alongside aggregate best-validation Dice.
  - figure_id: fig-xwide-dropout-p0075-epoch40-sample-001-overlay
    source_run_id: run_20260601_162117_66bf89
    source_artifact_path: outputs/prediction_samples/sample_001_overlay.png
    reason: Second saved validation overlay for the 40-epoch p=0.075 dropout Run; useful for comparing the longer-budget model against the 30-epoch base and the refinement regression.
  - figure_id: fig-xwide-dropout-p0075-epoch40-sample-001-heatmap
    source_run_id: run_20260601_162117_66bf89
    source_artifact_path: outputs/prediction_samples/sample_001_probability_heatmap.png
    reason: Probability heatmap for the second saved validation sample, selected to inspect whether the longer training budget sharpens or broadens contrail probabilities while improving both precision and recall.
```

## Decision

Promote `run_20260601_162117_66bf89` as the current best in-contract Result. The Candidate Experiment completed cleanly, changed only the approved `max_epochs` training knob, exceeded the primary comparison target by a meaningful margin, improved precision and recall together, and kept the best epoch at the final epoch. This supports the conclusion that the p=0.075 extra-wide Line Target U-Net benefits from a 40-epoch budget under the current Harness policy.

Do not continue immediate architecture tinkering on the refinement branch. The training-budget probe produced a clearer gain than the full-resolution refinement block and avoided the Boundary Target contract gap.

## Next proposed change

Request a bounded failure-bucket Post-Run Evaluation for `run_20260601_162117_66bf89` before launching another Candidate Experiment. Compare against the existing p=0.075 30-epoch evaluation and earlier p=0.10 dropout evaluation, focusing on threshold-sweep optimum, empty-mask false positives, false-negative-heavy positives, and missed-positive masks. If diagnostics confirm that the gain is not mainly threshold-specific or false-positive-heavy, use the 40-epoch p=0.075 model as the new in-contract base for the next approved hypothesis.
