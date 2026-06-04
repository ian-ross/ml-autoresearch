# 2026-06-01 high-resolution refinement regression

## Hypothesis

The current best in-contract Result, `run_20260601_085755_25cd06`, improved best-validation Dice and recall with the extra-wide base-64 U-Net, Line Target auxiliary weight 0.10, and bottleneck `Dropout2d(p=0.075)`, but the failure-bucket review still showed tiny missed positives and under-segmentation on some larger masks. The hypothesis for `single_frame_xwide_unet_line_aux_w010_dropout_p0075_refine` was that a lightweight full-resolution residual refinement block after the final decoder stage would recover small contrail fragments and sharpen thin structures without needing Boundary Target runtime support.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_xwide_unet_line_aux_w010_dropout_p0075_refine`
- Candidate Experiment path: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_refine`
- Relevant Experiment Proposal: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_refine/PROPOSAL.md`
- Primary Comparison Target: `run_20260601_085755_25cd06` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_rerun`
- Contract choices: Single-Frame RGB Input, mask logits output, Line Target auxiliary output with `weighted_bce` weight 0.10, deterministic shuffle Sampling Policy, no augmentation, `bce_dice`, AdamW, learning rate 0.001, requested/effective batch size 8, max epochs 30.

## Run(s)

- Run ID: `run_20260601_122336_17faa3`
- Dataset mode/subset: GVCCS Working Validation Split
- Harness/backend notes: completed on CUDA with no Resource Failure retry; requested and effective batch size were both 8. Best-validation model artifact persisted at `outputs/models/best_epoch_model.pt`. Model summary reports 7,868,098 parameters, still below the 10,000,000 parameter budget but slightly above the 7,785,794-parameter p=0.075 base because of the refinement block.

## Key metrics

Best-validation metrics, selected by max `val/dice`:

| Run | Candidate Experiment | Best epoch | val/dice | val/iou | val/precision | val/recall | val/loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `run_20260530_180658_0af8a8` | `single_frame_xwide_unet_line_aux_w010_dropout` | 29 | 0.833852 | 0.715049 | 0.856300 | 0.812552 | 0.510518 |
| `run_20260601_085755_25cd06` | `single_frame_xwide_unet_line_aux_w010_dropout_p0075_rerun` | 30 | 0.834603 | 0.716153 | 0.846543 | 0.822995 | 0.508752 |
| `run_20260601_122336_17faa3` | `single_frame_xwide_unet_line_aux_w010_dropout_p0075_refine` | 28 | 0.833781 | 0.714944 | 0.840078 | 0.827578 | 0.504853 |

The refinement Run regressed slightly from the p=0.075 comparison target by best-validation Dice (`-0.00082`) while increasing recall (`+0.00458`) and lowering precision (`-0.00646`). It is also marginally below the earlier p=0.10 dropout model by best-validation Dice. The best epoch was 28 rather than the final epoch.

Final completed-epoch metrics for `run_20260601_122336_17faa3` dropped further: `val/dice` 0.827474, `val/iou` 0.705719, `val/precision` 0.858937, `val/recall` 0.798234, and `val/loss` 0.514650. The best-to-final gap suggests the refinement block did not improve the 30-epoch stability that made the p=0.075 base attractive.

## Qualitative observations

The saved first-N prediction samples look better than the p=0.075 base on these two examples, but they do not reflect the aggregate validation regression:

- `val/000000`: Dice 0.8533 and IoU 0.7442, stronger than the p=0.075 base sample 000 Dice 0.8154.
- `val/000001`: Dice 0.8811 and IoU 0.7875, stronger than the p=0.075 base sample 001 Dice 0.8321.

This mismatch suggests the refinement block can sharpen or improve some visible early positive examples, but it likely broadens predictions or shifts calibration elsewhere on the Working Validation Split. Because the aggregate best-validation Result is worse and the final epoch falls off, the first-N improvement is not enough evidence to continue this exact refinement direction without a more targeted diagnostic.

![refinement sample 000 overlay](../runs/run_20260601_122336_17faa3/outputs/prediction_samples/sample_000_overlay.png)

![refinement sample 001 overlay](../runs/run_20260601_122336_17faa3/outputs/prediction_samples/sample_001_overlay.png)

![refinement sample 001 heatmap](../runs/run_20260601_122336_17faa3/outputs/prediction_samples/sample_001_probability_heatmap.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-xwide-refinement-sample-000-overlay
    source_run_id: run_20260601_122336_17faa3
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: First saved validation overlay for the high-resolution refinement Run; selected because this sample improved qualitatively despite aggregate Dice regression.
  - figure_id: fig-xwide-refinement-sample-001-overlay
    source_run_id: run_20260601_122336_17faa3
    source_artifact_path: outputs/prediction_samples/sample_001_overlay.png
    reason: Second saved validation overlay for the refinement Run; selected because its high sample Dice contrasts with the worse full-validation best Dice.
  - figure_id: fig-xwide-refinement-sample-001-heatmap
    source_run_id: run_20260601_122336_17faa3
    source_artifact_path: outputs/prediction_samples/sample_001_probability_heatmap.png
    reason: Probability heatmap for the second saved validation sample, useful for checking whether the refinement block sharpens visible contrail probabilities while hurting aggregate calibration.
```

## Decision

Reject `single_frame_xwide_unet_line_aux_w010_dropout_p0075_refine` as a bad research result relative to the current best in-contract base. It completed cleanly and stayed within the Candidate Experiment Contract, but it did not meet the proposal success criterion of matching or exceeding `run_20260601_085755_25cd06` by best-validation Dice, and it reintroduced a best-to-final validation gap.

Keep `run_20260601_085755_25cd06` as the preferred in-contract base. Do not spend another immediate Candidate Experiment on this exact full-resolution refinement block; the observed gain on first-N examples is outweighed by weaker aggregate Result and stability.

## Next proposed change

If Boundary Target runtime support remains blocked, prefer either a bounded Post-Run Evaluation of this refinement Run only if humans need to understand the first-N/aggregate mismatch, or a different in-contract hypothesis with a clearer mechanism for tiny missed positives and calibration. Avoid further local dropout-rate tuning and avoid emulating Boundary Target supervision inside candidate architecture code.
