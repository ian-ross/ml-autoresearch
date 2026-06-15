# 2026-06-15 reduce-on-plateau early-stopping training-policy improvement

## Hypothesis

`single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es` tested whether the current best documented p=0.075 extra-wide U-Net with Line Target auxiliary loss would benefit from the newly available Harness-owned training-policy slice: reduce-on-plateau scheduling plus early stopping with best-checkpoint restoration. The expected effect was modest improvement or at least a cleaner restored final model relative to the fixed-60-epoch parent `run_20260602_203450_c05550`, without changing the Model Architecture.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es`
- Candidate Experiment path: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es`
- Relevant Experiment Proposal: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es/PROPOSAL.md`
- Primary Comparison Target: `run_20260602_203450_c05550` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60`
- Contract choices: Single-Frame RGB Input, mask logits output, Line Target auxiliary output with `weighted_bce` weight 0.10, deterministic shuffle Sampling Policy, no augmentation, `bce_dice`, AdamW, learning rate 0.001, requested/effective batch size 8, max epochs 80, `reduce_on_plateau` scheduler, validation-Dice early stopping with best-checkpoint restoration.

## Run(s)

- Run ID: `run_20260614_124226_05e3eb`
- Dataset mode/subset: GVCCS Working Validation Split
- Harness/backend notes: completed on CUDA with Docker backend. Model summary reports 7,785,794 parameters, unchanged from the p=0.075 parent and below the 10,000,000 parameter budget. The scheduler reduced learning rate to 0.00025 at the best epoch and 0.000125 by the final epoch. Early stopping was enabled but did not stop before the 80-epoch maximum; best-checkpoint restoration was recorded as enabled and restored.

## Key metrics

Best-validation metrics, selected by max `val/dice`:

| Run | Candidate Experiment | Best epoch | val/dice | val/iou | val/precision | val/recall | val/loss | val/total_loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `run_20260602_203450_c05550` | `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60` | 50 | 0.849036 | 0.737673 | 0.867065 | 0.831741 | 0.490690 | 0.500491 |
| `run_20260614_124226_05e3eb` | `single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es` | 71 | 0.861622 | 0.756886 | 0.883582 | 0.840728 | 0.484870 | 0.495275 |

The scheduler/early-stopping run is a clear improvement over the fixed-60-epoch p=0.075 parent: best-validation Dice improves by about `+0.01259`, IoU by `+0.01921`, precision by `+0.01652`, and recall by `+0.00899`. This is stronger than the proposal's success threshold of `val/dice >= 0.8490` and exceeds the strong-success threshold of `val/dice >= 0.8510`.

Final completed-epoch metrics for `run_20260614_124226_05e3eb` were also strong but below the selected best epoch: final `val/dice` 0.854125, `val/iou` 0.745392, `val/precision` 0.876363, `val/recall` 0.832989, `val/loss` 0.274166, and `val/total_loss` 0.284945. The best-to-final Dice gap is about `0.00750`, so best-checkpoint restoration is important even though early stopping did not fire before the epoch cap.

## Qualitative observations

The saved first-N prediction samples support the aggregate improvement, though they are not enough to assess the known failure buckets:

- `val/000000`: Dice 0.8794 and IoU 0.7848, improving over the fixed-60 parent sample 000 Dice 0.8633 / IoU 0.7595.
- `val/000001`: Dice 0.8889 and IoU 0.8000, improving over the fixed-60 parent sample 001 Dice 0.8571 / IoU 0.7500.

The aggregate best-validation metrics show a favorable simultaneous precision and recall improvement. However, prior failure-bucket evaluations found that aggregate Dice can hide empty-mask affected-sample spread, tiny missed positives, broad false-positive spillover, and under-segmentation of large positives. This run should receive a bounded whole-validation failure-bucket evaluation before using it as a foundation for the next architecture change.

![plateau/es sample 000 overlay](../runs/run_20260614_124226_05e3eb/outputs/prediction_samples/sample_000_overlay.png)

![plateau/es sample 001 overlay](../runs/run_20260614_124226_05e3eb/outputs/prediction_samples/sample_001_overlay.png)

![plateau/es sample 001 heatmap](../runs/run_20260614_124226_05e3eb/outputs/prediction_samples/sample_001_probability_heatmap.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-plateau-es-sample-000-overlay
    source_run_id: run_20260614_124226_05e3eb
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: First saved validation overlay for the scheduler/early-stopping Run; selected because its first-N Dice improved over the fixed-60 p=0.075 parent sample.
  - figure_id: fig-plateau-es-sample-001-overlay
    source_run_id: run_20260614_124226_05e3eb
    source_artifact_path: outputs/prediction_samples/sample_001_overlay.png
    reason: Second saved validation overlay for the scheduler/early-stopping Run; selected because it shows stronger first-N improvement and provides a direct qualitative check against the parent note's sample 001 concern.
  - figure_id: fig-plateau-es-sample-001-heatmap
    source_run_id: run_20260614_124226_05e3eb
    source_artifact_path: outputs/prediction_samples/sample_001_probability_heatmap.png
    reason: Probability heatmap for the second saved validation sample, selected to inspect confidence distribution after reduce-on-plateau scheduling and best-checkpoint restoration.
```

## Decision

Promote `run_20260614_124226_05e3eb` as the new current best in-contract Result by best-validation Dice. The improvement is large for a training-policy-only change and improves both precision and recall relative to the documented fixed-60 p=0.075 baseline. Treat the scheduler/early-stopping slice as valuable, but do not infer that the known qualitative failure modes are solved until whole-validation diagnostics are available.

## Next proposed change

Request a bounded failure-bucket Post-Run Evaluation for `run_20260614_124226_05e3eb`, comparing it to `run_20260602_203450_c05550` on threshold sweep, false-negative-heavy positives, missed-positive masks, false-positive-heavy positives, empty-mask false positives, and empty-mask affected-sample count. If diagnostics confirm the gain without worsening empty-sky spread, use this run as the primary Comparison Target for the next architecture hypothesis.
