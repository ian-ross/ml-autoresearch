# 2026-06-03 xwide p=0.075 dropout 60-epoch improvement

## Hypothesis

`single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60` tested whether the recall-safer p=0.075 extra-wide U-Net with Line Target auxiliary weight 0.10 was still under-trained at 40 epochs. The expectation was that extending only the Harness-owned `max_epochs` knob from 40 to 60 might recover or exceed the p=0.05 40-epoch leaderboard Dice while preserving recall closer to the p=0.075 baseline.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60`
- Candidate Experiment path: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60`
- Relevant Experiment Proposal: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60/PROPOSAL.md`
- Primary Comparison Target: `run_20260601_162117_66bf89` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40`
- Global-best context: `run_20260602_101047_294476` / `single_frame_xwide_unet_line_aux_w010_dropout_p005_epoch40`
- Contract choices: Single-Frame RGB Input, mask logits output, Line Target auxiliary output with `weighted_bce` weight 0.10, deterministic shuffle Sampling Policy, no augmentation, `bce_dice`, AdamW, learning rate 0.001, requested/effective batch size 8, max epochs 60.

## Run(s)

- Run ID: `run_20260602_203450_c05550`
- Dataset mode/subset: GVCCS Working Validation Split
- Harness/backend notes: completed on CUDA with Docker backend and no Resource Failure retry. Best-validation model artifact persisted at `outputs/models/best_epoch_model.pt`. Model summary reports 7,785,794 parameters, unchanged from the p=0.075 40-epoch comparison target and below the 10,000,000 parameter budget.

## Key metrics

Best-validation metrics, selected by max `val/dice`:

| Run | Candidate Experiment | Best epoch | val/dice | val/iou | val/precision | val/recall | val/loss | val/total_loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `run_20260601_162117_66bf89` | `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40` | 40 | 0.843361 | 0.729148 | 0.855490 | 0.831572 | 0.500193 | 0.510662 |
| `run_20260602_101047_294476` | `single_frame_xwide_unet_line_aux_w010_dropout_p005_epoch40` | 39 | 0.844391 | 0.730689 | 0.866790 | 0.823120 | 0.499712 | 0.509931 |
| `run_20260602_203450_c05550` | `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60` | 50 | 0.849036 | 0.737673 | 0.867065 | 0.831741 | 0.490690 | 0.500491 |

The 60-epoch p=0.075 run improves best-validation Dice by about `+0.00567` over the p=0.075 40-epoch primary comparison target and about `+0.00464` over the p=0.05 40-epoch global-best context. It also preserves the p=0.075 recall profile: recall is essentially unchanged versus the primary target (`+0.00017`) and about `+0.00862` above the p=0.05 run, while precision rises to roughly the p=0.05 level.

Final completed-epoch metrics for `run_20260602_203450_c05550` are below the selected best epoch but still strong: final `val/dice` 0.845143, `val/iou` 0.731817, `val/precision` 0.870333, `val/recall` 0.821371, `val/loss` 0.503301, and `val/total_loss` 0.514672. The best-to-final Dice gap is about `0.00389`, under the proposal's `0.005` instability threshold, but the final epoch shifts toward higher precision and lower recall. Best-validation selection matters for this Result.

## Qualitative observations

The saved first-N prediction samples are consistent with a small aggregate improvement over the p=0.075 40-epoch comparison target, though they do not capture the final-epoch recall dip:

- `val/000000`: Dice 0.8633 and IoU 0.7595, nearly matching the p=0.075 40-epoch sample 000 Dice 0.8645 / IoU 0.7614.
- `val/000001`: Dice 0.8571 and IoU 0.7500, slightly above the p=0.075 40-epoch sample 001 Dice 0.8533 / IoU 0.7442 but far below the p=0.05 sample 001 Dice 0.9116.

The aggregate best-validation metrics are more compelling than the first-N qualitative samples. The run appears to combine p=0.05-like precision with p=0.075-like recall at the best epoch, but a whole-validation failure-bucket evaluation is needed to check whether this higher Dice hides new missed-positive or empty-mask false-positive behavior.

![p0075 epoch60 sample 000 overlay](../runs/run_20260602_203450_c05550/outputs/prediction_samples/sample_000_overlay.png)

![p0075 epoch60 sample 001 overlay](../runs/run_20260602_203450_c05550/outputs/prediction_samples/sample_001_overlay.png)

![p0075 epoch60 sample 001 heatmap](../runs/run_20260602_203450_c05550/outputs/prediction_samples/sample_001_probability_heatmap.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-xwide-dropout-p0075-epoch60-sample-000-overlay
    source_run_id: run_20260602_203450_c05550
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: First saved validation overlay for the 60-epoch p=0.075 dropout Run; selected to compare the longer training budget against the p=0.075 40-epoch baseline on a stable first-N example.
  - figure_id: fig-xwide-dropout-p0075-epoch60-sample-001-overlay
    source_run_id: run_20260602_203450_c05550
    source_artifact_path: outputs/prediction_samples/sample_001_overlay.png
    reason: Second saved validation overlay for the 60-epoch p=0.075 dropout Run; selected because this sample improved only modestly despite the aggregate best-validation Dice gain.
  - figure_id: fig-xwide-dropout-p0075-epoch60-sample-001-heatmap
    source_run_id: run_20260602_203450_c05550
    source_artifact_path: outputs/prediction_samples/sample_001_probability_heatmap.png
    reason: Probability heatmap for the second saved validation sample, selected to inspect confidence distribution after extending the p=0.075 model to 60 epochs.
```

## Decision

Promote `run_20260602_203450_c05550` as the new current best in-contract Result by best-validation Dice and as the preferred p=0.075 recall-sensitive baseline. It exceeded both the primary comparison target and the p=0.05 leaderboard context, preserved best-epoch recall near the p=0.075 40-epoch baseline, and kept the final-vs-best Dice gap within the proposal guardrail.

Do not continue simple fixed-epoch extension beyond 60 epochs without additional diagnostics. The best epoch was 50 and the final epoch shifted toward lower recall, suggesting that further gains are more likely to need Harness-owned scheduler/early-stopping support or a new architecture hypothesis than another longer fixed budget.

## Next proposed change

Request a bounded failure-bucket Post-Run Evaluation for `run_20260602_203450_c05550`, comparing it to `run_20260601_162117_66bf89` and `run_20260602_101047_294476` on threshold sweep, false-negative-heavy positives, missed-positive masks, false-positive-heavy positives, and empty-mask false positives. If diagnostics confirm that the best-epoch gain is not mainly threshold-specific or false-positive-heavy, use the 60-epoch p=0.075 model as the baseline for the next in-contract architecture change; otherwise prefer a Capability Request for scheduler/early-stopping before more training-budget tuning.
