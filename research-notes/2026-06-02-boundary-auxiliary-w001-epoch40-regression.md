# 2026-06-02 boundary auxiliary w=0.01 epoch-40 regression

## Hypothesis

`single_frame_xwide_unet_line_boundary_aux_w010_w001_dropout_p0075_epoch40` tested whether dropping the Boundary Target auxiliary weight from 0.03 to 0.01 would make boundary supervision act as weak edge-local regularization rather than the high-precision / low-recall shift seen in `run_20260601_230602_718182`.

The intended comparison target was the safer 40-epoch p=0.075 extra-wide Line Target U-Net, `run_20260601_162117_66bf89`. The current leaderboard context is `run_20260602_101047_294476`, the p=0.05 dropout variant, but that Result is already precision-biased and lower-recall.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_xwide_unet_line_boundary_aux_w010_w001_dropout_p0075_epoch40`
- Candidate Experiment path: `candidates/single_frame_xwide_unet_line_boundary_aux_w010_w001_dropout_p0075_epoch40`
- Relevant Experiment Proposal: `candidates/single_frame_xwide_unet_line_boundary_aux_w010_w001_dropout_p0075_epoch40/PROPOSAL.md`
- Primary Comparison Target: `run_20260601_162117_66bf89` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40`
- Contract choices: Single-Frame RGB Input, mask logits output, Line Target auxiliary output with `weighted_bce` weight 0.10, Boundary Target auxiliary output with `weighted_bce` weight 0.01, deterministic shuffle Sampling Policy, no augmentation, `bce_dice`, AdamW, learning rate 0.001, requested/effective batch size 8, max epochs 40.

## Run(s)

- Run ID: `run_20260602_143432_e3119d`
- Dataset mode/subset: GVCCS Working Validation Split
- Harness/backend notes: completed on CUDA with no Resource Failure retry. Best-validation model artifact persisted at `outputs/models/best_epoch_model.pt`. Model summary reports 7,785,859 parameters, effectively unchanged from the boundary w=0.03 variant and below the 10,000,000 parameter budget.

## Key metrics

Best-validation metrics, selected by max `val/dice`:

| Run | Candidate Experiment | Boundary weight | Best epoch | val/dice | val/iou | val/precision | val/recall | val/loss | val/total_loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `run_20260601_162117_66bf89` | `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40` | none | 40 | 0.843361 | 0.729148 | 0.855490 | 0.831572 | 0.500193 | 0.510662 |
| `run_20260601_230602_718182` | `single_frame_xwide_unet_line_boundary_aux_w010_w003_dropout_p0075_epoch40` | 0.03 | 40 | 0.841019 | 0.725654 | 0.873355 | 0.810993 | 0.504293 | 0.519100 |
| `run_20260602_143432_e3119d` | `single_frame_xwide_unet_line_boundary_aux_w010_w001_dropout_p0075_epoch40` | 0.01 | 40 | 0.843161 | 0.728849 | 0.866428 | 0.821111 | 0.497533 | 0.509004 |

The boundary w=0.01 run nearly matches the p=0.075 baseline Dice but does not exceed it: `-0.00020` Dice and `-0.00030` IoU. It improves precision by `+0.01094`, but recall drops by `-0.01046`, failing the proposal guardrail that recall remain within 0.005 of the p=0.075 baseline (`>=0.826572`).

Compared with boundary w=0.03, lowering the boundary weight helped: Dice improved by `+0.00214`, recall recovered by `+0.01012`, and total loss improved. However, even the very low boundary weight still preserves the same directional effect: tighter, higher-precision masks at the default threshold, with too much recall loss for a general-purpose successor.

Final completed-epoch metrics are identical to best-validation metrics because the best epoch was epoch 40.

## Qualitative observations

The saved first-N prediction samples are mixed and again not representative enough to override aggregate metrics:

- `val/000000`: Dice 0.8993 and IoU 0.8171, better than both the p=0.075 baseline sample reported earlier (Dice 0.8645 / IoU 0.7614) and boundary w=0.03 (Dice 0.8759 / IoU 0.7792).
- `val/000001`: Dice 0.8286 and IoU 0.7073, worse than the p=0.075 baseline sample (Dice 0.8533 / IoU 0.7442) and boundary w=0.03 (Dice 0.8571 / IoU 0.7500).

This pattern matches the metrics: a weak boundary head can make some visible masks cleaner, but the campaign still pays for it through lower recall across the Working Validation Split.

![boundary w001 sample 000 overlay](../runs/run_20260602_143432_e3119d/outputs/prediction_samples/sample_000_overlay.png)

![boundary w001 sample 001 overlay](../runs/run_20260602_143432_e3119d/outputs/prediction_samples/sample_001_overlay.png)

![boundary w001 sample 001 heatmap](../runs/run_20260602_143432_e3119d/outputs/prediction_samples/sample_001_probability_heatmap.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-boundary-w001-epoch40-sample-000-overlay
    source_run_id: run_20260602_143432_e3119d
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: First saved validation overlay for the boundary w=0.01 Run; selected because this sample improved strongly even though aggregate Dice only nearly matched the non-boundary baseline.
  - figure_id: fig-boundary-w001-epoch40-sample-001-overlay
    source_run_id: run_20260602_143432_e3119d
    source_artifact_path: outputs/prediction_samples/sample_001_overlay.png
    reason: Second saved validation overlay for the boundary w=0.01 Run; selected because this sample regressed and illustrates the mixed qualitative behavior behind the aggregate recall loss.
  - figure_id: fig-boundary-w001-epoch40-sample-001-heatmap
    source_run_id: run_20260602_143432_e3119d
    source_artifact_path: outputs/prediction_samples/sample_001_probability_heatmap.png
    reason: Probability heatmap for the regressing saved sample, selected to inspect whether the very-low-weight Boundary Target still shifts outputs toward tighter masks.
```

## Decision

Do not promote `run_20260602_143432_e3119d`. It is a useful negative ablation: reducing Boundary Target weight from 0.03 to 0.01 almost recovered baseline Dice, but it did not preserve recall and did not beat either the p=0.075 safer baseline or the p=0.05 leaderboard Result.

Stop immediate Boundary Target auxiliary follow-up for this extra-wide Line Target U-Net family. The repeated boundary variants show a consistent precision-biased effect at the default threshold rather than the desired recall-preserving edge regularization.

## Next proposed change

Prefer a non-boundary, recall-oriented in-contract change. The next proposal should use `run_20260601_162117_66bf89` as the safer recall baseline while reporting `run_20260602_101047_294476` as the leaderboard best, and should target tiny missed positives or large-mask under-segmentation without adding another precision-biased auxiliary signal.
