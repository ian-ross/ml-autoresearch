# Research Note: Lower-weight Line Target auxiliary Small U-Net

## Summary

`single_frame_small_unet_line_auxiliary_w010` tested whether reducing the Harness-derived Line Target auxiliary loss weight from `0.25` to `0.10` would preserve the line-auxiliary benefit while reducing overly permissive mask predictions. The Run completed successfully as `run_20260510_165335_b458c3` and is the best mounted Result by validation Dice: `0.7843769771696992`.

## Source Run

- Run: `run_20260510_165335_b458c3`
- Candidate Experiment: `single_frame_small_unet_line_auxiliary_w010`
- Candidate source copy: `/history/runs/run_20260510_165335_b458c3/candidate/`
- Resolved manifest: `/history/runs/run_20260510_165335_b458c3/resolved_manifest.yaml`
- Final metrics: `/history/runs/run_20260510_165335_b458c3/outputs/final_metrics.json`
- Best metrics: `/history/runs/run_20260510_165335_b458c3/outputs/best_metrics.json`
- Prediction samples: `/history/runs/run_20260510_165335_b458c3/outputs/prediction_samples/samples.json`

## Candidate setup

- Input mode: `single_frame_rgb`
- Output form: `mask_logits`
- Auxiliary target: `line_logits` trained with `weighted_bce`, weight `0.10`
- Sampling policy: `deterministic_shuffle`
- Augmentation policy: `none`
- Training: AdamW, learning rate `0.001`, batch size `16`, max epochs `30`
- Effective batch size: `16`; no resource retry was needed.

## Result metrics

| Metric | Value |
| --- | ---: |
| Best epoch | `30` |
| `val/dice` | `0.7843769771696992` |
| `val/iou` | `0.6452468918723383` |
| `val/precision` | `0.7717854808163968` |
| `val/recall` | `0.7973861422181288` |
| `val/loss` | `0.5639248168630936` |
| `val/total_loss` | `0.5785634115196555` |
| `train/loss` | `0.53199625346268` |
| `train/mask_loss` | `0.519328256022855` |
| `train/aux/line_loss` | `0.012667998311405915` |
| `val/aux/line_loss` | `0.014638596443472802` |

Validation Dice improved through training and reached its maximum at epoch 30. The mounted local Run history identifies this Run as the best completed Result by `val/dice`.

## Comparison context

The intended comparison targets were:

- in-contract baseline: `run_20260506_045020_84aac5`
- prior line-auxiliary challenger: `run_20260507_193004_0b4688`

Those exact Run artifacts are not mounted in `/history/runs/`, so this note does not claim an artifact-verified delta against them. From the mounted Runs, `run_20260510_165335_b458c3` is the clear current best.

## Qualitative observations

The 16 saved prediction samples have mean sample Dice about `0.5636`, with a wide spread from near-zero to `0.8706`. Positive samples often show useful thin-structure localization, while failure cases remain concentrated in very sparse or empty-mask examples.

Observed examples:

- `sample_000` (`val/000000`) has Dice `0.8193`, with 43 ground-truth positive pixels and 40 predicted positive pixels, suggesting good scale control on a thin positive sample.
- `sample_013` (`val/000013`) has Dice `0.8706`, with 47 ground-truth positive pixels and 38 predicted positive pixels, another strong sparse-positive example.
- `sample_003` and `sample_004` have weaker Dice (`0.4866` and `0.2759`) and under-predict broader or more difficult positive masks.
- `sample_012` and `sample_015` are empty-mask cases with 12 and 13 predicted positive pixels respectively, indicating residual false positives on negative examples.
- `sample_014` has only 2 ground-truth positive pixels and no predicted positives, illustrating missed tiny positives.

## Research Figures

```yaml
research-figures:
  - figure_id: fig_sample_000_overlay
    source_run_id: run_20260510_165335_b458c3
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: Good thin positive-sample segmentation with high Dice and similar predicted/ground-truth positive pixel counts.
  - figure_id: fig_sample_000_heatmap
    source_run_id: run_20260510_165335_b458c3
    source_artifact_path: outputs/prediction_samples/sample_000_probability_heatmap.png
    reason: Shows confidence distribution for a high-quality positive thin-structure prediction.
  - figure_id: fig_sample_013_overlay
    source_run_id: run_20260510_165335_b458c3
    source_artifact_path: outputs/prediction_samples/sample_013_overlay.png
    reason: Best saved qualitative example by sample Dice; useful for judging whether lower auxiliary weight preserves line-like localization.
  - figure_id: fig_sample_003_overlay
    source_run_id: run_20260510_165335_b458c3
    source_artifact_path: outputs/prediction_samples/sample_003_overlay.png
    reason: Weaker adjacent-window positive example showing remaining under-segmentation risk.
  - figure_id: fig_sample_012_overlay
    source_run_id: run_20260510_165335_b458c3
    source_artifact_path: outputs/prediction_samples/sample_012_overlay.png
    reason: Empty-mask false-positive example; relevant to the hypothesis that lower line-auxiliary weight may reduce permissive predictions.
  - figure_id: fig_sample_014_overlay
    source_run_id: run_20260510_165335_b458c3
    source_artifact_path: outputs/prediction_samples/sample_014_overlay.png
    reason: Tiny-positive missed-mask case, showing sensitivity limits on extremely sparse positives.
```

## Decision

Treat `single_frame_small_unet_line_auxiliary_w010` as the current best mounted Result and a promising successor to the prior line-auxiliary candidate. The lower auxiliary weight should remain in the active search branch.

Before proposing the next Candidate Experiment, request or run an approved Whole-Validation Failure Analysis for `run_20260510_165335_b458c3` to quantify false-positive and false-negative buckets across the full Working Validation Split. That diagnostic should decide whether the next hypothesis should tune threshold/auxiliary weight further, add an approved augmentation policy, or pursue the next contract-supported loss or target variation.
