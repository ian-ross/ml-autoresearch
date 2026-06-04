# 2026-06-02 boundary auxiliary w=0.03 epoch-40 regression

## Hypothesis

`single_frame_xwide_unet_line_boundary_aux_w010_w003_dropout_p0075_epoch40` tested whether adding a conservative Harness-derived Boundary Target auxiliary head at weight 0.03 to the current 40-epoch p=0.075 extra-wide Line Target U-Net would sharpen edge-local features, reduce broad false-positive spillover, and recover false-negative-heavy positives without overwhelming the successful Line Target supervision.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_xwide_unet_line_boundary_aux_w010_w003_dropout_p0075_epoch40`
- Candidate Experiment path: `candidates/single_frame_xwide_unet_line_boundary_aux_w010_w003_dropout_p0075_epoch40`
- Relevant Experiment Proposal: `candidates/single_frame_xwide_unet_line_boundary_aux_w010_w003_dropout_p0075_epoch40/PROPOSAL.md`
- Primary Comparison Target: `run_20260601_162117_66bf89` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40`
- Contract choices: Single-Frame RGB Input, mask logits output, Line Target auxiliary output with `weighted_bce` weight 0.10, Boundary Target auxiliary output with `weighted_bce` weight 0.03, deterministic shuffle Sampling Policy, no augmentation, `bce_dice`, AdamW, learning rate 0.001, requested/effective batch size 8, max epochs 40.

## Run(s)

- Run ID: `run_20260601_230602_718182`
- Dataset mode/subset: GVCCS Working Validation Split
- Harness/backend notes: completed on CUDA with no Resource Failure retry. This confirms that Boundary Target auxiliary training is now operational for this Candidate Experiment path. Best-validation model artifact persisted at `outputs/models/best_epoch_model.pt`. Model summary reports 7,785,859 parameters, effectively unchanged from the comparison target apart from one 1x1 auxiliary head and below the 10,000,000 parameter budget.

## Key metrics

Best-validation metrics, selected by max `val/dice`:

| Run | Candidate Experiment | Best epoch | val/dice | val/iou | val/precision | val/recall | val/loss | val/total_loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `run_20260601_162117_66bf89` | `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40` | 40 | 0.843361 | 0.729148 | 0.855490 | 0.831572 | 0.500193 | 0.510662 |
| `run_20260601_230602_718182` | `single_frame_xwide_unet_line_boundary_aux_w010_w003_dropout_p0075_epoch40` | 40 | 0.841019 | 0.725654 | 0.873355 | 0.810993 | 0.504293 | 0.519100 |

The boundary-auxiliary run is the second-best completed Result by best-validation Dice, but it does not beat its primary comparison target. It regresses `val/dice` by about `-0.00234` and `val/iou` by about `-0.00349`. Precision improves substantially by about `+0.01787`, but recall drops by about `-0.02058`, violating the proposal guardrail that neither precision nor recall should fall by more than 0.005. Validation loss and total loss also worsen.

Final completed-epoch metrics are identical to best-validation metrics because the best epoch was the final epoch 40. The result therefore does not show an early best-to-final instability; instead, the Boundary Target auxiliary loss appears to have shifted the trained model toward a more conservative/high-precision operating point at the default threshold.

## Qualitative observations

The first-N prediction samples improved slightly relative to the 40-epoch non-boundary comparison target despite the aggregate regression:

- `val/000000`: Dice 0.8759 and IoU 0.7792, compared with 0.8645 / 0.7614 in the comparison target note.
- `val/000001`: Dice 0.8571 and IoU 0.7500, compared with 0.8533 / 0.7442 in the comparison target note.

These two saved examples are not representative enough to override the full Working Validation Split metrics. They do, however, fit the likely mechanism: a boundary head may cleanly sharpen some visible masks while losing enough recall elsewhere to reduce aggregate Dice.

![boundary w003 sample 000 overlay](../runs/run_20260601_230602_718182/outputs/prediction_samples/sample_000_overlay.png)

![boundary w003 sample 001 overlay](../runs/run_20260601_230602_718182/outputs/prediction_samples/sample_001_overlay.png)

![boundary w003 sample 001 heatmap](../runs/run_20260601_230602_718182/outputs/prediction_samples/sample_001_probability_heatmap.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-boundary-w003-epoch40-sample-000-overlay
    source_run_id: run_20260601_230602_718182
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: First saved validation overlay for the boundary-auxiliary Run; selected because the first-N sample improved even though aggregate Dice regressed.
  - figure_id: fig-boundary-w003-epoch40-sample-001-overlay
    source_run_id: run_20260601_230602_718182
    source_artifact_path: outputs/prediction_samples/sample_001_overlay.png
    reason: Second saved validation overlay for the boundary-auxiliary Run; useful for checking whether visible sample quality matches the precision-biased aggregate metrics.
  - figure_id: fig-boundary-w003-epoch40-sample-001-heatmap
    source_run_id: run_20260601_230602_718182
    source_artifact_path: outputs/prediction_samples/sample_001_probability_heatmap.png
    reason: Probability heatmap for the second saved sample, selected to inspect whether Boundary Target supervision sharpens probabilities while reducing aggregate recall.
```

## Decision

Do not promote `run_20260601_230602_718182`; keep `run_20260601_162117_66bf89` as the current best Result. The Boundary Target auxiliary capability trained successfully, so this is useful research evidence rather than a contract failure, but weight 0.03 did not satisfy the proposal success criteria. The observed precision gain with recall loss suggests the conservative boundary auxiliary acts more like regularization toward tighter masks than a broad Dice improvement for the current best architecture.

## Next proposed change

Do not request immediate failure-bucket evaluation for this regressing boundary variant. Return to non-boundary hypotheses around the current best 40-epoch p=0.075 Line Target U-Net, or revisit Boundary Target only with a clearly different question such as a much smaller boundary weight or threshold-sweep diagnostic after a broader need for high-precision behavior is established.
