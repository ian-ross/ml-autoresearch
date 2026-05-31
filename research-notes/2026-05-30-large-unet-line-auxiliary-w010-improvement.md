# 2026-05-30 large U-Net Line Target auxiliary improvement

## Hypothesis

Increasing the current best Line Target auxiliary U-Net from base 24 to base 32 would test whether more representation capacity remains a useful in-contract lever for Ground-Camera Contrail Detection. The expected effect was a small best-validation Dice gain over the base-24 comparison target without a severe precision/recall collapse or Resource Failure.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_large_unet_line_auxiliary_w010`
- Candidate Experiment path: `candidates/single_frame_large_unet_line_auxiliary_w010`
- Relevant Experiment Proposal: `candidates/single_frame_large_unet_line_auxiliary_w010/PROPOSAL.md`
- Contract choices: Single-Frame RGB Input, mask logits output, Line Target auxiliary output with `weighted_bce` weight 0.10, deterministic shuffle Sampling Policy, no augmentation, `bce_dice`, AdamW, batch size 16, max epochs 30.

## Run(s)

- Run ID: `run_20260530_073538_f7c8b7`
- Dataset mode/subset: GVCCS Working Validation Split
- Harness/backend notes: completed on CUDA; best-validation model artifact persisted at `outputs/models/best_epoch_model.pt`.
- Primary Comparison Target: `run_20260529_155844_d8ebec` / `single_frame_medium_unet_line_auxiliary_w010`.

## Key metrics

Best-validation metrics, selected by max `val/dice`:

| Run | Candidate Experiment | Best epoch | val/dice | val/iou | val/precision | val/recall | val/loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `run_20260529_155844_d8ebec` | `single_frame_medium_unet_line_auxiliary_w010` | 28 | 0.799453 | 0.665907 | 0.842995 | 0.760187 | 0.543476 |
| `run_20260530_073538_f7c8b7` | `single_frame_large_unet_line_auxiliary_w010` | 30 | 0.815802 | 0.688907 | 0.814527 | 0.817081 | 0.528378 |

The large U-Net improved best-validation Dice by about `+0.01635` and IoU by about `+0.02300` over the medium comparison target. The improvement came with a recall increase of about `+0.05689` and a precision decrease of about `-0.02847`, producing a much more balanced precision/recall profile than the medium base-24 run.

Final completed-epoch metrics for `run_20260530_073538_f7c8b7` match the best-validation metrics at epoch 30, so there is no final-vs-best degradation in this Run.

## Qualitative observations

The saved first-N prediction samples were both strong according to per-sample metrics in `outputs/prediction_samples/samples.json`:

- `val/000000`: Dice 0.8750, IoU 0.7778.
- `val/000001`: Dice 0.8414, IoU 0.7262.

These first-N samples do not expose obvious failure buckets by themselves. Because this Run is now the best Result and shifted the best family toward higher recall, the next useful diagnostic is a bounded failure-bucket or whole-validation review to check whether the recall gain introduced unacceptable empty-mask false positives or diffuse over-segmentation outside the first-N samples.

## Research Figures

The following existing Harness artifacts are referenced for provenance rather than copied into this note.

```research-figures
figures:
  - figure_id: fig-large-unet-sample-000-overlay
    source_run_id: run_20260530_073538_f7c8b7
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: First saved validation overlay for the new best large Line Target auxiliary U-Net; paired sample metrics are Dice 0.8750 and IoU 0.7778.
  - figure_id: fig-large-unet-sample-001-overlay
    source_run_id: run_20260530_073538_f7c8b7
    source_artifact_path: outputs/prediction_samples/sample_001_overlay.png
    reason: Second saved validation overlay for the new best large Line Target auxiliary U-Net; paired sample metrics are Dice 0.8414 and IoU 0.7262.
```

## Decision

The hypothesis is supported. Base-32 capacity scaling within the Line Target auxiliary U-Net family produced a clear new best Result under the current Candidate Experiment Contract, exceeding the proposal success threshold of `+0.003` Dice over the base-24 comparison target and avoiding Resource Failure.

Promote `run_20260530_073538_f7c8b7` as the current best Result for the Research Loop. Avoid further augmentation changes until the new base-32 model has been diagnostically checked, because recent `light_photometric`, `light_geometric`, and `light_combined` augmentation variants did not improve the previous best family.

## Next proposed change

Request a bounded Post-Run Evaluation for `run_20260530_073538_f7c8b7`, preferably failure-bucket review or whole-validation failure analysis, to inspect false-positive-heavy, false-negative-heavy, empty-mask false-positive, and missed-positive cases for the new high-recall best model. If diagnostics look acceptable, the next architecture hypothesis can test whether base-32 is the capacity sweet spot or whether a safer refinement, such as modest regularization or a base-32 variant with a slightly adjusted auxiliary balance, improves precision without sacrificing the new recall gain.
