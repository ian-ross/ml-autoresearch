# 2026-05-30 extra-wide U-Net Line Target auxiliary improvement

## Hypothesis

Increasing the current best base-48 Line Target auxiliary U-Net to base 64 tested whether one more in-contract capacity step would improve Contrail Mask Dice on the GVCCS Working Validation Split. The expected benefit was better thin-contrail and ambiguous-background segmentation from added representation capacity, while the main risks were overfitting and resource pressure.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_xwide_unet_line_auxiliary_w010`
- Candidate Experiment path: `candidates/single_frame_xwide_unet_line_auxiliary_w010`
- Relevant Experiment Proposal: `candidates/single_frame_xwide_unet_line_auxiliary_w010/PROPOSAL.md`
- Primary Comparison Target: `run_20260530_101019_def893` / `single_frame_wide_unet_line_auxiliary_w010`
- Contract choices: Single-Frame RGB Input, mask logits output, Line Target auxiliary output with `weighted_bce` weight 0.10, deterministic shuffle Sampling Policy, no augmentation, `bce_dice`, AdamW, learning rate 0.001, requested/effective batch size 8, max epochs 30.

## Run(s)

- Run ID: `run_20260530_134005_6f20b1`
- Dataset mode/subset: GVCCS Working Validation Split
- Harness/backend notes: completed on CUDA; best-validation model artifact persisted at `outputs/models/best_epoch_model.pt`; model summary reports 7,785,794 parameters, below the 10,000,000 parameter budget.

## Key metrics

Best-validation metrics, selected by max `val/dice`:

| Run | Candidate Experiment | Best epoch | val/dice | val/iou | val/precision | val/recall | val/loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `run_20260530_101019_def893` | `single_frame_wide_unet_line_auxiliary_w010` | 29 | 0.829165 | 0.708183 | 0.850682 | 0.808710 | 0.512781 |
| `run_20260530_134005_6f20b1` | `single_frame_xwide_unet_line_auxiliary_w010` | 28 | 0.832669 | 0.713310 | 0.862413 | 0.804908 | 0.510368 |

The extra-wide base-64 U-Net improved best-validation Dice by about `+0.00350` and IoU by about `+0.00513` over the base-48 comparison target. The improvement came from higher precision (`+0.01173`) with a small recall decrease (`-0.00380`).

Final completed-epoch metrics for `run_20260530_134005_6f20b1` were substantially below the best epoch: final `val/dice` 0.808548, `val/iou` 0.678624, `val/precision` 0.868045, `val/recall` 0.756684, and `val/loss` 0.539795. Decisions should therefore use the persisted best-epoch model, and the final-vs-best gap is a warning that this capacity step may be less stable near the end of the fixed 30-epoch budget.

## Qualitative observations

The saved first-N prediction samples were acceptable but weaker than the aggregate best-validation result might suggest:

- `val/000000`: Dice 0.7846, IoU 0.6456.
- `val/000001`: Dice 0.8333, IoU 0.7143.

Compared with the prior base-48 note, the first saved sample is noticeably weaker while the second remains strong. This does not invalidate the whole-validation gain, but it reinforces the need for bounded failure-bucket diagnostics before treating base 64 as robustly better. The metrics pattern suggests the extra-wide model is more precision-oriented and may be losing some recall on harder or smaller positives.

## Research Figures

The following existing Harness artifacts are referenced for provenance rather than copied into this note.

```research-figures
figures:
  - figure_id: fig-xwide-unet-sample-000-overlay
    source_run_id: run_20260530_134005_6f20b1
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: First saved validation overlay for the extra-wide Line Target auxiliary U-Net; paired sample metrics are Dice 0.7846 and IoU 0.6456, illustrating that the best aggregate Run still has nontrivial sample-level errors.
  - figure_id: fig-xwide-unet-sample-001-overlay
    source_run_id: run_20260530_134005_6f20b1
    source_artifact_path: outputs/prediction_samples/sample_001_overlay.png
    reason: Second saved validation overlay for the extra-wide Line Target auxiliary U-Net; paired sample metrics are Dice 0.8333 and IoU 0.7143.
  - figure_id: fig-xwide-unet-sample-000-heatmap
    source_run_id: run_20260530_134005_6f20b1
    source_artifact_path: outputs/prediction_samples/sample_000_probability_heatmap.png
    reason: Probability heatmap for the weaker first saved validation sample, useful for inspecting the higher-precision lower-recall behavior suggested by aggregate metrics.
```

## Decision

The hypothesis is cautiously supported. Base-64 capacity scaling produced a small new best-validation Dice Result, stayed within the Candidate Experiment Contract and parameter budget, and avoided Resource Failure. However, the gain over base 48 is much smaller than the previous base-32 to base-48 jump, and the final-vs-best degradation is larger, suggesting diminishing returns and possible late-epoch instability.

Promote `run_20260530_134005_6f20b1` as the current best by best-validation Dice, but do not continue width scaling blindly. Treat base 64 as a candidate capacity ceiling unless diagnostics show clear failure-bucket improvement over base 48.

## Next proposed change

Request a bounded failure-bucket Post-Run Evaluation for `run_20260530_134005_6f20b1` and compare it directly with `eval_eval_2026_05_30_wide_unet_failure_buckets`. Focus on missed-positive masks, very-low-recall positives, and empty-mask false positives. If the diagnostic advantage is marginal or worse than base 48, pivot away from width scaling toward in-contract training refinements or a human-gated scheduler/early-stopping capability slice.
