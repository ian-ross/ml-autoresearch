# 2026-05-31 xwide U-Net bottleneck dropout improvement

## Hypothesis

Adding a small in-architecture `Dropout2d(p=0.10)` at the bottleneck of the extra-wide base-64 Line Target auxiliary U-Net tested whether light regularization could keep the current capacity ceiling's precision gains while reducing late-epoch instability and recovering recall. The expected result was similar or slightly higher best-validation Dice than `run_20260530_134005_6f20b1`, a smaller final-vs-best Dice gap, and fewer signs of overconfident precision bias.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_xwide_unet_line_aux_w010_dropout`
- Candidate Experiment path: `candidates/single_frame_xwide_unet_line_aux_w010_dropout`
- Relevant Experiment Proposal: `candidates/single_frame_xwide_unet_line_aux_w010_dropout/PROPOSAL.md`
- Primary Comparison Target: `run_20260530_134005_6f20b1` / `single_frame_xwide_unet_line_auxiliary_w010`
- Secondary context: `run_20260530_101019_def893` / `single_frame_wide_unet_line_auxiliary_w010`
- Contract choices: Single-Frame RGB Input, mask logits output, Line Target auxiliary output with `weighted_bce` weight 0.10, deterministic shuffle Sampling Policy, no augmentation, `bce_dice`, AdamW, learning rate 0.001, requested/effective batch size 8, max epochs 30.

## Run(s)

- Run ID: `run_20260530_180658_0af8a8`
- Dataset mode/subset: GVCCS Working Validation Split
- Harness/backend notes: completed on CUDA with no Resource Failure retry; best-validation model artifact persisted at `outputs/models/best_epoch_model.pt`; model summary reports 7,785,794 parameters, unchanged from the xwide comparison and below the 10,000,000 parameter budget.

## Key metrics

Best-validation metrics, selected by max `val/dice`:

| Run | Candidate Experiment | Best epoch | val/dice | val/iou | val/precision | val/recall | val/loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `run_20260530_101019_def893` | `single_frame_wide_unet_line_auxiliary_w010` | 29 | 0.829165 | 0.708183 | 0.850682 | 0.808710 | 0.512781 |
| `run_20260530_134005_6f20b1` | `single_frame_xwide_unet_line_auxiliary_w010` | 28 | 0.832669 | 0.713310 | 0.862413 | 0.804908 | 0.510368 |
| `run_20260530_180658_0af8a8` | `single_frame_xwide_unet_line_aux_w010_dropout` | 29 | 0.833852 | 0.715049 | 0.856300 | 0.812552 | 0.510518 |

The dropout variant improves best-validation Dice by about `+0.00118` over the xwide comparison and `+0.00469` over the wide comparison. More importantly, the precision/recall balance shifts in the desired direction: compared with plain xwide, precision drops modestly (`-0.00611`) but remains above the wide comparison, while recall improves by `+0.00764` and also exceeds the wide comparison.

Final completed-epoch metrics for `run_20260530_180658_0af8a8` were `val/dice` 0.820063, `val/iou` 0.695005, `val/precision` 0.842684, `val/recall` 0.798624, and `val/loss` 0.528300. The best-to-final Dice gap is about `0.01379`, materially smaller than the plain xwide gap of about `0.02412` but still larger than the wide base-48 gap. This supports the regularization hypothesis while still arguing for best-epoch model use and additional diagnostics.

## Qualitative observations

The saved first-N prediction samples are stronger than the corresponding plain xwide first-N samples:

- `val/000000`: Dice 0.8462, IoU 0.7333.
- `val/000001`: Dice 0.8613, IoU 0.7564.

For context, the plain xwide note recorded `val/000000` Dice 0.7846 and `val/000001` Dice 0.8333. These two saved samples are not a full-validation diagnostic, but they are consistent with the aggregate pattern: dropout did not merely lower confidence globally; it appears to preserve high-quality masks on the first saved positives while improving recall-oriented aggregate metrics.

## Research Figures

The following existing Harness artifacts are referenced for provenance rather than copied into this note.

```research-figures
figures:
  - figure_id: fig-xwide-dropout-sample-000-overlay
    source_run_id: run_20260530_180658_0af8a8
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: First saved validation overlay for the xwide dropout variant; paired sample metrics are Dice 0.8462 and IoU 0.7333, useful for comparing against the weaker first saved plain-xwide sample.
  - figure_id: fig-xwide-dropout-sample-001-overlay
    source_run_id: run_20260530_180658_0af8a8
    source_artifact_path: outputs/prediction_samples/sample_001_overlay.png
    reason: Second saved validation overlay for the xwide dropout variant; paired sample metrics are Dice 0.8613 and IoU 0.7564, illustrating that the regularized model retains strong qualitative masks on saved positives.
  - figure_id: fig-xwide-dropout-sample-000-heatmap
    source_run_id: run_20260530_180658_0af8a8
    source_artifact_path: outputs/prediction_samples/sample_000_probability_heatmap.png
    reason: Probability heatmap for the first saved validation sample, useful for checking whether bottleneck dropout softened or fragmented the predicted contrail region.
```

## Decision

The hypothesis is supported. Bottleneck dropout is now the best observed in-contract Result by best-validation Dice, improves recall over the plain xwide capacity ceiling, and reduces the late-epoch degradation that motivated the experiment. It should replace `run_20260530_134005_6f20b1` as the current best by best-validation Dice, pending bounded failure-bucket diagnostics.

Do not continue pure width scaling. The useful change was regularization at the existing base-64 capacity, not additional capacity. The remaining risk is that the recall gain may increase false positives or leave small-positive misses unresolved, which requires a Post-Run Evaluation rather than another Candidate Experiment.

## Next proposed change

Request a bounded failure-bucket Post-Run Evaluation for `run_20260530_180658_0af8a8` and compare it with `eval_eval_2026_05_30_xwide_unet_failure_buckets` and `eval_eval_2026_05_30_wide_unet_failure_buckets`. Focus on missed-positive masks, false-negative-heavy samples, empty-mask false positives, and the threshold sweep optimum. If diagnostics confirm better recall without a large false-positive penalty, use the xwide dropout model as the new base for any later in-contract refinements; otherwise pivot to a human-gated threshold selection, scheduler, or early-stopping capability slice.
