# 2026-05-30 wide U-Net Line Target auxiliary improvement

## Hypothesis

Increasing the current best Line Target auxiliary U-Net from base 32 to base 48 tested whether straightforward capacity scaling remained useful under the Candidate Experiment Contract. The expected effect was a modest best-validation Dice gain over `run_20260530_073538_f7c8b7` without severe precision/recall collapse or Resource Failure.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_wide_unet_line_auxiliary_w010`
- Candidate Experiment path: `candidates/single_frame_wide_unet_line_auxiliary_w010`
- Relevant Experiment Proposal: `candidates/single_frame_wide_unet_line_auxiliary_w010/PROPOSAL.md`
- Primary Comparison Target: `run_20260530_073538_f7c8b7` / `single_frame_large_unet_line_auxiliary_w010`
- Contract choices: Single-Frame RGB Input, mask logits output, Line Target auxiliary output with `weighted_bce` weight 0.10, deterministic shuffle Sampling Policy, no augmentation, `bce_dice`, AdamW, learning rate 0.001, requested/effective batch size 12, max epochs 30.

## Run(s)

- Run ID: `run_20260530_101019_def893`
- Dataset mode/subset: GVCCS Working Validation Split
- Harness/backend notes: completed on CUDA; best-validation model artifact persisted at `outputs/models/best_epoch_model.pt`; model summary reports 4,380,914 parameters, below the 10,000,000 parameter budget.

## Key metrics

Best-validation metrics, selected by max `val/dice`:

| Run | Candidate Experiment | Best epoch | val/dice | val/iou | val/precision | val/recall | val/loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `run_20260530_073538_f7c8b7` | `single_frame_large_unet_line_auxiliary_w010` | 30 | 0.815802 | 0.688907 | 0.814527 | 0.817081 | 0.528378 |
| `run_20260530_101019_def893` | `single_frame_wide_unet_line_auxiliary_w010` | 29 | 0.829165 | 0.708183 | 0.850682 | 0.808710 | 0.512781 |

The wide base-48 U-Net improved best-validation Dice by about `+0.01336` and IoU by about `+0.01928` over the base-32 comparison target. The improvement shifted the precision/recall balance toward higher precision (`+0.03616`) with a small recall decrease (`-0.00837`).

Final completed-epoch metrics for `run_20260530_101019_def893` were slightly below the best epoch: final `val/dice` 0.826942, `val/iou` 0.704945, `val/precision` 0.858678, `val/recall` 0.797468, and `val/loss` 0.515518. Decisions should therefore prefer the persisted best-epoch model, but the final-vs-best gap is small compared with the gain over the base-32 comparison target.

## Qualitative observations

The saved first-N prediction samples were both strong according to `outputs/prediction_samples/samples.json`:

- `val/000000`: Dice 0.8511, IoU 0.7407.
- `val/000001`: Dice 0.8497, IoU 0.7386.

These first-N samples are consistent and high-quality, though they do not by themselves expose whole-validation failure buckets. Compared with the previous base-32 note, the saved sample metrics are less peaked on sample 0 but slightly stronger on sample 1, matching the aggregate picture of a more precise but still high-recall model.

## Research Figures

The following existing Harness artifacts are referenced for provenance rather than copied into this note.

```research-figures
figures:
  - figure_id: fig-wide-unet-sample-000-overlay
    source_run_id: run_20260530_101019_def893
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: First saved validation overlay for the new best wide Line Target auxiliary U-Net; paired sample metrics are Dice 0.8511 and IoU 0.7407.
  - figure_id: fig-wide-unet-sample-001-overlay
    source_run_id: run_20260530_101019_def893
    source_artifact_path: outputs/prediction_samples/sample_001_overlay.png
    reason: Second saved validation overlay for the new best wide Line Target auxiliary U-Net; paired sample metrics are Dice 0.8497 and IoU 0.7386.
  - figure_id: fig-wide-unet-sample-000-heatmap
    source_run_id: run_20260530_101019_def893
    source_artifact_path: outputs/prediction_samples/sample_000_probability_heatmap.png
    reason: Probability heatmap for the first saved validation sample, useful for inspecting the higher-precision behavior suggested by aggregate metrics.
```

## Decision

The hypothesis is supported. Base-48 capacity scaling within the Line Target auxiliary U-Net family produced a new best Result by best-validation Dice, exceeded the proposal success threshold of `+0.002` over the base-32 comparison target, stayed within the Candidate Experiment Contract, and did not trigger a Resource Failure.

Promote `run_20260530_101019_def893` as the current best Result for the Research Loop. Simple capacity scaling has now improved from base 16 to 24 to 32 to 48, but the latest gain came with lower recall and higher precision, so the next step should verify whether this new balance is broadly beneficial rather than only strong on aggregate Dice.

## Next proposed change

Request a bounded Post-Run Evaluation for `run_20260530_101019_def893`, preferably failure-bucket review or whole-validation failure analysis, focused on false-negative-heavy and missed-positive-mask cases to check whether the recall drop hides thin-contrail misses. If diagnostics look acceptable, consider either one cautious in-contract capacity follow-up or a precision/recall calibration-oriented hypothesis using existing Harness-owned options rather than returning immediately to augmentation presets, which recently regressed in this architecture family.
