# 2026-05-29 medium U-Net Line Target auxiliary improvement

## Hypothesis

The `single_frame_medium_unet_line_auxiliary_w010` Candidate Experiment tested whether a modest capacity increase would improve the current best lower-weight Line Target auxiliary U-Net after the available augmentation presets failed to produce a better Result. The hypothesis was that a base-24 U-Net could represent thin or faint contrails better than the base-16 comparison target while preserving the useful Harness-derived Line Target auxiliary signal at weight 0.10.

## Candidate Experiment(s)

- Candidate Experiment path or ID: `candidates/single_frame_medium_unet_line_auxiliary_w010`
- Relevant Experiment Proposal: `candidates/single_frame_medium_unet_line_auxiliary_w010/PROPOSAL.md`
- Primary Comparison Target: `run_20260510_165335_b458c3` (`single_frame_small_unet_line_auxiliary_w010`), best-validation `val/dice` 0.7843769771696992.
- Secondary context: `run_20260525_202245_c3bcd5` (`single_frame_small_unet_line_auxiliary_w010_light_photometric`) nearly matched the comparison target with best/final `val/dice` 0.7823704848421766 but did not beat it.

## Run(s)

- Run ID: `run_20260529_155844_d8ebec`
- Dataset mode/subset: GVCCS Working Validation Split
- Harness/backend notes: completed Docker/GPU Run using `input_mode: single_frame_rgb`, primary `bce_dice` loss, Line Target auxiliary `weighted_bce` at weight 0.10, deterministic shuffle, and no augmentation. Batch size requested/effective: 16; max epochs: 30; no resource retry beyond the successful first attempt. The model summary reports 1,096,634 parameters, within the 10,000,000 parameter budget.

## Key metrics

| Run | Candidate Experiment | Epoch source | val/dice | val/iou | val/precision | val/recall | val/loss |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `run_20260510_165335_b458c3` | `single_frame_small_unet_line_auxiliary_w010` | best epoch 30 | 0.7843769771696992 | 0.6452468918723383 | 0.7717854808163968 | 0.7973861422181288 | 0.5639248168630936 |
| `run_20260525_202245_c3bcd5` | `single_frame_small_unet_line_auxiliary_w010_light_photometric` | best/final epoch 30 | 0.7823704848421766 | 0.6425357426891697 | 0.7912549779131358 | 0.7736832929168169 | 0.5693556516941116 |
| `run_20260529_155844_d8ebec` | `single_frame_medium_unet_line_auxiliary_w010` | best epoch 28 | 0.7994525273696043 | 0.665906634760569 | 0.8429952005805884 | 0.760187086855412 | 0.5434756466587701 |
| `run_20260529_155844_d8ebec` | `single_frame_medium_unet_line_auxiliary_w010` | final epoch 30 | 0.7786779644696994 | 0.6375697333026549 | 0.8428723045641596 | 0.723569865886675 | 0.564426931325805 |

The medium-capacity Run supports the proposal's primary success criterion at the best epoch: best-validation Dice improved over the comparison target by about 0.0151, exceeding the requested +0.003 margin. The improvement comes with much higher precision but lower recall than the prior best (`0.8430` precision / `0.7602` recall versus `0.7718` precision / `0.7974` recall). The final epoch regressed below the prior best, so decisions should use the persisted best-epoch model and avoid interpreting epoch 30 as the best model state.

## Qualitative observations

The saved first-N prediction samples are strong and consistent with the aggregate best-epoch improvement, although they are only two early validation examples:

- `val/000000`: Dice 0.8919, IoU 0.8049.
- `val/000001`: Dice 0.8189, IoU 0.6933.

Both saved samples exceed the corresponding light-photometric first-N examples reported in the previous Research Note (`0.8406` and `0.7771` Dice). The aggregate metrics suggest the medium model is more conservative than the previous best, trading recall for precision. This may be beneficial if false positives were a limiting issue, but the recall drop should be checked with a whole-validation failure-bucket diagnostic before assuming the capacity increase uniformly improves thin-contrail recovery.

Example first-N overlays and probability heatmap:

![Sample 0 overlay](../runs/run_20260529_155844_d8ebec/outputs/prediction_samples/sample_000_overlay.png)

![Sample 0 probability heatmap](../runs/run_20260529_155844_d8ebec/outputs/prediction_samples/sample_000_probability_heatmap.png)

![Sample 1 overlay](../runs/run_20260529_155844_d8ebec/outputs/prediction_samples/sample_001_overlay.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-medium-unet-sample-000-overlay
    source_run_id: run_20260529_155844_d8ebec
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: Shows the strongest saved first-N validation prediction from the medium-capacity Run, illustrating high-quality mask overlap after increasing U-Net width.
  - figure_id: fig-medium-unet-sample-000-heatmap
    source_run_id: run_20260529_155844_d8ebec
    source_artifact_path: outputs/prediction_samples/sample_000_probability_heatmap.png
    reason: Provides probability-level context for the high-Dice sample to inspect the more conservative high-precision behavior suggested by aggregate metrics.
  - figure_id: fig-medium-unet-sample-001-overlay
    source_run_id: run_20260529_155844_d8ebec
    source_artifact_path: outputs/prediction_samples/sample_001_overlay.png
    reason: Shows a second saved first-N validation prediction above 0.8 Dice, supporting that the improvement is not limited to a single qualitative example.
```

## Decision

Promote `run_20260529_155844_d8ebec` to the current best in-contract Result by best-validation Dice, with the important caveat that the final epoch regressed and the precision/recall balance shifted toward high precision and lower recall. The Candidate Experiment stayed within the Candidate Experiment Contract and did not trigger a Resource Failure. Treat capacity scaling from base 16 to base 24 as a useful in-contract lever for this family.

## Next proposed change

Before proposing another architecture change, request a bounded Whole-Validation Failure Analysis or failure-bucket Post-Run Evaluation for `run_20260529_155844_d8ebec`, focusing on whether the recall drop causes missed positive masks or thin-contrail false negatives despite the improved best Dice. If diagnostics are acceptable, the next Experiment Proposal could test a narrow training-knob follow-up for the medium model, such as preserving best-epoch behavior or improving recall, within the existing Harness-owned contract surface.
