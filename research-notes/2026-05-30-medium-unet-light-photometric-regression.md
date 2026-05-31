# 2026-05-30 medium U-Net light photometric augmentation regression

## Hypothesis

The `single_frame_medium_unet_line_auxiliary_w010_light_photometric` Candidate Experiment tested whether the Harness-owned `light_photometric` augmentation preset could improve robustness for the current best medium Line Target auxiliary U-Net without the regressions seen from geometric augmentation. The primary Comparison Target was `run_20260529_155844_d8ebec` (`single_frame_medium_unet_line_auxiliary_w010`), whose best-validation `val/dice` was 0.7994525273696043.

## Candidate Experiment(s)

- Candidate Experiment path or ID: `candidates/single_frame_medium_unet_line_auxiliary_w010_light_photometric`
- Relevant Experiment Proposal: `candidates/single_frame_medium_unet_line_auxiliary_w010_light_photometric/PROPOSAL.md`
- Primary Comparison Target: `run_20260529_155844_d8ebec` / `single_frame_medium_unet_line_auxiliary_w010`
- Secondary context: the small-U-Net `light_photometric` Run `run_20260525_202245_c3bcd5` was a near miss, while geometric-containing augmentation presets regressed.

## Run(s)

- Run ID: `run_20260530_043004_3e7aca`
- Dataset mode/subset: GVCCS Working Validation Split
- Harness/backend notes: completed Docker/GPU Run using `input_mode: single_frame_rgb`, primary `bce_dice` loss, Harness-derived Line Target auxiliary `weighted_bce` at weight 0.10, deterministic shuffle, and `light_photometric` augmentation. Batch size requested/effective: 16; max epochs: 30; no resource retry beyond the successful first attempt. The model summary reports 1,096,634 parameters, matching the unaugmented medium-family baseline and staying within the 10,000,000 parameter budget.

## Key metrics

| Run | Candidate Experiment | Epoch source | val/dice | val/iou | val/precision | val/recall | val/loss |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `run_20260510_165335_b458c3` | `single_frame_small_unet_line_auxiliary_w010` | best epoch 30 | 0.7843769771696992 | 0.6452468918723383 | 0.7717854808163968 | 0.7973861422181288 | 0.5639248168630936 |
| `run_20260529_155844_d8ebec` | `single_frame_medium_unet_line_auxiliary_w010` | best epoch 28 | 0.7994525273696043 | 0.665906634760569 | 0.8429952005805884 | 0.760187086855412 | 0.5434756466587701 |
| `run_20260529_210959_8b9a18` | `single_frame_medium_unet_line_auxiliary_w005` | best epoch 28 | 0.7945989165999967 | 0.659198774202797 | 0.8317512055349874 | 0.7606237182899099 | 0.5469463209040399 |
| `run_20260530_043004_3e7aca` | `single_frame_medium_unet_line_auxiliary_w010_light_photometric` | best epoch 29 | 0.7970676596635833 | 0.6626038996014364 | 0.8082276668042074 | 0.7862116496743524 | 0.556723261498706 |
| `run_20260530_043004_3e7aca` | `single_frame_medium_unet_line_auxiliary_w010_light_photometric` | final epoch 30 | 0.7874375578215196 | 0.6493995941411677 | 0.7789114115015496 | 0.7961524283405604 | 0.5631353898660234 |

The light-photometric medium Run did not meet the proposal's promotion criterion. Its best-validation Dice was 0.00238 below the unaugmented medium baseline, which is within the match band but came with a large precision drop from 0.8430 to 0.8082. Recall improved from 0.7602 to 0.7862, but not enough to offset the precision loss in aggregate Dice. The Run remains above the older small-U-Net line-auxiliary baseline, but it does not replace the current best medium baseline.

## Qualitative observations

The saved first-N validation samples are mixed relative to the unaugmented medium baseline:

- `val/000000`: Dice 0.7360, IoU 0.5823 for the light-photometric medium Run versus 0.8919 Dice for the unaugmented medium Run.
- `val/000001`: Dice 0.8281, IoU 0.7067 for the light-photometric medium Run versus 0.8189 Dice for the unaugmented medium Run.

These first-N samples should not be over-weighted, but they are consistent with the aggregate precision/recall shift: photometric augmentation appears to make the model less conservative, recovering some positives while allowing more false-positive area. The first saved overlay is substantially worse than the unaugmented medium example, while the second is slightly stronger.

Example first-N overlays and probability heatmap:

![Sample 0 overlay](../runs/run_20260530_043004_3e7aca/outputs/prediction_samples/sample_000_overlay.png)

![Sample 0 probability heatmap](../runs/run_20260530_043004_3e7aca/outputs/prediction_samples/sample_000_probability_heatmap.png)

![Sample 1 overlay](../runs/run_20260530_043004_3e7aca/outputs/prediction_samples/sample_001_overlay.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-medium-unet-light-photometric-sample-000-overlay
    source_run_id: run_20260530_043004_3e7aca
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: Shows the first saved validation prediction where the light-photometric medium Run is much weaker than the unaugmented medium baseline sample.
  - figure_id: fig-medium-unet-light-photometric-sample-000-heatmap
    source_run_id: run_20260530_043004_3e7aca
    source_artifact_path: outputs/prediction_samples/sample_000_probability_heatmap.png
    reason: Provides probability-level context for the weaker first sample and the aggregate shift toward lower precision.
  - figure_id: fig-medium-unet-light-photometric-sample-001-overlay
    source_run_id: run_20260530_043004_3e7aca
    source_artifact_path: outputs/prediction_samples/sample_001_overlay.png
    reason: Shows a second saved validation prediction where the light-photometric medium Run remains competitive despite the aggregate regression.
```

## Decision

Do not promote `run_20260530_043004_3e7aca` to current best. Keep `run_20260529_155844_d8ebec` (`single_frame_medium_unet_line_auxiliary_w010`) as the current best in-contract Result by best-validation Dice. Treat `light_photometric` augmentation as not yet useful for the medium U-Net family under the current training policy: it improves recall relative to the unaugmented medium Run but sacrifices too much precision and does not improve Dice.

## Next proposed change

Avoid further immediate augmentation-preset variants for this family. The available augmentation evidence now suggests combined and geometric presets regress strongly, and photometric augmentation is at best a precision/recall tradeoff rather than a best-Dice improvement. The next useful step is to return to the unaugmented medium w0.10 baseline and use the completed failure-bucket Post-Run Evaluation for `run_20260529_155844_d8ebec` to decide whether its lower recall is a serious qualitative problem. If recall-specific failures are acceptable, continue architecture exploration from the unaugmented medium baseline; if they are severe, prefer a different in-contract lever rather than more augmentation.
