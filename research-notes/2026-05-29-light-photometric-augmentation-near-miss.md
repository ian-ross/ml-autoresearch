# 2026-05-29 light-photometric augmentation near miss

## Hypothesis

The `single_frame_small_unet_line_auxiliary_w010_light_photometric` Candidate Experiment tested the final narrow augmentation-component control for the current Small U-Net plus lower-weight Line Target auxiliary branch. After `light_combined` and `light_geometric` both regressed, this Run asked whether Harness-owned photometric-only perturbations could preserve the unaugmented branch's strength while avoiding geometric augmentation harm.

## Candidate Experiment(s)

- Candidate Experiment path or ID: `candidates/single_frame_small_unet_line_auxiliary_w010_light_photometric`
- Relevant Experiment Proposal: `candidates/single_frame_small_unet_line_auxiliary_w010_light_photometric/PROPOSAL.md`
- Primary Comparison Target: `run_20260510_165335_b458c3` (`single_frame_small_unet_line_auxiliary_w010`), best-validation `val/dice` 0.7843769771696992.

## Run(s)

- Run ID: `run_20260525_202245_c3bcd5`
- Dataset mode/subset: GVCCS Working Validation Split
- Harness/backend notes: completed Docker/GPU Run using `input_mode: single_frame_rgb`, primary `bce_dice` loss, Line Target auxiliary `weighted_bce` at weight 0.10, deterministic shuffle, and effective `augmentation_policy: light_photometric`. Batch size requested/effective: 16; max epochs: 30; no resource retry beyond the successful first attempt.

## Key metrics

| Run | Candidate Experiment | Epoch source | val/dice | val/iou | val/precision | val/recall | val/loss |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `run_20260510_165335_b458c3` | `single_frame_small_unet_line_auxiliary_w010` | best epoch 30 | 0.7843769771696992 | 0.6452468918723383 | 0.7717854808163968 | 0.7973861422181288 | 0.5639248168630936 |
| `run_20260523_163405_8e18eb` | `single_frame_small_unet_line_auxiliary_w010_light_combined` | best epoch 26 | 0.717276204322084 | 0.5591821144496796 | 0.75108987313162 | 0.6863759233042218 | 0.6413536987687236 |
| `run_20260525_123213_541981` | `single_frame_small_unet_line_auxiliary_w010_light_geometric` | best/final epoch 30 | 0.7292688877723468 | 0.5738970902301385 | 0.7374074904570065 | 0.7213079718981288 | 0.6231930805803664 |
| `run_20260525_202245_c3bcd5` | `single_frame_small_unet_line_auxiliary_w010_light_photometric` | best/final epoch 30 | 0.7823704848421766 | 0.6425357426891697 | 0.7912549779131358 | 0.7736832929168169 | 0.5693556516941116 |

The photometric-only Run nearly matched but did not beat the unaugmented comparison target: best-validation Dice was lower by about 0.0020 and missed the proposal's success criterion of improving by at least 0.003. It clearly outperformed the two previous augmented variants, improving over `light_geometric` by about 0.0531 Dice and over `light_combined` by about 0.0651 Dice.

## Qualitative observations

The saved first-N prediction samples are stronger than the earlier augmented Runs and consistent with the near-match aggregate Result:

- `val/000000`: Dice 0.8406, IoU 0.7250.
- `val/000001`: Dice 0.7771, IoU 0.6354.

Compared with the unaugmented comparison target, the photometric Run trades slightly higher precision for lower recall (`val/precision` 0.7913 vs 0.7718; `val/recall` 0.7737 vs 0.7974). That trade-off is not enough to replace the current best Result, but it suggests photometric augmentation is much less harmful than the available geometric-containing presets.

Example first-N overlays and probability heatmap:

![Sample 0 overlay](../runs/run_20260525_202245_c3bcd5/outputs/prediction_samples/sample_000_overlay.png)

![Sample 0 probability heatmap](../runs/run_20260525_202245_c3bcd5/outputs/prediction_samples/sample_000_probability_heatmap.png)

![Sample 1 overlay](../runs/run_20260525_202245_c3bcd5/outputs/prediction_samples/sample_001_overlay.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-light-photometric-sample-000-overlay
    source_run_id: run_20260525_202245_c3bcd5
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: Shows the strongest saved first-N validation prediction from the light-photometric Run, supporting that photometric augmentation can preserve high-quality masks on some samples.
  - figure_id: fig-light-photometric-sample-000-heatmap
    source_run_id: run_20260525_202245_c3bcd5
    source_artifact_path: outputs/prediction_samples/sample_000_probability_heatmap.png
    reason: Provides probability-level context for the same high-Dice sample to inspect confidence after photometric-only augmentation.
  - figure_id: fig-light-photometric-sample-001-overlay
    source_run_id: run_20260525_202245_c3bcd5
    source_artifact_path: outputs/prediction_samples/sample_001_overlay.png
    reason: Shows a second first-N validation prediction near the aggregate Dice, illustrating the remaining gap to the unaugmented comparison target.
```

## Decision

Do not replace the current best Result. Keep `run_20260510_165335_b458c3` as the best known in-contract Run. Treat `run_20260525_202245_c3bcd5` as a completed near-miss research outcome rather than a candidate bug: it stayed within contract and avoided the severe augmentation regression, but it did not satisfy the declared improvement threshold.

## Next proposed change

Stop immediate augmentation-component variants for this architecture. The available Harness-owned augmentation presets now appear ordered as photometric-only near-neutral, geometric harmful, and combined most harmful. The next Experiment Proposal should shift to a different in-contract lever while keeping the lower-weight Line Target auxiliary branch and `run_20260510_165335_b458c3` as the primary Comparison Target; reasonable next directions are a modest architecture-capacity change, a bounded training-knob change, or a Capability Request only if the desired next lever is outside the current Candidate Experiment Contract.
