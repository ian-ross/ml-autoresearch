# 2026-05-25 light-combined augmentation regression

## Hypothesis

The `single_frame_small_unet_line_auxiliary_w010_light_combined` Candidate Experiment tested whether the Harness-owned `light_combined` Augmentation Policy would improve robustness for the current best lower-weight Line Target auxiliary Small U-Net without changing the architecture, loss, optimizer, shuffle policy, or training budget.

## Candidate Experiment(s)

- Candidate Experiment path or ID: `candidates/single_frame_small_unet_line_auxiliary_w010_light_combined`
- Relevant Experiment Proposal: `candidates/single_frame_small_unet_line_auxiliary_w010_light_combined/PROPOSAL.md`
- Primary Comparison Target: `run_20260510_165335_b458c3` (`single_frame_small_unet_line_auxiliary_w010`), best-validation `val/dice` 0.7843769771696992.

## Run(s)

- Run ID: `run_20260523_163405_8e18eb`
- Dataset mode/subset: GVCCS Working Validation Split
- Harness/backend notes: completed Run using `input_mode: single_frame_rgb`, primary `bce_dice` loss, Line Target auxiliary `weighted_bce` at weight 0.10, deterministic shuffle, and effective `augmentation_policy: light_combined`. Batch size requested/effective: 16; max epochs: 30.

## Key metrics

| Run | Candidate Experiment | Epoch source | val/dice | val/iou | val/precision | val/recall | val/loss |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `run_20260510_165335_b458c3` | `single_frame_small_unet_line_auxiliary_w010` | best | 0.7843769771696992 | 0.6452468918723383 | 0.7717854808163968 | 0.7973861422181288 | unavailable here |
| `run_20260523_163405_8e18eb` | `single_frame_small_unet_line_auxiliary_w010_light_combined` | best epoch 26 | 0.717276204322084 | 0.5591821144496796 | 0.75108987313162 | 0.6863759233042218 | 0.6413536987687236 |
| `run_20260523_163405_8e18eb` | `single_frame_small_unet_line_auxiliary_w010_light_combined` | final epoch 30 | 0.7135662549156633 | 0.5546855853574437 | 0.6704639903049499 | 0.762591116135282 | 0.6536754936508606 |

The best epoch under `light_combined` augmentation regressed by about 0.0671 Dice relative to the unaugmented lower-weight Line Target auxiliary comparison target. This misses the proposal success criterion of improving over 0.7843769771696992 by at least 0.003.

## Qualitative observations

The saved first-N prediction samples are internally consistent with the aggregate regression rather than revealing a harness failure. Both sampled validation frames have moderate Dice near the final aggregate result:

- `val/000000`: Dice 0.7234, IoU 0.5667.
- `val/000001`: Dice 0.7167, IoU 0.5584.

The final-epoch metric shift also suggests calibration/threshold instability: precision dropped to 0.6705 while recall rose to 0.7626 between the best epoch and final epoch. The best epoch already underperformed the comparison Run, so this should be treated as a bad research result for the `light_combined` policy on this architecture rather than as a repairable candidate bug.

Example first-N overlays and probability heatmaps:

![Sample 0 overlay](../runs/run_20260523_163405_8e18eb/outputs/prediction_samples/sample_000_overlay.png)

![Sample 0 probability heatmap](../runs/run_20260523_163405_8e18eb/outputs/prediction_samples/sample_000_probability_heatmap.png)

![Sample 1 overlay](../runs/run_20260523_163405_8e18eb/outputs/prediction_samples/sample_001_overlay.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-light-combined-sample-000-overlay
    source_run_id: run_20260523_163405_8e18eb
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: Shows a representative first-N validation prediction from the regressed light-combined augmentation Run, with sample Dice near the final aggregate value.
  - figure_id: fig-light-combined-sample-000-heatmap
    source_run_id: run_20260523_163405_8e18eb
    source_artifact_path: outputs/prediction_samples/sample_000_probability_heatmap.png
    reason: Provides probability-level context for the same representative validation sample to inspect confidence/calibration after augmentation.
  - figure_id: fig-light-combined-sample-001-overlay
    source_run_id: run_20260523_163405_8e18eb
    source_artifact_path: outputs/prediction_samples/sample_001_overlay.png
    reason: Shows a second first-N validation prediction with similar sample Dice, supporting that the sampled qualitative behavior matches the aggregate regression.
```

## Decision

Do not replace the current best Result. Keep `run_20260510_165335_b458c3` as the best known in-contract Run. Classify `run_20260523_163405_8e18eb` as a completed but poor research outcome (`bad_research_result`) for this hypothesis: the Candidate Experiment ran successfully and stayed within the contract, but `light_combined` augmentation substantially reduced best-validation Dice.

## Next proposed change

Avoid repeating `light_combined` on this architecture. If augmentation remains the next uncertainty to reduce, test narrower Harness-owned presets separately, especially `light_geometric` versus `light_photometric`, because the combined preset may be too strong or may mix helpful and harmful transforms. Otherwise move to a different in-contract architecture/loss hypothesis while continuing to use `run_20260510_165335_b458c3` as the primary Comparison Target.
