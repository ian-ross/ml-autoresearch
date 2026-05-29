# 2026-05-25 light-geometric augmentation regression

## Hypothesis

The `single_frame_small_unet_line_auxiliary_w010_light_geometric` Candidate Experiment tested whether isolating Harness-owned light geometric augmentation would recover the robustness benefit hoped for from augmentation while avoiding the larger regression seen with `light_combined` augmentation.

## Candidate Experiment(s)

- Candidate Experiment path or ID: `candidates/single_frame_small_unet_line_auxiliary_w010_light_geometric`
- Relevant Experiment Proposal: `candidates/single_frame_small_unet_line_auxiliary_w010_light_geometric/PROPOSAL.md`
- Primary Comparison Target: `run_20260510_165335_b458c3` (`single_frame_small_unet_line_auxiliary_w010`), best-validation `val/dice` 0.7843769771696992.

## Run(s)

- Run ID: `run_20260525_123213_541981`
- Dataset mode/subset: GVCCS Working Validation Split
- Harness/backend notes: completed Run using `input_mode: single_frame_rgb`, primary `bce_dice` loss, Line Target auxiliary `weighted_bce` at weight 0.10, deterministic shuffle, and effective `augmentation_policy: light_geometric`. Batch size requested/effective: 16; max epochs: 30; GPU backend completed without resource retry.

## Key metrics

| Run | Candidate Experiment | Epoch source | val/dice | val/iou | val/precision | val/recall | val/loss |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `run_20260510_165335_b458c3` | `single_frame_small_unet_line_auxiliary_w010` | best epoch 30 | 0.7843769771696992 | 0.6452468918723383 | 0.7717854808163968 | 0.7973861422181288 | 0.5639248168630936 |
| `run_20260523_163405_8e18eb` | `single_frame_small_unet_line_auxiliary_w010_light_combined` | best epoch 26 | 0.717276204322084 | 0.5591821144496796 | 0.75108987313162 | 0.6863759233042218 | 0.6413536987687236 |
| `run_20260525_123213_541981` | `single_frame_small_unet_line_auxiliary_w010_light_geometric` | best/final epoch 30 | 0.7292688877723468 | 0.5738970902301385 | 0.7374074904570065 | 0.7213079718981288 | 0.6231930805803664 |

`light_geometric` improves over the earlier `light_combined` best-validation Dice by about 0.0120, but it remains about 0.0551 Dice below the unaugmented lower-weight Line Target auxiliary comparison target. It therefore misses the proposal success criterion of matching or improving the current best Result.

## Qualitative observations

The saved first-N prediction samples are consistent with a completed but lower-quality research outcome rather than a Harness failure. Both sampled validation frames score in the same range as the aggregate Result:

- `val/000000`: Dice 0.7832, IoU 0.6437.
- `val/000001`: Dice 0.7333, IoU 0.5789.

Compared with the unaugmented comparison target, the aggregate regression is driven by lower recall as well as lower precision (`val/recall` 0.7213 vs 0.7974; `val/precision` 0.7374 vs 0.7718). The geometric-only preset appears less harmful than the combined preset, but not useful enough to displace the unaugmented policy for this Small U-Net plus Line Target auxiliary family.

Example first-N overlays and probability heatmap:

![Sample 0 overlay](../runs/run_20260525_123213_541981/outputs/prediction_samples/sample_000_overlay.png)

![Sample 0 probability heatmap](../runs/run_20260525_123213_541981/outputs/prediction_samples/sample_000_probability_heatmap.png)

![Sample 1 overlay](../runs/run_20260525_123213_541981/outputs/prediction_samples/sample_001_overlay.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-light-geometric-sample-000-overlay
    source_run_id: run_20260525_123213_541981
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: Shows a representative first-N validation prediction from the light-geometric augmentation Run, with sample Dice near the comparison target but insufficient aggregate improvement.
  - figure_id: fig-light-geometric-sample-000-heatmap
    source_run_id: run_20260525_123213_541981
    source_artifact_path: outputs/prediction_samples/sample_000_probability_heatmap.png
    reason: Provides probability-level context for the same representative validation sample to inspect confidence after geometric-only augmentation.
  - figure_id: fig-light-geometric-sample-001-overlay
    source_run_id: run_20260525_123213_541981
    source_artifact_path: outputs/prediction_samples/sample_001_overlay.png
    reason: Shows a second first-N validation prediction whose Dice tracks the aggregate regression more closely.
```

## Decision

Do not replace the current best Result. Keep `run_20260510_165335_b458c3` as the best known in-contract Run. Treat `run_20260525_123213_541981` as a completed poor research outcome (`bad_research_result`) for this hypothesis: the Candidate Experiment stayed within contract and completed, but geometric-only augmentation still reduced best-validation Dice.

## Next proposed change

Avoid further immediate geometric augmentation variants for this architecture. The augmentation evidence now suggests `light_combined` is harmful and `light_geometric` is only a partial recovery. If augmentation uncertainty must be closed, a single `light_photometric` control could isolate whether the combined preset's non-geometric transforms were the main problem; otherwise, shift the next Experiment Proposal back to non-augmentation in-contract changes while retaining `run_20260510_165335_b458c3` as the primary Comparison Target.
