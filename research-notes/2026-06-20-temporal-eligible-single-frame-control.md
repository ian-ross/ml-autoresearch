# 2026-06-20 temporal-eligible single-frame control

## Hypothesis

`single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context_temporal_eligible_control` tested whether the recent temporal ASPP-context gain was mostly caused by the `temporal_eligible_center` frame-selection subset rather than by centered three-frame temporal input. The candidate kept the successful single-frame ASPP-context architecture and training policy unchanged, and changed only the Harness-owned frame selection policy from `all_target_frames` to `temporal_eligible_center`.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context_temporal_eligible_control`
- Candidate Experiment path: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context_temporal_eligible_control`
- Relevant Experiment Proposal: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context_temporal_eligible_control/PROPOSAL.md`
- Primary Comparison Target: `run_20260618_204556_e6a60b` / `temporal_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context`
- Secondary reference: `run_20260615_140810_2bee94` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context`

## Run(s)

- Run ID: `run_20260620_092332_3ee9a1`
- Dataset mode/subset: GVCCS Working Validation Split using `single_frame_rgb` with explicit `temporal_eligible_center` frame selection; 15,401 train samples and 3,850 validation samples, matching the temporal run's subset.
- Harness/backend notes: completed on CUDA with Docker backend. The model summary reports a `[3, 128, 128]` single-frame input and 9,360,450 parameters. Reduce-on-plateau scheduling and early stopping with best-checkpoint restoration were enabled; early stopping fired after 72 completed epochs and restored the best checkpoint from epoch 60.

## Key metrics

Best-validation metrics selected by max `val/dice`:

| Run | Candidate Experiment | Input mode | Frame selection | Best epoch | val/dice | val/iou | val/precision | val/recall | val/loss | val/total_loss |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `run_20260615_140810_2bee94` | single-frame ASPP-context | single-frame | all target frames | 63 | 0.866737 | 0.764816 | 0.870191 | 0.863310 | 0.470763 | 0.479140 |
| `run_20260618_204556_e6a60b` | temporal ASPP-context | centered 3-frame clip | temporal-eligible centers | 63 | 0.873707 | 0.775737 | 0.877774 | 0.869677 | 0.467766 | 0.475748 |
| `run_20260620_092332_3ee9a1` | matched single-frame control | single-frame | temporal-eligible centers | 60 | 0.871672 | 0.772534 | 0.874025 | 0.869331 | 0.471771 | 0.480062 |

The matched single-frame control is much closer to the temporal run than to the all-target-frame single-frame reference. Relative to `run_20260615_140810_2bee94`, using the temporal-eligible subset improved best Dice by about `+0.00493`, IoU by `+0.00772`, precision by `+0.00383`, and recall by `+0.00602` without changing input channels. Relative to the temporal run, it remains lower by about `-0.00204` Dice, `-0.00320` IoU, and `-0.00375` precision, while recall is nearly identical (`-0.00035`).

This supports a mixed interpretation: excluding sequence-boundary frames accounts for a large part of the apparent temporal-family gain, but temporal channels still provide a small controlled advantage on the same validation subset, mainly through precision/IoU rather than recall.

Final completed-epoch metrics after best-checkpoint restoration were `val/dice` 0.871658, `val/iou` 0.772513, `val/precision` 0.882104, `val/recall` 0.861457, `val/loss` 0.214315, and `val/total_loss` 0.222720. As with prior restored-checkpoint runs, model selection should rely on persisted best metrics while final metrics describe the restored final artifact state.

## Qualitative observations

The first saved validation samples are broadly consistent with a strong aggregate run. Sample `val/000000` is stronger for this single-frame control than for the temporal run's first-N note, with Dice 0.909091. Sample `val/000001` remains a harder example at Dice 0.823529, similar to the earlier temporal first-N sample. The saved overlays should be interpreted cautiously because they are first-N samples, not failure-bucket diagnostics.

![Temporal-eligible control sample 000 overlay](../runs/run_20260620_092332_3ee9a1/outputs/prediction_samples/sample_000_overlay.png)

![Temporal-eligible control sample 001 overlay](../runs/run_20260620_092332_3ee9a1/outputs/prediction_samples/sample_001_overlay.png)

![Temporal-eligible control sample 001 heatmap](../runs/run_20260620_092332_3ee9a1/outputs/prediction_samples/sample_001_probability_heatmap.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-temporal-eligible-control-sample-000-overlay
    source_run_id: run_20260620_092332_3ee9a1
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: First saved validation overlay for the matched single-frame temporal-eligible control; selected because it shows a high-Dice first-N case on the same subset as the temporal run.
  - figure_id: fig-temporal-eligible-control-sample-001-overlay
    source_run_id: run_20260620_092332_3ee9a1
    source_artifact_path: outputs/prediction_samples/sample_001_overlay.png
    reason: Second saved validation overlay for the matched control; selected because it remains a harder first-N example despite strong aggregate metrics.
  - figure_id: fig-temporal-eligible-control-sample-001-heatmap
    source_run_id: run_20260620_092332_3ee9a1
    source_artifact_path: outputs/prediction_samples/sample_001_probability_heatmap.png
    reason: Probability heatmap for the harder matched-control first-N sample, selected to inspect whether residual error is confidence or threshold related.
```

## Decision

Treat `run_20260620_092332_3ee9a1` as an important controlled comparison rather than a new global best. It shows that the temporal-eligible frame-selection subset itself materially improves the single-frame ASPP-context result, so earlier temporal-vs-all-frame comparisons overstated the contribution of temporal input. However, `run_20260618_204556_e6a60b` remains the best observed Result on the temporal-eligible subset and retains a small but meaningful controlled lead.

## Next proposed change

Before adding larger temporal architecture changes, request bounded failure-bucket diagnostics for the matched single-frame control or compare existing evaluation buckets against the temporal run. The immediate research question is whether the temporal model's remaining `~0.002` Dice advantage comes from fewer false-positive-heavy positives, fewer missed small positives, or only threshold/precision calibration.
