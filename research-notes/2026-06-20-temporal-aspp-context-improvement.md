# 2026-06-20 temporal ASPP-context improvement

## Hypothesis

`temporal_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context` tested whether switching the current best single-frame ASPP-context extra-wide Line Target U-Net to Harness-owned centered three-frame temporal RGB clip input would preserve the ASPP-context recall gain while improving precision. The key premise was that adjacent GVCCS frames could help distinguish persistent linear contrails from cloud-edge texture without adding the decoder-side gate that previously trimmed true positives.

## Candidate Experiment(s)

- Candidate Experiment ID: `temporal_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context`
- Candidate Experiment path: `candidates/temporal_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context`
- Relevant Experiment Proposal: `candidates/temporal_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context/PROPOSAL.md`
- Primary Comparison Target: `run_20260615_140810_2bee94` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context`
- Secondary references: `run_20260618_113001_287165` / ASPP mask gate and `run_20260614_124226_05e3eb` / plateau/es parent

## Run(s)

- Run ID: `run_20260618_204556_e6a60b`
- Dataset mode/subset: GVCCS Working Validation Split using `centered_temporal_rgb_clip` and required `temporal_eligible_center` frame selection. This produced 15,401 train samples and 3,850 validation samples, versus 15,558/3,889 for prior all-target-frame single-frame Runs.
- Harness/backend notes: completed on CUDA with Docker backend. The model summary reports a `[9, 128, 128]` channel-stacked temporal input and 9,363,906 parameters, below the 10,000,000 parameter budget. Reduce-on-plateau scheduling and early stopping with best-checkpoint restoration were enabled; early stopping fired after 75 completed epochs and restored the best checkpoint from epoch 63.

## Key metrics

Best-validation metrics selected by max `val/dice`:

| Run | Candidate Experiment | Frame selection | Best epoch | val/dice | val/iou | val/precision | val/recall | val/loss | val/total_loss |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `run_20260614_124226_05e3eb` | plateau/es parent | all target frames | 71 | 0.861622 | 0.756886 | 0.883582 | 0.840728 | 0.484870 | 0.495275 |
| `run_20260615_140810_2bee94` | single-frame ASPP-context target | all target frames | 63 | 0.866737 | 0.764816 | 0.870191 | 0.863310 | 0.470763 | 0.479140 |
| `run_20260618_113001_287165` | single-frame ASPP mask gate | all target frames | 56 | 0.863796 | 0.760247 | 0.877939 | 0.850101 | 0.472557 | 0.481186 |
| `run_20260618_204556_e6a60b` | temporal ASPP-context | temporal-eligible centers | 63 | 0.873707 | 0.775737 | 0.877774 | 0.869677 | 0.467766 | 0.475748 |

The temporal candidate met the proposal's strong numeric success criteria: best `val/dice` exceeded the ASPP-context target by about `+0.00697`, precision was above `0.875`, and recall was above `0.858`. Relative to the single-frame ASPP-context target, the temporal Run improved Dice, IoU, precision, recall, mask loss, and total loss. It also matched the mask gate's precision almost exactly while avoiding the gate's recall loss.

The main caveat is comparison validity: temporal input requires `temporal_eligible_center`, so the validation set is slightly smaller and excludes sequence-boundary frames. This should be treated as a promising family result rather than a fully controlled proof that temporal channels alone beat the all-target-frame single-frame model.

Final completed-epoch metrics remained close to the best checkpoint: final `val/dice` 0.873029, `val/iou` 0.774668, `val/precision` 0.883899, `val/recall` 0.862422, `val/loss` 0.210129, and `val/total_loss` 0.218351. The final metric scale differs from the best-epoch loss values after best-checkpoint restoration, so promotion should rely on the persisted best metrics for model selection and final metrics for the final restored artifact state.

## Qualitative observations

The two saved first-N prediction samples are not uniformly strong despite the aggregate improvement. Sample `val/000000` has Dice 0.853659 and sample `val/000001` has Dice 0.829268, both lower than the aggregate validation Dice. These first temporal-eligible validation centers therefore look like useful hard examples rather than representative successes. The overlays and heatmaps should be used to inspect whether temporal input still misses thin positive structure or whether the remaining errors are mostly boundary thickness/threshold issues.

![Temporal sample 000 overlay](../runs/run_20260618_204556_e6a60b/outputs/prediction_samples/sample_000_overlay.png)

![Temporal sample 001 overlay](../runs/run_20260618_204556_e6a60b/outputs/prediction_samples/sample_001_overlay.png)

![Temporal sample 001 heatmap](../runs/run_20260618_204556_e6a60b/outputs/prediction_samples/sample_001_probability_heatmap.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-temporal-sample-000-overlay
    source_run_id: run_20260618_204556_e6a60b
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: First saved temporal validation overlay; selected because it is a hard first-N example despite the Run's aggregate Dice improvement.
  - figure_id: fig-temporal-sample-001-overlay
    source_run_id: run_20260618_204556_e6a60b
    source_artifact_path: outputs/prediction_samples/sample_001_overlay.png
    reason: Second saved temporal validation overlay; selected because its low sample Dice highlights remaining qualitative failure modes after temporal context is added.
  - figure_id: fig-temporal-sample-001-heatmap
    source_run_id: run_20260618_204556_e6a60b
    source_artifact_path: outputs/prediction_samples/sample_001_probability_heatmap.png
    reason: Probability heatmap for the lower-Dice first-N temporal sample, selected to inspect confidence and threshold behavior around residual errors.
```

## Decision

Promote `run_20260618_204556_e6a60b` as the best observed in-contract Result for the temporal-eligible validation subset and as the leading candidate family for further study. Do not yet declare it an apples-to-apples global replacement for `run_20260615_140810_2bee94`, because the data policy changed from all target frames to temporal-eligible centers.

This is a successful completed Run, not a candidate bug, resource failure, or bad research result. It provides strong evidence that temporal context is worth pursuing under the current Candidate Experiment Contract.

## Next proposed change

Request a bounded failure-bucket Post-Run Evaluation for `run_20260618_204556_e6a60b` before making another temporal architecture change. The diagnostic should check whether the aggregate gain comes from fewer false-positive-heavy examples, fewer missed positive masks, or broad threshold/shape improvements on the temporal-eligible validation subset. A later matched single-frame control using `temporal_eligible_center` would also be valuable if the contract/harness supports it, because it would isolate temporal input from frame-selection effects.
