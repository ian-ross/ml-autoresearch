# 2026-06-16 ASPP-style bottleneck context aggregate improvement

## Hypothesis

`single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context` tested whether adding a lightweight dilated bottleneck context block to the current plateau/es extra-wide Line Target U-Net would improve missed small/faint positives while preserving the precision profile of `run_20260614_124226_05e3eb`. The expected benefit was higher recall and Dice from multi-scale bottleneck context without the precision loss previously seen from high-resolution detail fusion.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context`
- Candidate Experiment path: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context`
- Relevant Experiment Proposal: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context/PROPOSAL.md`
- Primary Comparison Target: `run_20260614_124226_05e3eb` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es`

## Run(s)

- Run ID: `run_20260615_140810_2bee94`
- Dataset mode/subset: GVCCS Working Validation Split, 15,558 train samples and 3,889 validation samples.
- Harness/backend notes: completed on CUDA with Docker backend. The model summary reports 9,360,450 parameters, below the 10,000,000 parameter budget but substantially larger than the 7,785,794-parameter plateau/es comparison target. Reduce-on-plateau scheduling and early stopping with best-checkpoint restoration were enabled; early stopping fired after 72 completed epochs and restored the best checkpoint from epoch 63.

## Key metrics

Best-validation metrics selected by max `val/dice`:

| Run | Candidate Experiment | Best epoch | val/dice | val/iou | val/precision | val/recall | val/loss | val/total_loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `run_20260614_124226_05e3eb` | `single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es` | 71 | 0.861622 | 0.756886 | 0.883582 | 0.840728 | 0.484870 | 0.495275 |
| `run_20260615_140810_2bee94` | `single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context` | 63 | 0.866737 | 0.764816 | 0.870191 | 0.863310 | 0.470763 | 0.479140 |

The bottleneck-context Run improves best-validation Dice by about `+0.00511`, IoU by `+0.00793`, and recall by `+0.02258` over the plateau/es comparison target. Precision drops by about `-0.01339`, which misses the proposal's precision guardrail (`>= 0.8780`) despite meeting the strong-success Dice and recall targets.

Final completed-epoch metrics for `run_20260615_140810_2bee94` were lower than the selected best checkpoint: final `val/dice` 0.855609, `val/iou` 0.747654, `val/precision` 0.872259, `val/recall` 0.839582, `val/loss` 0.246437, and `val/total_loss` 0.256075. The best-to-final Dice gap is about `0.01113`, so best-checkpoint restoration is important for interpreting this Result.

## Qualitative observations

The saved first-N prediction samples are favorable but not sufficient to validate the missed-positive hypothesis. Compared with the plateau/es target, both saved sample Dice scores improve:

| Sample | Plateau/es Dice | ASPP-context Dice | Plateau/es IoU | ASPP-context IoU |
| --- | ---: | ---: | ---: | ---: |
| `val/000000` | 0.879433 | 0.890511 | 0.784810 | 0.802632 |
| `val/000001` | 0.888889 | 0.943662 | 0.800000 | 0.893333 |

The aggregate pattern matches the intended direction for sensitivity: recall rises sharply and first-N overlays improve. The tradeoff is a meaningful precision drop, so this Run should not be promoted solely from aggregate validation metrics. A bounded failure-bucket evaluation is needed to determine whether the recall gain corresponds to fewer missed positive masks or instead comes with unacceptable false-positive spread, especially on empty masks and false-positive-heavy positives.

![ASPP-context sample 000 overlay](../runs/run_20260615_140810_2bee94/outputs/prediction_samples/sample_000_overlay.png)

![ASPP-context sample 001 overlay](../runs/run_20260615_140810_2bee94/outputs/prediction_samples/sample_001_overlay.png)

![ASPP-context sample 001 heatmap](../runs/run_20260615_140810_2bee94/outputs/prediction_samples/sample_001_probability_heatmap.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-aspp-context-sample-000-overlay
    source_run_id: run_20260615_140810_2bee94
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: First saved validation overlay for the bottleneck-context Run; selected because its first-N Dice improved over the plateau/es comparison target while aggregate precision fell.
  - figure_id: fig-aspp-context-sample-001-overlay
    source_run_id: run_20260615_140810_2bee94
    source_artifact_path: outputs/prediction_samples/sample_001_overlay.png
    reason: Second saved validation overlay for the bottleneck-context Run; selected because it shows the largest first-N improvement and helps inspect whether added context sharpens or over-expands the mask.
  - figure_id: fig-aspp-context-sample-001-heatmap
    source_run_id: run_20260615_140810_2bee94
    source_artifact_path: outputs/prediction_samples/sample_001_probability_heatmap.png
    reason: Probability heatmap for the second saved validation sample, selected to inspect confidence distribution after adding dilated bottleneck context.
```

## Decision

Treat `run_20260615_140810_2bee94` as a promising but unconfirmed aggregate improvement. It is the best observed best-validation Dice so far, and the recall gain is directly relevant to the missed-positive problem, but the precision loss is large enough that promotion should wait for whole-validation failure-bucket diagnostics. The candidate is not a clear failure; it is a mixed success that needs bounded evaluation before becoming the next primary Comparison Target.

## Next proposed change

Request a bounded failure-bucket Post-Run Evaluation for `run_20260615_140810_2bee94`, comparing it to `run_20260614_124226_05e3eb` on threshold sweep, missed positive masks, false-negative-heavy positives, false-positive-heavy positives, empty-mask false positives, and empty-mask affected-sample count. If diagnostics show fewer missed positives without broad empty-sky false-positive spread, promote the bottleneck-context model; if the precision drop is driven by empty-mask or diffuse false positives, abandon this context width/dilation setting and consider a smaller context block or a Harness-owned data-policy capability instead.
