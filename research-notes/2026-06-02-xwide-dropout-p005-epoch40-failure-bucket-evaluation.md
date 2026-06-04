# 2026-06-02 xwide p=0.05 dropout 40-epoch failure-bucket evaluation

## Hypothesis

The bounded Post-Run Evaluation tested whether the 40-epoch extra-wide base-64 U-Net with Line Target auxiliary weight 0.10 and very light bottleneck `Dropout2d(p=0.05)` is a robust successor to the p=0.075 40-epoch comparison target, or whether its small best-validation Dice gain is mainly a precision-biased tradeoff with worse missed-positive and false-negative-heavy behavior.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_xwide_unet_line_aux_w010_dropout_p005_epoch40`
- Candidate Experiment path: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p005_epoch40`
- Relevant Experiment Proposal: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p005_epoch40/PROPOSAL.md`
- Evaluation Request: `evaluation-requests/eval-2026-06-02-xwide-dropout-p005-epoch40-failure-buckets.yaml`
- Primary Comparison Target: `run_20260601_162117_66bf89` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40`

## Run(s)

- Target Run ID: `run_20260602_101047_294476`
- Target Evaluation ID: `eval_eval_2026_06_02_xwide_dropout_p005_epoch40_failure_buckets`
- Comparison Run ID: `run_20260601_162117_66bf89`
- Comparison Evaluation ID: `eval_eval_2026_06_01_xwide_dropout_p0075_epoch40_failure_buckets`
- Dataset mode/subset: GVCCS Working Validation Split, whole-validation failure-bucket review over 3,889 validation samples.
- Harness/backend notes: request-gated Post-Run Evaluation completed and wrote `aggregate_metrics.json`, `threshold_sweep.json`, `per_sample_metrics.jsonl`, and bounded diagnostic artifacts under the parent Run.

## Key metrics

Whole-validation metrics at the default threshold 0.5:

| Run | Evaluation | Dropout p | val/dice | val/iou | val/precision | val/recall | Empty-mask FP pixels | Empty-mask samples with FP | Missed-positive samples | Best threshold by Dice |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `run_20260601_162117_66bf89` | `eval_eval_2026_06_01_xwide_dropout_p0075_epoch40_failure_buckets` | 0.075 | 0.843362 | 0.729150 | 0.855472 | 0.831591 | 1,210 | 143 / 1,004 | 83 / 2,885 | 0.40 (`dice` 0.843619) |
| `run_20260602_101047_294476` | `eval_eval_2026_06_02_xwide_dropout_p005_epoch40_failure_buckets` | 0.050 | 0.844387 | 0.730684 | 0.866780 | 0.823122 | 1,119 | 157 / 1,004 | 79 / 2,885 | 0.30 (`dice` 0.845291) |

The p=0.05 model preserves its small Dice gain under whole-validation evaluation: `+0.00102` Dice and `+0.00153` IoU versus the p=0.075 40-epoch target at threshold 0.5. The gain is clearly precision-biased: precision improves by `+0.01131`, while recall drops by `-0.00847`. Empty-mask false-positive pixels improve from 1,210 to 1,119, but the number of empty validation masks with any false positives worsens from 143 to 157.

The threshold sweep does not show a fragile threshold artifact, but it reinforces the calibration shift. The p=0.05 best threshold is lower (`0.30`, Dice `0.845291`) than the p=0.075 target (`0.40`, Dice `0.843619`). At threshold 0.30, p=0.05 has Dice `0.845291`, precision `0.849286`, and recall `0.841334`; at threshold 0.50 it has Dice `0.844387`, precision `0.866780`, and recall `0.823122`. Lowering the threshold can recover recall, but the default operating point is more conservative than the comparison target.

## Qualitative observations

The selected failure buckets confirm a tradeoff rather than a clean replacement:

- Missed-positive selected artifacts remain tiny positives. The worst selected miss is `val/000221` with 162 positive pixels and zero predicted pixels, matching the earlier 30-epoch p=0.075 missed-positive outlier and worse than the selected misses in the p=0.075 40-epoch evaluation.
- Empty-mask false positives are individually small in the selected artifacts (`42`, `39`, `39`, and `36` false-positive pixels), consistent with the improved aggregate empty-mask false-positive pixel count. However, these small false positives are distributed across more empty-mask samples than the p=0.075 40-epoch comparison.
- False-positive-heavy positives still include broad spillover, especially `val/001487` with 963 false-positive pixels and Dice `0.6611`, and `val/001488` with 744 false-positive pixels and Dice `0.3866`.
- False-negative-heavy large positives remain a substantial weakness. Selected samples `val/002499`, `val/002504`, and `val/002500` each have roughly 984-1,165 false-negative pixels with Dice around `0.7725`-`0.7735`.

Overall, p=0.05 is now the top Result by best-validation and whole-validation Dice, and it reduces aggregate false positives. It should not replace p=0.075 as the general-purpose design baseline because the gain is small, recall worsens, and the qualitative review shows persistent missed/tiny-positive and under-segmentation failures.

![p005 missed positive](../runs/run_20260602_101047_294476/outputs/evaluations/eval_eval_2026_06_02_xwide_dropout_p005_epoch40_failure_buckets/diagnostic_samples/sample_000_overlay.png)

![p005 empty-mask false positive](../runs/run_20260602_101047_294476/outputs/evaluations/eval_eval_2026_06_02_xwide_dropout_p005_epoch40_failure_buckets/diagnostic_samples/sample_007_overlay.png)

![p005 false-positive-heavy positive](../runs/run_20260602_101047_294476/outputs/evaluations/eval_eval_2026_06_02_xwide_dropout_p005_epoch40_failure_buckets/diagnostic_samples/sample_013_overlay.png)

![p005 false-negative-heavy large positive](../runs/run_20260602_101047_294476/outputs/evaluations/eval_eval_2026_06_02_xwide_dropout_p005_epoch40_failure_buckets/diagnostic_samples/sample_017_overlay.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-p005-missed-positive-worst
    source_evaluation_id: eval_eval_2026_06_02_xwide_dropout_p005_epoch40_failure_buckets
    source_artifact_path: diagnostic_samples/sample_000_overlay.png
    reason: Worst selected missed-positive artifact for val/000221 with 162 positive pixels and no predicted positives; selected because the p=0.05 gain remains vulnerable to tiny-mask misses.
  - figure_id: fig-p005-empty-mask-fp
    source_evaluation_id: eval_eval_2026_06_02_xwide_dropout_p005_epoch40_failure_buckets
    source_artifact_path: diagnostic_samples/sample_007_overlay.png
    reason: Empty-mask false-positive artifact for val/003111 with 42 false-positive pixels; selected because aggregate empty-mask false-positive pixels improved while the number of affected empty masks increased.
  - figure_id: fig-p005-false-positive-heavy-positive
    source_evaluation_id: eval_eval_2026_06_02_xwide_dropout_p005_epoch40_failure_buckets
    source_artifact_path: diagnostic_samples/sample_013_overlay.png
    reason: False-positive-heavy artifact for val/001487 with 963 false-positive pixels; selected to show that broad spillover remains despite higher aggregate precision.
  - figure_id: fig-p005-false-negative-heavy-large-positive
    source_evaluation_id: eval_eval_2026_06_02_xwide_dropout_p005_epoch40_failure_buckets
    source_artifact_path: diagnostic_samples/sample_017_overlay.png
    reason: False-negative-heavy artifact for val/002500 with 1,165 false-negative pixels; selected to show persistent under-segmentation on large positives.
```

## Decision

Promote `run_20260602_101047_294476` as the current top Result by best-validation Dice, but treat `run_20260601_162117_66bf89` as the safer general-purpose baseline for recall-sensitive architectural follow-up. The p=0.05 dropout setting is useful evidence that slightly weaker regularization can reduce aggregate false-positive burden and raise Dice, but it does not solve the remaining failure modes and worsens default-threshold recall.

Do not keep lowering bottleneck dropout. The next in-contract hypothesis should avoid further precision bias and should target missed positives or under-segmentation more directly.

## Next proposed change

Use the p=0.075 40-epoch architecture as the safer comparison target for recall-oriented changes, while reporting p=0.05 as the leaderboard best. If Boundary Target auxiliary training remains available and stable, retry a conservative boundary auxiliary variant with very low boundary weight only if the agent-side contract and training path agree; otherwise prefer an in-contract decoder/refinement change intended to recover recall without increasing empty-mask false-positive pixels.
