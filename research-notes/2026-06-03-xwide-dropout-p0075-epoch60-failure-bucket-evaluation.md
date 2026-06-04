# 2026-06-03 xwide p=0.075 dropout 60-epoch failure-bucket evaluation

## Hypothesis

The bounded Post-Run Evaluation tested whether the 60-epoch extra-wide base-64 U-Net with Line Target auxiliary weight 0.10 and bottleneck `Dropout2d(p=0.075)` is a robust new baseline, or whether the best-validation Dice gain from `run_20260602_203450_c05550` is mainly threshold-specific or offset by worse missed-positive, false-negative-heavy, or empty-mask false-positive behavior.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60`
- Candidate Experiment path: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60`
- Relevant Experiment Proposal: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60/PROPOSAL.md`
- Evaluation Request: `evaluation-requests/eval-2026-06-03-xwide-dropout-p0075-epoch60-failure-buckets.yaml`
- Primary Comparison Target: `run_20260601_162117_66bf89` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40`
- Leaderboard context: `run_20260602_101047_294476` / `single_frame_xwide_unet_line_aux_w010_dropout_p005_epoch40`

## Run(s)

- Target Run ID: `run_20260602_203450_c05550`
- Target Evaluation ID: `eval_eval_2026_06_03_xwide_dropout_p0075_epoch60_failure_buckets`
- Comparison Evaluation IDs: `eval_eval_2026_06_01_xwide_dropout_p0075_epoch40_failure_buckets`, `eval_eval_2026_06_02_xwide_dropout_p005_epoch40_failure_buckets`
- Dataset mode/subset: GVCCS Working Validation Split, whole-validation failure-bucket review over 3,889 validation samples.
- Harness/backend notes: request-gated Post-Run Evaluation completed and wrote `aggregate_metrics.json`, `threshold_sweep.json`, `per_sample_metrics.jsonl`, and bounded diagnostic artifacts under the parent Run.

## Key metrics

Whole-validation metrics at default threshold 0.5:

| Run | Evaluation | Dropout p / epochs | val/dice | val/iou | val/precision | val/recall | Empty-mask FP pixels | Empty-mask samples with FP | Missed-positive samples | Best threshold by Dice |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `run_20260601_162117_66bf89` | `eval_eval_2026_06_01_xwide_dropout_p0075_epoch40_failure_buckets` | 0.075 / 40 | 0.843362 | 0.729150 | 0.855472 | 0.831591 | 1,210 | 143 / 1,004 | 83 / 2,885 | 0.40 (`dice` 0.843619) |
| `run_20260602_101047_294476` | `eval_eval_2026_06_02_xwide_dropout_p005_epoch40_failure_buckets` | 0.050 / 40 | 0.844387 | 0.730684 | 0.866780 | 0.823122 | 1,119 | 157 / 1,004 | 79 / 2,885 | 0.30 (`dice` 0.845291) |
| `run_20260602_203450_c05550` | `eval_eval_2026_06_03_xwide_dropout_p0075_epoch60_failure_buckets` | 0.075 / 60 | 0.849040 | 0.737679 | 0.867067 | 0.831747 | 1,107 | 184 / 1,004 | 74 / 2,885 | 0.35 (`dice` 0.849576) |

The 60-epoch p=0.075 model preserves the Run-level improvement under whole-validation evaluation. At threshold 0.5 it improves Dice by `+0.00568` versus the 40-epoch p=0.075 parent and `+0.00465` versus the p=0.05 40-epoch leaderboard context. The gain is not just threshold-specific: the best sweep threshold moves only to 0.35, and the default-threshold Dice is within `0.00054` of the best-threshold Dice.

The recall/precision tradeoff is favorable relative to the p=0.05 model: precision is essentially tied with p=0.05 (`+0.00029`), while recall is `+0.00862` higher and almost identical to the p=0.075 40-epoch parent (`+0.00016`). Missed-positive samples also improve from 83 to 74 versus the parent, and from 79 to 74 versus p=0.05. Empty-mask false-positive pixels improve slightly versus both comparisons, but the number of empty masks with any false positive worsens materially to 184 / 1,004.

## Qualitative observations

The selected failure buckets support promoting the 60-epoch p=0.075 model, with one important caution:

- Missed-positive artifacts remain tiny positives. The worst selected misses are `val/000220` with 114 positive pixels, `val/000693` with 56 positive pixels, and `val/002034` with 47 positive pixels, all with zero predicted pixels at threshold 0.5. This is still a persistent failure mode, but the aggregate missed-positive count is lower than both comparison evaluations.
- Empty-mask false positives are individually small in the selected artifacts (`37`, `35`, `33`, and `30` false-positive pixels), and total empty-mask false-positive pixels are the best of the three compared evaluations. However, these tiny false positives are distributed across many more empty masks than before, so the model is not a strict improvement for empty-sky cleanliness.
- False-positive-heavy positives still include the same broad spillover sequence seen in earlier evaluations. `val/001487` has 963 false-positive pixels and Dice `0.5873`; `val/001488` has 738 false-positive pixels and Dice `0.6114`.
- False-negative-heavy large positives persist. Selected large positives include `val/002500` with 1,320 false-negative pixels and Dice `0.7481`, and `val/002496` with 883 false-negative pixels and Dice `0.7477`.

Overall, the longer p=0.075 training budget produces the cleanest aggregate Dice/recall balance seen so far, but it does not solve tiny-mask misses, broad spillover on some positive sequences, or under-segmentation of large positives.

![p0075 epoch60 missed positive](../runs/run_20260602_203450_c05550/outputs/evaluations/eval_eval_2026_06_03_xwide_dropout_p0075_epoch60_failure_buckets/diagnostic_samples/sample_000_overlay.png)

![p0075 epoch60 empty-mask false positive](../runs/run_20260602_203450_c05550/outputs/evaluations/eval_eval_2026_06_03_xwide_dropout_p0075_epoch60_failure_buckets/diagnostic_samples/sample_005_overlay.png)

![p0075 epoch60 false-positive-heavy positive](../runs/run_20260602_203450_c05550/outputs/evaluations/eval_eval_2026_06_03_xwide_dropout_p0075_epoch60_failure_buckets/diagnostic_samples/sample_009_overlay.png)

![p0075 epoch60 false-negative-heavy large positive](../runs/run_20260602_203450_c05550/outputs/evaluations/eval_eval_2026_06_03_xwide_dropout_p0075_epoch60_failure_buckets/diagnostic_samples/sample_014_overlay.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-p0075-epoch60-missed-positive-worst
    source_evaluation_id: eval_eval_2026_06_03_xwide_dropout_p0075_epoch60_failure_buckets
    source_artifact_path: diagnostic_samples/sample_000_overlay.png
    reason: Worst selected missed-positive artifact for val/000220 with 114 positive pixels and no predicted positives; selected because tiny-mask misses remain despite the aggregate missed-positive count improving.
  - figure_id: fig-p0075-epoch60-empty-mask-fp
    source_evaluation_id: eval_eval_2026_06_03_xwide_dropout_p0075_epoch60_failure_buckets
    source_artifact_path: diagnostic_samples/sample_005_overlay.png
    reason: Empty-mask false-positive artifact for val/003181 with 37 false-positive pixels; selected because total empty-mask false-positive pixels improved while the number of affected empty masks worsened.
  - figure_id: fig-p0075-epoch60-false-positive-heavy-positive
    source_evaluation_id: eval_eval_2026_06_03_xwide_dropout_p0075_epoch60_failure_buckets
    source_artifact_path: diagnostic_samples/sample_009_overlay.png
    reason: False-positive-heavy artifact for val/001487 with 963 false-positive pixels; selected to show persistent broad spillover on positive masks.
  - figure_id: fig-p0075-epoch60-false-negative-heavy-large-positive
    source_evaluation_id: eval_eval_2026_06_03_xwide_dropout_p0075_epoch60_failure_buckets
    source_artifact_path: diagnostic_samples/sample_014_overlay.png
    reason: False-negative-heavy artifact for val/002500 with 1,320 false-negative pixels; selected to show persistent under-segmentation on large positives.
```

## Decision

Promote `run_20260602_203450_c05550` as the current best in-contract Result and as the preferred baseline for the next Candidate Experiment. The whole-validation evaluation confirms that the 60-epoch p=0.075 model improves Dice and IoU at threshold 0.5, is not highly threshold fragile, preserves the recall-sensitive p=0.075 behavior, and improves missed-positive counts versus both immediate comparison Runs.

Do not pursue further simple fixed-epoch extension as the next move. The remaining errors point to model behavior rather than insufficient fixed training time: small positive masks can still be fully missed, false positives can spread across more empty masks, and large positives can be under-segmented.

## Next proposed change

Use `run_20260602_203450_c05550` as the Comparison Target for the next in-contract architecture hypothesis. Prefer a small, auditable architecture change aimed at reducing tiny-mask misses and large-mask under-segmentation without increasing the spread of empty-mask false positives. Report empty-mask affected-sample count, not only total false-positive pixels, in the next Research Note because this evaluation exposed that those can move in opposite directions.
