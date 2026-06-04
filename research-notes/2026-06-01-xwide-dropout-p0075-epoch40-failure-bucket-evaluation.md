# 2026-06-01 xwide p=0.075 dropout 40-epoch failure-bucket evaluation

## Hypothesis

The bounded Post-Run Evaluation tested whether the 40-epoch extra-wide base-64 U-Net with Line Target auxiliary weight 0.10 and bottleneck `Dropout2d(p=0.075)` preserves its large best-validation gain without becoming threshold-sensitive or creating a material false-positive / missed-positive regression. The main comparison targets are the 30-epoch p=0.075 base (`run_20260601_085755_25cd06`) and the earlier 30-epoch p=0.10 dropout base (`run_20260530_180658_0af8a8`).

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40`
- Candidate Experiment path: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40`
- Relevant Experiment Proposal: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40/PROPOSAL.md`
- Evaluation Request: `evaluation-requests/eval-2026-06-01-xwide-dropout-p0075-epoch40-failure-buckets.yaml`

## Run(s)

- Target Run ID: `run_20260601_162117_66bf89`
- Target Evaluation ID: `eval_eval_2026_06_01_xwide_dropout_p0075_epoch40_failure_buckets`
- Comparison Run IDs: `run_20260601_085755_25cd06` and `run_20260530_180658_0af8a8`
- Dataset mode/subset: GVCCS Working Validation Split, whole-validation failure-bucket review over 3,889 validation samples.
- Harness/backend notes: request-gated Post-Run Evaluation completed and wrote `aggregate_metrics.json`, `threshold_sweep.json`, `per_sample_metrics.jsonl`, and bounded diagnostic artifacts under the parent Run.

## Key metrics

Whole-validation metrics at the default threshold 0.5:

| Run | Evaluation | val/dice | val/iou | val/precision | val/recall | Empty-mask FP pixels | Empty-mask samples with FP | Best threshold by Dice |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `run_20260530_180658_0af8a8` | `eval_eval_2026_05_31_xwide_dropout_failure_buckets` | 0.833856 | 0.715054 | 0.856288 | 0.812569 | 1,739 | 220 / 1,004 | 0.30 (`dice` 0.834962) |
| `run_20260601_085755_25cd06` | `eval_eval_2026_06_01_xwide_dropout_p0075_failure_buckets` | 0.834604 | 0.716155 | 0.846532 | 0.823008 | 1,621 | 135 / 1,004 | 0.40 (`dice` 0.834900) |
| `run_20260601_162117_66bf89` | `eval_eval_2026_06_01_xwide_dropout_p0075_epoch40_failure_buckets` | 0.843362 | 0.729150 | 0.855472 | 0.831591 | 1,210 | 143 / 1,004 | 0.40 (`dice` 0.843619) |

The 40-epoch Result preserves the training-note improvement under whole-validation evaluation. Relative to the 30-epoch p=0.075 base, it improves Dice by about `+0.00876`, precision by about `+0.00894`, and recall by about `+0.00858`. Relative to the older p=0.10 dropout base, it improves Dice by about `+0.00951` and recall by about `+0.01902` while almost matching precision (`-0.00082`).

The threshold sweep supports the default-threshold Result rather than exposing a threshold-tuning artifact. The best Dice is at threshold 0.40 (`0.843619`), only about `+0.00026` above threshold 0.5 (`0.843362`). Empty-mask false-positive pixels also improve materially at threshold 0.5: 1,210 for the 40-epoch model versus 1,621 for the 30-epoch p=0.075 model and 1,739 for the 30-epoch p=0.10 model. The number of empty-mask samples with any false positives rises slightly versus the 30-epoch p=0.075 base (143 vs 135), but remains far below the p=0.10 base (220).

## Qualitative observations

The selected failure buckets show that longer training reduces the aggregate error burden but does not eliminate familiar outliers:

- Empty-mask false-positive selected artifacts are less severe than the 30-epoch p=0.075 outlier noted earlier. The largest selected 40-epoch empty-mask artifact is `val/003505` with 168 false-positive pixels, followed by 126 and 110 pixels, compared with the prior p=0.075 selected empty-mask outlier of 329 false-positive pixels.
- Missed-positive selected artifacts remain mostly tiny masks. The 40-epoch run misses `val/000174` with 70 positive pixels, `val/000059` with 62, `val/000693` with 56, and `val/002364` with 54. This is not solved by longer training, but the missed-positive examples are smaller than the worst selected p=0.075 30-epoch miss of 162 pixels.
- False-positive-heavy positives remain a visible failure mode. `val/001487` has 913 false-positive pixels and 275 false negatives with Dice 0.5852, worse on false positives than the 30-epoch p=0.075 selected version of the same sample. This suggests some individual broad-mask outliers can worsen even as aggregate empty-mask false positives and overall Dice improve.
- False-negative-heavy large positives remain present, including `val/002527` with 927 false-negative pixels and Dice 0.7851, and `val/001796` with 1029 false-negative pixels and Dice 0.8186. These are still plausible targets for Boundary Target auxiliary training once the contract gap is resolved.

Overall, the evaluation strengthens the case for `run_20260601_162117_66bf89` as the current best in-contract base. The gain is broad across Dice, precision, recall, IoU, threshold stability, and empty-mask false-positive pixels, while the remaining qualitative risks are specific outlier buckets rather than a global calibration or false-positive collapse.

![epoch40 empty-mask false positive](../runs/run_20260601_162117_66bf89/outputs/evaluations/eval_eval_2026_06_01_xwide_dropout_p0075_epoch40_failure_buckets/diagnostic_samples/sample_000_overlay.png)

![epoch40 missed tiny positive](../runs/run_20260601_162117_66bf89/outputs/evaluations/eval_eval_2026_06_01_xwide_dropout_p0075_epoch40_failure_buckets/diagnostic_samples/sample_003_overlay.png)

![epoch40 false-positive-heavy positive](../runs/run_20260601_162117_66bf89/outputs/evaluations/eval_eval_2026_06_01_xwide_dropout_p0075_epoch40_failure_buckets/diagnostic_samples/sample_008_overlay.png)

![epoch40 false-negative-heavy large positive](../runs/run_20260601_162117_66bf89/outputs/evaluations/eval_eval_2026_06_01_xwide_dropout_p0075_epoch40_failure_buckets/diagnostic_samples/sample_010_overlay.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-epoch40-empty-mask-fp
    source_evaluation_id: eval_eval_2026_06_01_xwide_dropout_p0075_epoch40_failure_buckets
    source_artifact_path: diagnostic_samples/sample_000_overlay.png
    reason: Empty-mask false-positive artifact for val/003505; selected because aggregate empty-mask false-positive pixels improved while this bucket remains a residual risk.
  - figure_id: fig-epoch40-missed-tiny-positive
    source_evaluation_id: eval_eval_2026_06_01_xwide_dropout_p0075_epoch40_failure_buckets
    source_artifact_path: diagnostic_samples/sample_003_overlay.png
    reason: Missed-positive artifact for val/000174 with 70 positive pixels missed; illustrates that tiny positive masks remain difficult even after the 40-epoch gain.
  - figure_id: fig-epoch40-false-positive-heavy-positive
    source_evaluation_id: eval_eval_2026_06_01_xwide_dropout_p0075_epoch40_failure_buckets
    source_artifact_path: diagnostic_samples/sample_008_overlay.png
    reason: False-positive-heavy artifact for val/001487; selected because this individual broad-mask outlier worsens despite stronger aggregate metrics.
  - figure_id: fig-epoch40-false-negative-heavy-large-positive
    source_evaluation_id: eval_eval_2026_06_01_xwide_dropout_p0075_epoch40_failure_buckets
    source_artifact_path: diagnostic_samples/sample_010_overlay.png
    reason: False-negative-heavy artifact for val/002527; selected to show persistent under-segmentation on large positive masks that may motivate Boundary Target auxiliary training.
```

## Decision

Promote `run_20260601_162117_66bf89` as the current best in-contract Result and as the base for the next Candidate Experiment. The failure-bucket review confirms that the 40-epoch gain is robust at the default threshold, close to the threshold-sweep optimum, and not driven by aggregate false-positive expansion. Longer training also reduces empty-mask false-positive pixels while improving both precision and recall.

Do not extend the same training-budget line again immediately. The best threshold remains near 0.4 and the best epoch was final epoch 40, but the campaign has already consumed a substantially larger Run budget for this family. The next useful uncertainty is architectural or contract-capability driven, not simply more epochs.

## Next proposed change

Use `run_20260601_162117_66bf89` as the comparison target for the next in-contract Candidate Experiment. If Boundary Target auxiliary training remains unavailable to the agent-side validator/training path, prefer a conservative in-contract architecture hypothesis aimed at the remaining failure buckets: preserve the base-64 extra-wide U-Net, Line Target weight 0.10, `Dropout2d(p=0.075)`, and 40 epochs, but test a small decoder-side attention or refinement change only if it stays under the 10M parameter budget and does not use unsupported auxiliary targets.
