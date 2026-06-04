# 2026-06-01 xwide p=0.075 dropout failure-bucket evaluation

## Hypothesis

The bounded Post-Run Evaluation tested whether `run_20260601_085755_25cd06`, the extra-wide base-64 U-Net with Line Target auxiliary weight 0.10 and lighter bottleneck `Dropout2d(p=0.075)`, keeps the p=0.10 dropout model's qualitative failure profile while improving recall and best-validation Dice. The main concern was that the small Dice gain might come from broader masks and a material false-positive penalty.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_xwide_unet_line_aux_w010_dropout_p0075_rerun`
- Candidate Experiment path: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_rerun`
- Relevant Experiment Proposal: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_rerun/PROPOSAL.md`
- Primary Comparison Target: `run_20260530_180658_0af8a8` / `single_frame_xwide_unet_line_aux_w010_dropout`
- Evaluation Request: `evaluation-requests/eval-2026-06-01-xwide-dropout-p0075-failure-buckets.yaml`

## Run(s)

- Target Run ID: `run_20260601_085755_25cd06`
- Target Evaluation ID: `eval_eval_2026_06_01_xwide_dropout_p0075_failure_buckets`
- Comparison Run ID: `run_20260530_180658_0af8a8`
- Comparison Evaluation ID: `eval_eval_2026_05_31_xwide_dropout_failure_buckets`
- Dataset mode/subset: GVCCS Working Validation Split, whole-validation failure-bucket review over 3,889 validation samples.
- Harness/backend notes: request-gated Post-Run Evaluation completed and wrote `aggregate_metrics.json`, `threshold_sweep.json`, `per_sample_metrics.jsonl`, and bounded diagnostic artifacts under the parent Run.

## Key metrics

Whole-validation metrics at the default threshold 0.5:

| Run | Evaluation | val/dice | val/iou | val/precision | val/recall | Empty-mask FP pixels | Empty-mask samples with FP | Best threshold by Dice |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `run_20260530_180658_0af8a8` | `eval_eval_2026_05_31_xwide_dropout_failure_buckets` | 0.833856 | 0.715054 | 0.856288 | 0.812569 | 1,739 | 220 / 1,004 | 0.30 (`dice` 0.834962) |
| `run_20260601_085755_25cd06` | `eval_eval_2026_06_01_xwide_dropout_p0075_failure_buckets` | 0.834604 | 0.716155 | 0.846532 | 0.823008 | 1,621 | 135 / 1,004 | 0.40 (`dice` 0.834900) |

The p=0.075 Run preserves the small Dice improvement at whole-validation evaluation time (`+0.000748`) and gains recall (`+0.010439`) while losing precision (`-0.009757`) relative to p=0.10. Importantly, the expected false-positive penalty is not visible in the aggregate empty-mask metrics: empty-mask false-positive pixels decrease from 1,739 to 1,621, and the number of empty-mask validation samples with any false positives decreases from 220 to 135 at threshold 0.5.

The threshold sweep is also less dependent on an unusually low threshold than the p=0.10 comparison: p=0.075 peaks near threshold 0.40 and remains close to its default-threshold Dice, whereas p=0.10 peaks at 0.30. This supports treating the p=0.075 Result as a slightly more stable default-threshold base, not merely a threshold-tuning artifact.

## Qualitative observations

The bounded diagnostic samples show a mixed but acceptable failure-profile shift:

- Empty-mask false-positive aggregate behavior improves, but the selected worst empty-mask artifact is more severe for p=0.075 on `val/003797` (329 false-positive pixels) than the p=0.10 artifact on the same sample (73 false-positive pixels). This means the aggregate improvement coexists with a few sharper empty-mask outliers.
- Missed-positive selected artifacts remain concentrated on tiny masks. p=0.075 includes `val/000221` with 162 missed pixels plus repeated tiny misses (`val/000692`, `val/002294`, `val/000693`), while p=0.10 selected misses were 67, 62, 56, and 48 pixels. The new model improves aggregate recall despite at least one worse tiny-mask miss.
- False-positive-heavy selected samples are not uniformly worse. On `val/001487`, p=0.075 has similar false-positive pixels to p=0.10 (724 vs 717) but many fewer false negatives (222 vs 540), raising sample Dice from 0.4769 to 0.6532. Other selected false-positive-heavy samples remain broad-mask cases with several hundred false-positive pixels.
- False-negative-heavy artifacts still show under-segmentation on large positives. p=0.075 introduces severe examples on `val/001796` and `val/001797`, but also has high-Dice false-negative-heavy examples on `val/001775` and `val/001779`.

Overall, p=0.075 does not look like a simple broadening failure. It shifts the precision/recall tradeoff toward recall, lowers aggregate empty-mask false positives, and leaves a small set of outlier misses and false-positive-heavy masks as the main remaining qualitative risks.

![p0075 empty-mask false positive outlier](../runs/run_20260601_085755_25cd06/outputs/evaluations/eval_eval_2026_06_01_xwide_dropout_p0075_failure_buckets/diagnostic_samples/sample_000_overlay.png)

![p0075 missed tiny positive](../runs/run_20260601_085755_25cd06/outputs/evaluations/eval_eval_2026_06_01_xwide_dropout_p0075_failure_buckets/diagnostic_samples/sample_001_overlay.png)

![p0075 false-positive-heavy with improved recall](../runs/run_20260601_085755_25cd06/outputs/evaluations/eval_eval_2026_06_01_xwide_dropout_p0075_failure_buckets/diagnostic_samples/sample_009_overlay.png)

![p0075 false-negative-heavy large mask](../runs/run_20260601_085755_25cd06/outputs/evaluations/eval_eval_2026_06_01_xwide_dropout_p0075_failure_buckets/diagnostic_samples/sample_010_overlay.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-p0075-empty-mask-fp-outlier
    source_evaluation_id: eval_eval_2026_06_01_xwide_dropout_p0075_failure_buckets
    source_artifact_path: diagnostic_samples/sample_000_overlay.png
    reason: Empty-mask false-positive and worst-by-Dice artifact for val/003797; selected because aggregate empty-mask false positives improved despite this stronger outlier.
  - figure_id: fig-p0075-missed-tiny-positive
    source_evaluation_id: eval_eval_2026_06_01_xwide_dropout_p0075_failure_buckets
    source_artifact_path: diagnostic_samples/sample_001_overlay.png
    reason: Missed-positive and worst-by-Dice artifact for val/000221 with 162 positive pixels missed; illustrates the remaining tiny-mask recall failure mode.
  - figure_id: fig-p0075-false-positive-heavy-recall-tradeoff
    source_evaluation_id: eval_eval_2026_06_01_xwide_dropout_p0075_failure_buckets
    source_artifact_path: diagnostic_samples/sample_009_overlay.png
    reason: False-positive-heavy artifact for val/001487 where p=0.075 retains many false positives but reduces false negatives relative to the p=0.10 comparison.
  - figure_id: fig-p0075-false-negative-heavy-large-mask
    source_evaluation_id: eval_eval_2026_06_01_xwide_dropout_p0075_failure_buckets
    source_artifact_path: diagnostic_samples/sample_010_overlay.png
    reason: False-negative-heavy artifact for val/001796 showing under-segmentation on a large positive mask, one of the remaining failure modes after the p=0.075 improvement.
```

## Decision

Promote `run_20260601_085755_25cd06` as the preferred in-contract base over `run_20260530_180658_0af8a8`. The p=0.075 model's whole-validation failure-bucket review supports the earlier best-validation Result: it improves Dice and recall, reduces aggregate empty-mask false positives, and has a default-threshold performance close to the threshold-sweep optimum. The remaining risks are specific outliers and tiny missed positives rather than a broad false-positive collapse.

Do not continue architecture-only dropout tuning immediately. The p=0.075 vs p=0.10 frontier is narrow, and the next useful uncertainty is whether an approved auxiliary target or other Harness-owned capability slice can reduce tiny missed positives and boundary under-segmentation without sacrificing the improved aggregate profile.

## Next proposed change

Use `run_20260601_085755_25cd06` as the comparison target/base for the next approved capability-slice experiment, especially Boundary Target auxiliary training if/when approved. If Boundary Target remains unavailable, prefer a new in-contract hypothesis that specifically addresses tiny missed positives or calibration while preserving the p=0.075 default-threshold profile; avoid another pure dropout-rate sweep unless a new diagnostic justifies it.
