# 2026-05-31 xwide dropout failure-bucket evaluation

## Hypothesis

The failure-bucket Post-Run Evaluation for the extra-wide base-64 U-Net with Line Target auxiliary weight 0.10 and bottleneck `Dropout2d(p=0.10)` tested whether the best-validation Dice improvement in `run_20260530_180658_0af8a8` represents a robust operating point rather than a recall gain bought by unacceptable false positives. The expected confirmation signal was: whole-validation metrics at threshold 0.5 at least match the prior plain xwide Run, the threshold sweep remains stable, and diagnostic buckets do not show a material increase in missed positives, false-negative-heavy cases, or empty-mask false positives.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_xwide_unet_line_aux_w010_dropout`
- Candidate source: `/history/candidates/single_frame_xwide_unet_line_aux_w010_dropout/`
- Relevant Experiment Proposal: `/history/candidates/single_frame_xwide_unet_line_aux_w010_dropout/PROPOSAL.md`
- Primary Comparison Target: `single_frame_xwide_unet_line_auxiliary_w010` / Run `run_20260530_134005_6f20b1`
- Secondary context: `single_frame_wide_unet_line_auxiliary_w010` / Run `run_20260530_101019_def893`

## Run(s)

- Target Run ID: `run_20260530_180658_0af8a8`
- Post-Run Evaluation ID: `eval_eval_2026_05_31_xwide_dropout_failure_buckets`
- Evaluation Request: `eval_2026_05_31_xwide_dropout_failure_buckets`
- Dataset/split: GVCCS Working Validation Split (`val`), 3,889 evaluated samples
- Harness/backend notes: request-gated `failure_bucket_review`; primary threshold 0.5; artifact budget 24 samples; evaluation completed successfully.

## Key metrics

| Run | Evaluation | threshold | val/dice | val/iou | val/precision | val/recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `run_20260530_101019_def893` wide base-48 | `eval_eval_2026_05_30_wide_unet_failure_buckets` | 0.50 | 0.829163 | 0.708180 | 0.850680 | 0.808707 |
| `run_20260530_134005_6f20b1` xwide base-64 | `eval_eval_2026_05_30_xwide_unet_failure_buckets` | 0.50 | 0.832670 | 0.713311 | 0.862413 | 0.804909 |
| `run_20260530_180658_0af8a8` xwide dropout | `eval_eval_2026_05_31_xwide_dropout_failure_buckets` | 0.50 | 0.833856 | 0.715054 | 0.856288 | 0.812569 |
| `run_20260530_180658_0af8a8` xwide dropout | threshold sweep best | 0.30 | 0.834962 | 0.716683 | 0.838005 | 0.831942 |

The dropout evaluation confirms the best-epoch training metrics. At the default threshold, dropout improves whole-validation Dice by `+0.00119` over plain xwide and `+0.00469` over wide. Its precision drops relative to plain xwide (`-0.00613`) but remains above wide, while recall improves over both plain xwide (`+0.00766`) and wide (`+0.00386`). The threshold sweep optimum moves from 0.25 for plain xwide to 0.30 for dropout, with a slightly higher best sweep Dice (`0.834962` vs `0.834485`), so the model remains threshold-stable rather than becoming brittle.

## Qualitative observations

The diagnostic bucket set includes `worst_by_dice`, `best_by_dice`, `false_positive_heavy`, `false_negative_heavy`, `empty_mask_false_positives`, and `missed_positive_masks`.

The dropout Run still has small positive masks that are completely missed at threshold 0.5: `val/000692` (67 positive pixels), `val/000059` (62), `val/000693` (56), and `val/000696` (48). This is not eliminated, but the largest missed-positive sample in the plain xwide diagnostic was `val/000592` with 100 positive pixels; that case no longer appears in the bounded missed-positive artifacts. The remaining misses are mostly very small masks and overlap with known difficult examples from prior wide/xwide diagnostics.

The main tradeoff is an increase in the largest sampled empty-mask false positive relative to plain xwide. Dropout's largest sampled empty-mask false positive is `val/003797` with 73 predicted positive pixels, compared with 52 for plain xwide and 73 for the wide base-48 model. The next sampled dropout empty-mask false positives are 56, 47, and 37 predicted positive pixels. This suggests the recall recovery partly spends false-positive budget, but not beyond the earlier wide baseline.

False-negative-heavy samples remain substantial on larger positive masks. Dropout includes `val/002495` with 1,001 false-negative pixels and Dice 0.7410, `val/002499` with 1,014 false-negative pixels and Dice 0.7704, and `val/002500` with 1,060 false-negative pixels and Dice 0.7892. This is mixed relative to plain xwide: `val/002527` with 1,118 false-negative pixels is absent from the dropout diagnostic, but `val/002495` reappears. The aggregate recall gain is real, but thin or low-contrast regions remain the dominant error mode.

![Dropout missed positive overlay](../runs/run_20260530_180658_0af8a8/outputs/evaluations/eval_eval_2026_05_31_xwide_dropout_failure_buckets/diagnostic_samples/sample_001_overlay.png)

![Dropout empty-mask false positive overlay](../runs/run_20260530_180658_0af8a8/outputs/evaluations/eval_eval_2026_05_31_xwide_dropout_failure_buckets/diagnostic_samples/sample_000_overlay.png)

![Dropout false-negative-heavy overlay](../runs/run_20260530_180658_0af8a8/outputs/evaluations/eval_eval_2026_05_31_xwide_dropout_failure_buckets/diagnostic_samples/sample_011_overlay.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-xwide-dropout-missed-positive-sample-001
    source_evaluation_id: eval_eval_2026_05_31_xwide_dropout_failure_buckets
    source_artifact_path: diagnostic_samples/sample_001_overlay.png
    reason: Shows a remaining missed-positive failure for dropout at val/000692, with 67 positive pixels and zero predicted positives at threshold 0.5.
  - figure_id: fig-xwide-dropout-empty-mask-fp-sample-000
    source_evaluation_id: eval_eval_2026_05_31_xwide_dropout_failure_buckets
    source_artifact_path: diagnostic_samples/sample_000_overlay.png
    reason: Shows the largest sampled dropout empty-mask false positive, val/003797 with 73 predicted positive pixels, to assess the false-positive cost of the recall gain.
  - figure_id: fig-xwide-dropout-fn-heavy-sample-011
    source_evaluation_id: eval_eval_2026_05_31_xwide_dropout_failure_buckets
    source_artifact_path: diagnostic_samples/sample_011_overlay.png
    reason: Shows a false-negative-heavy larger-mask case, val/002495 with 1,001 false-negative pixels, illustrating the main remaining error mode after dropout.
```

## Decision

Treat `run_20260530_180658_0af8a8` as the current best in-contract base. The Post-Run Evaluation supports the earlier Research Note: bottleneck dropout improves Dice and recall without producing a false-positive penalty worse than the older wide baseline. It does not solve small-positive misses or false-negative-heavy larger masks, so another pure width or simple dropout-only variant is unlikely to be the best next use of budget.

## Next proposed change

The next Candidate Experiment should make a bounded in-contract refinement to the xwide-dropout base that targets remaining false negatives without greatly increasing empty-mask false positives. Prefer a small architecture-local change such as adding lightweight decoder attention/gating or modest deep-supervision-like architecture support if already within the Candidate Experiment Contract. If those are not contract-safe, pivot to a Capability Request for a Harness-owned threshold selection, scheduler, early-stopping, or boundary-target slice rather than encoding policy changes in candidate code.
