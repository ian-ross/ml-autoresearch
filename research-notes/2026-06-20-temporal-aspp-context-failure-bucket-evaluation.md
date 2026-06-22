# 2026-06-20 temporal ASPP-context failure-bucket evaluation

## Hypothesis

The bounded failure-bucket Post-Run Evaluation for `run_20260618_204556_e6a60b` tested whether the centered three-frame temporal ASPP-context U-Net's aggregate improvement is broad and threshold-stable on the temporal-eligible GVCCS Working Validation subset, or whether it hides concentrated failure buckets such as empty-mask false positives or fully missed small positive masks.

## Candidate Experiment(s)

- Candidate Experiment ID: `temporal_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context`
- Candidate Experiment path: `candidates/temporal_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context`
- Relevant Experiment Proposal: `candidates/temporal_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context/PROPOSAL.md`
- Primary Comparison Target: `run_20260615_140810_2bee94` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context`

## Run(s)

- Target Run ID: `run_20260618_204556_e6a60b`
- Evaluation Request: `evaluation-requests/eval-2026-06-20-temporal-aspp-context-failure-buckets.yaml`
- Evaluation ID: `eval_eval_2026_06_20_temporal_aspp_context_failure_buckets`
- Evaluation artifact root: `runs/run_20260618_204556_e6a60b/outputs/evaluations/eval_eval_2026_06_20_temporal_aspp_context_failure_buckets/`
- Dataset mode/subset: GVCCS Working Validation Split using `centered_temporal_rgb_clip` and `temporal_eligible_center` frame selection; 3,850 validation samples.

## Key metrics

Whole-validation aggregate metrics at threshold 0.5:

| Run | Evaluation | Dice | IoU | Precision | Recall | Samples |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `run_20260618_204556_e6a60b` | `eval_eval_2026_06_20_temporal_aspp_context_failure_buckets` | 0.873707 | 0.775737 | 0.877783 | 0.869669 | 3850 |

Threshold sweep summary:

| Threshold | Dice | IoU | Precision | Recall |
| ---: | ---: | ---: | ---: | ---: |
| 0.35 | 0.873565 | 0.775513 | 0.866354 | 0.880896 |
| 0.45 | 0.873771 | 0.775837 | 0.874178 | 0.873364 |
| 0.50 | 0.873707 | 0.775737 | 0.877783 | 0.869669 |
| 0.60 | 0.873170 | 0.774891 | 0.884826 | 0.861818 |
| 0.75 | 0.871138 | 0.771696 | 0.896701 | 0.846992 |

The best Dice in the sweep occurs at threshold 0.45 (`0.873771`), only `+0.000064` above the default threshold. This confirms that the temporal result is not a threshold-local artifact; the 0.35--0.60 neighborhood is nearly flat while trading recall for precision.

Failure-bucket and per-sample summaries at threshold 0.5:

| Quantity | Value |
| --- | ---: |
| Empty-mask samples | 1052 |
| Empty masks with any predicted positives | 123 |
| Mean predicted-positive pixels on empty masks | 0.441 |
| Maximum predicted-positive pixels on an empty mask | 27 |
| Positive-mask samples | 2798 |
| Missed positive masks with zero predicted pixels | 39 |
| Mean per-sample false-positive pixels | 61.93 |
| Mean per-sample false-negative pixels | 66.65 |
| Mean false-positive pixels on positive-mask samples | 85.04 |
| Mean false-negative pixels on positive-mask samples | 91.72 |

## Qualitative observations

The evaluation supports the aggregate improvement from the prior note. Empty-mask false positives remain bounded: only 123 of 1,052 empty temporal-eligible validation samples have any positive prediction, the mean empty-mask predicted-positive count is below one pixel, and the worst selected empty-mask false positive contains 27 pixels. This argues against the temporal model's improvement being driven by broad empty-sky hallucination.

Residual failures are concentrated in tiny or difficult positive masks. The worst diagnostic sample `val/003020` is a fully missed 60-pixel positive mask; other worst-by-Dice samples contain 22--32 positive pixels with little or no overlap. The model still has 39 fully missed positive masks at threshold 0.5. False-positive-heavy diagnostics on positive masks also remain, with selected samples showing hundreds of false-positive pixels around real contrail cases rather than empty-sky spread.

Selected diagnostic artifacts:

![Temporal missed positive overlay](../runs/run_20260618_204556_e6a60b/outputs/evaluations/eval_eval_2026_06_20_temporal_aspp_context_failure_buckets/diagnostic_samples/sample_000_overlay.png)

![Temporal empty-mask false-positive overlay](../runs/run_20260618_204556_e6a60b/outputs/evaluations/eval_eval_2026_06_20_temporal_aspp_context_failure_buckets/diagnostic_samples/sample_004_overlay.png)

![Temporal false-positive-heavy overlay](../runs/run_20260618_204556_e6a60b/outputs/evaluations/eval_eval_2026_06_20_temporal_aspp_context_failure_buckets/diagnostic_samples/sample_011_overlay.png)

![Temporal false-negative-heavy overlay](../runs/run_20260618_204556_e6a60b/outputs/evaluations/eval_eval_2026_06_20_temporal_aspp_context_failure_buckets/diagnostic_samples/sample_012_overlay.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-temporal-eval-missed-positive-overlay
    source_evaluation_id: eval_eval_2026_06_20_temporal_aspp_context_failure_buckets
    source_artifact_path: diagnostic_samples/sample_000_overlay.png
    reason: Worst-by-Dice missed-positive diagnostic; selected to show the remaining fully missed small positive masks despite the temporal aggregate improvement.
  - figure_id: fig-temporal-eval-empty-mask-fp-overlay
    source_evaluation_id: eval_eval_2026_06_20_temporal_aspp_context_failure_buckets
    source_artifact_path: diagnostic_samples/sample_004_overlay.png
    reason: Empty-mask false-positive diagnostic; selected because empty-sky false positives are bounded and do not appear to explain the model's tradeoff.
  - figure_id: fig-temporal-eval-fp-heavy-overlay
    source_evaluation_id: eval_eval_2026_06_20_temporal_aspp_context_failure_buckets
    source_artifact_path: diagnostic_samples/sample_011_overlay.png
    reason: False-positive-heavy positive-mask diagnostic; selected to show residual over-extension around real contrail cases.
  - figure_id: fig-temporal-eval-fn-heavy-overlay
    source_evaluation_id: eval_eval_2026_06_20_temporal_aspp_context_failure_buckets
    source_artifact_path: diagnostic_samples/sample_012_overlay.png
    reason: False-negative-heavy diagnostic; selected to show that some positive masks still lose substantial true-positive area under the default threshold.
```

## Decision

Keep `run_20260618_204556_e6a60b` as the leading in-contract temporal-family Result. The Post-Run Evaluation validates that its gain is threshold-stable and not dominated by empty-mask hallucination. The main caveat remains comparison validity: temporal input uses `temporal_eligible_center`, so this is not an apples-to-apples full-validation replacement for the single-frame ASPP-context Run.

## Next proposed change

The next Candidate Experiment should isolate the data-policy effect before adding more temporal architecture complexity: run a matched single-frame ASPP-context control on `temporal_eligible_center` frame selection if the current contract permits that combination. If it does not, request a small Harness-owned capability to allow a single-frame model to use temporal-eligible center frame selection for controlled comparisons.
