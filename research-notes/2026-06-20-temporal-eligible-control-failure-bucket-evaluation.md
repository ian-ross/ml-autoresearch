# 2026-06-20 temporal-eligible single-frame control failure-bucket evaluation

## Hypothesis

The bounded failure-bucket Post-Run Evaluation for `run_20260620_092332_3ee9a1` tested whether the matched single-frame ASPP-context control on `temporal_eligible_center` has the same error profile as the centered three-frame temporal ASPP-context run, or whether the temporal model's remaining controlled advantage is explained by specific failure buckets.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context_temporal_eligible_control`
- Candidate Experiment path: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context_temporal_eligible_control`
- Relevant Experiment Proposal: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context_temporal_eligible_control/PROPOSAL.md`
- Primary Comparison Target: `run_20260618_204556_e6a60b` / `temporal_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context`

## Run(s)

- Target Run ID: `run_20260620_092332_3ee9a1`
- Evaluation Request: `evaluation-requests/eval-2026-06-20-temporal-eligible-control-failure-buckets.yaml`
- Evaluation ID: `eval_eval_2026_06_20_temporal_eligible_control_failure_buckets`
- Evaluation artifact root: `runs/run_20260620_092332_3ee9a1/outputs/evaluations/eval_eval_2026_06_20_temporal_eligible_control_failure_buckets/`
- Dataset mode/subset: GVCCS Working Validation Split using `single_frame_rgb` and `temporal_eligible_center` frame selection; 3,850 validation samples.
- Comparison evaluation: `eval_eval_2026_06_20_temporal_aspp_context_failure_buckets` for `run_20260618_204556_e6a60b`, same 3,850-sample temporal-eligible validation subset.

## Key metrics

Whole-validation aggregate metrics at threshold 0.5:

| Run | Evaluation | Input mode | Dice | IoU | Precision | Recall | Samples |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `run_20260620_092332_3ee9a1` | `eval_eval_2026_06_20_temporal_eligible_control_failure_buckets` | single-frame | 0.871672 | 0.772535 | 0.874014 | 0.869343 | 3850 |
| `run_20260618_204556_e6a60b` | `eval_eval_2026_06_20_temporal_aspp_context_failure_buckets` | centered 3-frame clip | 0.873707 | 0.775737 | 0.877783 | 0.869669 | 3850 |

The temporal run remains ahead by about `+0.00203` Dice, `+0.00320` IoU, `+0.00377` precision, and only `+0.00033` recall on the matched validation subset.

Threshold sweep summary:

| Run | Best threshold by Dice | Best Dice | Dice at 0.50 | Notes |
| --- | ---: | ---: | ---: | --- |
| `run_20260620_092332_3ee9a1` | 0.50 | 0.871672 | 0.871672 | Very flat from 0.35--0.60, with the default threshold already optimal in the sweep. |
| `run_20260618_204556_e6a60b` | 0.45 | 0.873771 | 0.873707 | Also flat from 0.35--0.60; threshold tuning adds only `+0.000064` Dice. |

Failure-bucket and per-sample summaries at threshold 0.5:

| Quantity | Single-frame control | Temporal run | Interpretation |
| --- | ---: | ---: | --- |
| Empty-mask samples | 1052 | 1052 | Same subset. |
| Empty masks with any predicted positives | 71 | 123 | Control triggers on fewer empty masks, but see max below. |
| Mean predicted-positive pixels on empty masks | 0.391 | 0.441 | Both are low; empty-sky hallucination is bounded for both. |
| Maximum predicted-positive pixels on an empty mask | 85 | 27 | Control has rarer but larger worst empty-mask false positives. |
| Positive-mask samples | 2798 | 2798 | Same subset. |
| Missed positive masks with zero predicted pixels | 51 | 39 | Temporal input reduces fully missed positive masks. |
| Mean per-sample false-positive pixels | 64.09 | 61.93 | Temporal has a small aggregate FP advantage. |
| Mean per-sample false-negative pixels | 66.82 | 66.65 | Nearly tied, slight temporal advantage. |
| Mean false-positive pixels on positive-mask samples | 88.04 | 85.04 | Temporal is modestly cleaner on positive examples. |
| Mean false-negative pixels on positive-mask samples | 91.94 | 91.72 | Nearly tied. |

## Qualitative observations

The matched single-frame control's failure-bucket profile is similar to the temporal run but not identical. Empty-mask false positives are not the main reason for the temporal advantage: the control has fewer empty masks with any predicted positives, yet its aggregate Dice and IoU are lower. The control's worst empty-mask false positive, `val/001160`, predicts 85 positive pixels, which is larger than the temporal evaluation's worst selected empty-mask false-positive case.

The clearest controlled difference is on small positives. The control fully misses 51 positive masks at threshold 0.5 versus 39 for the temporal run. Its selected missed-positive diagnostics include 61-, 33-, and 27-pixel masks with zero predicted positives. This suggests the centered temporal input is helping recover a small number of faint or tiny positives, while leaving the broader false-negative-heavy positive-mask examples nearly tied.

False-positive-heavy positive-mask artifacts remain in both runs. The control's selected `val/003247` and `val/003246` examples still show hundreds of false-positive pixels around real contrail masks, while the temporal run's selected false-positive-heavy examples were different and sometimes lower-Dice. This bucket does not support a simple claim that temporal input uniformly suppresses over-extension, but the aggregate positive-mask FP mean is modestly better for the temporal run.

Selected diagnostic artifacts:

![Control empty-mask false-positive overlay](../runs/run_20260620_092332_3ee9a1/outputs/evaluations/eval_eval_2026_06_20_temporal_eligible_control_failure_buckets/diagnostic_samples/sample_000_overlay.png)

![Control missed-positive overlay](../runs/run_20260620_092332_3ee9a1/outputs/evaluations/eval_eval_2026_06_20_temporal_eligible_control_failure_buckets/diagnostic_samples/sample_001_overlay.png)

![Control false-negative-heavy overlay](../runs/run_20260620_092332_3ee9a1/outputs/evaluations/eval_eval_2026_06_20_temporal_eligible_control_failure_buckets/diagnostic_samples/sample_011_overlay.png)

![Control false-positive-heavy overlay](../runs/run_20260620_092332_3ee9a1/outputs/evaluations/eval_eval_2026_06_20_temporal_eligible_control_failure_buckets/diagnostic_samples/sample_014_overlay.png)

![Temporal missed-positive reference overlay](../runs/run_20260618_204556_e6a60b/outputs/evaluations/eval_eval_2026_06_20_temporal_aspp_context_failure_buckets/diagnostic_samples/sample_000_overlay.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-control-empty-mask-fp-overlay
    source_evaluation_id: eval_eval_2026_06_20_temporal_eligible_control_failure_buckets
    source_artifact_path: diagnostic_samples/sample_000_overlay.png
    reason: Worst selected empty-mask false-positive case for the matched single-frame control; selected because it shows rarer but larger empty-mask hallucination than the temporal run.
  - figure_id: fig-control-missed-positive-overlay
    source_evaluation_id: eval_eval_2026_06_20_temporal_eligible_control_failure_buckets
    source_artifact_path: diagnostic_samples/sample_001_overlay.png
    reason: Missed-positive diagnostic for the matched control; selected because the control has more fully missed positive masks than the temporal run.
  - figure_id: fig-control-fn-heavy-overlay
    source_evaluation_id: eval_eval_2026_06_20_temporal_eligible_control_failure_buckets
    source_artifact_path: diagnostic_samples/sample_011_overlay.png
    reason: False-negative-heavy positive-mask diagnostic for the matched control; selected to illustrate residual large-positive under-segmentation after controlling for frame selection.
  - figure_id: fig-control-fp-heavy-overlay
    source_evaluation_id: eval_eval_2026_06_20_temporal_eligible_control_failure_buckets
    source_artifact_path: diagnostic_samples/sample_014_overlay.png
    reason: False-positive-heavy positive-mask diagnostic for the matched control; selected to show over-extension around real contrail masks remains in the single-frame control.
  - figure_id: fig-temporal-missed-positive-reference-overlay
    source_evaluation_id: eval_eval_2026_06_20_temporal_aspp_context_failure_buckets
    source_artifact_path: diagnostic_samples/sample_000_overlay.png
    reason: Temporal run missed-positive reference; selected to compare the shared small-positive failure mode while noting that the temporal run has fewer fully missed positives overall.
```

## Decision

Keep `run_20260618_204556_e6a60b` as the current best Result and treat temporal input as a real but modest controlled improvement after isolating the `temporal_eligible_center` data-policy effect. The failure-bucket evidence points to fewer fully missed small positive masks and slightly cleaner positive-mask predictions, not to threshold calibration or empty-mask false-positive suppression, as the most plausible source of the temporal run's remaining advantage.

## Next proposed change

The next Candidate Experiment should target the small-positive recovery mechanism without substantially increasing false positives. A conservative in-contract direction is a temporal refinement that emphasizes fine-scale features in the temporal ASPP-context family, while preserving the current line auxiliary weight, dropout, scheduler, early stopping, and `temporal_eligible_center` policy so the comparison remains controlled against `run_20260618_204556_e6a60b`.
