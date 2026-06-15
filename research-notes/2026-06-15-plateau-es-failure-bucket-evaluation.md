# 2026-06-15 plateau/es failure-bucket evaluation

## Hypothesis

The bounded failure-bucket Post-Run Evaluation for `run_20260614_124226_05e3eb` tested whether the reduce-on-plateau plus best-checkpoint-restored training-policy improvement for the extra-wide p=0.075 Line Target U-Net was robust across the full GVCCS Working Validation Split, rather than being a narrow threshold-0.5 gain or hiding worse known failure buckets.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es`
- Candidate Experiment path: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es`
- Relevant Experiment Proposal: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es/PROPOSAL.md`
- Primary Comparison Target: `run_20260602_203450_c05550` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60`

## Run(s)

- Target Run ID: `run_20260614_124226_05e3eb`
- Evaluation Request: `evaluation-requests/eval-2026-06-15-plateau-es-failure-buckets.yaml`
- Evaluation ID: `eval_eval_2026_06_15_plateau_es_failure_buckets`
- Evaluation artifact root: `runs/run_20260614_124226_05e3eb/outputs/evaluations/eval_eval_2026_06_15_plateau_es_failure_buckets/`
- Comparison evaluation: `eval_eval_2026_06_03_xwide_dropout_p0075_epoch60_failure_buckets` for `run_20260602_203450_c05550`

## Key metrics

Whole-validation aggregate metrics at threshold 0.5:

| Run | Evaluation | Dice | IoU | Precision | Recall | Samples |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `run_20260602_203450_c05550` | `eval_eval_2026_06_03_xwide_dropout_p0075_epoch60_failure_buckets` | 0.849040 | 0.737679 | 0.867067 | 0.831747 | 3889 |
| `run_20260614_124226_05e3eb` | `eval_eval_2026_06_15_plateau_es_failure_buckets` | 0.861625 | 0.756890 | 0.883590 | 0.840726 | 3889 |

The training-policy run preserves the earlier best-validation improvement under whole-validation evaluation: Dice improves by about `+0.01259`, IoU by `+0.01921`, precision by `+0.01652`, and recall by `+0.00898` over the fixed-60 parent at the default threshold.

Threshold sweep summary:

| Run | Best threshold by Dice | Best sweep Dice | Dice @ 0.25 | Dice @ 0.35 | Dice @ 0.50 | Dice @ 0.75 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `run_20260602_203450_c05550` | 0.35 | 0.849576 | 0.849168 | 0.849576 | 0.849040 | 0.844549 |
| `run_20260614_124226_05e3eb` | 0.25 | 0.862611 | 0.862611 | 0.862357 | 0.861625 | 0.858191 |

The gain is not a single-threshold artifact. The plateau/es Run remains above the parent across the inspected threshold band, with the best threshold shifting lower from 0.35 to 0.25.

Failure-bucket and per-sample summaries at threshold 0.5:

| Quantity | Parent fixed-60 | Plateau/es | Direction |
| --- | ---: | ---: | --- |
| Empty-mask samples | 1004 | 1004 | unchanged |
| Empty masks with any predicted positives | 184 | 130 | improved |
| Mean predicted-positive pixels on empty masks | 1.103 | 0.846 | improved |
| Maximum predicted-positive pixels on an empty mask | 37 | 50 | worse tail case |
| Positive-mask samples | 2885 | 2885 | unchanged |
| Missed positive masks with zero predicted pixels | 74 | 73 | essentially unchanged/slightly improved |
| Mean per-sample false-positive pixels | 64.13 | 55.71 | improved |
| Mean per-sample false-negative pixels | 84.62 | 80.10 | improved |

## Qualitative observations

The diagnostic buckets show a mostly favorable whole-validation profile. The new Run reduces empty-mask affected-sample count and average false-positive load, while also reducing mean false negatives. The remaining concern is not broad empty-sky spread but a worse isolated empty-mask maximum: the worst empty mask has 50 predicted-positive pixels versus 37 in the fixed-60 parent.

Missed-positive behavior is not solved. The count of fully missed positive masks changes only from 74 to 73, and the worst-by-Dice bucket still contains small or faint positive masks with zero predicted pixels. This suggests the scheduler/early-stopping policy made the existing model better calibrated overall but did not address the small-positive detection failure mode.

Selected diagnostic artifacts:

![plateau/es missed positive overlay](../runs/run_20260614_124226_05e3eb/outputs/evaluations/eval_eval_2026_06_15_plateau_es_failure_buckets/diagnostic_samples/sample_000_overlay.png)

![plateau/es false-positive-heavy overlay](../runs/run_20260614_124226_05e3eb/outputs/evaluations/eval_eval_2026_06_15_plateau_es_failure_buckets/diagnostic_samples/sample_004_overlay.png)

![plateau/es empty-mask false-positive overlay](../runs/run_20260614_124226_05e3eb/outputs/evaluations/eval_eval_2026_06_15_plateau_es_failure_buckets/diagnostic_samples/sample_016_overlay.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-plateau-es-missed-positive-overlay
    source_evaluation_id: eval_eval_2026_06_15_plateau_es_failure_buckets
    source_artifact_path: diagnostic_samples/sample_000_overlay.png
    reason: Worst-by-Dice missed-positive diagnostic for the plateau/es Run; selected to show that the improved training policy still fully misses some small positive masks.
  - figure_id: fig-plateau-es-false-positive-heavy-overlay
    source_evaluation_id: eval_eval_2026_06_15_plateau_es_failure_buckets
    source_artifact_path: diagnostic_samples/sample_004_overlay.png
    reason: False-positive-heavy positive-mask diagnostic; selected because the Run improves mean false positives overall but still has high-FP positive cases.
  - figure_id: fig-plateau-es-empty-mask-fp-overlay
    source_evaluation_id: eval_eval_2026_06_15_plateau_es_failure_buckets
    source_artifact_path: diagnostic_samples/sample_016_overlay.png
    reason: Empty-mask false-positive diagnostic; selected because empty-mask affected-sample count improves while the worst empty-mask tail case worsens.
```

## Decision

Promote `run_20260614_124226_05e3eb` as the primary in-contract Comparison Target for the next architecture hypothesis. The whole-validation evaluation confirms that the aggregate gain is robust across threshold sweep and improves the average false-positive/false-negative profile. Do not spend the next step on threshold-only tuning: a lower threshold can slightly increase Dice, but the dominant unresolved research issue is missed small positives rather than threshold calibration alone.

## Next proposed change

Use the plateau/es training policy as the default training configuration for the next Candidate Experiment. The next architecture hypothesis should target missed small positives while preserving the improved precision profile, for example by adding a lightweight multi-scale context/detail path within the existing U-Net family rather than revisiting boundary auxiliary loss, which previously regressed.
