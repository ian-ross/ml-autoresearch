# 2026-06-01 xwide U-Net lighter bottleneck dropout improvement

## Hypothesis

Reducing bottleneck-only `Dropout2d` in the current best extra-wide base-64 U-Net with Line Target auxiliary weight 0.10 from `p=0.10` to `p=0.075` tested whether the model could keep the useful recall gain from dropout while recovering some precision and improving best-validation Dice at the default threshold. This note covers the completed rerun after the first p=0.075 attempt was interrupted by an operator and recorded as a non-scientific Harness failure.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_xwide_unet_line_aux_w010_dropout_p0075_rerun`
- Candidate Experiment path: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_rerun`
- Relevant Experiment Proposal: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p0075_rerun/PROPOSAL.md`
- Primary Comparison Target: `run_20260530_180658_0af8a8` / `single_frame_xwide_unet_line_aux_w010_dropout`
- Secondary context: `run_20260530_134005_6f20b1` / `single_frame_xwide_unet_line_auxiliary_w010`, `run_20260531_041919_8dbd72` / `single_frame_xwide_unet_line_aux_w010_attention_dropout`, and `run_20260531_160756_9f5a12` / `single_frame_xwide_unet_line_aux_w010_dropout_p015`.
- Contract choices: Single-Frame RGB Input, mask logits output, Line Target auxiliary output with `weighted_bce` weight 0.10, deterministic shuffle Sampling Policy, no augmentation, `bce_dice`, AdamW, learning rate 0.001, requested/effective batch size 8, max epochs 30.

## Run(s)

- Run ID: `run_20260601_085755_25cd06`
- Dataset mode/subset: GVCCS Working Validation Split
- Harness/backend notes: completed on CUDA with no Resource Failure retry; requested and effective batch size were both 8. Best-validation model artifact persisted at `outputs/models/best_epoch_model.pt`. Model summary reports 7,785,794 parameters, unchanged from the p=0.10 and p=0.15 dropout comparisons and below the 10,000,000 parameter budget.

## Key metrics

Best-validation metrics, selected by max `val/dice`:

| Run | Candidate Experiment | Best epoch | val/dice | val/iou | val/precision | val/recall | val/loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `run_20260530_134005_6f20b1` | `single_frame_xwide_unet_line_auxiliary_w010` | 28 | 0.832669 | 0.713310 | 0.862413 | 0.804908 | 0.510368 |
| `run_20260530_180658_0af8a8` | `single_frame_xwide_unet_line_aux_w010_dropout` | 29 | 0.833852 | 0.715049 | 0.856300 | 0.812552 | 0.510518 |
| `run_20260531_041919_8dbd72` | `single_frame_xwide_unet_line_aux_w010_attention_dropout` | 28 | 0.831409 | 0.711463 | 0.840498 | 0.822515 | 0.510823 |
| `run_20260531_160756_9f5a12` | `single_frame_xwide_unet_line_aux_w010_dropout_p015` | 28 | 0.831613 | 0.711762 | 0.844943 | 0.818698 | 0.511055 |
| `run_20260601_085755_25cd06` | `single_frame_xwide_unet_line_aux_w010_dropout_p0075_rerun` | 30 | 0.834603 | 0.716153 | 0.846543 | 0.822995 | 0.508752 |

The p=0.075 rerun improves best-validation Dice by about `+0.00075` over the p=0.10 comparison target and becomes the best observed in-contract Result. The mechanism is not the expected precision recovery: precision falls by about `-0.00976` relative to p=0.10, but recall rises by about `+0.01044`, and the Dice tradeoff is slightly better overall. Precision still clears the proposal guardrail of approximately `>=0.8457`, remains above the attention-dropout and p=0.15 variants, and recall is the highest among the compared Runs.

Final completed-epoch metrics for `run_20260601_085755_25cd06` are the same as best-validation metrics because the best epoch was the final epoch 30: `val/dice` 0.834603, `val/iou` 0.716153, `val/precision` 0.846543, `val/recall` 0.822995, and `val/loss` 0.508752. This eliminates the p=0.10 model's best-to-final Dice gap of about `0.01379` in this Run and supports lighter bottleneck dropout as a more stable training choice under the current 30-epoch policy.

## Qualitative observations

The saved first-N prediction samples are mixed relative to p=0.10:

- `val/000000`: Dice 0.8154, IoU 0.6883, weaker than the p=0.10 sample 000 Dice 0.8462 but stronger than the attention-gated variant's sample 000 Dice 0.7603.
- `val/000001`: Dice 0.8321, IoU 0.7125, weaker than the p=0.10 sample 001 Dice 0.8613 but stronger than the p=0.15 and attention-gated sample 001 results recorded in earlier notes.

These first-N samples do not by themselves explain the aggregate improvement. They suggest the new best Result is driven by broader validation-set recall and stability rather than visibly better masks on the two saved positives. A bounded failure-bucket Post-Run Evaluation is therefore needed before treating p=0.075 as a qualitatively safer base.

![p0075 dropout rerun sample 000 overlay](../runs/run_20260601_085755_25cd06/outputs/prediction_samples/sample_000_overlay.png)

![p0075 dropout rerun sample 001 overlay](../runs/run_20260601_085755_25cd06/outputs/prediction_samples/sample_001_overlay.png)

![p0075 dropout rerun sample 001 heatmap](../runs/run_20260601_085755_25cd06/outputs/prediction_samples/sample_001_probability_heatmap.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-xwide-dropout-p0075-rerun-sample-000-overlay
    source_run_id: run_20260601_085755_25cd06
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: First saved validation overlay for the p=0.075 dropout rerun; sample Dice 0.8154 helps check whether the aggregate Dice gain corresponds to visibly improved first-N masks.
  - figure_id: fig-xwide-dropout-p0075-rerun-sample-001-overlay
    source_run_id: run_20260601_085755_25cd06
    source_artifact_path: outputs/prediction_samples/sample_001_overlay.png
    reason: Second saved validation overlay for the p=0.075 dropout rerun; sample Dice 0.8321 supports qualitative comparison against p=0.10, p=0.15, and attention-dropout variants.
  - figure_id: fig-xwide-dropout-p0075-rerun-sample-001-heatmap
    source_run_id: run_20260601_085755_25cd06
    source_artifact_path: outputs/prediction_samples/sample_001_probability_heatmap.png
    reason: Probability heatmap for the second saved validation sample, useful for checking whether the lighter dropout model broadens or softens contrail probabilities while improving aggregate recall.
```

## Decision

Promote `run_20260601_085755_25cd06` as the current best in-contract Result by best-validation Dice, with caution. The candidate completed cleanly, met the main Dice and recall criteria, stayed within the precision guardrail, and ended with the best epoch at the final epoch. However, the original precision-recovery hypothesis is only partially supported: p=0.075 wins by increasing recall further, not by restoring precision relative to p=0.10.

Stop the local bottleneck-dropout-rate sweep for now. The p=0.075, p=0.10, and p=0.15 results show a narrow architecture-only frontier where small regularization changes trade precision, recall, and stability. Further dropout-rate tuning risks overfitting the Working Validation Split without a better diagnostic signal.

## Next proposed change

Request a bounded failure-bucket Post-Run Evaluation for `run_20260601_085755_25cd06` before launching another Candidate Experiment. Compare against the existing p=0.10 dropout evaluation for `run_20260530_180658_0af8a8`, focusing on missed-positive masks, false-negative-heavy samples, empty-mask false positives, and threshold-sweep optimum. If p=0.075 improves recall without a large false-positive penalty, use it as the new base for a boundary-target auxiliary experiment or other human-approved capability slice; if it mainly broadens masks, keep p=0.10 as the safer base despite slightly lower Dice.
