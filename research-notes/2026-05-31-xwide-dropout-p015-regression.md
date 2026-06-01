# 2026-05-31 xwide U-Net stronger bottleneck dropout regression

## Hypothesis

Increasing bottleneck-only `Dropout2d` in the current best extra-wide base-64 U-Net with Line Target auxiliary weight 0.10 from `p=0.10` to `p=0.15` tested whether slightly stronger regularization would preserve the useful recall gain from dropout while improving calibration or reducing late-epoch overconfidence. The expected signal was best-validation Dice within noise of `run_20260530_180658_0af8a8`, precision no worse than the attention-gated variant, and recall still above the plain xwide comparison.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_xwide_unet_line_aux_w010_dropout_p015`
- Candidate Experiment path: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p015`
- Relevant Experiment Proposal: `candidates/single_frame_xwide_unet_line_aux_w010_dropout_p015/PROPOSAL.md`
- Primary Comparison Target: `run_20260530_180658_0af8a8` / `single_frame_xwide_unet_line_aux_w010_dropout`
- Secondary context: `run_20260531_041919_8dbd72` / `single_frame_xwide_unet_line_aux_w010_attention_dropout` and `run_20260530_134005_6f20b1` / `single_frame_xwide_unet_line_auxiliary_w010`
- Contract choices: Single-Frame RGB Input, mask logits output, Line Target auxiliary output with `weighted_bce` weight 0.10, deterministic shuffle Sampling Policy, no augmentation, `bce_dice`, AdamW, learning rate 0.001, requested/effective batch size 8, max epochs 30.

## Run(s)

- Run ID: `run_20260531_160756_9f5a12`
- Dataset mode/subset: GVCCS Working Validation Split
- Harness/backend notes: completed on CUDA with no Resource Failure retry; best-validation model artifact persisted at `outputs/models/best_epoch_model.pt`; model summary reports 7,785,794 parameters, unchanged from the p=0.10 dropout comparison and below the 10,000,000 parameter budget.

## Key metrics

Best-validation metrics, selected by max `val/dice`:

| Run | Candidate Experiment | Best epoch | val/dice | val/iou | val/precision | val/recall | val/loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `run_20260530_134005_6f20b1` | `single_frame_xwide_unet_line_auxiliary_w010` | 28 | 0.832669 | 0.713310 | 0.862413 | 0.804908 | 0.510368 |
| `run_20260530_180658_0af8a8` | `single_frame_xwide_unet_line_aux_w010_dropout` | 29 | 0.833852 | 0.715049 | 0.856300 | 0.812552 | 0.510518 |
| `run_20260531_041919_8dbd72` | `single_frame_xwide_unet_line_aux_w010_attention_dropout` | 28 | 0.831409 | 0.711463 | 0.840498 | 0.822515 | 0.510823 |
| `run_20260531_160756_9f5a12` | `single_frame_xwide_unet_line_aux_w010_dropout_p015` | 28 | 0.831613 | 0.711762 | 0.844943 | 0.818698 | 0.511055 |

The stronger-dropout variant misses its success criterion by regressing best-validation Dice by about `-0.00224` relative to the p=0.10 dropout comparison target (`0.831613` vs `0.833852`, below the proposed `>=0.83235` guardrail). It lands near the attention-gated variant but with a less extreme precision/recall tradeoff: precision is slightly higher than attention dropout (`+0.00445`) while recall is lower (`-0.00382`). Compared with plain xwide, it improves recall (`+0.01379`) but spends too much precision (`-0.01747`) to improve Dice.

Final completed-epoch metrics for `run_20260531_160756_9f5a12` were `val/dice` 0.828032, `val/iou` 0.706531, `val/precision` 0.832425, `val/recall` 0.823685, and `val/loss` 0.515490. The best-to-final Dice gap is about `0.00358`, much smaller than the p=0.10 dropout gap, so stronger dropout improved late-epoch stability but lowered the best reachable validation Dice.

## Qualitative observations

The saved first-N prediction samples are mixed relative to the p=0.10 dropout comparison:

- `val/000000`: Dice 0.8428, IoU 0.7283, close to the p=0.10 sample 000 Dice 0.8462.
- `val/000001`: Dice 0.8028, IoU 0.6706, clearly weaker than the p=0.10 sample 001 Dice 0.8613.

These first-N samples are not a full-validation diagnostic, but they match the aggregate result: stronger dropout did not catastrophically fail, but it suppressed the best p=0.10 model's quality enough to lose Dice. The final-epoch recall is high, yet the associated precision drop suggests that simply increasing bottleneck dropout shifts the model toward broader or less selective masks rather than solving the remaining calibration issue.

![p015 dropout sample 000 overlay](../runs/run_20260531_160756_9f5a12/outputs/prediction_samples/sample_000_overlay.png)

![p015 dropout sample 001 overlay](../runs/run_20260531_160756_9f5a12/outputs/prediction_samples/sample_001_overlay.png)

![p015 dropout sample 001 heatmap](../runs/run_20260531_160756_9f5a12/outputs/prediction_samples/sample_001_probability_heatmap.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-xwide-dropout-p015-sample-000-overlay
    source_run_id: run_20260531_160756_9f5a12
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: First saved validation overlay for the p=0.15 dropout variant; sample Dice 0.8428 shows this stronger regularizer can still produce a good mask on one saved positive example.
  - figure_id: fig-xwide-dropout-p015-sample-001-overlay
    source_run_id: run_20260531_160756_9f5a12
    source_artifact_path: outputs/prediction_samples/sample_001_overlay.png
    reason: Second saved validation overlay for the p=0.15 dropout variant; sample Dice 0.8028 illustrates the qualitative weakening relative to the p=0.10 dropout comparison sample.
  - figure_id: fig-xwide-dropout-p015-sample-001-heatmap
    source_run_id: run_20260531_160756_9f5a12
    source_artifact_path: outputs/prediction_samples/sample_001_probability_heatmap.png
    reason: Probability heatmap for the weaker saved sample, useful for checking whether stronger bottleneck dropout softened or broadened the predicted contrail region.
```

## Decision

Do not promote `run_20260531_160756_9f5a12`. The run completed cleanly and stayed within the Candidate Experiment Contract, but the research hypothesis is not supported: `p=0.15` bottleneck dropout sacrifices enough precision and best-validation Dice to underperform `p=0.10`.

Keep `run_20260530_180658_0af8a8` / `single_frame_xwide_unet_line_aux_w010_dropout` as the current best in-contract base. The recent architecture-only variants now show a plateau around the same precision/recall frontier: attention and stronger dropout can buy recall or final-epoch stability, but neither improves the best Result.

## Next proposed change

Pause this local bottleneck-dropout sweep; do not try still higher dropout rates without a new reason. The most useful next step is a bounded Post-Run Evaluation or human-gated capability slice that addresses threshold selection, scheduler/early stopping, Boundary Target, or additional loss options. If another in-contract Candidate Experiment is required before capability expansion, it should reduce model complexity or regularization around the p=0.10 base with a strict precision guardrail rather than continuing to trade precision for recall.
