# 2026-05-06 small U-Net shuffled realistic training

## Hypothesis

The small U-Net Candidate Experiment should train much more reliably when the Harness uses reproducible shuffled training order and improved qualitative prediction sample diagnostics. The expected effect was higher validation Dice/IoU than the prior sequential realistic run, plus prediction samples that show whether the model is actually localizing Contrail Masks rather than producing a degenerate all-positive or all-negative output.

## Candidate Experiment(s)

- Candidate Experiment path: `candidates/single_frame_small_unet_realistic_training_shuffled`
- Architecture lineage: same base-16 Small U-Net as `candidates/single_frame_small_unet_realistic_training`
- Contract choices: Single-Frame RGB Input, `mask_logits` output, `bce_dice`, AdamW, learning rate `0.001`, batch size `16`, max epochs `30`
- Data Policy: `data.sampling_policy: deterministic_shuffle`
- Prior Research Notes:
  - `research-notes/2026-05-04-small-unet-tiny-subset-comparison.md`
  - `research-notes/2026-05-05-small-unet-realistic-training.md`

## Run(s)

- Main Run ID: `run_20260506_045020_84aac5`
- Earlier shuffled Run for context: `run_20260505_172954_febf68`
- Dataset mode/subset: GVCCS Dataset via Docker-mounted `/data`; full Harness train/validation split.
- Harness/backend notes: Docker backend using `ml-autoresearch-runner:local`; GPU enabled by Harness configuration; CUDA used for training.
- Prediction Sample Policy: `adjacent_and_scattered`
- Max prediction samples: `16` for the main Run
- Artifact path: `runs/run_20260506_045020_84aac5/outputs/`

## Key metrics

| Run | Candidate Experiment | Epoch view | val/dice | val/iou | val/precision | val/recall | val/loss |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `run_20260505_092224_fa11f8` | `single_frame_small_unet_realistic_training` | prior sequential best, epoch 22 | 0.0932767 | 0.0489199 | 0.4297077 | 0.0523165 | 1.0936881 |
| `run_20260505_172954_febf68` | `single_frame_small_unet_realistic_training_shuffled` | best/final, epoch 30 | 0.7848568 | 0.6458966 | 0.8108725 | 0.7604586 | 0.5644617 |
| `run_20260506_045020_84aac5` | `single_frame_small_unet_realistic_training_shuffled` | best/final, epoch 30 | 0.7784179 | 0.6372211 | 0.8036883 | 0.7546883 | 0.5701003 |

For `run_20260506_045020_84aac5`, `best_metrics.json` and `final_metrics.json` agree: the best-validation epoch was the final epoch 30, selected by max `val/dice`. The best-epoch model artifact is `outputs/models/best_epoch_model.pt`.

## Qualitative observations

The main Run produced 16 prediction samples with adjacent-and-scattered selection and probability heatmaps. This is much more useful than the earlier first-N two-sample artifact set.

Sample-level metrics show credible localization on many validation examples:

- adjacent-window samples: 10 samples, mean Dice about `0.649`, range `0.402` to `0.835`;
- scattered-singleton samples: 6 samples, mixed behavior, including strong examples up to Dice `0.881` and several near-zero examples;
- all samples combined: median Dice about `0.710`, mean Dice about `0.554`.

Several adjacent samples look qualitatively good and have Dice in the `0.70`-`0.83` range, indicating that the model is detecting real contrail structure rather than merely exploiting foreground frequency. The low-Dice scattered samples are useful hard cases for the next Research Loop step: they may be true missed contrails, negative/near-empty masks, thresholding failures, or unusual scene conditions. The probability heatmaps should be inspected before deciding which category they fall into.

The training curve was also encouraging. Validation Dice rose from `0.371` at epoch 1 to `0.653` at epoch 6, `0.726` at epoch 11, and `0.778` by epoch 30, with precision and recall both ending near balanced values (`0.804` precision, `0.755` recall). Unlike the prior sequential realistic run, the final epoch did not collapse below an earlier best epoch.

## Decision

Promote `single_frame_small_unet_realistic_training_shuffled` to the current in-contract baseline for Ground-Camera Contrail Detection on GVCCS.

This Result is no longer just a Harness/training-process validation. It is a credible single-frame GVCCS segmentation baseline: best/final Dice is near `0.78`, IoU near `0.64`, and qualitative samples show meaningful mask localization. The large improvement over the prior sequential realistic run suggests that the new Data Policy and improved run setup materially changed training behavior.

Do not expand the Candidate Experiment Contract yet. The current envelope is sufficient to produce a useful baseline. The next step should compare targeted changes against this shuffled small U-Net baseline.

## Next proposed change

Before changing architecture, use this Run as the baseline and inspect the failure cases:

- review the near-zero scattered samples and their probability heatmaps;
- classify failures as false negatives, false positives on empty masks, thresholding issues, or ambiguous labels;
- consider a threshold-sweep/evaluation diagnostic if probability heatmaps show plausible below-threshold detections;
- rerun or extend with more prediction samples if qualitative coverage is still too sparse.

Candidate Experiment ideas to compare next:

1. same architecture with lower learning rate such as `0.0003`, if the goal is smoother late training;
2. modestly wider U-Net or light regularization change within the existing parameter budget;
3. Harness-owned class-imbalance or threshold diagnostics before adding temporal input or auxiliary heads.
