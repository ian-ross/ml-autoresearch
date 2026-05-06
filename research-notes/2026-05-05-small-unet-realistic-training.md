# 2026-05-05 small U-Net realistic training

## Hypothesis

The small U-Net Candidate Experiment should become meaningfully more informative when moved from the tiny 6-train/2-val smoke-sized subset to a realistic GVCCS training budget. With full-data training, 30 epochs, batch size 16, and GPU execution, the model should move away from the degenerate near-all-negative behavior seen in `run_20260504_202102_9b2093` while staying inside the existing Candidate Experiment Contract.

## Candidate Experiment(s)

- Candidate Experiment path: `candidates/single_frame_small_unet_realistic_training`
- Architecture lineage: reuses the base-16 Small U-Net from `candidates/single_frame_small_unet`
- Prior Research Note: `research-notes/2026-05-04-small-unet-tiny-subset-comparison.md`

## Run(s)

- Run ID: `run_20260505_092224_fa11f8`
- Dataset mode/subset: GVCCS Dataset via Docker-mounted `/data`; full Harness split reported as 15,558 train samples and 3,889 validation samples.
- Harness/backend notes: Docker backend using `ml-autoresearch-runner:local`; GPU enabled by Harness configuration; CUDA used for training.
- Contract: Single-Frame RGB Input, `mask_logits` output, `bce_dice`, AdamW, learning rate `0.001`, batch size `16`, max epochs `30`.
- Artifact paths: `runs/run_20260505_092224_fa11f8/outputs/`.

## Key metrics

| Run | Candidate Experiment | Epoch view | val/dice | val/iou | val/precision | val/recall | val/loss |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `run_20260505_092224_fa11f8` | `candidates/single_frame_small_unet_realistic_training` | best observed, epoch 22 | 0.0932767 | 0.0489199 | 0.4297077 | 0.0523165 | 1.0936881 |
| `run_20260505_092224_fa11f8` | `candidates/single_frame_small_unet_realistic_training` | final, epoch 30 | 0.0508167 | 0.0260708 | 0.7952412 | 0.0262470 | 1.1542968 |
| `run_20260504_202102_9b2093` | `candidates/single_frame_small_unet` | tiny-subset final, epoch 1 | ~0.0 | ~0.0 | 1.0 | ~0.0 | 1.5889179 |

## Qualitative observations

The realistic training budget did produce learning signal compared with the one-epoch tiny-subset run. Validation Dice rose from effectively zero to a best observed value of about `0.093` at epoch 22, with nonzero recall and nontrivial precision. This is still weak segmentation performance, but it is no longer just the tiny-run all-negative failure mode.

The final epoch is worse than the best observed epoch. Dice peaked around epoch 22 and then declined to `0.0508` by epoch 30 while precision increased and recall fell. This suggests the model became increasingly conservative late in training. Future decisions should compare by best-validation metrics, not only by `final_metrics.json`.

The saved first-N prediction samples were not representative enough for diagnosis: both sampled validation examples had per-sample Dice/IoU near zero in `outputs/prediction_samples/samples.json`, even though aggregate validation metrics show some positive detection elsewhere. This exposed a training-process issue: qualitative artifacts need more deliberate sample selection and probability visualization.

## Training-process improvements closed after this Run

This Run motivated several Harness/process improvements that should be used for the next Human-Guided Research Iteration:

- [#31 Add candidate-selectable Sampling Policy to Candidate Experiment Contract](https://github.com/ian-ross/ml-autoresearch/issues/31): enables reproducible `deterministic_shuffle` training order instead of relying only on sequential ordering.
- [#32 Add adjacent-and-scattered Prediction Sample Policy for GVCCS diagnostics](https://github.com/ian-ross/ml-autoresearch/issues/32): improves qualitative diagnostics beyond first-N validation samples.
- [#33 Add probability heatmaps to prediction sample artifacts](https://github.com/ian-ross/ml-autoresearch/issues/33): makes near-threshold or weak contrail responses inspectable instead of only binary masks.
- [#34 Report best-validation metrics separately from final metrics](https://github.com/ian-ross/ml-autoresearch/issues/34): prevents the epoch-22 result from being hidden by the weaker epoch-30 final metrics.
- [#35 Persist best-epoch model artifact for later evaluation](https://github.com/ian-ross/ml-autoresearch/issues/35): preserves the selected best-validation weights for follow-up evaluation and comparison.

The recorded Run predates these artifacts in its output tree, so its best epoch above was reconstructed from `metrics.jsonl` rather than read from `best_metrics.json`.

## Decision

Keep the small U-Net as the current in-contract baseline, but treat this Result as a weak baseline rather than a satisfactory model. The realistic training budget improved over the tiny subset and validated GPU/full-data Harness execution, but best Dice near `0.093` and low recall indicate that the model/training setup is still missing most contrail pixels.

Do not expand the Candidate Experiment Contract yet. The immediate bottleneck is observability and training-process discipline, now addressed by issues #31-#35, plus likely tuning within the existing envelope.

## Next proposed change

Rerun `candidates/single_frame_small_unet_realistic_training` after the process improvements, using:

- `data.sampling_policy: deterministic_shuffle`;
- `--prediction-sample-policy adjacent_and_scattered`;
- probability heatmap artifacts;
- best-validation metric reporting and best-epoch checkpoint persistence.

Then compare best-validation Dice and qualitative heatmaps before changing architecture. If the same conservative/low-recall behavior persists, try a lower AdamW learning rate such as `0.0003` or a Harness-owned class-imbalance adjustment while keeping Single-Frame RGB Input and mask-only output.
