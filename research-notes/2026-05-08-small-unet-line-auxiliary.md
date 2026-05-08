# 2026-05-08 small U-Net line auxiliary target

## Hypothesis

Adding a Harness-derived per-pixel Line Target auxiliary loss to the current shuffled small U-Net baseline should encourage thinner contrail-structure sensitivity and improve recall on faint or narrow contrails, while preserving primary Contrail Mask validation Dice as the comparison metric.

The risk was that the auxiliary `line_logits` head and weighted BCE loss might make the model over-predict line-like structures, increasing false-positive pixels or visibly thickening predicted masks.

## Candidate Experiment(s)

- Candidate Experiment path: `candidates/single_frame_small_unet_line_auxiliary`
- Candidate description: `candidates/single_frame_small_unet_line_auxiliary/README.md`
- Architecture lineage: `candidates/single_frame_small_unet_realistic_training_shuffled`
- Contract choices: Single-Frame RGB Input, primary `mask_logits`, auxiliary `line_logits`, primary `bce_dice`, auxiliary `weighted_bce`, auxiliary weight `0.25`, AdamW, learning rate `0.001`, batch size `16`, max epochs `30`
- Data Policy: `data.sampling_policy: deterministic_shuffle`
- Comparison baseline: `run_20260506_045020_84aac5`
- Baseline Post-Run Evaluation: `eval_20260507_104724_a96db5`

## Run(s)

- Main Run ID: `run_20260507_193004_0b4688`
- Post-Run Evaluation ID: `eval_20260508_100735_6fea74`
- Evaluation path: `runs/run_20260507_193004_0b4688/outputs/evaluations/eval_20260508_100735_6fea74/`
- Dataset mode/subset: GVCCS Dataset via Docker-mounted `/data`; full Harness train/validation split; evaluation validation split has `3889` samples.
- Harness/backend notes: Docker backend using `ml-autoresearch-runner:local`; GPU enabled by Harness configuration; CUDA used for training. An initial Docker smoke-test issue was fixed before this successful Run: in-container smoke testing now derives `output_spec` from the Run-scoped `resolved_manifest.yaml`, so auxiliary outputs are required during smoke testing.
- Prediction Sample Policy: `adjacent_and_scattered`
- Max prediction samples: `16`
- Artifact path: `runs/run_20260507_193004_0b4688/outputs/`

## Key metrics

Training selected epoch 29 as best by validation Dice. Final epoch 30 was close but had slightly lower Dice and higher recall.

| Run | Candidate Experiment | Epoch view | val/dice | val/iou | val/precision | val/recall | val/loss |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `run_20260506_045020_84aac5` | `single_frame_small_unet_realistic_training_shuffled` | best/final, epoch 30 | 0.7784179 | 0.6372211 | 0.8036883 | 0.7546883 | 0.5701003 |
| `run_20260507_193004_0b4688` | `single_frame_small_unet_line_auxiliary` | best, epoch 29 | 0.7826883 | 0.6429646 | 0.8040935 | 0.7623933 | 0.5706279 |
| `run_20260507_193004_0b4688` | `single_frame_small_unet_line_auxiliary` | final, epoch 30 | 0.7812116 | 0.6409739 | 0.7685671 | 0.7942791 | 0.5666773 |

Auxiliary-loss fields for the best epoch:

- `train/mask_loss`: `0.5172614`
- `train/aux/line_loss`: `0.0292385`
- `train/loss`: `0.5464999`
- `val/aux/line_loss`: `0.0367005`
- `val/total_loss`: `0.6073284`

Whole-Validation Failure Analysis at threshold `0.5`:

| Evaluation | threshold | sample_count | dice | iou | precision | recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline `eval_20260507_104724_a96db5` | 0.50 | 3889 | 0.7784263 | 0.6372324 | 0.8036859 | 0.7547062 |
| line aux `eval_20260508_100735_6fea74` | 0.50 | 3889 | 0.7826974 | 0.6429768 | 0.8040921 | 0.7624117 |

Pixel-count comparison at threshold `0.5`:

| Count | Baseline | Line aux | Delta |
| --- | ---: | ---: | ---: |
| positive pixels | 1,955,883 | 1,955,883 | 0 |
| predicted positive pixels | 1,836,684 | 1,854,499 | +17,815 |
| false-positive pixels | 360,567 | 363,311 | +2,744 |
| false-negative pixels | 479,766 | 464,695 | -15,071 |
| empty-mask samples with any FP | 506 | 178 | -328 |
| empty-mask FP pixels | 6,091 | 2,252 | -3,839 |

Threshold-sweep comparison around the operating region:

| threshold | baseline dice | line aux dice | baseline precision | line aux precision | baseline recall | line aux recall | baseline empty FP px | line aux empty FP px |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.30 | 0.7788 | 0.7829 | 0.7797 | 0.7789 | 0.7780 | 0.7869 | 7,410 | 2,788 |
| 0.35 | 0.7791 | 0.7832 | 0.7865 | 0.7860 | 0.7717 | 0.7805 | 7,043 | 2,630 |
| 0.40 | 0.7791 | 0.7832 | 0.7927 | 0.7923 | 0.7660 | 0.7743 | 6,704 | 2,492 |
| 0.45 | 0.7788 | 0.7831 | 0.7982 | 0.7983 | 0.7603 | 0.7684 | 6,369 | 2,358 |
| 0.50 | 0.7784 | 0.7827 | 0.8037 | 0.8041 | 0.7547 | 0.7624 | 6,091 | 2,252 |
| 0.55 | 0.7779 | 0.7821 | 0.8090 | 0.8098 | 0.7490 | 0.7562 | 5,814 | 2,143 |
| 0.60 | 0.7771 | 0.7811 | 0.8144 | 0.8156 | 0.7431 | 0.7495 | 5,546 | 2,033 |

Best threshold in the evaluation sweep:

- Baseline: Dice `0.7790809` at threshold `0.40`
- Line aux: Dice `0.7832294` at threshold `0.35`

## Qualitative observations

Visual inspection of prediction and diagnostic overlay plots gave an initial impression that the line-auxiliary candidate may produce more false-positive pixels or slightly more permissive masks. The aggregate metrics partially support that impression: total false-positive pixels increased by `2,744` pixels across the full validation split, and predicted-positive pixels increased by `17,815`.

However, the increase is small relative to the total false-positive count, precision is essentially unchanged, and false-negative pixels dropped by `15,071`. The model also substantially reduced false positives on empty-mask validation samples: empty samples with any false-positive pixels fell from `506` to `178`, and empty-sample false-positive pixels fell from `6,091` to `2,252`.

This suggests the visual impression is probably detecting a real local behavior on positive-mask images: predictions may be slightly thicker or more inclusive around true contrail structures. That behavior does not currently look like broad hallucination on empty skies. It appears to trade a small number of additional false-positive pixels near positive examples for a larger reduction in false-negative pixels and a modest Dice/IoU improvement.

The threshold sweep is also favorable. The line-auxiliary candidate beats the baseline across the inspected threshold range from `0.30` to `0.60`, and its best swept threshold is a little lower (`0.35`), consistent with a slightly more recall-friendly model.

## Decision

Treat `single_frame_small_unet_line_auxiliary` as a promising challenger and modest improvement over the current shuffled small U-Net baseline, but do not over-interpret it as a decisive baseline replacement from one Run.

The Result supports the value of Harness-owned per-pixel Auxiliary Targets: primary validation Dice improved from about `0.7784` to `0.7827` in whole-validation evaluation, recall improved by about `0.0077`, and precision did not materially degrade. The auxiliary Run also improved empty-mask false-positive diagnostics.

The main caution is qualitative mask shape: if the line target encourages thicker or more permissive predictions around positives, the next Research Loop step should tune that tradeoff rather than simply increasing auxiliary influence.

## Next proposed change

Run a small auxiliary-weight sweep using the same candidate lineage and training setup:

1. `line_logits` auxiliary weight `0.10`
2. `line_logits` auxiliary weight `0.15`
3. optionally repeat `0.25` with a second seed/run if the Harness exposes seed control or if stochastic variation needs estimating

Primary comparison should remain validation Dice over the Contrail Mask, with additional attention to:

- total false-positive and false-negative pixels;
- empty-mask false-positive pixels and empty-mask samples with any FP;
- qualitative overlays on positive samples where the `0.25` model appears too thick or permissive;
- whether the best threshold stays below `0.5` or returns closer to the baseline operating region.
