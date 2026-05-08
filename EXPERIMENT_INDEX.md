# Experiment Index

This index links Candidate Experiment descriptions to the Research Notes that interpret their Runs and Post-Run Evaluations.

**Maintenance rule:** update this file whenever a new Candidate Experiment is introduced or a new Research Note is written.

## Current baseline

- Current in-contract GVCCS baseline: [`candidates/single_frame_small_unet_realistic_training_shuffled`](candidates/single_frame_small_unet_realistic_training_shuffled/README.md)
- Most recent Research Note: [`research-notes/2026-05-08-small-unet-line-auxiliary.md`](research-notes/2026-05-08-small-unet-line-auxiliary.md)
- Baseline Run: `run_20260506_045020_84aac5`
- Latest baseline Post-Run Evaluation: `eval_20260507_104724_a96db5`
- Current challenger Run: `run_20260507_193004_0b4688`

## Candidate Experiments and notes

| Candidate Experiment | Description | Related Research Notes | Key Runs / Evaluations | Status |
| --- | --- | --- | --- | --- |
| `tests/fixtures/candidates/single_frame_unet_baseline` | Harness fixture candidate; no standalone README. | [`2026-05-04 baseline GVCCS tiny subset`](research-notes/2026-05-04-baseline-gvccs-tiny-subset.md); [`2026-05-04 small U-Net tiny subset comparison`](research-notes/2026-05-04-small-unet-tiny-subset-comparison.md) | `run_20260504_192239_b2ebd1` | Harness/debug baseline only; not a model-quality baseline. |
| `candidates/single_frame_small_unet` | [`README.md`](candidates/single_frame_small_unet/README.md) | [`2026-05-04 small U-Net tiny subset comparison`](research-notes/2026-05-04-small-unet-tiny-subset-comparison.md) | `run_20260504_202102_9b2093` | Tiny-subset architecture check; inconclusive for model quality. |
| `candidates/single_frame_small_unet_realistic_training` | [`README.md`](candidates/single_frame_small_unet_realistic_training/README.md) | [`2026-05-05 small U-Net realistic training`](research-notes/2026-05-05-small-unet-realistic-training.md) | `run_20260505_092224_fa11f8` | Weak realistic baseline; superseded by shuffled training. |
| `candidates/single_frame_small_unet_realistic_training_shuffled` | [`README.md`](candidates/single_frame_small_unet_realistic_training_shuffled/README.md) | [`2026-05-06 small U-Net shuffled realistic training`](research-notes/2026-05-06-small-unet-shuffled-realistic-training.md); [`2026-05-07 small U-Net post-run evaluation`](research-notes/2026-05-07-small-unet-post-run-evaluation.md) | `run_20260505_172954_febf68`; `run_20260506_045020_84aac5`; `eval_20260507_104724_a96db5` | Current in-contract GVCCS baseline. |
| `candidates/single_frame_small_unet_line_auxiliary` | [`README.md`](candidates/single_frame_small_unet_line_auxiliary/README.md) | [`2026-05-08 small U-Net line auxiliary target`](research-notes/2026-05-08-small-unet-line-auxiliary.md) | `run_20260507_193004_0b4688`; `eval_20260508_100735_6fea74`; compared to `run_20260506_045020_84aac5` / `eval_20260507_104724_a96db5` | Promising challenger; modest Dice/recall gain, tune auxiliary weight next. |
| `candidates/single_frame_small_unet_line_auxiliary_w010` | [`README.md`](candidates/single_frame_small_unet_line_auxiliary_w010/README.md) | Pending full GVCCS Research Note. | Pending GVCCS Run; compare to `run_20260506_045020_84aac5` and `run_20260507_193004_0b4688`. | Introduced; tests lower Line Target auxiliary weight `0.10`. |

## Chronological Research Notes

- [`2026-05-04 baseline GVCCS tiny subset`](research-notes/2026-05-04-baseline-gvccs-tiny-subset.md)
- [`2026-05-04 small U-Net tiny subset comparison`](research-notes/2026-05-04-small-unet-tiny-subset-comparison.md)
- [`2026-05-05 small U-Net realistic training`](research-notes/2026-05-05-small-unet-realistic-training.md)
- [`2026-05-06 small U-Net shuffled realistic training`](research-notes/2026-05-06-small-unet-shuffled-realistic-training.md)
- [`2026-05-07 small U-Net post-run evaluation`](research-notes/2026-05-07-small-unet-post-run-evaluation.md)
- [`2026-05-08 small U-Net line auxiliary target`](research-notes/2026-05-08-small-unet-line-auxiliary.md)
