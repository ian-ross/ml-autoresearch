# 2026-05-07 small U-Net post-run evaluation

## Hypothesis

Whole-Validation Failure Analysis for `run_20260506_045020_84aac5` should clarify whether the shuffled small U-Net baseline is genuinely useful across the full GVCCS validation split, and whether its default binary threshold of `0.5` is materially too conservative.

The expectation after inspecting the original prediction samples was that the model was already localizing Contrail Masks credibly, but that a threshold sweep and broader failure-bucket diagnostics would expose whether the next Research Loop step should focus on threshold selection, false positives, false negatives, or architecture/training changes.

## Candidate Experiment(s)

- Candidate Experiment path: `candidates/single_frame_small_unet_realistic_training_shuffled`
- Candidate description: `candidates/single_frame_small_unet_realistic_training_shuffled/README.md`
- Source Run candidate copy: `runs/run_20260506_045020_84aac5/candidate/README.md`
- Prior Research Note: `research-notes/2026-05-06-small-unet-shuffled-realistic-training.md`

## Run(s)

- Source Run ID: `run_20260506_045020_84aac5`
- Post-Run Evaluation ID: `eval_20260507_104724_a96db5`
- Evaluation path: `runs/run_20260506_045020_84aac5/outputs/evaluations/eval_20260507_104724_a96db5/`
- Evaluation mode: Whole-Validation Failure Analysis on the validation split.
- Source model artifact: `runs/run_20260506_045020_84aac5/outputs/models/best_epoch_model.pt`
- Dataset mode/subset: GVCCS Dataset validation split, `3889` samples.
- Harness/backend notes: evaluation was launched through Docker/rootless-container-root, then ran the in-container native Post-Run Evaluation operation against mounted Run artifacts and GVCCS data.

## Key metrics

| Evaluation | threshold | sample_count | dice | iou | precision | recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `eval_20260507_104724_a96db5` | 0.50 | 3889 | 0.7784263 | 0.6372324 | 0.8036859 | 0.7547062 |
| best threshold in sweep | 0.40 | 3889 | 0.7790809 | 0.6381102 | 0.7926636 | 0.7659558 |

Threshold-sweep comparison around the operating region:

| threshold | dice | precision | recall |
| ---: | ---: | ---: | ---: |
| 0.30 | 0.7788424 | 0.7797092 | 0.7779775 |
| 0.35 | 0.7790694 | 0.7865293 | 0.7717496 |
| 0.40 | 0.7790809 | 0.7926636 | 0.7659558 |
| 0.45 | 0.7787875 | 0.7982179 | 0.7602807 |
| 0.50 | 0.7784263 | 0.8036859 | 0.7547062 |
| 0.55 | 0.7778581 | 0.8090144 | 0.7490126 |

Additional validation-set counts:

- Positive-mask samples: `2885`
- Empty-mask samples: `1004`
- Missed positive masks at threshold `0.5`: `71 / 2885` (`2.5%`)
- Empty-mask samples with any false-positive pixels at threshold `0.5`: `506 / 1004`
- Total false-positive pixels on empty-mask samples at threshold `0.5`: `6091` across approximately `65.8M` empty-mask pixels, indicating many tiny specks rather than large hallucinated masks.

## Qualitative observations

The diagnostic samples look strong for a first serious single-frame detection model. The selected failure-bucket artifacts include the expected hard cases: tiny false-positive specks on empty masks, a small number of missed positive masks, and representative best/worst Dice examples. The qualitative impression is consistent with the aggregate metrics: the model is often detecting real contrail structure rather than exploiting a degenerate foreground/background prior.

The threshold sweep shows that `0.5` is slightly conservative, but only marginally. Moving from `0.5` to `0.4` improves Dice by about `0.00065`, trading a little precision for a little recall. This is too small to make threshold tuning the main research lever. If a downstream use case strongly prefers recall, an operating threshold around `0.35`-`0.4` is defensible, but `0.5` remains a reasonable default for ongoing research comparisons.

The empty-mask false-positive count initially looks high at the sample level, but the pixel count is very low relative to all empty-mask pixels. This suggests that the visible issue is mostly small speckle-like false positives rather than broad cloud/sky hallucination. That failure mode is worth tracking, but it does not undermine the baseline.

## Decision

Keep `single_frame_small_unet_realistic_training_shuffled` as the current in-contract GVCCS baseline.

Do not spend the next Human-Guided Research Iteration mainly on threshold selection. The threshold curve is flat near the optimum, and the current default threshold is already near the best swept Dice. The more valuable next step is a Candidate Experiment that targets either small false positives on empty masks or recall on faint/thin contrails while preserving the current baseline for comparison.

## Next proposed change

Compare a targeted Candidate Experiment against this baseline, with one of the following priorities:

1. reduce tiny false positives on empty masks, for example through light regularization or a loss/training adjustment inside the existing contract;
2. improve recall on faint or thin contrails, likely through auxiliary line/boundary targets if the Harness-owned contract path is ready;
3. test a modest model/training variation such as a lower AdamW learning rate (`0.0003`) only if the goal is to understand whether the current late-training balance can be improved without changing architecture.

For now, prefer a model/training improvement over threshold-only tuning.
