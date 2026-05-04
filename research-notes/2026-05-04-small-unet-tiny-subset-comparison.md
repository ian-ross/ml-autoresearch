# 2026-05-04 small U-Net tiny subset comparison

## Hypothesis

A small standard U-Net-style Model Architecture with encoder/decoder structure and skip connections should be a more credible single-frame segmentation baseline than the 233-parameter Harness fixture model, while staying within the current Candidate Experiment Contract.

For this tiny Human-Guided Research Iteration run, the main expectation was not a reliable validation score. The useful question was whether a more realistic in-contract Candidate Experiment could pass the Harness, run on the same tiny GVCCS subset, and produce artifacts suitable for comparison against the initial baseline.

## Candidate Experiment(s)

- Baseline Candidate Experiment path: `tests/fixtures/candidates/single_frame_unet_baseline`
- Variation Candidate Experiment path: `candidates/single_frame_small_unet`
- Relevant prior Research Note: `research-notes/2026-05-04-baseline-gvccs-tiny-subset.md`

## Run(s)

- Baseline Run ID: `run_20260504_192239_b2ebd1`
- Variation Run ID: `run_20260504_202102_9b2093`
- Dataset mode/subset: GVCCS Dataset via `--data-root`; tiny subset with `--max-samples 8`, producing 6 train samples and 2 validation samples.
- Harness/backend notes: Docker backend using `ml-autoresearch-runner:local`; rootless Docker ownership mode; GPU disabled by default.
- Contract: both Candidate Experiments used Single-Frame RGB Input, mask-logits output, `bce_dice`, AdamW, batch size 2, and 1 epoch.

## Key metrics

| Run | Candidate Experiment | Parameters | val/dice | val/iou | val/precision | val/recall | val/loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `run_20260504_192239_b2ebd1` | `tests/fixtures/candidates/single_frame_unet_baseline` | 233 | 0.0092345 | 0.0046387 | 0.0046387 | 1.0 | 1.7275207 |
| `run_20260504_202102_9b2093` | `candidates/single_frame_small_unet` | 488001 | 0.0000000006579 | 0.0000000006579 | 1.0 | 0.0000000006579 | 1.5889179 |

## Qualitative observations

The variation is architecturally more realistic than the baseline fixture: `outputs/model_summary.json` reports 488,001 parameters and the model has a conventional U-Net-style encoder/decoder with skip connections. It is still comfortably inside the 10M parameter smoke-test budget.

The validation behavior changed, but not in a way that supports a model-quality conclusion:

- The fixture baseline effectively predicted the positive class too broadly: recall was 1.0, precision was near the positive-pixel fraction, and Dice/IoU were near zero.
- The small U-Net variation swung the opposite way under this one-epoch tiny-subset run: precision was 1.0, recall was effectively zero, and Dice/IoU were effectively zero.

Because both Runs used only 6 training samples and 2 validation samples for one epoch, these metrics should not be interpreted as evidence that the U-Net architecture is worse than the fixture. They are evidence that the tiny setup is useful for Harness comparison and artifact generation, but too small to judge segmentation performance.

The artifacts were sufficient for the intended comparison. Metrics, model summaries, logs, and prediction sample metadata all made the failure mode visible and preserved enough context to choose the next step.

## Decision

Treat the comparison as **inconclusive for model quality** and **successful for Research Loop mechanics**.

The small U-Net did not improve tiny-subset Dice/IoU in this specific one-epoch Run. However, it is a more appropriate in-contract baseline architecture than the fixture model, and the comparison confirms that the Harness can run and compare independently authored Candidate Experiments without expanding the Candidate Experiment Contract.

Do not expand the contract yet. The current limitations are more likely due to the deliberately tiny training budget than to missing contract features such as temporal inputs, auxiliary heads, pretrained weights, or alternative losses.

## Next proposed change

Continue with the small U-Net-style Candidate Experiment, but run a more informative baseline before changing the contract:

- increase `max_samples` beyond 8;
- consider more than 1 epoch if wall-clock budget allows;
- keep Single-Frame RGB Input, mask-logits output, `bce_dice`, and AdamW;
- inspect whether prediction samples show any movement toward localized contrail masks.

If the small U-Net remains degenerate after a modestly larger run, the next Research Note should decide whether to adjust Harness-owned training budget/defaults, class-imbalance handling within the existing `bce_dice` path, or Candidate Experiment architecture details before considering broader contract expansion.
