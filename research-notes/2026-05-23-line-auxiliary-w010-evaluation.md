# 2026-05-23 lower-weight Line Target auxiliary evaluation follow-up

## Hypothesis

The lower-weight Line Target auxiliary Small U-Net (`single_frame_small_unet_line_auxiliary_w010`) remains the current best in-contract GVCCS Result. The follow-up diagnostic question was whether its remaining validation errors suggest a threshold-only change, another Line Target auxiliary-weight tune, an approved augmentation policy, or a different contract-supported variation.

## Candidate Experiment(s)

- Candidate Experiment path or ID: `candidates/single_frame_small_unet_line_auxiliary_w010`
- Relevant Experiment Proposal: candidate-local `PROPOSAL.md` in the submitted Candidate Experiment source.

## Run(s) and Post-Run Evaluations

- Run ID: `run_20260510_165335_b458c3`
- Dataset mode/subset: GVCCS Working Validation Split
- Harness/backend notes: completed Run with best-validation checkpoint artifact at `outputs/models/best_epoch_model.pt`.
- Post-Run Evaluation IDs:
  - `eval_20260511_042848_cb206d`: Whole-Validation Failure Analysis with aggregate metrics, threshold sweep, per-sample metrics, and diagnostic sample figures.
  - `eval_eval_2026_05_22_run_20260510_failure_buckets`: request-gated `failure_bucket_review`; completed as a linkage/summary artifact for the bounded diagnostic question, but its `summary.json` states that mode-specific metric computation can deepen behind this request gate rather than adding new bucket metrics beyond the existing Whole-Validation Failure Analysis artifacts.

## Key metrics

| Source | threshold | val/dice | val/iou | val/precision | val/recall | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Run best metrics | default | 0.7843769771696992 | 0.6452468918723383 | 0.7717854808163968 | 0.7973861422181288 | Best epoch 30 selected by `val/dice`. |
| Whole-validation aggregate | 0.50 | 0.7843861272606625 | 0.6452592758694735 | 0.771780207903964 | 0.7974106835633932 | Full validation evaluation over 3889 samples. |
| Threshold sweep best | 0.65 | 0.7854235562487357 | 0.6466645720733188 | 0.7911654211396292 | 0.7797644337621534 | Small Dice gain from higher threshold, trading recall for precision. |

The threshold sweep shows only a very small Dice improvement from moving the binary threshold from 0.50 to 0.65 (+0.00104 Dice on the whole validation evaluation). This is useful operationally but not enough to justify treating threshold tuning as the main next architecture hypothesis.

## Qualitative observations

The failure buckets show two distinct residual error modes:

- Missed-positive/worst-Dice samples exist where small positive masks are entirely missed at threshold 0.50 (`val/000318`, `val/000319`). These are false-negative-dominated but have only 402 and 303 positive pixels, respectively, so they may be thin/faint cases or ambiguous positives.
- Empty-mask false positives remain common but small in pixel rate: at threshold 0.50, 422 of 1004 empty-mask samples have at least one false-positive pixel, with 6354 total false-positive pixels across empty-mask samples. Raising the threshold reduces this monotonically, but the best Dice threshold still leaves 390 empty-mask samples with false positives.
- Larger positive-mask errors include both false-positive-heavy (`val/001489`, 1402 FP pixels) and false-negative-heavy (`val/002494`, 1083 FN pixels; `val/002501`, 1108 FN pixels) examples. This mixed error profile argues against a pure threshold response: higher thresholds improve precision but worsen recall, and the model already misses some positive masks entirely.

Example diagnostic overlays:

![Missed positive mask overlay](../runs/run_20260510_165335_b458c3/outputs/evaluations/eval_20260511_042848_cb206d/diagnostic_samples/sample_000_overlay.png)

![False-positive-heavy overlay](../runs/run_20260510_165335_b458c3/outputs/evaluations/eval_20260511_042848_cb206d/diagnostic_samples/sample_004_overlay.png)

![False-negative-heavy overlay](../runs/run_20260510_165335_b458c3/outputs/evaluations/eval_20260511_042848_cb206d/diagnostic_samples/sample_005_overlay.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-missed-positive-000318
    source_evaluation_id: eval_20260511_042848_cb206d
    source_artifact_path: diagnostic_samples/sample_000_overlay.png
    reason: Shows a worst-Dice missed-positive case where the model predicts no positive pixels for a small positive mask at threshold 0.50.
  - figure_id: fig-false-positive-heavy-001489
    source_evaluation_id: eval_20260511_042848_cb206d
    source_artifact_path: diagnostic_samples/sample_004_overlay.png
    reason: Shows a false-positive-heavy positive-mask case, demonstrating that precision errors remain even in non-empty images.
  - figure_id: fig-false-negative-heavy-002494
    source_evaluation_id: eval_20260511_042848_cb206d
    source_artifact_path: diagnostic_samples/sample_005_overlay.png
    reason: Shows a false-negative-heavy case, supporting the conclusion that simply raising threshold would worsen an existing residual error mode.
```

## Decision

Keep `single_frame_small_unet_line_auxiliary_w010` as the current best Result. Do not spend the next Candidate Experiment on threshold tuning alone: the best threshold only slightly improves Dice and trades away recall while missed-positive cases remain. The request-gated failure-bucket evaluation did not add new mode-specific metrics beyond confirming artifact linkage, so use the existing Whole-Validation Failure Analysis as the operative diagnostic evidence.

## Next proposed change

Next Experiment Proposal should stay within the Candidate Experiment Contract and test an approved augmentation/data-policy variation, if available in the manifest allowlist, against `run_20260510_165335_b458c3` as the Comparison Target. The rationale is that mixed false-positive and false-negative residuals look more like robustness/generalization errors than a single calibration or Line Target weight problem. If no approved augmentation policy is available, propose the next small in-contract architecture/loss variant rather than lowering/raising the Line Target auxiliary weight again immediately.
