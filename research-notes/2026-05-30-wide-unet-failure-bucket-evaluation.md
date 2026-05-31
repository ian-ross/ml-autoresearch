# 2026-05-30 wide U-Net failure-bucket evaluation

## Hypothesis

The Post-Run Evaluation for `run_20260530_101019_def893` tested whether the new best wide base-48 Line Target auxiliary U-Net's aggregate Dice improvement hid unacceptable recall-side failures. Because the Run improved Dice while shifting from the base-32 model toward higher precision and slightly lower recall, the diagnostic question was whether false-negative-heavy and missed-positive-mask cases became severe enough to block further exploration from this family.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_wide_unet_line_auxiliary_w010`
- Candidate Experiment path: `candidates/single_frame_wide_unet_line_auxiliary_w010`
- Relevant Experiment Proposal: `candidates/single_frame_wide_unet_line_auxiliary_w010/PROPOSAL.md`
- Primary comparison context: `run_20260530_073538_f7c8b7` / `single_frame_large_unet_line_auxiliary_w010`

## Run(s)

- Target Run ID: `run_20260530_101019_def893`
- Evaluation Request: `evaluation-requests/eval-2026-05-30-wide-unet-failure-buckets.yaml`
- Evaluation ID: `eval_eval_2026_05_30_wide_unet_failure_buckets`
- Dataset mode/subset: GVCCS Working Validation Split, `val`, 3889 samples
- Harness/backend notes: request-gated `failure_bucket_review` Post-Run Evaluation using the persisted best-epoch model at `outputs/models/best_epoch_model.pt`, primary threshold 0.5, threshold sweep 0.05--0.95, artifact budget 24.

## Key metrics

Whole-validation aggregate metrics at threshold 0.5:

| Run / Evaluation | val/dice | val/iou | val/precision | val/recall | Sample count |
| --- | ---: | ---: | ---: | ---: | ---: |
| `run_20260530_073538_f7c8b7` / `eval_eval_2026_05_30_large_unet_failure_buckets` | 0.815793 | 0.688895 | 0.814528 | 0.817063 | 3889 |
| `run_20260530_101019_def893` / `eval_eval_2026_05_30_wide_unet_failure_buckets` | 0.829163 | 0.708180 | 0.850680 | 0.808707 | 3889 |

The evaluation confirms the Run-level result: base-48 remains better than base-32 on whole-validation Dice and IoU, with substantially higher precision and modestly lower recall. The threshold sweep found the best Dice at threshold 0.35 (`0.829920`), but the gain over threshold 0.5 (`0.829163`) is small; threshold 0.5 remains a reasonable default unless calibration becomes an explicit Harness-owned policy question.

Failure-bucket counts at threshold 0.5 also do not show a recall collapse relative to the base-32 comparison:

| Evaluation | Empty-mask samples with false positives | Empty-mask false-positive pixels | Missed-positive samples with zero true positives | Positive samples with recall < 0.25 |
| --- | ---: | ---: | ---: | ---: |
| `eval_eval_2026_05_30_large_unet_failure_buckets` | 263 / 1004 | 3626 | 121 / 2885 | 188 / 2885 |
| `eval_eval_2026_05_30_wide_unet_failure_buckets` | 202 / 1004 | 1531 | 111 / 2885 | 183 / 2885 |

Despite the aggregate recall decrease, the wide model has fewer empty-mask false-positive samples, fewer empty-mask false-positive pixels, fewer complete missed-positive samples, and slightly fewer very-low-recall positive samples than the base-32 comparison in this evaluation.

## Qualitative observations

The selected diagnostic samples include both real failure modes and strong successes:

- `sample_000_overlay.png` (`val/003222`) is an empty-mask false positive with 73 predicted positive pixels. This is one of the worst-Dice samples, but the false-positive area is bounded rather than a large diffuse mask.
- `sample_002_overlay.png` (`val/000059`) is a missed-positive-mask case with 62 positive pixels and no predicted positives. This supports continued attention to faint or small contrails, but the aggregate count of complete misses is not worse than the base-32 comparison.
- `sample_012_overlay.png` (one of the best-by-Dice diagnostic samples) demonstrates that the same model can produce near-perfect masks on easier positives, so the failure mode appears sample-dependent rather than a global inability to segment contrails.

The diagnostic therefore supports the wide model as a genuine current best rather than a brittle aggregate-only gain. The remaining errors are concentrated in small positives and empty-mask false positives, which may be better addressed by capacity/regularization or future loss/scheduler capability slices than by more augmentation-preset variants, given recent augmentation regressions.

## Research Figures

The following existing Harness artifacts are referenced for provenance rather than copied into this note.

```research-figures
figures:
  - figure_id: fig-wide-eval-empty-mask-fp-overlay
    source_evaluation_id: eval_eval_2026_05_30_wide_unet_failure_buckets
    source_artifact_path: diagnostic_samples/sample_000_overlay.png
    reason: Empty-mask false-positive diagnostic sample (`val/003222`) showing a bounded but real false-positive failure selected by the failure-bucket review.
  - figure_id: fig-wide-eval-missed-positive-overlay
    source_evaluation_id: eval_eval_2026_05_30_wide_unet_failure_buckets
    source_artifact_path: diagnostic_samples/sample_002_overlay.png
    reason: Missed-positive-mask diagnostic sample (`val/000059`) illustrating the recall-side failure mode that motivated the evaluation.
  - figure_id: fig-wide-eval-best-positive-overlay
    source_evaluation_id: eval_eval_2026_05_30_wide_unet_failure_buckets
    source_artifact_path: diagnostic_samples/sample_012_overlay.png
    reason: Best-by-Dice diagnostic sample demonstrating that the wide model can produce high-quality masks on easier positive examples.
```

## Decision

Keep `run_20260530_101019_def893` as the current best Result. The bounded Post-Run Evaluation does not reveal a blocking recall-side regression; it instead strengthens the case that base-48 capacity scaling improved the family while reducing empty-mask false-positive burden relative to base-32.

## Next proposed change

Continue from the base-48 Line Target auxiliary family. A cautious next Experiment Proposal could test one further in-contract capacity step only if it stays safely under the 10M parameter budget and uses a conservative batch size, or test an in-contract precision/recall refinement that preserves the base-48 architecture. Avoid immediate augmentation-preset follow-ups unless a new hypothesis explains why previous photometric/geometric/combined augmentation regressions should not recur.
