# 2026-05-30 xwide U-Net failure-bucket evaluation

## Hypothesis

The extra-wide base-64 single-frame U-Net with Line Target auxiliary weight 0.10 was expected to extend the capacity-scaling gains from the base-48 wide model while retaining acceptable whole-validation failure behavior. The key diagnostic question was whether the small aggregate Dice gain came with a brittle high-precision / lower-recall tradeoff that worsens missed positives or false-negative-heavy examples.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_xwide_unet_line_auxiliary_w010`
- Candidate source: `/history/candidates/single_frame_xwide_unet_line_auxiliary_w010/`
- Relevant Experiment Proposal: `/history/candidates/single_frame_xwide_unet_line_auxiliary_w010/PROPOSAL.md`
- Comparison target: `single_frame_wide_unet_line_auxiliary_w010` / Run `run_20260530_101019_def893`

## Run(s)

- Target Run ID: `run_20260530_134005_6f20b1`
- Comparison Run ID: `run_20260530_101019_def893`
- Post-Run Evaluation ID: `eval_eval_2026_05_30_xwide_unet_failure_buckets`
- Evaluation Request: `eval_2026_05_30_xwide_unet_failure_buckets`
- Dataset/split: GVCCS Working Validation Split (`val`), 3,889 evaluated samples
- Harness/backend notes: Docker-backed completed Run, effective batch size 8, no Resource Failure retry

## Key metrics

| Run | Evaluation | threshold | val/dice | val/iou | val/precision | val/recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `run_20260530_101019_def893` wide base-48 | `eval_eval_2026_05_30_wide_unet_failure_buckets` | 0.50 | 0.829163 | 0.708180 | 0.850680 | 0.808707 |
| `run_20260530_134005_6f20b1` xwide base-64 | `eval_eval_2026_05_30_xwide_unet_failure_buckets` | 0.50 | 0.832670 | 0.713311 | 0.862413 | 0.804909 |
| `run_20260530_134005_6f20b1` xwide base-64 | threshold sweep best | 0.25 | 0.834485 | 0.715980 | 0.836755 | 0.832227 |

The xwide model improves full-validation Dice by +0.00351 over the wide comparison at the default threshold, but the default-threshold gain is precision-led: precision rises by +0.01173 while recall falls by -0.00380. The threshold sweep indicates the xwide model has useful threshold slack: at threshold 0.25 it improves both Dice and recall relative to its default operating point, while retaining precision close to the wide model.

## Qualitative observations

The xwide failure-bucket sample set contains the expected bounded buckets: `worst_by_dice`, `best_by_dice`, `false_positive_heavy`, `false_negative_heavy`, `empty_mask_false_positives`, and `missed_positive_masks`.

Compared with the wide model, xwide reduces the worst empty-mask false positives in the diagnostic artifacts: the largest sampled empty-mask false positive drops from 73 predicted positive pixels in wide (`val/003222`) to 52 in xwide (`val/003443`). This supports the precision improvement and suggests the extra width did not merely smear more contrail predictions onto negative skies.

The main concern is recall brittleness on small positive masks. Xwide still has four sampled `missed_positive_masks`, including `val/000592` with 100 positive pixels and zero predicted positives, plus `val/000059`, `val/000693`, and `val/000301`. The wide model also had four missed-positive diagnostic samples, but xwide introduces a different 100-pixel miss (`val/000592`) while retaining some known small-mask misses. The `false_negative_heavy` xwide artifacts also include substantial misses on larger masks, such as `val/002527` with 1,118 false-negative pixels.

![Xwide missed positive overlay](../runs/run_20260530_134005_6f20b1/outputs/evaluations/eval_eval_2026_05_30_xwide_unet_failure_buckets/diagnostic_samples/sample_000_overlay.png)

![Xwide empty-mask false positive overlay](../runs/run_20260530_134005_6f20b1/outputs/evaluations/eval_eval_2026_05_30_xwide_unet_failure_buckets/diagnostic_samples/sample_004_overlay.png)

![Wide empty-mask false positive overlay](../runs/run_20260530_101019_def893/outputs/evaluations/eval_eval_2026_05_30_wide_unet_failure_buckets/diagnostic_samples/sample_000_overlay.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-xwide-missed-positive-sample-000
    source_evaluation_id: eval_eval_2026_05_30_xwide_unet_failure_buckets
    source_artifact_path: diagnostic_samples/sample_000_overlay.png
    reason: Shows an xwide missed-positive failure where a 100-pixel positive mask at val/000592 is predicted as empty at threshold 0.5.
  - figure_id: fig-xwide-empty-mask-fp-sample-004
    source_evaluation_id: eval_eval_2026_05_30_xwide_unet_failure_buckets
    source_artifact_path: diagnostic_samples/sample_004_overlay.png
    reason: Shows the largest xwide sampled empty-mask false positive bucket case, useful for checking whether precision gains reduce negative-sky false positives.
  - figure_id: fig-wide-empty-mask-fp-sample-000
    source_evaluation_id: eval_eval_2026_05_30_wide_unet_failure_buckets
    source_artifact_path: diagnostic_samples/sample_000_overlay.png
    reason: Provides the wide base-48 comparison empty-mask false-positive artifact that had more predicted positive pixels than the xwide counterpart.
```

## Decision

Keep `single_frame_xwide_unet_line_auxiliary_w010` as the current best capacity ceiling for in-contract refinements, but stop pure width scaling for now. The xwide Result is a real aggregate improvement and improves false-positive behavior, yet its default operating point is more precision-biased and still misses small positive masks. Further capacity alone is unlikely to address the recall/threshold brittleness.

## Next proposed change

Prefer a bounded in-contract follow-up that tests whether the xwide architecture can preserve its precision gains while recovering recall, for example by selecting a lower primary threshold if the contract/run policy exposes it, or by requesting/using Harness-owned training-policy capability such as scheduler or early-stopping support. If no threshold or training-policy control is currently candidate-selectable, a Capability Request for threshold selection or scheduler/early-stopping is more justified than another width-only Candidate Experiment.
