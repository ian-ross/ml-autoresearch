# 2026-05-31 xwide attention-dropout regression

## Hypothesis

Adding lightweight decoder attention gates to the current best extra-wide base-64 U-Net with Line Target auxiliary weight 0.10 and bottleneck `Dropout2d(p=0.10)` tested whether skip-feature gating could reduce irrelevant sky/background detail while preserving thin contrail structure. The expected signal was a small best-validation Dice gain over `run_20260530_180658_0af8a8`, or at least similar Dice with higher recall and precision no worse than the older wide baseline.

## Candidate Experiment(s)

- Candidate Experiment ID: `single_frame_xwide_unet_line_aux_w010_attention_dropout`
- Candidate Experiment path: `candidates/single_frame_xwide_unet_line_aux_w010_attention_dropout`
- Relevant Experiment Proposal: `candidates/single_frame_xwide_unet_line_aux_w010_attention_dropout/PROPOSAL.md`
- Primary Comparison Target: `run_20260530_180658_0af8a8` / `single_frame_xwide_unet_line_aux_w010_dropout`
- Secondary context: `run_20260530_134005_6f20b1` / `single_frame_xwide_unet_line_auxiliary_w010` and `run_20260530_101019_def893` / `single_frame_wide_unet_line_auxiliary_w010`
- Contract choices: Single-Frame RGB Input, mask logits output, Line Target auxiliary output with `weighted_bce` weight 0.10, deterministic shuffle Sampling Policy, no augmentation, `bce_dice`, AdamW, learning rate 0.001, requested/effective batch size 8, max epochs 30.

## Run(s)

- Run ID: `run_20260531_041919_8dbd72`
- Dataset mode/subset: GVCCS Working Validation Split
- Harness/backend notes: completed on CUDA with no Resource Failure retry; best-validation model artifact persisted at `outputs/models/best_epoch_model.pt`; model summary reports 7,915,941 parameters, below the 10,000,000 parameter budget.

## Key metrics

Best-validation metrics, selected by max `val/dice`:

| Run | Candidate Experiment | Best epoch | val/dice | val/iou | val/precision | val/recall | val/loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `run_20260530_101019_def893` | `single_frame_wide_unet_line_auxiliary_w010` | 29 | 0.829165 | 0.708183 | 0.850682 | 0.808710 | 0.512781 |
| `run_20260530_134005_6f20b1` | `single_frame_xwide_unet_line_auxiliary_w010` | 28 | 0.832669 | 0.713310 | 0.862413 | 0.804908 | 0.510368 |
| `run_20260530_180658_0af8a8` | `single_frame_xwide_unet_line_aux_w010_dropout` | 29 | 0.833852 | 0.715049 | 0.856300 | 0.812552 | 0.510518 |
| `run_20260531_041919_8dbd72` | `single_frame_xwide_unet_line_aux_w010_attention_dropout` | 28 | 0.831409 | 0.711463 | 0.840498 | 0.822515 | 0.510823 |

The attention-gated variant regresses best-validation Dice by about `-0.00244` relative to the xwide dropout comparison target. It does improve recall by about `+0.00996`, but precision falls by about `-0.01580` and also drops below the older wide baseline. This misses the proposal success criterion: the recall gain appears to spend too much false-positive budget.

Final completed-epoch metrics for `run_20260531_041919_8dbd72` were `val/dice` 0.828109, `val/iou` 0.706644, `val/precision` 0.849981, `val/recall` 0.807335, and `val/loss` 0.517817. The best-to-final Dice gap is about `0.00330`, which is much smaller than the xwide dropout gap, so attention gates did not worsen late-epoch stability. The issue is the best model's precision/recall tradeoff, not training instability.

## Qualitative observations

The saved first-N prediction samples are weaker than the xwide dropout comparison samples:

- `val/000000`: Dice 0.7603, IoU 0.6133.
- `val/000001`: Dice 0.7969, IoU 0.6623.

For context, the xwide dropout note recorded `val/000000` Dice 0.8462 and `val/000001` Dice 0.8613. These first-N samples are not a full-validation diagnostic, but they are consistent with the aggregate metrics: the attention-gated model is more recall-oriented and less precise, with visibly less reliable saved positive masks.

![Attention-dropout sample 000 overlay](../runs/run_20260531_041919_8dbd72/outputs/prediction_samples/sample_000_overlay.png)

![Attention-dropout sample 001 overlay](../runs/run_20260531_041919_8dbd72/outputs/prediction_samples/sample_001_overlay.png)

![Attention-dropout sample 000 heatmap](../runs/run_20260531_041919_8dbd72/outputs/prediction_samples/sample_000_probability_heatmap.png)

## Research Figures

```research-figures
figures:
  - figure_id: fig-xwide-attention-dropout-sample-000-overlay
    source_run_id: run_20260531_041919_8dbd72
    source_artifact_path: outputs/prediction_samples/sample_000_overlay.png
    reason: First saved validation overlay for the attention-gated xwide dropout variant; sample Dice 0.7603 shows weaker qualitative behavior than the xwide dropout comparison sample.
  - figure_id: fig-xwide-attention-dropout-sample-001-overlay
    source_run_id: run_20260531_041919_8dbd72
    source_artifact_path: outputs/prediction_samples/sample_001_overlay.png
    reason: Second saved validation overlay for the attention-gated variant; sample Dice 0.7969 supports the aggregate regression relative to xwide dropout.
  - figure_id: fig-xwide-attention-dropout-sample-000-heatmap
    source_run_id: run_20260531_041919_8dbd72
    source_artifact_path: outputs/prediction_samples/sample_000_probability_heatmap.png
    reason: Probability heatmap for the first saved validation sample, useful for checking whether decoder attention gating broadened or softened predicted contrail regions in a lower-precision model.
```

## Decision

Do not promote `run_20260531_041919_8dbd72` as the current best. The run completed cleanly and the architecture stayed within contract, but the research hypothesis is not supported: decoder attention gates increased recall at the cost of enough precision to reduce Dice below the xwide dropout baseline and below the proposal's precision guardrail.

Keep `run_20260530_180658_0af8a8` / `single_frame_xwide_unet_line_aux_w010_dropout` as the current best in-contract base. Do not continue adding decoder-gating complexity under the current architecture-only contract unless a later diagnostic specifically justifies a more constrained version.

## Next proposed change

Prefer a bounded Capability Request or Evaluation Request rather than another immediate architecture-complexity variant. The most useful next step is human-gated Harness support for threshold selection, scheduler/early stopping, Boundary Target, or additional loss options, because recent in-contract architecture-only changes are mostly trading precision against recall around a plateau. If another Candidate Experiment is required before a capability slice, it should be a simpler regularization or calibration-oriented change to the xwide dropout base with an explicit precision guardrail, not more skip-gate complexity.
