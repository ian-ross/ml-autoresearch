# Experiment Proposal: extra-wide Line Target U-Net with stronger bottleneck dropout

## Hypothesis

The current best in-contract base, `single_frame_xwide_unet_line_aux_w010_dropout` / `run_20260530_180658_0af8a8`, improved Dice and recall with bottleneck `Dropout2d(p=0.10)`, while the later attention-gated variant traded too much precision for recall. A slightly stronger bottleneck-only dropout rate (`p=0.15`) may preserve the useful regularization signal while reducing empty-mask false positives and late-epoch overconfidence without adding new decoder complexity.

## Comparison Target

Primary comparison target: `run_20260530_180658_0af8a8` / `single_frame_xwide_unet_line_aux_w010_dropout`, especially best-validation `val/dice=0.833852` and the request-gated failure-bucket evaluation `eval_eval_2026_05_31_xwide_dropout_failure_buckets` at threshold 0.5.

Secondary context: `run_20260531_041919_8dbd72` / `single_frame_xwide_unet_line_aux_w010_attention_dropout`, which improved recall but regressed Dice and precision, and `run_20260530_134005_6f20b1` / plain xwide, which had stronger precision but lower recall.

## Expected Effect

Expected effect is a small regularization/calibration-oriented change, not a capacity change. The desired signal is best-validation Dice at least matching the p=0.10 dropout base within noise, precision closer to plain xwide than the attention-gated run, and no worse recall than plain xwide. The main risk is underfitting or excessive confidence suppression, visible as lower Dice and lower recall than the p=0.10 comparison.

## Implementation Sketch

Start from `single_frame_xwide_unet_line_aux_w010_dropout`. Keep the base-64 U-Net, mask head, Line Target auxiliary head, deterministic shuffle, no augmentation, `bce_dice`, AdamW, learning rate, batch size, and epoch budget unchanged. Change only the bottleneck regularizer from `nn.Dropout2d(p=0.10)` to `nn.Dropout2d(p=0.15)` after the deepest encoder block before decoder upsampling.

## Contract Features Used

- Single-Frame RGB Input (`input_mode: single_frame_rgb`).
- Mask logits primary output (`output_form: mask_logits`).
- Harness-derived Line Target auxiliary output (`line_logits`) with `weighted_bce` weight `0.10`.
- Harness-owned `bce_dice` primary loss and `adamw` optimizer.
- Harness-owned deterministic shuffle Sampling Policy and `none` Augmentation Policy.
- Candidate code limited to Model Architecture; no custom data loading, training loop, loss, filesystem access, networking, or artifact writing.

## Budget Requested

- GVCCS Working Validation Split Run under the existing per-Run policy.
- `batch_size: 8`, matching the current best and avoiding new resource pressure.
- `max_epochs: 30`.
- Expected parameter count unchanged from p=0.10 dropout and below the 10M parameter budget.

## Success Criteria

- Candidate passes static validation and Harness smoke testing.
- Run completes without Resource Failure.
- Best-validation `val/dice` is within `0.0015` of the p=0.10 dropout comparison (`>=0.83235`) and ideally exceeds it.
- Precision does not fall below the attention-gated variant's best-epoch precision (`0.8405`) and ideally remains near or above the p=0.10 dropout precision (`0.8563`).
- Recall remains above the plain xwide comparison (`>0.8049`) while avoiding the attention-gated precision collapse.
- Final-vs-best Dice gap is no worse than the p=0.10 dropout gap.

## Fallback/Next Decision

If p=0.15 improves or preserves Dice while improving the precision/recall tradeoff, request a bounded failure-bucket Post-Run Evaluation before promoting it. If it regresses, keep p=0.10 dropout as the current best and pause architecture-only regularization variants in favor of human-gated Harness capability slices such as threshold selection, scheduler/early stopping, Boundary Target, or additional losses.
