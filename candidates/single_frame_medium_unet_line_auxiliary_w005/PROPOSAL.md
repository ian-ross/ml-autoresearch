# Experiment Proposal: lower-weight Line Target auxiliary medium U-Net

## Hypothesis

The current-best medium-capacity U-Net (`run_20260529_155844_d8ebec`) improved best-validation Dice over the base-16 line-auxiliary model, but it did so with a conservative precision/recall trade-off: best-epoch precision rose to 0.842995 while recall fell to 0.760187. Reducing the Harness-derived Line Target auxiliary weight from 0.10 to 0.05 may keep the useful thin-structure inductive bias while relaxing auxiliary regularization enough to recover recall and preserve or improve Dice.

## Comparison Target

Primary Comparison Target: `run_20260529_155844_d8ebec` / `single_frame_medium_unet_line_auxiliary_w010`, with best-validation `val/dice` 0.7994525273696043, `val/precision` 0.8429952005805884, and `val/recall` 0.760187086855412.

Secondary context: `run_20260510_165335_b458c3` / `single_frame_small_unet_line_auxiliary_w010` had lower best-validation Dice (0.7843769771696992) but higher recall (0.7973861422181288), so recall recovery is a meaningful diagnostic for the medium family.

## Expected Effect

Lowering the auxiliary Line Target loss weight should reduce pressure to prefer very line-confident, high-precision masks while retaining some centerline guidance. The expected outcome is similar or improved best-validation Dice relative to the medium w0.10 model, with better recall and no severe precision collapse. The main risk is that lowering the line auxiliary signal removes useful geometric regularization and regresses toward less stable masks.

## Implementation Sketch

Reuse the `single_frame_medium_unet_line_auxiliary_w010` Model Architecture exactly: base-24 U-Net encoder/decoder with skip connections, a primary `mask_logits` head, and a `line_logits` auxiliary head. Change only the manifest-declared Line Target auxiliary `weight` from `0.10` to `0.05`. Keep `input_mode: single_frame_rgb`, `output_form: mask_logits`, deterministic shuffle sampling, no augmentation, primary `bce_dice`, auxiliary `weighted_bce`, AdamW learning rate 0.001, batch size 16, and max epochs 30.

## Contract Features Used

- Model Architecture defined in `model.py` only.
- Single-Frame RGB Input (`input_mode: single_frame_rgb`).
- Primary mask logits output (`output_form: mask_logits`).
- Harness-derived Line Target auxiliary output (`line_logits`) with `weighted_bce` weight 0.05.
- Harness-owned deterministic shuffle Sampling Policy.
- Harness-owned no-augmentation policy (`augmentation_policy: none`).
- Harness-owned `bce_dice` primary loss, `adamw` optimizer, learning rate, batch size, and epoch budget.

No custom data loading, transforms, training loop, loss implementation, filesystem access, network access, checkpoint download, or artifact writing is requested.

## Budget Requested

One standard GVCCS training Run under the existing wall-clock/resource policy: batch size 16, max epochs 30, no Experiment Batch, and no Post-Run Evaluation requested in this proposal.

## Success Criteria

Consider the hypothesis supported if the completed Run improves best-validation `val/dice` over 0.7994525273696043 by at least 0.003, or matches within 0.003 while materially recovering recall toward the small-model comparison without a large precision collapse. Treat final-vs-best separation explicitly because the comparison target's final epoch regressed below its best epoch.

## Fallback/Next Decision

If the Run improves Dice or preserves Dice while improving recall, use it as the next medium-family baseline and consider either a best-epoch-focused training-policy Capability Request or a further bounded auxiliary-weight comparison. If it regresses on Dice and recall, treat the w0.10 Line Target weight as the better medium setting and stop immediate auxiliary-weight reductions. If it fails from resource pressure, classify the failure rather than hiding resource workarounds in candidate code.
