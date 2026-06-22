# Experiment Proposal: ASPP-context xwide U-Net with decoder-side mask gate

## Hypothesis

Adding a lightweight decoder-side spatial gate to the current best ASPP-context extra-wide Line Target U-Net will recover part of the precision lost by `run_20260615_140810_2bee94` while preserving most of its recall and Dice gain over the plateau/es parent. The ASPP-context failure-bucket evaluation showed that the main regression is increased false-positive pixels on positive-mask samples, not broad empty-mask hallucination. A mask-branch-only gate learned from final decoder features should suppress diffuse expansions around real contrail cases without removing the bottleneck context that improved missed positives.

## Comparison Target

Primary Comparison Target: `run_20260615_140810_2bee94` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context`.

Secondary reference: `run_20260614_124226_05e3eb` / `single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es`, because it had higher precision but lower recall before the ASPP-context change.

## Expected Effect

Expected outcome is similar or slightly lower recall than the ASPP-context target, improved precision, and stable best-validation Dice. The desired failure-bucket direction is fewer false-positive pixels on positive-mask samples while keeping missed positive masks below the plateau/es level.

## Implementation Sketch

Start from the ASPP-context base-64 extra-wide U-Net with p=0.075 bottleneck dropout, 96-channel dilated bottleneck context branches, and Line Target auxiliary output. Keep the encoder, bottleneck context, decoder, Line Target head, manifest training policy, and data policy unchanged. Add a small `SpatialMaskGate` after the final decoder block and before the primary mask head. The gate uses a 3x3 convolution from 64 channels to 16 channels, batch normalization, ReLU, a 1x1 projection to one spatial gate channel, and sigmoid. The primary mask head receives final features scaled by `0.5 + gate`; the line head remains attached to ungated final features.

## Contract Features Used

- `single_frame_rgb` Input Mode.
- `mask_logits` primary output.
- Harness-derived Line Target auxiliary output `line_logits` with `weighted_bce` at weight `0.10`.
- Harness-owned `bce_dice` primary loss.
- Harness-owned `adamw` optimizer with learning rate `0.001`.
- Harness-owned deterministic shuffle Sampling Policy.
- No augmentation (`augmentation_policy: none`).
- Harness-owned reduce-on-plateau scheduler.
- Harness-owned early stopping with best-checkpoint restoration.

No custom data loading, custom training loop, custom loss, filesystem access, network access, pretrained weights, or artifact writing is introduced.

## Budget Requested

One standard GVCCS Working Validation Run with batch size 8 and up to 80 epochs, matching the ASPP-context comparison target. The gate adds only a small number of parameters and is expected to remain below the current 10M-parameter budget.

## Success Criteria

Strong success: best-validation `val/dice` matches or exceeds `0.8667` while `val/precision` improves over `0.8702` and `val/recall` remains at least `0.8550`.

Minimal success: best-validation `val/dice` remains at least `0.8640` with improved precision over the ASPP-context target and recall still above the plateau/es target (`0.8407`).

Failure/regression: best-validation `val/dice` below `0.8640`, recall collapses to the plateau/es level or worse, precision does not improve over the ASPP-context target, or the candidate fails validation/resource limits.

## Fallback/Next Decision

If the gate improves precision without losing most of the ASPP-context recall gain, request a bounded failure-bucket Post-Run Evaluation to check false-positive-heavy positives and missed positive masks before promotion. If it regresses, abandon decoder-side gating for this family and try a smaller context-capacity variant or a Harness-owned loss/data-policy capability request rather than stacking more architecture-side refinements.
