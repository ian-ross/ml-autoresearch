# single_frame_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context_temporal_eligible_control

Matched single-frame control Candidate Experiment for Ground-Camera Contrail Detection.

## Model Architecture

This candidate reuses the successful extra-wide base-64 U-Net with p=0.075 bottleneck dropout, a lightweight parallel dilated bottleneck context block, and a Line Target auxiliary head. The context block uses dilation rates 1, 2, 4, and 6 at bottleneck resolution and fuses the branches back before the decoder.

## Manifest choices

- Input: `single_frame_rgb`
- Output: `mask_logits`
- Auxiliary target: Harness-derived `line` target via `line_logits`, `weighted_bce`, weight `0.10`
- Data policy: deterministic shuffle, `temporal_eligible_center` frame selection, no augmentation
- Training: `bce_dice`, AdamW, learning rate `0.001`, batch size `8`, max epochs `80`
- Training policy: reduce-on-plateau scheduler and early stopping with best-checkpoint restoration

## Contract assumptions

The Candidate Experiment code defines only Model Architecture. The Harness owns data loading, temporal-eligible frame selection, target construction, losses, optimizer, scheduler, early stopping, metrics, artifacts, and validation split handling.

## Known limitations

This is intentionally not a temporal model. It cannot use neighboring frames; it only restricts the single-frame data policy to temporal-eligible center Target Frames so the temporal ASPP-context result can be compared against a matched single-frame control.
