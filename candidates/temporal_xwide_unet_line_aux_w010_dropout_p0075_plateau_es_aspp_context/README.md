# temporal_xwide_unet_line_aux_w010_dropout_p0075_plateau_es_aspp_context

Temporal Candidate Experiment for Ground-Camera Contrail Detection.

## Model Architecture

This candidate keeps the current best extra-wide base-64 ASPP-context U-Net with p=0.075 bottleneck dropout and a Line Target auxiliary head, but changes the input mode from a single RGB frame to a Harness-owned centered three-frame RGB clip. The Research Problem adapter supplies the clip as channel-stacked RGB (`[9, 128, 128]`), and the model predicts the Contrail Mask for the center Target Frame only.

## Manifest choices

- Input: `centered_temporal_rgb_clip`
- Output: `mask_logits`
- Auxiliary target: Harness-derived `line` target via `line_logits`, `weighted_bce`, weight `0.10`
- Data policy: deterministic shuffle, `temporal_eligible_center` frame selection, no augmentation
- Training: `bce_dice`, AdamW, learning rate `0.001`, batch size `8`, max epochs `80`
- Training policy: reduce-on-plateau scheduler and early stopping with best-checkpoint restoration

## Contract assumptions

The Candidate Experiment code defines only Model Architecture. The Harness owns frame sequence discovery, temporal clip construction, center-frame target selection, data loading, target construction, losses, optimizer, scheduler, early stopping, metrics, artifacts, and validation split handling.

## Known limitations

This Run is not a perfectly matched comparison to all-target-frame single-frame Runs because temporal input requires `temporal_eligible_center` frame selection. Promotion should therefore depend on both aggregate metrics and, if promising, bounded failure-bucket diagnostics or a matched single-frame control.
