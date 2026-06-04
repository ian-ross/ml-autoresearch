# single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60_head_dropout_p005

Candidate Experiment for Ground-Camera Contrail Detection.

This candidate preserves the current best extra-wide base-64 single-frame U-Net with Harness-derived Line Target auxiliary supervision and 60 training epochs, then adds light `Dropout2d(p=0.05)` on the final decoder feature map before both output heads. The goal is precision-safe regularization: reduce brittle high-resolution texture activation and empty-mask false positives without changing data policy, loss, optimizer, or training-loop behavior.

## Contract assumptions

- Input mode: `single_frame_rgb`
- Outputs: `mask_logits` and requested `line_logits`
- Losses, Line Target derivation, data loading, training, validation, metrics, and artifacts are Harness-owned.
- Candidate code only defines the model architecture.

## Known limitations

Head dropout may suppress faint positives and lower recall. The first Run should be compared against `run_20260602_203450_c05550`, and if aggregate metrics are close, a failure-bucket evaluation is needed to inspect empty-mask false positives and missed positives.
