# single_frame_small_unet_line_auxiliary_w010

Small U-Net Candidate Experiment for an auxiliary-weight follow-up to `single_frame_small_unet_line_auxiliary`.

Intent:

- reuse the same base-16 single-frame Small U-Net architecture and shared decoder representation as `single_frame_small_unet_line_auxiliary`;
- keep the shuffled realistic-training setup for direct comparison: `data.sampling_policy: deterministic_shuffle`, `batch_size: 16`, `max_epochs: 30`, AdamW learning rate `0.001`;
- emit primary `mask_logits` for Contrail Mask evaluation;
- emit auxiliary `line_logits` trained by the Harness against the v1 Line Target;
- reduce auxiliary `weighted_bce` influence from `0.25` to `0.10`.

Hypothesis: a smaller Line Target auxiliary weight may preserve the modest Dice/recall gain from `run_20260507_193004_0b4688` while reducing the visually observed tendency toward thicker or more permissive positive-sample masks.

Comparison targets:

- baseline: `run_20260506_045020_84aac5`
- prior line-auxiliary run: `run_20260507_193004_0b4688`

Primary decision metric remains validation Dice over the Contrail Mask. Secondary diagnostics should include total false-positive pixels, total false-negative pixels, empty-mask false positives, and qualitative overlays on positive samples.
