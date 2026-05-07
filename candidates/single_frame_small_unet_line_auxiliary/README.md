# single_frame_small_unet_line_auxiliary

Small U-Net Candidate Experiment extending the current shuffled realistic-training baseline with a Harness-owned Line Target auxiliary objective.

Intent:

- reuse the base-16 single-frame Small U-Net lineage from `single_frame_small_unet_realistic_training_shuffled`;
- keep `data.sampling_policy: deterministic_shuffle`, `batch_size: 16`, `max_epochs: 30`, and AdamW learning rate `0.001` for direct comparison to baseline Run `run_20260506_045020_84aac5`;
- emit primary `mask_logits` for Contrail Mask evaluation;
- add a decoder-shared `line_logits` head trained by the Harness against the v1 Line Target;
- request `weighted_bce` auxiliary loss with weight `0.25`.

Hypothesis: line-structure supervision may improve recall on thin/faint contrails while preserving the primary validation Dice comparison on the Contrail Mask.

Run status: candidate introduced and covered by synthetic Harness tests. A full GVCCS training Run should compare against `run_20260506_045020_84aac5` and inspect whether precision degrades materially.
