# Experiment Proposal: Line-auxiliary Small U-Net with light combined augmentation

## Hypothesis

The current best `single_frame_small_unet_line_auxiliary_w010` branch has mixed residual false-positive and false-negative errors after whole-validation failure analysis. Applying the Harness-owned `light_combined` Augmentation Policy during training should improve robustness to modest geometric and photometric variation without changing Candidate Experiment authority or the successful lower-weight Line Target auxiliary setup.

## Comparison Target

Primary Comparison Target: `run_20260510_165335_b458c3` (`single_frame_small_unet_line_auxiliary_w010`), current best validation Dice `0.7843769771696992` using deterministic shuffle, no augmentation, and Line Target auxiliary weight `0.10`.

Global-best comparison is the same mounted Run at this Autonomy Step.

## Expected Effect

The expected effect is a modest improvement in best-validation `val/dice` by reducing validation overfitting and improving generalization across faint/thin positives and empty-mask negatives. Secondary expectations are no large precision/recall collapse relative to the comparison Run and qualitatively fewer small false-positive fragments on empty masks while preserving recall on positive masks.

## Implementation Sketch

Reuse the proven base-16 single-frame Small U-Net architecture with shared decoder features, primary `mask_logits`, and auxiliary `line_logits`. Keep training loss, optimizer, learning rate, batch size, max epochs, and deterministic shuffle unchanged. Add only the manifest-level Harness-owned data-policy choice:

```yaml
data:
  sampling_policy: deterministic_shuffle
  augmentation_policy: light_combined
```

No candidate-owned transforms, data loading, losses, or training-loop changes are introduced.

## Contract Features Used

- `input_mode: single_frame_rgb`
- `output_form: mask_logits`
- Harness-derived Line Target auxiliary output `line_logits`
- Auxiliary loss `weighted_bce` with weight `0.10`
- Primary loss `bce_dice`
- Optimizer `adamw`
- Training knobs: learning rate `0.001`, batch size `16`, max epochs `30`
- Data Policy: `sampling_policy: deterministic_shuffle`
- Data Policy: `augmentation_policy: light_combined`

## Budget Requested

One standard GVCCS Working Validation Split Run with the same budget as `run_20260510_165335_b458c3`: batch size `16`, max epochs `30`, and the Harness default wall-clock/resource policy for this campaign.

## Success Criteria

- Primary: best-validation `val/dice` improves over `0.7843769771696992` by at least `0.003` without Harness or Candidate Contract violations.
- Secondary: best-validation precision and recall remain within roughly `0.02` absolute of the comparison Run unless Dice improves enough to justify the trade-off.
- Diagnostic: follow-up qualitative prediction samples or Whole-Validation Failure Analysis should show no obvious increase in missed-positive masks or empty-mask false-positive fragments.

## Fallback/Next Decision

If the Run fails for candidate or contract reasons, classify the failure before any repair. If the Run completes but does not improve, treat `light_combined` as too broad or not useful for this architecture and consider a narrower Harness-owned augmentation preset (`light_geometric` or `light_photometric`) only if diagnostics identify which component is likely beneficial; otherwise move to a different in-contract architecture/loss hypothesis rather than repeating Line Target weight tuning immediately.
