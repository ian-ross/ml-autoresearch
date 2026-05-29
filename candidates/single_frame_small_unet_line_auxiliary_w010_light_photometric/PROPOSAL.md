# Experiment Proposal: Line-auxiliary Small U-Net with light photometric augmentation

## Hypothesis

The unaugmented lower-weight Line Target auxiliary Small U-Net remains the current best in-contract Result, while `light_combined` and `light_geometric` augmentation both regressed. Because `light_geometric` recovered only part of the `light_combined` loss and still underperformed, a single remaining isolation test of the Harness-owned `light_photometric` Augmentation Policy can determine whether conservative brightness/contrast/noise perturbations help or harm this branch without geometric mirroring.

## Comparison Target

Primary Comparison Target: `run_20260510_165335_b458c3` (`single_frame_small_unet_line_auxiliary_w010`), current best validation Dice `0.7843769771696992` using deterministic shuffle, no augmentation, and Line Target auxiliary weight `0.10`.

Secondary context: `run_20260523_163405_8e18eb` (`light_combined`) reached best-validation Dice `0.717276204322084`, and `run_20260525_123213_541981` (`light_geometric`) reached best-validation Dice `0.7292688877723468`. This Candidate Experiment is the final narrow augmentation-component control for this architecture before shifting away from augmentation if it also regresses.

## Expected Effect

If photometric variation is the useful part of augmentation, the Run should maintain or improve best-validation `val/dice` relative to the unaugmented comparison target while preserving recall on faint contrails. If it regresses, the result will close the current augmentation question by showing that the available light augmentation presets are not a productive next lever for this Small U-Net plus Line Target auxiliary branch.

## Implementation Sketch

Reuse the proven base-16 single-frame Small U-Net architecture with shared decoder features, primary `mask_logits`, and auxiliary `line_logits`. Keep primary loss, auxiliary loss, optimizer, learning rate, batch size, max epochs, deterministic shuffle, architecture, and Line Target weight unchanged. Add only this manifest-level Harness-owned data-policy choice:

```yaml
data:
  sampling_policy: deterministic_shuffle
  augmentation_policy: light_photometric
```

No candidate-owned transforms, data loading, losses, samplers, runtime downloads, filesystem inspection, or training-loop changes are introduced.

## Contract Features Used

- `input_mode: single_frame_rgb`
- `output_form: mask_logits`
- Harness-derived Line Target auxiliary output `line_logits`
- Auxiliary loss `weighted_bce` with weight `0.10`
- Primary loss `bce_dice`
- Optimizer `adamw`
- Training knobs: learning rate `0.001`, batch size `16`, max epochs `30`
- Data Policy: `sampling_policy: deterministic_shuffle`
- Data Policy: `augmentation_policy: light_photometric`

## Budget Requested

One standard GVCCS Working Validation Split Run with the same budget as the comparison target: batch size `16`, max epochs `30`, and the Harness default wall-clock/resource policy for this campaign.

## Success Criteria

- Primary: best-validation `val/dice` improves over `0.7843769771696992` by at least `0.003` without Harness or Candidate Contract violations.
- Secondary: best-validation precision and recall remain within roughly `0.02` absolute of the comparison Run unless Dice improves enough to justify the trade-off.
- Diagnostic: materially outperform the regressed `light_geometric` Run (`0.7292688877723468`) and avoid the recall loss seen in the augmented variants.

## Fallback/Next Decision

If the Run fails for candidate or contract reasons, classify the failure before any repair. If the Run completes but regresses like the prior augmentation variants, keep `run_20260510_165335_b458c3` as the best Result, stop immediate augmentation variants for this architecture, and move the next proposal to a different in-contract architecture/loss/training-knob hypothesis or to a Capability Request if progress is blocked by missing contract surface. If it improves, compare precision/recall and qualitative samples before deciding whether photometric augmentation should become the default for this branch.
