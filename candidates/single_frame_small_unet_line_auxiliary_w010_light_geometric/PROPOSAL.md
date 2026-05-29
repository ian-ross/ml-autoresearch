# Experiment Proposal: Line-auxiliary Small U-Net with light geometric augmentation

## Hypothesis

The completed `light_combined` augmentation Run regressed substantially relative to the current best unaugmented lower-weight Line Target auxiliary Small U-Net. Because `light_combined` confounds geometric mirroring with photometric perturbations, a narrower Harness-owned `light_geometric` Augmentation Policy can test whether conservative image/mask-aligned horizontal mirroring is useful by itself without introducing photometric noise that may have harmed calibration or faint-contrail recall.

## Comparison Target

Primary Comparison Target: `run_20260510_165335_b458c3` (`single_frame_small_unet_line_auxiliary_w010`), current best validation Dice `0.7843769771696992` using deterministic shuffle, no augmentation, and Line Target auxiliary weight `0.10`.

Secondary context: `run_20260523_163405_8e18eb` (`single_frame_small_unet_line_auxiliary_w010_light_combined`) reached only best-validation Dice `0.717276204322084`, so this Candidate Experiment is an isolation test of the geometric component rather than a repeat of the combined preset.

## Expected Effect

The expected effect is either a modest best-validation `val/dice` improvement over the unaugmented comparison Run, or clear evidence that geometric mirroring alone is not responsible for useful robustness in this architecture. If helpful, the preset should improve orientation robustness without the precision/recall instability observed in the completed combined-augmentation Run. Precision and recall should remain close to the comparison target, with no large recall loss on positive masks.

## Implementation Sketch

Reuse the proven base-16 single-frame Small U-Net architecture with shared decoder features, primary `mask_logits`, and auxiliary `line_logits`. Keep primary loss, auxiliary loss, optimizer, learning rate, batch size, max epochs, deterministic shuffle, architecture, and Line Target weight unchanged. Add only this manifest-level Harness-owned data-policy choice:

```yaml
data:
  sampling_policy: deterministic_shuffle
  augmentation_policy: light_geometric
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
- Data Policy: `augmentation_policy: light_geometric`

## Budget Requested

One standard GVCCS Working Validation Split Run with the same budget as the comparison target: batch size `16`, max epochs `30`, and the Harness default wall-clock/resource policy for this campaign.

## Success Criteria

- Primary: best-validation `val/dice` improves over `0.7843769771696992` by at least `0.003` without Harness or Candidate Contract violations.
- Secondary: best-validation precision and recall remain within roughly `0.02` absolute of the comparison Run unless Dice improves enough to justify the trade-off.
- Comparative: best-validation Dice should be materially above the regressed `light_combined` Run (`0.717276204322084`), confirming that the narrower preset avoids the combined policy's observed degradation.

## Fallback/Next Decision

If the Run fails for candidate or contract reasons, classify the failure before any repair. If the Run completes but regresses like `light_combined`, avoid geometric augmentation for this architecture and either test `light_photometric` as the remaining isolated augmentation component or move to a different in-contract architecture/loss hypothesis. If it matches but does not exceed the unaugmented comparison target, keep `run_20260510_165335_b458c3` as the best Result and use the outcome as evidence that augmentation is not the highest-value next lever for this branch.
