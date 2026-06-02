# Per-pixel Auxiliary Targets Harness Extension Plan

This plan records the agreed local implementation sequence for adding per-pixel Auxiliary Target support to the Candidate Experiment Contract. It follows ADR 0005. The initial Line Target slice is complete, and the public surface has since expanded to include the Boundary Target.

## Goal

Enable Candidate Experiments to request Harness-derived per-pixel Auxiliary Targets for additional Harness-owned loss terms: `line_logits` trained against a v1 Line Target and `boundary_logits` trained against a v1 Boundary Target. The primary prediction remains `mask_logits` for the Contrail Mask, and primary model comparison remains based on validation Dice over the Contrail Mask.

## Non-goals for first slice

- No image-level or patch-level auxiliary objectives.
- No presence head.
- No arbitrary auxiliary target names beyond the implemented `line` and `boundary` targets.
- No candidate-owned target derivation or custom loss code.
- No exact skeletonization dependency for the v1 Line Target.
- No visual line-target or line-prediction artifacts initially.
- No change to primary Post-Run Evaluation metrics.

## Agreed contract shape

Candidate manifest:

```yaml
output_form: mask_logits
auxiliary_targets:
  - name: line
    output: line_logits
    loss: weighted_bce
    weight: 0.10
  - name: boundary
    output: boundary_logits
    loss: weighted_bce
    weight: 0.05
```

Rules:

- `auxiliary_targets` defaults to `[]`.
- Allowed names: `line` and `boundary`.
- `line` requires `output: line_logits`; `boundary` requires `output: boundary_logits`.
- Implemented auxiliary loss: `weighted_bce`.
- Candidate may configure auxiliary loss `weight` within Harness-owned bounds.
- Positive weighting for `weighted_bce` is Harness-owned policy.
- Requested auxiliary outputs are required.
- Output names are exact: missing, misspelled, or extra outputs fail smoke testing.
- Tensor output shorthand remains valid only for mask-only candidates.

Resolved output spec passed to `build_model(input_spec, output_spec)` should remain backward-compatible:

```python
{
    "form": "mask_logits",
    "shape": [1, 128, 128],
    "auxiliary_outputs": [
        {"target": "line", "name": "line_logits", "shape": [1, 128, 128]},
        {"target": "boundary", "name": "boundary_logits", "shape": [1, 128, 128]},
    ],
}
```

## Implementation slices

### 1. Manifest and Resolved Manifest support

- Add an `AuxiliaryTargetManifest` model.
- Add `CandidateManifest.auxiliary_targets` defaulting to `[]`.
- Validate implemented public surface:
  - `name == "line"` with `output == "line_logits"`
  - `name == "boundary"` with `output == "boundary_logits"`
  - `loss == "weighted_bce"`
  - bounded `weight`: `0.0 <= weight <= 1.0`.
- Ensure existing mask-only candidates remain valid without changes.
- Ensure Resolved Manifest preserves normalized `auxiliary_targets`.
- Add tests for valid line and boundary targets, default empty list, unknown target rejection, wrong output rejection, wrong loss rejection, and out-of-bounds weight rejection.

### 2. Output-spec and smoke-test support

- Build `output_spec` from `resolved_manifest.yaml`, not from a global fixed `OUTPUT_SPEC`, for run-scoped operations.
- Preserve the existing mask-only spec shape and add `auxiliary_outputs` when requested.
- Update smoke testing to pass the manifest-derived output spec into `build_model`.
- Replace mask-only extraction with an expected-output extraction/validation helper that:
  - accepts tensor shorthand only when no auxiliary outputs are requested;
  - requires dictionaries for auxiliary-output candidates;
  - requires exactly the expected keys;
  - validates each output has floating dtype and shape `[B, 1, H, W]`.
- Keep `mask_logits` extraction available for primary evaluation logic.
- Add tests for required `line_logits` / `boundary_logits`, missing requested auxiliary logits, extra output keys, bad shape, and backward compatibility for existing mask-only candidates.

### 3. Training support for v1 Line Target and Boundary Target

- Implement Harness-owned v1 Line Target derivation from the Contrail Mask.
- Use a simple tolerance-band/dilation-style Line Target; do not add skeletonization dependencies yet.
- Implement Harness-owned v1 Boundary Target derivation from the Contrail Mask as a deterministic one-pixel edge band using dilation minus erosion.
- Implement `weighted_bce` for auxiliary logits with Harness-owned positive weighting.
- Compute training total loss:
  - primary mask loss: `bce_dice(mask_logits, contrail_mask)`;
  - weighted auxiliary line loss, when requested: `weight * weighted_bce(line_logits, line_target)`;
  - weighted auxiliary boundary loss, when requested: `weight * weighted_bce(boundary_logits, boundary_target)`;
  - total loss = primary + weighted auxiliary losses.
- Preserve `val/loss` as primary mask loss for comparability with prior Runs.
- Record auxiliary/total loss fields when applicable:
  - training batch or epoch total loss;
  - mask loss;
  - `aux/line_loss` and/or `aux/boundary_loss`;
  - `val/aux/line_loss` and/or `val/aux/boundary_loss`;
  - `val/total_loss`.
- Keep best-checkpoint selection by `val/dice`.
- Add synthetic training tests proving loss computation and metric/artifact fields exist.

### 4. Post-Run Evaluation and prediction artifact tolerance

- Update Post-Run Evaluation model-output handling to tolerate expected auxiliary outputs while using only `mask_logits` for primary metrics.
- Update prediction sample artifact generation similarly.
- Do not add auxiliary-target-specific diagnostic images in the first slice.
- Add tests showing auxiliary-output models can be evaluated and produce ordinary primary mask evaluation artifacts.

### 5. First auxiliary Candidate Experiment

- Add `candidates/single_frame_small_unet_line_auxiliary`.
- Reuse the current shuffled small U-Net baseline lineage.
- Add a small `line_logits` head sharing the decoder representation.
- Manifest uses `auxiliary_targets` with `line_logits`, `weighted_bce`, and a starting auxiliary weight of `0.25`.
- Train against the current baseline Run `run_20260506_045020_84aac5`.
- Compare:
  - primary validation Dice/IoU/precision/recall;
  - recall on faint/thin examples via qualitative diagnostics;
  - whether precision degrades materially;
  - whether Post-Run Evaluation changes the threshold/recall picture.
- Write a Research Note and update `EXPERIMENT_INDEX.md`.

## Acceptance criteria

A first vertical slice is complete when:

1. Existing mask-only Candidate Experiments pass unchanged.
2. Synthetic line-auxiliary and boundary-auxiliary Candidate Experiments pass manifest validation and smoke testing.
3. Missing or extra auxiliary output names fail smoke testing clearly.
4. Training computes primary + auxiliary losses without candidate-owned loss code.
5. Run metrics preserve `val/loss` as primary mask loss and include auxiliary/total loss fields for auxiliary Runs.
6. Post-Run Evaluation tolerates auxiliary-output models and reports primary mask metrics.
7. Documentation remains consistent with ADR 0005 and `docs/candidate-experiment-contract.md`.
