# Candidate Experiment Contract

This document describes the currently implemented local Candidate Experiment source contract. A Candidate Experiment is submitted as a local directory, not as an archive.

The current tracer-bullet implementation supports Single-Frame RGB Input, mask-only primary output, optional Harness-derived per-pixel Line Target auxiliary output, `bce_dice`, auxiliary `weighted_bce`, and `adamw`. Broader v1 contract surface such as temporal inputs, additional auxiliary targets, additional losses, and pretrained weight requests is documented as planned capability in `docs/harness-capabilities.md` and `docs/top-level-plan.md`.

Per-pixel Auxiliary Target support is recorded in `docs/adr/0005-per-pixel-auxiliary-targets-in-the-candidate-experiment-contract.md`. The first implemented public surface is the Line Target only.

## Minimal layout

```text
candidate/
├── manifest.yaml
└── model.py
```

`manifest.yaml` declares allowed Harness-owned contract choices. `model.py` must expose `build_model(input_spec, output_spec)`, but issue #1 validation does not import or execute candidate Python code.

## Minimal manifest

```yaml
name: single_frame_unet_baseline
description: Tiny single-frame mask-only baseline for harness validation.
input_mode: single_frame_rgb
output_form: mask_logits
data:
  sampling_policy: sequential
  augmentation_policy: none
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
```

## Allowed files

Required:

- `manifest.yaml`
- `model.py`

Allowed:

- additional `.py` helper files
- `README.md`
- `PROPOSAL.md` (required in autonomous execution mode)

Rejected:

- symlinks
- hidden files or directories
- checkpoints such as `*.pt`, `*.pth`, `*.ckpt`
- archives
- shell scripts
- notebooks
- dataset files
- arbitrary config blobs

## Auxiliary targets

Candidate manifests may request the first Harness-owned per-pixel Auxiliary Target:

```yaml
auxiliary_targets:
  - name: line
    output: line_logits
    loss: weighted_bce
    weight: 0.25
```

Rules for the initial surface:

- `auxiliary_targets` defaults to `[]`.
- The only allowed target name is `line`.
- `line` requires `output: line_logits`.
- The only allowed auxiliary loss is `weighted_bce`.
- `weight` must be between `0.0` and `1.0`.
- Auxiliary-output models must return exactly `mask_logits` and requested auxiliary output keys; tensor shorthand remains valid only for mask-only candidates.
- Primary validation comparison remains based on Contrail Mask metrics, especially `val/dice`.

## Experiment Proposal contract

`run-candidate` with `--require-proposal/--no-require-proposal` records a mandatory `PROPOSAL.md` in autonomous mode.
The `PROPOSAL.md` must include these required sections or metadata keys:

- `Hypothesis`
- `Comparison Target`
- `Expected Effect`
- `Implementation Sketch`
- `Contract Features Used`
- `Budget Requested`
- `Success Criteria`
- `Fallback/Next Decision`

A proposal copied with the Candidate Experiment can contain additional narrative and implementation detail.

## Repair Candidate lineage

A Repair Candidate is a distinct Candidate Experiment submitted to fix a prior candidate bug or contract issue without overwriting previously submitted source. Repair Candidates must preserve the original Experiment Proposal hypothesis and Comparison Target. If the hypothesis or Comparison Target changes, create a new Experiment Proposal instead of using repair lineage.

Repair Candidates declare structured lineage in `manifest.yaml`:

```yaml
repair:
  original_proposal_id: single_frame_unet_baseline
  original_candidate_id: single_frame_unet_baseline_v1
  motivating_run_id: run_20260501_120000_abcdef
  failure_classification: candidate_bug
  preserves_original_hypothesis: true
  preserves_comparison_target: true
```

`failure_classification` must use the Run Failure Classification vocabulary from `docs/run-lifecycle.md`. In autonomous execution mode, the Harness initially permits at most two Repair Candidates per `original_proposal_id`; scientific changes require a new Experiment Proposal and lineage.

## Data policy

Candidate manifests may select Harness-owned Sampling Policy and Augmentation Policy presets:

```yaml
data:
  sampling_policy: deterministic_shuffle
  augmentation_policy: light_combined
```

Allowed Sampling Policy values are `sequential` and `deterministic_shuffle`. If omitted, `data.sampling_policy` resolves to `sequential` for compatibility with older manifests. `deterministic_shuffle` only affects training example order and is reproducible; validation order stays stable for reproducible metrics and qualitative diagnostics.

Allowed Augmentation Policy presets are `none`, `light_geometric`, `light_photometric`, and `light_combined`. If omitted, `data.augmentation_policy` resolves to `none`. The Resolved Manifest records both the requested `augmentation_policy` and the Harness-applied `augmentation_policy_effective`. Presets are currently Ground-Camera Contrail Detection / GVCCS-specific: `light_geometric` applies conservative image/mask-aligned horizontal mirroring, `light_photometric` applies conservative brightness/contrast/noise perturbations to images only, and `light_combined` applies both. These trusted Harness transforms are applied to training examples only; validation examples remain unaugmented and stable.

Composable augmentation policies or candidate-defined transform DSLs are deferred until justified by structured Capability Requests.

## Dataset and mount authority

Candidate manifests cannot request data roots, bind mounts, arbitrary filesystem paths, custom data loaders, custom samplers, custom transforms, or custom training loops. For GVCCS training, the host CLI accepts `--data-root`; the Harness validates it and, for Docker execution, mounts it read-only at `/data` for the in-container Harness-owned GVCCS adapter. Candidate code receives only the tensors supplied by the Harness.
