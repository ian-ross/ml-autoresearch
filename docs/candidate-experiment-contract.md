# Candidate Experiment Contract

This document describes the currently implemented local Candidate Experiment source contract. A Candidate Experiment is submitted as a local directory, not as an archive.

The current tracer-bullet implementation supports Single-Frame RGB Input, mask-only primary output, optional Harness-derived per-pixel Line Target and Boundary Target auxiliary outputs, `bce_dice`, auxiliary `weighted_bce`, and `adamw`. Broader v1 contract surface such as temporal inputs, additional primary losses, and pretrained weight requests is documented as planned capability in `docs/harness-capabilities.md` and `docs/top-level-plan.md`.

Per-pixel Auxiliary Target support is recorded in `docs/adr/0005-per-pixel-auxiliary-targets-in-the-candidate-experiment-contract.md`. The implemented public auxiliary-target surface includes Line Target (`line_logits`) and Boundary Target (`boundary_logits`).

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

Optional Harness-owned training policy controls may also be declared under `training.scheduler` and `training.early_stopping`; omitted values resolve to constant learning rate and disabled early stopping.

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

Candidate manifests may request Harness-owned per-pixel Auxiliary Targets:

```yaml
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

Rules for the implemented surface:

- `auxiliary_targets` defaults to `[]`.
- Allowed target names are `line` and `boundary`.
- `line` requires `output: line_logits`; `boundary` requires `output: boundary_logits`.
- The only implemented auxiliary loss is `weighted_bce`.
- `weight` must be between `0.0` and `1.0`.
- Auxiliary-output models must return exactly `mask_logits` and requested auxiliary output keys; tensor shorthand remains valid only for mask-only candidates.
- The Harness derives Line Target and Boundary Target tensors from the primary Contrail Mask; Candidate Experiment code must not derive targets or implement auxiliary losses.
- Primary validation comparison remains based on Contrail Mask metrics, especially `val/dice`.

## Training policy

Candidate manifests may select a narrow Harness-owned learning-rate scheduler preset:

```yaml
training:
  scheduler:
    policy: reduce_on_plateau
    factor: 0.5
    patience: 3
    min_lr: 0.00001
```

Allowed scheduler policies are `constant_lr` (default), `cosine_decay`, and `reduce_on_plateau`. Scheduler scalar parameters are bounded by manifest validation; Candidate Experiment code must not implement schedulers or custom training-loop logic.

Candidate manifests may enable Harness-owned early stopping on the Research Problem's working-validation selection metric (for GVCCS, `val/dice`, maximized):

```yaml
training:
  max_epochs: 60
  early_stopping:
    enabled: true
    patience: 10
    min_delta: 0.001
    restore_best_checkpoint: true
```

When enabled, `patience` must be less than `max_epochs`; `min_delta` is the minimum selection-metric improvement needed to reset patience. The Harness records the resolved scheduler and early-stopping policy, stop reason, completed epochs, and whether the best-validation checkpoint was restored for the final in-memory model state used by post-training artifacts.

## Experiment Proposal contract

`submit-candidate` and `run-candidate` default to autonomous-mode proposal validation (`--require-proposal`). Use `--no-require-proposal` only for manual compatibility flows that intentionally omit a candidate-local proposal. When proposal validation is enabled, the Candidate Experiment directory must include `PROPOSAL.md` with these required sections or metadata keys:

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
  frame_selection_policy: temporal_eligible_center
  augmentation_policy: light_combined
```

Allowed Sampling Policy values are `sequential` and `deterministic_shuffle`. If omitted, `data.sampling_policy` resolves to `sequential` for compatibility with older manifests. `deterministic_shuffle` only affects training example order and is reproducible; validation order stays stable for reproducible metrics and qualitative diagnostics.

Allowed Frame Selection Policy values are declared by the configured Research Problem Spec. In the GVCCS example package, `single_frame_rgb` defaults to `all_target_frames`, while `centered_temporal_rgb_clip` resolves to and requires `temporal_eligible_center`. That example policy selects only Target Frames with complete previous/next stride-1 neighbors inside one inferred Frame Sequence; it never pads, duplicates, or crosses gaps. Single-frame Candidate Experiments may explicitly request `temporal_eligible_center` as a matched control for temporal Candidate Experiments when the configured Spec allows it.

Allowed Augmentation Policy presets are declared by the configured Research Problem Spec. In the GVCCS example package, supported presets are `none`, `light_geometric`, `light_photometric`, and `light_combined`; omitted `data.augmentation_policy` resolves to `none`. The Resolved Manifest records both the requested `augmentation_policy` and adapter-applied `augmentation_policy_effective` when supplied. The example presets apply trusted Research Problem package transforms to training examples only; validation examples remain unaugmented and stable.

Composable augmentation policies or candidate-defined transform DSLs are deferred until justified by structured Capability Requests.

## Dataset and mount authority

Candidate manifests cannot request data roots, bind mounts, arbitrary filesystem paths, custom data loaders, custom samplers, custom transforms, or custom training loops. The Harness resolves dataset location through the configured Research Problem `data_config` (e.g., `dataset_root`) in `ml-autoresearch.toml`; for Docker execution, the Harness validates and mounts it read-only at `/data` for the in-container Research Problem adapter. Candidate code receives only the tensors supplied by the Harness.
