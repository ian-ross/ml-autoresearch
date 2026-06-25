# Candidate Experiment Contract

This document describes the implemented Candidate Experiment source contract. A Candidate Experiment is submitted as a local directory, not an archive. The contract is generic: allowed input modes, output forms, losses, optimizers, data policies, metrics, and auxiliary targets come from the configured Research Problem Spec.

GVCCS-specific examples live temporarily in `docs/gvccs-features.md`.

## Validation phases

Candidate handling has two phases:

1. **Static contract validation** checks files, manifest fields, proposal sections, and spec allowlists without importing or executing candidate Python code.
2. **Harness smoke test/training** later imports `model.py` in a controlled Harness path and calls `build_model(input_spec, output_spec)` to check the model interface before or during execution.

## Minimal layout

```text
candidate/
├── manifest.yaml
└── model.py
```

`manifest.yaml` declares allowed Harness-owned contract choices. `model.py` must expose `build_model(input_spec, output_spec)` and return a `torch.nn.Module`.

## Minimal manifest

```yaml
name: single_frame_unet_baseline
description: Tiny baseline for harness validation.
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

The example values are valid only when the configured Research Problem Spec advertises them.

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

## Manifest authority

Candidate manifests may select only schema-supported values allowed by the active Research Problem Spec. Candidate code must not implement custom data loading, target construction, losses, optimizers, schedulers, training loops, filesystem probes, network calls, runtime pretrained-weight downloads, Docker calls, or ledger writes.

Implemented generic manifest surfaces include:

- `input_mode`
- `output_form`
- `data.sampling_policy`
- `data.frame_selection_policy`
- `data.augmentation_policy`
- `training.loss`
- `training.optimizer`
- bounded learning rate, batch size, and max epochs
- `training.scheduler`
- `training.early_stopping`
- provider-declared `auxiliary_targets`
- optional `repair` lineage

## Auxiliary targets

Candidate manifests may request provider-declared per-pixel auxiliary targets:

```yaml
auxiliary_targets:
  - name: line
    output: line_logits
    loss: weighted_bce
    weight: 0.10
```

Rules for the generic contract:

- `auxiliary_targets` defaults to `[]`.
- Target names, output names, losses, and output specs must be declared by the active Research Problem Spec.
- `weight` must be between `0.0` and `1.0`.
- Auxiliary-output models must return exactly the primary output plus requested auxiliary output keys.
- Tensor shorthand remains valid only for single-output candidates.
- Target construction and auxiliary loss implementation are provider/Harness-owned; candidate code must not derive targets or implement auxiliary losses.
- Primary validation comparison remains based on the Research Problem primary output and selection metric.

GVCCS-specific Line/Boundary Target semantics are described in `docs/gvccs-features.md`.

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

Implemented scheduler policies are `constant_lr` (default), `cosine_decay`, and `reduce_on_plateau`. Scheduler scalar parameters are bounded by manifest validation.

Candidate manifests may enable Harness-owned early stopping on the Research Problem working-validation selection metric:

```yaml
training:
  max_epochs: 60
  early_stopping:
    enabled: true
    patience: 10
    min_delta: 0.001
    restore_best_checkpoint: true
```

When enabled, `patience` must be less than `max_epochs`; `min_delta` is the minimum selection-metric improvement needed to reset patience. The Harness records the resolved scheduler and early-stopping policy, stop reason, completed epochs, and best-checkpoint restoration status.

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

The validator also accepts documented normalized section synonyms. A proposal copied with the Candidate Experiment may include additional narrative and implementation detail.

## Repair Candidate lineage

A Repair Candidate is a distinct Candidate Experiment submitted to fix a prior candidate bug or contract issue without overwriting previous source. Repair Candidates must preserve the original Experiment Proposal hypothesis and Comparison Target. If either changes, create a new Experiment Proposal instead of using repair lineage.

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

Candidate manifests may select spec-allowed Harness/provider-owned data policies:

```yaml
data:
  sampling_policy: deterministic_shuffle
  frame_selection_policy: temporal_eligible_center
  augmentation_policy: light_combined
```

Allowed values come from the configured Research Problem Spec. If omitted, `data.sampling_policy` resolves to `sequential` and `data.augmentation_policy` resolves to `none` when those defaults are supported. Frame-selection policies are provider-dependent; temporal modes such as `centered_temporal_rgb_clip` are implemented only by providers that advertise them.

## Dataset and mount authority

Candidate manifests cannot request data roots, bind mounts, arbitrary filesystem paths, custom data loaders, custom samplers, custom transforms, or custom training loops. The Harness resolves dataset location through the configured Research Problem `data_config` in `ml-autoresearch.toml`; for Docker execution, the Harness validates and mounts configured data read-only at `/data` for the in-container Research Problem adapter. Candidate code receives only tensors supplied by the Harness.
