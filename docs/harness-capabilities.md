# Harness Capabilities and Candidate Requirements

This document describes user-level expectations for the ML Autoresearch Harness. The Harness owns training, data loading, validation, execution policy, and artifact persistence. Candidate Experiments express research variation through the allowlisted Candidate Experiment Contract, not through arbitrary filesystem, network, Docker, dataset, MLflow, training-loop, or data-loading authority.

## Example Research Problem: Ground-Camera Contrail Detection

The initial example Research Problem package is binary semantic segmentation for the GVCCS Dataset: given ground-camera imagery, predict a Contrail Mask marking contrail pixels vs non-contrail pixels. This section documents the current example package behavior, not Harness defaults.

The primary prediction target is always the Contrail Mask for a Target Frame.

The GVCCS Dataset is from whole-sky cameras. Likely downstream use may involve conventional ground cameras with different optics, framing, distortion, exposure, and background distributions. This creates a possible domain shift, but GVCCS is the available large labelled dataset and is the training basis for the initial Research Problem. Evaluation on other camera types is a separate exercise outside the initial ML Autoresearch loop.

## Execution model

The current tracer-bullet implementation is synchronous and local, with Docker as the default Candidate Execution Boundary for `run-candidate`:

1. The agent or human submits a Candidate Experiment with `submit-candidate` or `run-candidate`.
2. The Harness validates the manifest/source, creates a Run ID, and copies accepted source into the Run directory.
3. The Harness performs a PyTorch smoke test through `build_model(input_spec, output_spec)`.
4. `run-candidate` trains synchronously against the configured Research Problem provider from `candidate-execution.toml`.
5. For Docker-backed Research Problem training, the host Harness validates `research_problem.data_config.dataset_root` in `candidate-execution.toml`, mounts it read-only at `/data`, and the in-container Research Problem adapter reads `/data`.
6. The agent follows Results through local observation commands: `list-runs`, `run-summary` / `get-run-summary`, and `get-best-runs`.

The native backend remains available as an explicit developer-unsafe escape hatch. Asynchronous queueing, MLflow persistence, and stronger production isolation are planned layers around the same Research Loop.

## Input modes

The Harness may provide these v1 Input Modes:

- **Single-Frame RGB Input** — one RGB image for the Target Frame.
- **Centered Temporal RGB Clip Input** — multiple RGB frames around the Target Frame; the model still predicts the Contrail Mask for the Target Frame only.

Candidate Experiments select from allowed Input Modes in their manifest. The Harness owns frame loading, clip construction, alignment, resizing/cropping, and batching.

## Output forms

A Candidate Experiment must always produce `mask_logits` for the Contrail Mask.

The current tracer-bullet implementation accepts either:

```python
# mask-only tensor form
Tensor  # shape [B, 1, H, W]
```

or:

```python
# mask-only dictionary form
{
    "mask_logits": Tensor,  # required, shape [B, 1, H, W]
}
```

The current tracer-bullet implementation also accepts optional auxiliary heads when requested by the manifest:

```python
{
    "mask_logits": Tensor,
    "line_logits": Tensor,       # when Line Target is requested
    "boundary_logits": Tensor,   # when Boundary Target is requested
}
```

`line_logits` is allowed only for the Harness-derived Line Target, and `boundary_logits` is allowed only for the Harness-derived Boundary Target. The Harness validates output names, shapes, and dtypes before real training and rejects unrequested or unknown output keys.

## Loss allowlist

Candidate Experiments may select only Harness-implemented loss functions. Candidate code must not implement custom losses in v1.

Allowed v1 primary mask losses:

- `bce_dice`
- `focal_dice`
- `focal_tversky`

The current tracer-bullet implementation supports only `bce_dice`; the broader v1 allowlist is planned contract surface.

Allowed v1 auxiliary losses for `line_logits` and `boundary_logits`:

- `focal_bce`
- `weighted_bce`

The current tracer-bullet implementation supports `weighted_bce` for the Line Target and Boundary Target. The manifest may provide bounded auxiliary loss weights, but the Harness owns implementation, validation, positive weighting, and defaults. `focal_bce` remains planned contract surface.

## Metrics

The v1 primary validation metric is:

- `val/dice`

The Harness should also record these secondary metrics:

- `val/iou`
- `val/precision`
- `val/recall`
- `val/loss`

If auxiliary outputs are enabled, the Harness may record diagnostic auxiliary metrics:

- `val/line_f1`
- `val/boundary_f1`

Primary model comparison should use `val/dice` unless a Research Problem definition explicitly says otherwise. Cloud-heavy false-positive metrics are expected later once dataset subsets or tags are defined.

## Augmentation and data policy allowlist

Candidate Experiments may select only Harness-implemented augmentation and data-policy options.

Allowed v1 geometric augmentations:

- `horizontal_flip`
- `small_rotation`
- `random_resized_crop`

Allowed v1 photometric augmentations:

- `brightness_contrast`
- `gamma`
- `blur`
- `noise`
- `jpeg_compression`

Allowed v1 temporal data-policy parameters:

- `clip_length`
- `frame_stride`
- centered Target Frame only

Allowed v1 sampling policies:

- `sequential`
- `deterministic_shuffle`

Manifests that omit `data.sampling_policy` resolve to `sequential`. `deterministic_shuffle` changes training example order reproducibly while validation order remains stable for reproducible metrics and qualitative diagnostics.

Implemented frame selection policies in the GVCCS example Research Problem package:

- `all_target_frames` — use every discovered Target Frame after the current split/max-sample policy; this is the default for `single_frame_rgb`.
- `temporal_eligible_center` — use only Target Frames with complete previous/next stride-1 neighbors inside one inferred Frame Sequence; this is the default and required policy for `centered_temporal_rgb_clip`, and can be selected by `single_frame_rgb` for matched controls.

Run metadata records the effective frame selection policy and the resulting train/validation sample counts for runs whose configured Research Problem adapter reports them.

Implemented augmentation presets in the GVCCS example Research Problem package:

- `none` — no augmentation, and the default for omitted `data.augmentation_policy`.
- `light_geometric` — conservative image/mask-aligned horizontal mirroring for training examples.
- `light_photometric` — conservative brightness/contrast/noise perturbations to training images only.
- `light_combined` — combines `light_geometric` and `light_photometric`.

These presets are trusted Research Problem package code reached through the checked Spec adapter, not Candidate Experiment code and not reusable Harness defaults. Validation examples are not augmented. Resolved Manifests record both selected `augmentation_policy` and `augmentation_policy_effective` when the adapter supplies that metadata.

Deferred policies include composable augmentation policies, arbitrary transform DSLs, positive/negative balancing, vertical flips, hard-negative mining from prior Results, and cloud-heavy negative sampling until the relevant metadata, subsets, Capability Requests, or artifact loop exist.

## Training knobs and resource bounds

Candidate Experiments may select bounded Harness-owned training knobs. Candidate code must not implement its own optimizer, scheduler, or training loop.

Allowed v1 optimizers:

- `adamw`
- `sgd_momentum`

The current tracer-bullet implementation supports only `adamw`; the broader v1 allowlist is planned contract surface.

Allowed v1 bounds:

- learning rate: `1e-5` to `3e-3`
- weight decay: `0` to `0.1`
- batch size: `1` to `32`, with the Harness allowed to lower it for GPU memory
- max epochs: `1` to `100`
- early stopping patience: `5` to `20`, if early stopping is enabled
- mixed precision: `on` or `off`, default `on`
- gradient clipping max norm: `0.1` to `10`, if enabled
- parameter budget: the current tracer-bullet smoke test enforces `10M` parameters; broader Harness policy may raise or override this per Research Problem later

Wall-clock budget policy is intentionally adjustable. A smaller early budget may be useful to push the agent toward many cheap experiments that identify gross architecture choices before longer training runs. For Docker-backed training Runs, wall-clock budget exhaustion should be handled through a Harness-owned graceful shutdown protocol: signal the training loop, allow a bounded grace period to finish a safe unit of work and write the best meaningful Result available, then force-terminate only if the grace period expires.

## Run artifacts

Every completed Run should produce enough artifacts for the Research Loop to compare, diagnose, and propose follow-up Candidate Experiments.

Required completed-Run artifacts:

- `final_metrics.json` — final completed epoch validation metrics; this remains the final epoch Result view, not the best-validation summary.
- `best_metrics.json` — best validation epoch selected by maximizing `val/dice`, including the selected epoch, selection metric name/value, metrics from that validation epoch, and the best-epoch model artifact path.
- `models/best_epoch_model.pt` — Harness-owned PyTorch checkpoint containing the best-validation epoch number, selection metric/value, and CPU-readable model `state_dict`. Candidate Experiment code cannot choose this path or policy. The Harness can later reload the copied Candidate Experiment architecture and this `state_dict` for evaluation beyond the original Run.
- `metrics.jsonl` — per-epoch or per-step metric history.
- `model_summary.json` — parameter count, input/output contract, and useful model summary information.
- `resolved_manifest.yaml` — fully resolved Candidate Experiment configuration after Harness defaults and validation.
- `run_metadata.json` — dataset/split identifiers, Harness version, code/image digests, timestamps, resource limits, and Run status.
- `prediction_samples/` — visual examples including input image or clip reference, ground truth mask, predicted mask, and overlay; include informative failures when possible. Run-level Prediction Sample Policy may be `first_n` or a configured Research Problem adapter policy such as the GVCCS example package's `adjacent_and_scattered` validation diagnostic policy.
- `logs/` — validation, smoke-test, training, timeout, and future persistence logs.

The current implementation writes operation-produced artifacts under `outputs/` while keeping `candidate/`, `resolved_manifest.yaml`, and `run_metadata.json` Harness-owned at the Run root. Dataset identifiers, data-root provenance, effective Data Policy metadata, and train/validation sample counts come from the configured Research Problem adapter.

Best-checkpoint persistence is optional Harness policy, not required for every Run, because checkpoint storage can become large.

Rejected or blocked Runs should still produce a clear status and reason in metadata/logs when possible.

## Auxiliary targets

Auxiliary targets are Harness-derived per-pixel training targets. They are used to add auxiliary losses that encourage useful geometry in the shared model representation. They are not separate end-user predictions and they do not replace the primary Contrail Mask prediction.

### Line Target

A Line Target emphasizes thin centerline-like contrail structure. The current Harness v1 derivation is a deterministic small tolerance band around positive mask pixels. Future Research Problem implementations may replace this with skeletonization or thinning while preserving Harness ownership of target derivation.

Candidate models may expose `line_logits` with shape `[B, 1, H, W]`. These are image-aligned per-pixel logits, not Hough-space logits or arbitrary line-parameter predictions.

### Boundary Target

A Boundary Target emphasizes contrail edge geometry. The current Harness v1 derivation is a deterministic one-pixel edge band computed from dilation minus erosion of the Contrail Mask. Future Research Problem implementations may replace this with distance-transform variants while preserving Harness ownership of target derivation.

Candidate models may expose `boundary_logits` with shape `[B, 1, H, W]`. These are image-aligned per-pixel logits, not object-detection boundaries or a separate segmentation class.

## Example auxiliary-loss use

If a model returns mask, line, and boundary logits, the Harness computes a combined training loss from the manifest-requested auxiliary targets. Conceptually:

```python
line_target = derive_line_target(contrail_mask)
boundary_target = derive_boundary_target(contrail_mask)

loss = (
    1.00 * bce_dice(mask_logits, contrail_mask)
  + line_weight * weighted_bce(line_logits, line_target)
  + boundary_weight * weighted_bce(boundary_logits, boundary_target)
)
```

The exact derivation method and loss weights are Harness-owned policy exposed through allowed manifest parameters. Primary evaluation remains based on `mask_logits` compared with the Contrail Mask.

## Safety and authority constraints

Candidate Experiments must not:

- implement custom training loops;
- implement custom data loading;
- choose arbitrary dataset paths;
- access arbitrary filesystem paths;
- access the network during training;
- write to MLflow;
- invoke Docker or shell commands;
- download pretrained weights at runtime;
- reference arbitrary checkpoint paths.

If new research freedom is needed, record a structured Capability Request using the format in `docs/capability-request-format.md`. Capability Requests are not self-approving; they only create an auditable request for separate human-supervised Harness work.

## Pretrained weights

Candidate Experiments may reference only Approved Weight Artifacts by stable identifier. New pretrained weights enter the system through Pretrained Weight Requests that record source, license, intended use, and audit information. Once approved, the Harness makes the artifact available under an approved ID.

## Docker Candidate Execution Boundary hardening status

The current Docker backend runs Candidate Experiment smoke tests and training with no network, a read-only root filesystem, dropped Linux capabilities, `no-new-privileges`, bounded memory/CPU/process limits, and Harness-owned explicit environment variables. By default the backend detects rootless Docker before launch: rootless Docker uses container `0:0`, which maps back to the invoking unprivileged host user so output artifacts remain user-owned; rootful Docker uses `--userns=host` with the host Harness uid/gid. `/outputs` and `/scratch` are the only writable container paths: `/outputs` is the run-scoped writable artifact mount, and `/scratch` is bounded tmpfs. `/candidate`, `/resolved_manifest.yaml`, `/run_metadata.json`, and the configured Research Problem data mount at `/data` are read-only. GPU access is disabled by default and is only enabled by Harness configuration.

Docker training wall-clock exhaustion uses a graceful timeout protocol. The Harness records timeout events, writes a sentinel in `/scratch`, waits a bounded grace period for the in-container Harness-owned training loop to stop at an end-of-batch checkpoint and write usable Results, and force-kills only after grace expires. Run metadata distinguishes normal completion, graceful timeout completion, and forced timeout failure.

Remaining limitations: no custom seccomp/AppArmor policy yet, no user-namespace remapping requirement yet, no hard artifact quota for `/outputs`, and no distributed/multi-container GPU policy.
