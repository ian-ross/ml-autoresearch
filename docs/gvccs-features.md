# GVCCS feature notes

These notes collect Ground-Camera Contrail Detection / GVCCS-specific behavior while this repository still carries example Research Problem documentation. They are not generic Harness requirements. This material should move to the GVCCS Research Problem repository; generic Harness docs should link here rather than embed GVCCS details.

## Provider boundary

The reusable Harness reaches GVCCS behavior through a configured filesystem Research Problem provider, for example `gvccs.research_problem:build_spec`. Generic Harness production code must not import GVCCS dataset types, hard-code GVCCS paths, or expose GVCCS-named production commands.

The GVCCS provider owns:

- dataset discovery and split/frame selection;
- input-mode construction;
- Contrail Mask target loading;
- GVCCS-specific auxiliary target construction;
- prediction sample rendering and evaluation adapters;
- brief documents and dataset profile artifacts.

## Data layout and mounts

GVCCS data is configured through the Research Problem provider `data_config`, usually with a `dataset_root`. For Docker-backed Harness execution, the Harness validates the host data root and mounts it read-only at `/data`; in-container provider config is rewritten to the mounted path. Candidate Experiment code must remain data-path agnostic.

External GVCCS package integration tests require a valid GVCCS package root. If `ML_AUTORESEARCH_GVCCS_PROBLEM_ROOT` is unset and the local fallback checkout is absent, those tests fail clearly rather than silently skip.

## Supported input modes

GVCCS examples currently include:

- `single_frame_rgb` — one RGB Target Frame, typically using all target frames.
- `centered_temporal_rgb_clip` — a channel-stacked centered temporal clip around the Target Frame. The provider selects only temporal-eligible center frames with complete previous/next stride-1 neighbors inside one inferred Frame Sequence; it does not pad, duplicate, or cross gaps.

Temporal input is provider-dependent. Generic Harness docs should say allowed input modes and frame-selection policies come from the active `ResearchProblemSpec`.

## Supported data policies

GVCCS provider examples include:

- sampling policies such as `sequential` and `deterministic_shuffle`;
- frame-selection policies such as `all_target_frames` and `temporal_eligible_center`;
- augmentation presets such as `none`, `light_geometric`, `light_photometric`, and `light_combined`.

These are trusted provider presets, not candidate-defined transform code or universal Harness defaults.

## Auxiliary targets

GVCCS exposes provider-declared per-pixel auxiliary targets used by candidate auxiliary heads:

- `line` with output `line_logits`;
- `boundary` with output `boundary_logits`.

Both use provider/Harness-owned target construction and the manifest-validated auxiliary loss `weighted_bce`. Candidate code may return the requested auxiliary logits but must not derive target tensors or implement auxiliary losses. Primary comparison remains based on the Contrail Mask output and the Research Problem selection metric.

## Candidate families

Existing GVCCS characterization tests cover single-frame mask-only candidates, temporal candidates, augmentation variants, dropout/residual variants, line auxiliary candidates, and line+boundary auxiliary candidates. Those examples are Research Problem history and should not be copied into generic Harness contracts except as explicitly labeled provider examples.
