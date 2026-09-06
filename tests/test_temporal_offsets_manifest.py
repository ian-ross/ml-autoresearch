from __future__ import annotations

from pathlib import Path

from ml_autoresearch.candidates import validate_candidate_directory
from ml_autoresearch.research_problems import ResearchProblemSpec, ResearchProblemSpecRegistry


def _spec() -> ResearchProblemSpec:
    return ResearchProblemSpec(
        id="ground_camera_contrail_detection",
        version="v0",
        input_modes=("single_frame_rgb", "centered_temporal_rgb_clip"),
        input_specs={
            "single_frame_rgb": {"mode": "single_frame_rgb", "shape": [3, 128, 128]},
            "centered_temporal_rgb_clip": {
                "mode": "centered_temporal_rgb_clip",
                "shape": [9, 128, 128],
                "layout": "channel_stacked_rgb",
                "target_frame": "offset_0_seconds",
            },
        },
        output_forms=("mask_logits",),
        output_specs={"mask_logits": {"form": "mask_logits", "shape": [1, 128, 128]}},
        losses=("bce_dice",),
        optimizers=("adamw",),
        sampling_policies=("sequential", "deterministic_shuffle"),
        frame_selection_policies=("all_target_frames", "temporal_eligible_center"),
        input_mode_frame_selection_defaults={
            "single_frame_rgb": "all_target_frames",
            "centered_temporal_rgb_clip": "temporal_eligible_center",
        },
        augmentation_policies=("none",),
        primary_metric="val/dice",
    )


def _write_candidate(path: Path, manifest: str) -> None:
    path.mkdir()
    (path / "manifest.yaml").write_text(manifest)
    (path / "model.py").write_text("def build_model(input_spec, output_spec):\n    raise RuntimeError('not used')\n")


def test_temporal_offsets_are_validated_and_normalized(tmp_path: Path):
    candidate = tmp_path / "candidate"
    _write_candidate(
        candidate,
        """
name: flexible_temporal
research_problem: ground_camera_contrail_detection
input_mode: centered_temporal_rgb_clip
output_form: mask_logits
data:
  sampling_policy: deterministic_shuffle
  temporal_offsets_seconds: [60, -60, 0, -30, 30]
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 4
  max_epochs: 5
""".strip(),
    )

    manifest = validate_candidate_directory(candidate, research_problem_registry=ResearchProblemSpecRegistry([_spec()]))

    assert manifest.data.frame_selection_policy == "temporal_eligible_center"
    assert manifest.data.temporal_offsets_seconds == [-60, -30, 0, 30, 60]


def test_temporal_offsets_update_smoke_input_spec_shape():
    spec = _spec()

    input_spec = spec.build_input_spec(
        {
            "input_mode": "centered_temporal_rgb_clip",
            "data": {"temporal_offsets_seconds_effective": [-60, -30, 0, 30, 60]},
        }
    )

    assert input_spec["shape"] == [15, 128, 128]
    assert input_spec["clip_length"] == 5
    assert input_spec["target_frame_index"] == 2
    assert input_spec["target_channel_start"] == 6


def _abi_frame_offset_spec() -> ResearchProblemSpec:
    return ResearchProblemSpec(
        id="goes_abi_contrail_segmentation",
        version="v0",
        input_modes=("abi_16ch_plus_sza", "abi_16ch_plus_sza_temporal"),
        input_specs={
            "abi_16ch_plus_sza": {"mode": "abi_16ch_plus_sza", "shape": [17, 256, 256]},
            "abi_16ch_plus_sza_temporal": {
                "mode": "abi_16ch_plus_sza_temporal",
                "shape": [51, 256, 256],
                "layout": "channel_stacked_temporal_frames",
                "temporal_offset_field": "temporal_frame_offsets",
                "default_temporal_frame_offsets": [-1, 0, 1],
                "allowed_temporal_frame_offsets": [[0], [-1, 0, 1], [-2, -1, 0, 1, 2]],
                "base_channels_per_frame": 17,
                "target_frame_offset": 0,
            },
        },
        output_forms=("mask_logits",),
        output_specs={"mask_logits": {"form": "mask_logits", "shape": [1, 256, 256]}},
        losses=("bce_dice",),
        optimizers=("adamw",),
        sampling_policies=("sequential",),
        frame_selection_policies=("all_target_frames", "temporal_eligible_center"),
        input_mode_frame_selection_defaults={
            "abi_16ch_plus_sza": "all_target_frames",
            "abi_16ch_plus_sza_temporal": "temporal_eligible_center",
        },
        augmentation_policies=("none",),
        primary_metric="val/dice",
    )


def test_provider_declared_frame_offsets_are_validated_and_defaulted(tmp_path: Path):
    candidate = tmp_path / "candidate"
    _write_candidate(
        candidate,
        """
name: abi_temporal
research_problem: goes_abi_contrail_segmentation
input_mode: abi_16ch_plus_sza_temporal
output_form: mask_logits
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 1
  max_epochs: 1
""".strip(),
    )

    manifest = validate_candidate_directory(candidate, research_problem_registry=ResearchProblemSpecRegistry([_abi_frame_offset_spec()]))

    assert manifest.data.frame_selection_policy == "temporal_eligible_center"
    assert manifest.data.temporal_frame_offsets == [-1, 0, 1]


def test_provider_declared_frame_offsets_reject_non_allowlisted_windows(tmp_path: Path):
    candidate = tmp_path / "candidate"
    _write_candidate(
        candidate,
        """
name: abi_temporal_bad_window
research_problem: goes_abi_contrail_segmentation
input_mode: abi_16ch_plus_sza_temporal
output_form: mask_logits
data:
  temporal_frame_offsets: [0, 1]
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 1
  max_epochs: 1
""".strip(),
    )

    try:
        validate_candidate_directory(candidate, research_problem_registry=ResearchProblemSpecRegistry([_abi_frame_offset_spec()]))
    except ValueError as exc:
        assert "data.temporal_frame_offsets" in str(exc)
        assert "allowed" in str(exc)
    else:  # pragma: no cover - documents intended failure
        raise AssertionError("non-allowlisted temporal_frame_offsets were accepted")


def test_provider_declared_frame_offsets_update_smoke_input_spec_shape():
    spec = _abi_frame_offset_spec()

    input_spec = spec.build_input_spec(
        {
            "input_mode": "abi_16ch_plus_sza_temporal",
            "data": {"temporal_frame_offsets_effective": [-2, -1, 0, 1, 2]},
        }
    )

    assert input_spec["shape"] == [85, 256, 256]
    assert input_spec["frame_count"] == 5
    assert input_spec["temporal_frame_offsets"] == [-2, -1, 0, 1, 2]
    assert input_spec["target_frame_index"] == 2
    assert input_spec["target_channel_start"] == 34
