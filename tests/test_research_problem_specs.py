import pytest

from ml_autoresearch.research_problems import (
    DEFAULT_RESEARCH_PROBLEM_ID,
    ResearchProblemSpec,
    ResearchProblemSpecRegistry,
    UnknownResearchProblemSpecError,
    get_default_research_problem_spec,
    get_research_problem_spec,
)


def test_default_registry_exposes_ground_camera_contrail_detection_spec() -> None:
    spec = get_default_research_problem_spec()

    assert spec.id == DEFAULT_RESEARCH_PROBLEM_ID
    assert spec.id == "ground_camera_contrail_detection"
    assert spec.version == "v0"
    assert spec.input_modes == ("single_frame_rgb", "centered_temporal_rgb_clip")
    assert spec.output_forms == ("mask_logits",)
    assert spec.auxiliary_targets == ("line", "boundary")
    assert spec.losses == ("bce_dice",)
    assert spec.auxiliary_losses == ("weighted_bce",)
    assert spec.optimizers == ("adamw",)
    assert spec.sampling_policies == ("sequential", "deterministic_shuffle")
    assert spec.frame_selection_policies == ("all_target_frames", "temporal_eligible_center")
    assert spec.input_mode_frame_selection_defaults == {
        "single_frame_rgb": "all_target_frames",
        "centered_temporal_rgb_clip": "temporal_eligible_center",
    }
    assert spec.augmentation_policies == (
        "none",
        "light_geometric",
        "light_photometric",
        "light_combined",
    )
    assert spec.primary_metric == "val/dice"

    assert get_research_problem_spec("ground_camera_contrail_detection") == spec


def test_registry_accepts_tiny_fake_research_problem_spec() -> None:
    fake = ResearchProblemSpec(
        id="fake_segmentation_problem",
        version="test-v0",
        input_modes=("tiny_rgb",),
        output_forms=("tiny_mask_logits",),
        auxiliary_targets=(),
        losses=("tiny_loss",),
        auxiliary_losses=(),
        optimizers=("sgd",),
        sampling_policies=("sequential",),
        augmentation_policies=("none",),
        primary_metric="val/tiny_score",
    )
    registry = ResearchProblemSpecRegistry(default_id=fake.id)

    registry.register(fake)

    assert registry.get("fake_segmentation_problem") == fake
    assert registry.default() == fake


def test_unknown_research_problem_spec_is_rejected() -> None:
    registry = ResearchProblemSpecRegistry(default_id="missing")

    with pytest.raises(UnknownResearchProblemSpecError, match="missing"):
        registry.get("missing")


def test_ground_camera_contrail_detection_builds_smoke_specs_from_manifest() -> None:
    spec = get_default_research_problem_spec()
    resolved_manifest = {
        "input_mode": "single_frame_rgb",
        "output_form": "mask_logits",
        "auxiliary_targets": [
            {"name": "line", "output": "line_logits"},
            {"name": "boundary", "output": "boundary_logits"},
        ],
    }

    assert spec.build_input_spec(resolved_manifest) == {"mode": "single_frame_rgb", "shape": [3, 128, 128]}
    assert spec.build_input_spec({"input_mode": "centered_temporal_rgb_clip"}) == {
        "mode": "centered_temporal_rgb_clip",
        "shape": [9, 128, 128],
        "clip_length": 3,
        "frame_stride": 1,
        "layout": "channel_stacked_rgb",
        "target_frame": "center",
    }
    assert spec.build_output_spec(resolved_manifest) == {
        "form": "mask_logits",
        "shape": [1, 128, 128],
        "auxiliary_outputs": [
            {"target": "line", "name": "line_logits", "shape": [1, 128, 128]},
            {"target": "boundary", "name": "boundary_logits", "shape": [1, 128, 128]},
        ],
    }


def test_fake_research_problem_spec_builds_distinct_smoke_specs() -> None:
    fake = ResearchProblemSpec(
        id="fake_temporal_problem",
        version="test-v0",
        input_modes=("tiny_clip_rgb",),
        input_specs={"tiny_clip_rgb": {"mode": "tiny_clip_rgb", "shape": [5, 3, 16, 16]}},
        output_forms=("tiny_mask_logits",),
        output_specs={"tiny_mask_logits": {"form": "tiny_mask_logits", "shape": [1, 16, 16]}},
        auxiliary_targets=("edge",),
        auxiliary_outputs={"edge": "edge_logits"},
        auxiliary_output_shapes={"edge": [1, 16, 16]},
        losses=("tiny_loss",),
        auxiliary_losses=("tiny_aux_loss",),
        optimizers=("sgd",),
        sampling_policies=("sequential",),
        augmentation_policies=("none",),
        primary_metric="val/tiny_score",
    )
    resolved_manifest = {
        "input_mode": "tiny_clip_rgb",
        "output_form": "tiny_mask_logits",
        "auxiliary_targets": [{"name": "edge", "output": "edge_logits"}],
    }

    assert fake.build_input_spec(resolved_manifest) == {"mode": "tiny_clip_rgb", "shape": [5, 3, 16, 16]}
    assert fake.build_output_spec(resolved_manifest) == {
        "form": "tiny_mask_logits",
        "shape": [1, 16, 16],
        "auxiliary_outputs": [{"target": "edge", "name": "edge_logits", "shape": [1, 16, 16]}],
    }
