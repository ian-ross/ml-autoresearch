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
    assert spec.input_modes == ("single_frame_rgb",)
    assert spec.output_forms == ("mask_logits",)
    assert spec.auxiliary_targets == ("line", "boundary")
    assert spec.losses == ("bce_dice",)
    assert spec.auxiliary_losses == ("weighted_bce",)
    assert spec.optimizers == ("adamw",)
    assert spec.sampling_policies == ("sequential", "deterministic_shuffle")
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
