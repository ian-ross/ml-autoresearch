import pytest

from ml_autoresearch.research_problems import (
    ResearchProblemSpec,
    ResearchProblemSpecRegistry,
    UnknownResearchProblemSpecError,
)


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
    registry = ResearchProblemSpecRegistry(active_id=fake.id)

    registry.register(fake)

    assert registry.get("fake_segmentation_problem") == fake
    assert registry.active() == fake


def test_unknown_research_problem_spec_is_rejected() -> None:
    registry = ResearchProblemSpecRegistry(active_id="missing")

    with pytest.raises(UnknownResearchProblemSpecError, match="missing"):
        registry.get("missing")


def test_no_active_research_problem_spec_is_rejected() -> None:
    registry = ResearchProblemSpecRegistry()

    with pytest.raises(UnknownResearchProblemSpecError, match="no active research problem spec configured"):
        registry.active()


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
