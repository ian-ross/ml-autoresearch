from pathlib import Path

import yaml

from ml_autoresearch.candidates import validate_candidate_directory
from ml_autoresearch.research_problems import ResearchProblemSpec, ResearchProblemSpecRegistry


def _single_fake_registry() -> ResearchProblemSpecRegistry:
    spec = ResearchProblemSpec(
        id="tiny_problem",
        version="v1",
        input_modes=("single_frame_rgb",),
        input_specs={"single_frame_rgb": {"mode": "single_frame_rgb", "shape": [3, 128, 128]}},
        output_forms=("mask_logits",),
        output_specs={"mask_logits": {"form": "mask_logits", "shape": [1, 128, 128]}},
        losses=("bce_dice",),
        auxiliary_losses=("weighted_bce",),
        optimizers=("adamw",),
        sampling_policies=("sequential",),
        augmentation_policies=("none",),
        primary_metric="val/dice",
    )
    return ResearchProblemSpecRegistry((spec,), active_id=spec.id)


def test_sample_candidate_fixture_satisfies_issue_1_contract(tmp_path: Path):
    fixture = Path("tests/fixtures/candidates/single_frame_unet_baseline")
    copied = tmp_path / "single_frame_unet_baseline"
    copied.mkdir()
    (copied / "manifest.yaml").write_text((fixture / "manifest.yaml").read_text())
    (copied / "model.py").write_text((fixture / "model.py").read_text())
    manifest_text = yaml.safe_load((copied / "manifest.yaml").read_text())
    manifest_text["research_problem"] = "tiny_problem"
    (copied / "manifest.yaml").write_text(yaml.safe_dump(manifest_text, sort_keys=False) + "\n")

    manifest = validate_candidate_directory(copied, research_problem_registry=_single_fake_registry())

    assert manifest.name == "single_frame_unet_baseline"
    assert manifest.input_mode == "single_frame_rgb"
    assert manifest.output_form == "mask_logits"
