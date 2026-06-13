from pathlib import Path

import pytest
import yaml

from ml_autoresearch.candidates import CandidateValidationError, validate_candidate_directory as _validate_candidate_directory
from ml_autoresearch.research_problems import (
    ResearchProblemSpec,
    ResearchProblemSpecRegistry,
)

DEFAULT_RESEARCH_PROBLEM_ID = "ground_camera_contrail_detection"


def _single_fake_registry() -> ResearchProblemSpecRegistry:
    spec = ResearchProblemSpec(
        id="tiny_problem",
        version="v1",
        input_modes=("single_frame_rgb",),
        input_specs={"single_frame_rgb": {"mode": "single_frame_rgb", "shape": [3, 128, 128]}},
        output_forms=("mask_logits",),
        output_specs={"mask_logits": {"form": "mask_logits", "shape": [1, 128, 128]}},
        auxiliary_targets=("line", "boundary"),
        auxiliary_outputs={"line": "line_logits", "boundary": "boundary_logits"},
        losses=("bce_dice",),
        auxiliary_losses=("weighted_bce",),
        optimizers=("adamw",),
        sampling_policies=("sequential", "deterministic_shuffle"),
        augmentation_policies=("none", "light_combined", "light_geometric", "light_photometric"),
        primary_metric="val/dice",
    )
    return ResearchProblemSpecRegistry((spec,), active_id=spec.id)



def validate_candidate_directory(candidate_dir, **kwargs):
    from research_problem_helpers import gvccs_registry

    kwargs.setdefault("research_problem_registry", gvccs_registry())
    return _validate_candidate_directory(candidate_dir, **kwargs)


def write_valid_candidate(root: Path) -> Path:
    return write_valid_candidate_without_research_problem(root, include_research_problem=True)


def write_valid_candidate_without_research_problem(root: Path, *, include_research_problem: bool = False) -> Path:
    candidate = root / "candidate"
    candidate.mkdir()
    manifest = {
        "name": "single_frame_unet_baseline",
        "description": "Tiny single-frame mask-only baseline for harness validation.",
        "input_mode": "single_frame_rgb",
        "output_form": "mask_logits",
        "training": {
            "loss": "bce_dice",
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "batch_size": 2,
            "max_epochs": 1,
        },
    }
    if include_research_problem:
        manifest["research_problem"] = DEFAULT_RESEARCH_PROBLEM_ID
    (candidate / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False).strip() + "\n")
    (candidate / "model.py").write_text(
        "def build_model(input_spec, output_spec):\n"
        "    raise NotImplementedError('Issue 1 does not import this file')\n"
    )
    return candidate


def write_valid_proposal(path: Path) -> None:
    path.write_text(
        """\
# Proposal

## Hypothesis
A simpler model should improve recall by reducing overfitting.

## Comparison Target
Compare against prior single-frame baseline by final val/dice.

## Expected Effect
Expected to increase val/dice and reduce false positives.

## Implementation Sketch
Adjust backbone depth and add residual skip.

## Contract Features Used
single_frame_rgb input, mask_logits output, bce_dice loss.

## Budget Requested
2h of compute plus one training run.

## Success Criteria
Increase val/dice by 0.05.

## Fallback/Next Decision
If failed, keep current best backbone and move to augmentation policy sweep.
"""
    )


def test_valid_candidate_directory_returns_normalized_manifest(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)

    manifest = validate_candidate_directory(candidate)

    assert manifest.name == "single_frame_unet_baseline"
    assert manifest.description == "Tiny single-frame mask-only baseline for harness validation."
    assert manifest.research_problem == DEFAULT_RESEARCH_PROBLEM_ID
    assert manifest.input_mode == "single_frame_rgb"
    assert manifest.output_form == "mask_logits"
    assert manifest.training.loss == "bce_dice"
    assert manifest.training.optimizer == "adamw"
    assert manifest.training.learning_rate == pytest.approx(0.001)
    assert manifest.training.batch_size == 2
    assert manifest.training.max_epochs == 1
    assert manifest.data.sampling_policy == "sequential"
    assert manifest.data.frame_selection_policy == "all_target_frames"
    assert manifest.auxiliary_targets == []
    assert manifest.data.augmentation_policy == "none"


def test_candidate_directory_accepts_valid_proposal_in_proposal_required_mode(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    write_valid_proposal(candidate / "PROPOSAL.md")

    manifest = validate_candidate_directory(candidate, require_proposal=True)

    assert manifest.name == "single_frame_unet_baseline"


def test_candidate_directory_rejects_missing_proposal_in_proposal_required_mode(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)

    with pytest.raises(CandidateValidationError, match="autonomous-mode requires"):
        validate_candidate_directory(candidate, require_proposal=True)


def test_candidate_directory_rejects_missing_readme_only_when_required(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)

    manifest = validate_candidate_directory(candidate)

    assert manifest.name == "single_frame_unet_baseline"
    with pytest.raises(CandidateValidationError, match="README.md"):
        validate_candidate_directory(candidate, require_readme=True)


def test_candidate_directory_accepts_readme_in_readme_required_mode(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "README.md").write_text("# Candidate\n\nStatic submission notes.\n")

    manifest = validate_candidate_directory(candidate, require_readme=True)

    assert manifest.name == "single_frame_unet_baseline"


def test_candidate_directory_rejects_proposal_missing_required_sections(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    write_valid_proposal(candidate / "PROPOSAL.md")
    (candidate / "PROPOSAL.md").write_text(
        """\
# Proposal

## Hypothesis
Try a new architecture for baseline.

## Comparison Target
Compare against previous run.
"""
    )

    with pytest.raises(CandidateValidationError) as excinfo:
        validate_candidate_directory(candidate, require_proposal=True)

    message = str(excinfo.value)
    assert "Expected Effect" in message
    assert "Implementation Sketch" in message
    assert "Contract Features Used" in message


def test_candidate_proposal_sections_require_exact_matches_not_substring_matches(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "PROPOSAL.md").write_text(
        """\
## Hypothesis about Comparison Target Effects
A hypothesis that combines multiple decision points.

## Comparison Target
This is a concise summary.
"""
    )

    with pytest.raises(CandidateValidationError) as excinfo:
        validate_candidate_directory(candidate, require_proposal=True)

    message = str(excinfo.value)
    assert "Expected Effect" in message
    assert "Implementation Sketch" in message
    assert "Contract Features Used" in message
    assert "Fallback Next Decision" in message


def test_missing_required_manifest_field_is_rejected(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "manifest.yaml").write_text(
        """
name: broken
output_form: mask_logits
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )

    with pytest.raises(CandidateValidationError, match="input_mode"):
        validate_candidate_directory(candidate)


def test_candidate_manifest_accepts_augmentation_policy(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "manifest.yaml").write_text(
        """
name: light_augmented
research_problem: ground_camera_contrail_detection
input_mode: single_frame_rgb
output_form: mask_logits
data:
  augmentation_policy: light_combined
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )

    manifest = validate_candidate_directory(candidate)

    assert manifest.data.augmentation_policy == "light_combined"


def test_candidate_manifest_accepts_explicit_default_research_problem(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    manifest = yaml.safe_load((candidate / "manifest.yaml").read_text())
    manifest["research_problem"] = "tiny_problem"
    (candidate / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False) + "\n")

    loaded = _validate_candidate_directory(candidate, research_problem_registry=_single_fake_registry())

    assert loaded.research_problem == "tiny_problem"


def test_candidate_manifest_resolves_missing_research_problem_from_singleton_registry(tmp_path: Path):
    candidate = write_valid_candidate_without_research_problem(tmp_path)
    registry = _single_fake_registry()

    loaded = _validate_candidate_directory(candidate, research_problem_registry=registry)

    assert loaded.research_problem == "tiny_problem"


def test_candidate_manifest_rejects_missing_research_problem_without_registry(tmp_path: Path):
    candidate = write_valid_candidate_without_research_problem(tmp_path)

    with pytest.raises(CandidateValidationError, match="research_problem is required"):
        _validate_candidate_directory(candidate, research_problem_registry=None)


def test_candidate_manifest_rejects_missing_research_problem_with_multiple_registered_specs(tmp_path: Path):
    single = _single_fake_registry().active()
    second = ResearchProblemSpec(
        id="second_problem",
        version="v1",
        input_modes=("single_frame_rgb",),
        input_specs={"single_frame_rgb": {"mode": "single_frame_rgb", "shape": [3, 128, 128]}},
        output_forms=("mask_logits",),
        output_specs={"mask_logits": {"form": "mask_logits", "shape": [1, 128, 128]}},
        auxiliary_targets=("line",),
        auxiliary_outputs={"line": "line_logits"},
        losses=("bce_dice",),
        auxiliary_losses=("weighted_bce",),
        optimizers=("adamw",),
        sampling_policies=("sequential",),
        augmentation_policies=("none",),
        primary_metric="val/dice",
    )
    registry = ResearchProblemSpecRegistry(active_id=single.id)
    registry.register(single)
    registry.register(second)

    candidate = write_valid_candidate_without_research_problem(tmp_path)

    with pytest.raises(CandidateValidationError, match="2 specs"):
        _validate_candidate_directory(candidate, research_problem_registry=registry)


def test_invalid_augmentation_policy_is_rejected(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "manifest.yaml").write_text(
        """
name: broken
research_problem: ground_camera_contrail_detection
input_mode: single_frame_rgb
output_form: mask_logits
data:
  augmentation_policy: candidate_custom_transform
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )

    with pytest.raises(CandidateValidationError) as excinfo:
        validate_candidate_directory(candidate)

    message = str(excinfo.value)
    assert "data.augmentation_policy" in message
    assert "candidate_custom_transform" in message
    assert DEFAULT_RESEARCH_PROBLEM_ID in message


def test_candidate_manifest_accepts_sampling_policy(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "manifest.yaml").write_text(
        """
name: shuffled
research_problem: ground_camera_contrail_detection
input_mode: single_frame_rgb
output_form: mask_logits
data:
  sampling_policy: deterministic_shuffle
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )

    manifest = validate_candidate_directory(candidate)

    assert manifest.data.sampling_policy == "deterministic_shuffle"


def test_candidate_manifest_accepts_temporal_eligible_frame_selection_for_single_frame_control(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "manifest.yaml").write_text(
        """
name: matched_single_frame_control
research_problem: ground_camera_contrail_detection
input_mode: single_frame_rgb
output_form: mask_logits
data:
  frame_selection_policy: temporal_eligible_center
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )

    manifest = validate_candidate_directory(candidate)

    assert manifest.data.frame_selection_policy == "temporal_eligible_center"


def test_temporal_input_defaults_to_temporal_eligible_frame_selection(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "manifest.yaml").write_text(
        """
name: temporal_candidate
research_problem: ground_camera_contrail_detection
input_mode: centered_temporal_rgb_clip
output_form: mask_logits
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )

    manifest = validate_candidate_directory(candidate)

    assert manifest.data.frame_selection_policy == "temporal_eligible_center"


def test_temporal_input_rejects_all_target_frame_selection(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "manifest.yaml").write_text(
        """
name: impossible_temporal_candidate
research_problem: ground_camera_contrail_detection
input_mode: centered_temporal_rgb_clip
output_form: mask_logits
data:
  frame_selection_policy: all_target_frames
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )

    with pytest.raises(CandidateValidationError) as excinfo:
        validate_candidate_directory(candidate)

    assert "data.frame_selection_policy" in str(excinfo.value)
    assert "temporal_eligible_center" in str(excinfo.value)


def test_invalid_sampling_policy_is_rejected(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "manifest.yaml").write_text(
        """
name: broken
research_problem: ground_camera_contrail_detection
input_mode: single_frame_rgb
output_form: mask_logits
data:
  sampling_policy: custom_candidate_sampler
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )

    with pytest.raises(CandidateValidationError) as excinfo:
        validate_candidate_directory(candidate)

    message = str(excinfo.value)
    assert "data.sampling_policy" in message
    assert "custom_candidate_sampler" in message
    assert DEFAULT_RESEARCH_PROBLEM_ID in message


def test_invalid_frame_selection_policy_is_rejected(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "manifest.yaml").write_text(
        """
name: broken
research_problem: ground_camera_contrail_detection
input_mode: single_frame_rgb
output_form: mask_logits
data:
  frame_selection_policy: candidate_boundary_padding
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )

    with pytest.raises(CandidateValidationError) as excinfo:
        validate_candidate_directory(candidate)

    message = str(excinfo.value)
    assert "data.frame_selection_policy" in message
    assert "candidate_boundary_padding" in message
    assert DEFAULT_RESEARCH_PROBLEM_ID in message


def test_unknown_contract_values_are_rejected(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "manifest.yaml").write_text(
        """
name: broken
research_problem: ground_camera_contrail_detection
input_mode: arbitrary_loader
output_form: mask_logits
training:
  loss: custom_loss
  optimizer: made_up_optimizer
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )

    with pytest.raises(CandidateValidationError) as excinfo:
        validate_candidate_directory(candidate)

    message = str(excinfo.value)
    assert "input_mode" in message
    assert "loss" in message
    assert "optimizer" in message
    assert DEFAULT_RESEARCH_PROBLEM_ID in message


def test_candidate_manifest_validates_against_fake_research_problem_spec(tmp_path: Path):
    fake_spec = ResearchProblemSpec(
        id="tiny_segmentation",
        version="v1",
        input_modes=("tiny_rgb",),
        output_forms=("tiny_mask_logits",),
        auxiliary_targets=("edge",),
        auxiliary_outputs={"edge": "edge_logits"},
        losses=("tiny_loss",),
        auxiliary_losses=("tiny_aux_loss",),
        optimizers=("tiny_optimizer",),
        sampling_policies=("tiny_order",),
        augmentation_policies=("tiny_aug",),
        primary_metric="val/tiny_dice",
    )
    registry = ResearchProblemSpecRegistry((fake_spec,), active_id=fake_spec.id)
    candidate = write_valid_candidate(tmp_path)
    (candidate / "manifest.yaml").write_text(
        """
name: fake_problem_candidate
research_problem: tiny_segmentation
input_mode: tiny_rgb
output_form: tiny_mask_logits
auxiliary_targets:
  - name: edge
    output: edge_logits
    loss: tiny_aux_loss
    weight: 0.5
data:
  sampling_policy: tiny_order
  augmentation_policy: tiny_aug
training:
  loss: tiny_loss
  optimizer: tiny_optimizer
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )

    manifest = validate_candidate_directory(candidate, research_problem_registry=registry)

    assert manifest.research_problem == "tiny_segmentation"
    assert manifest.input_mode == "tiny_rgb"
    assert manifest.output_form == "tiny_mask_logits"
    assert manifest.auxiliary_targets[0].output == "edge_logits"


def test_fake_research_problem_spec_rejects_values_not_in_its_allowlists(tmp_path: Path):
    fake_spec = ResearchProblemSpec(
        id="tiny_segmentation",
        version="v1",
        input_modes=("tiny_rgb",),
        output_forms=("tiny_mask_logits",),
        auxiliary_targets=("edge",),
        auxiliary_outputs={"edge": "edge_logits"},
        losses=("tiny_loss",),
        auxiliary_losses=("tiny_aux_loss",),
        optimizers=("tiny_optimizer",),
        sampling_policies=("tiny_order",),
        augmentation_policies=("tiny_aug",),
        primary_metric="val/tiny_dice",
    )
    registry = ResearchProblemSpecRegistry((fake_spec,), active_id=fake_spec.id)
    candidate = write_valid_candidate(tmp_path)
    manifest = yaml.safe_load((candidate / "manifest.yaml").read_text())
    manifest["research_problem"] = "tiny_segmentation"
    (candidate / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    with pytest.raises(CandidateValidationError) as excinfo:
        validate_candidate_directory(candidate, research_problem_registry=registry)

    message = str(excinfo.value)
    assert "tiny_segmentation" in message
    assert "input_mode" in message
    assert "output_form" in message
    assert "training.loss" in message
    assert "training.optimizer" in message


def test_candidate_manifest_accepts_scheduler_and_early_stopping_policy(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    manifest = yaml.safe_load((candidate / "manifest.yaml").read_text())
    manifest["training"]["max_epochs"] = 8
    manifest["training"]["scheduler"] = {"policy": "reduce_on_plateau", "factor": 0.5, "patience": 2, "min_lr": 0.00001}
    manifest["training"]["early_stopping"] = {"enabled": True, "patience": 3, "min_delta": 0.001, "restore_best_checkpoint": True}
    (candidate / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    loaded = validate_candidate_directory(candidate)

    assert loaded.training.scheduler.policy == "reduce_on_plateau"
    assert loaded.training.early_stopping.enabled is True
    assert loaded.training.early_stopping.patience == 3


def test_candidate_manifest_rejects_unapproved_scheduler_policy(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    manifest = yaml.safe_load((candidate / "manifest.yaml").read_text())
    manifest["training"]["scheduler"] = {"policy": "candidate_custom_scheduler"}
    (candidate / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    with pytest.raises(CandidateValidationError) as excinfo:
        validate_candidate_directory(candidate)

    message = str(excinfo.value)
    assert "scheduler.policy" in message
    assert "candidate_custom_scheduler" in message


def test_training_scalar_ranges_are_checked(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "manifest.yaml").write_text(
        """
name: broken
research_problem: ground_camera_contrail_detection
input_mode: single_frame_rgb
output_form: mask_logits
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 99
  batch_size: 0
  max_epochs: 0
""".strip()
        + "\n"
    )

    with pytest.raises(CandidateValidationError) as excinfo:
        validate_candidate_directory(candidate)

    message = str(excinfo.value)
    assert "learning_rate" in message
    assert "batch_size" in message
    assert "max_epochs" in message


@pytest.mark.parametrize(
    "relative_path",
    [
        ".secret",
        "train.sh",
        "notebook.ipynb",
        "weights.pt",
        "weights.pth",
        "weights.ckpt",
        "candidate.zip",
        "data.csv",
        "config.json",
    ],
)
def test_forbidden_candidate_files_are_rejected(tmp_path: Path, relative_path: str):
    candidate = write_valid_candidate(tmp_path)
    (candidate / relative_path).write_text("not allowed\n")

    with pytest.raises(CandidateValidationError, match="forbidden"):
        validate_candidate_directory(candidate)


def test_python_helpers_and_readme_are_allowed(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "blocks.py").write_text("class Block: pass\n")
    (candidate / "README.md").write_text("# sample candidate\n")

    validate_candidate_directory(candidate)


def test_symlinks_are_rejected(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "linked.py").symlink_to(candidate / "model.py")

    with pytest.raises(CandidateValidationError, match="symlink"):
        validate_candidate_directory(candidate)


def test_candidate_manifest_cannot_request_data_paths_or_mounts(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "manifest.yaml").write_text(
        """
name: broken
research_problem: ground_camera_contrail_detection
input_mode: single_frame_rgb
output_form: mask_logits
data_root: /host/data
mounts:
  - /host/data:/data
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )

    with pytest.raises(CandidateValidationError) as excinfo:
        validate_candidate_directory(candidate)

    message = str(excinfo.value)
    assert "data_root" in message
    assert "mounts" in message


def test_candidate_manifest_accepts_line_auxiliary_target(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "manifest.yaml").write_text(
        """
name: line_aux
research_problem: ground_camera_contrail_detection
input_mode: single_frame_rgb
output_form: mask_logits
auxiliary_targets:
  - name: line
    output: line_logits
    loss: weighted_bce
    weight: 0.25
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )

    manifest = validate_candidate_directory(candidate)

    assert len(manifest.auxiliary_targets) == 1
    auxiliary = manifest.auxiliary_targets[0]
    assert auxiliary.name == "line"
    assert auxiliary.output == "line_logits"
    assert auxiliary.loss == "weighted_bce"
    assert auxiliary.weight == pytest.approx(0.25)


def test_candidate_manifest_accepts_boundary_auxiliary_target(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "manifest.yaml").write_text(
        """
name: boundary_aux
research_problem: ground_camera_contrail_detection
input_mode: single_frame_rgb
output_form: mask_logits
auxiliary_targets:
  - name: boundary
    output: boundary_logits
    loss: weighted_bce
    weight: 0.10
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )

    manifest = validate_candidate_directory(candidate)

    assert len(manifest.auxiliary_targets) == 1
    auxiliary = manifest.auxiliary_targets[0]
    assert auxiliary.name == "boundary"
    assert auxiliary.output == "boundary_logits"
    assert auxiliary.loss == "weighted_bce"
    assert auxiliary.weight == pytest.approx(0.10)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output", "wrong_logits"),
        ("loss", "bce_dice"),
    ],
)
def test_invalid_auxiliary_target_values_are_rejected(tmp_path: Path, field: str, value: str):
    candidate = write_valid_candidate(tmp_path)
    auxiliary = {"name": "line", "output": "line_logits", "loss": "weighted_bce", "weight": 0.25}
    auxiliary[field] = value
    (candidate / "manifest.yaml").write_text(
        """
name: broken_aux
research_problem: ground_camera_contrail_detection
input_mode: single_frame_rgb
output_form: mask_logits
auxiliary_targets:
  - name: {name}
    output: {output}
    loss: {loss}
    weight: {weight}
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".format(**auxiliary).strip()
        + "\n"
    )

    with pytest.raises(CandidateValidationError) as excinfo:
        validate_candidate_directory(candidate)

    message = str(excinfo.value)
    assert f"auxiliary_targets.0.{field}" in message
    assert value in message


def test_auxiliary_target_name_must_match_output(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "manifest.yaml").write_text(
        """
name: mismatched_aux
research_problem: ground_camera_contrail_detection
input_mode: single_frame_rgb
output_form: mask_logits
auxiliary_targets:
  - name: boundary
    output: line_logits
    loss: weighted_bce
    weight: 0.25
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )

    with pytest.raises(CandidateValidationError) as excinfo:
        validate_candidate_directory(candidate)

    assert "boundary auxiliary target must use boundary_logits" in str(excinfo.value)


@pytest.mark.parametrize("weight", [-0.01, 1.01])
def test_auxiliary_target_weight_bounds_are_checked(tmp_path: Path, weight: float):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "manifest.yaml").write_text(
        f"""
name: broken_aux
research_problem: ground_camera_contrail_detection
input_mode: single_frame_rgb
output_form: mask_logits
auxiliary_targets:
  - name: line
    output: line_logits
    loss: weighted_bce
    weight: {weight}
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )

    with pytest.raises(CandidateValidationError) as excinfo:
        validate_candidate_directory(candidate)

    assert "auxiliary_targets.0.weight" in str(excinfo.value)


def test_repair_candidate_requires_preserving_original_hypothesis_and_comparison_target(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    manifest = yaml.safe_load((candidate / "manifest.yaml").read_text())
    manifest["repair"] = {
        "original_proposal_id": "proposal-original",
        "original_candidate_id": "candidate-original",
        "motivating_run_id": "run_20260501_120000_abcdef",
        "failure_classification": "candidate_bug",
        "preserves_original_hypothesis": False,
        "preserves_comparison_target": True,
    }
    (candidate / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    with pytest.raises(CandidateValidationError, match="preserves_original_hypothesis"):
        validate_candidate_directory(candidate)
