from pathlib import Path

import pytest
import yaml

from ml_autoresearch.candidates import CandidateValidationError, validate_candidate_directory


def write_valid_candidate(root: Path) -> Path:
    candidate = root / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        """
name: single_frame_unet_baseline
description: Tiny single-frame mask-only baseline for harness validation.
input_mode: single_frame_rgb
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
    assert manifest.input_mode == "single_frame_rgb"
    assert manifest.output_form == "mask_logits"
    assert manifest.training.loss == "bce_dice"
    assert manifest.training.optimizer == "adamw"
    assert manifest.training.learning_rate == pytest.approx(0.001)
    assert manifest.training.batch_size == 2
    assert manifest.training.max_epochs == 1
    assert manifest.data.sampling_policy == "sequential"
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


def test_invalid_augmentation_policy_is_rejected(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "manifest.yaml").write_text(
        """
name: broken
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


def test_candidate_manifest_accepts_sampling_policy(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "manifest.yaml").write_text(
        """
name: shuffled
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


def test_invalid_sampling_policy_is_rejected(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "manifest.yaml").write_text(
        """
name: broken
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


def test_unknown_contract_values_are_rejected(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "manifest.yaml").write_text(
        """
name: broken
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


def test_training_scalar_ranges_are_checked(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "manifest.yaml").write_text(
        """
name: broken
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "boundary"),
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


@pytest.mark.parametrize("weight", [-0.01, 1.01])
def test_auxiliary_target_weight_bounds_are_checked(tmp_path: Path, weight: float):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "manifest.yaml").write_text(
        f"""
name: broken_aux
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
