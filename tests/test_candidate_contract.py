from pathlib import Path

import pytest

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
