import json
import re
from pathlib import Path

import yaml

from ml_autoresearch.runs import RunStatus, submit_candidate


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
        "import torch\n"
        "from torch import nn\n"
        "class Tiny(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.conv = nn.Conv2d(3, 1, kernel_size=1)\n"
        "    def forward(self, x):\n"
        "        return self.conv(x)\n"
        "def build_model(input_spec, output_spec):\n"
        "    return Tiny()\n"
    )
    return candidate


def test_submit_candidate_creates_accepted_run_directory(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    runs_root = tmp_path / "runs"

    run = submit_candidate(candidate, runs_root)

    assert run.status == RunStatus.ACCEPTED
    assert re.fullmatch(r"run_\d{8}_\d{6}_[0-9a-f]{6}", run.run_id)
    run_dir = runs_root / run.run_id
    assert run.run_dir == run_dir
    assert (run_dir / "candidate" / "manifest.yaml").read_text() == (candidate / "manifest.yaml").read_text()
    assert (run_dir / "candidate" / "model.py").read_text() == (candidate / "model.py").read_text()
    assert (run_dir / "outputs" / "logs" / "validation.log").exists()
    assert set(child.name for child in run_dir.iterdir()) == {"candidate", "outputs", "resolved_manifest.yaml", "run_metadata.json"}

    resolved = yaml.safe_load((run_dir / "resolved_manifest.yaml").read_text())
    assert resolved["name"] == "single_frame_unet_baseline"
    assert resolved["description"] == "Tiny single-frame mask-only baseline for harness validation."
    assert resolved["input_mode"] == "single_frame_rgb"
    assert resolved["output_form"] == "mask_logits"
    assert resolved["data"]["sampling_policy"] == "sequential"
    assert resolved["training"]["loss"] == "bce_dice"

    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["run_id"] == run.run_id
    assert metadata["status"] == "accepted"
    assert metadata["candidate_source"]["path"] == str(candidate.resolve())
    assert metadata["rejection_reason"] is None
    assert metadata["created_at"]
    assert metadata["updated_at"]


def test_submit_candidate_creates_unique_run_ids(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    runs_root = tmp_path / "runs"

    first = submit_candidate(candidate, runs_root)
    second = submit_candidate(candidate, runs_root)

    assert first.run_id != second.run_id
    assert first.run_dir.exists()
    assert second.run_dir.exists()


def test_submit_candidate_records_rejected_run(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "train.sh").write_text("echo nope\n")
    runs_root = tmp_path / "runs"

    run = submit_candidate(candidate, runs_root)

    assert run.status == RunStatus.REJECTED
    assert run.rejection_reason
    assert "forbidden" in run.rejection_reason
    metadata = json.loads((run.run_dir / "run_metadata.json").read_text())
    assert metadata["status"] == "rejected"
    assert metadata["rejection_reason"] == run.rejection_reason
    assert (run.run_dir / "outputs" / "logs" / "validation.log").read_text()
    assert not (run.run_dir / "candidate" / "train.sh").exists()
