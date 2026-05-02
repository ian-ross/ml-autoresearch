import json
import subprocess
import sys
from pathlib import Path


def write_valid_candidate(root: Path) -> Path:
    candidate = root / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        """
name: cli_candidate
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
        "from torch import nn\n"
        "class Tiny(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.conv = nn.Conv2d(3, 1, kernel_size=1)\n"
        "    def forward(self, x):\n"
        "        return {'mask_logits': self.conv(x)}\n"
        "def build_model(input_spec, output_spec):\n"
        "    return Tiny()\n"
    )
    return candidate


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ml_autoresearch.cli", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_submit_candidate_cli_creates_run_and_prints_json(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    runs_root = tmp_path / "runs"

    completed = run_cli("submit-candidate", "--candidate", str(candidate), "--runs-root", str(runs_root))

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload["status"] == "accepted"
    assert payload["run_id"].startswith("run_")
    assert (runs_root / payload["run_id"] / "run_metadata.json").exists()
    assert completed.stderr == ""


def test_submit_candidate_cli_exits_nonzero_for_rejected_candidate(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    (candidate / "weights.pt").write_text("nope\n")
    runs_root = tmp_path / "runs"

    completed = run_cli("submit-candidate", "--candidate", str(candidate), "--runs-root", str(runs_root))

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["status"] == "rejected"
    assert "forbidden" in payload["rejection_reason"]
    assert (runs_root / payload["run_id"] / "run_metadata.json").exists()
