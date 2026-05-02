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


def test_submit_candidate_cli_accepts_explicit_native_backend(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    runs_root = tmp_path / "runs"

    completed = run_cli(
        "submit-candidate", "--candidate", str(candidate), "--runs-root", str(runs_root), "--backend", "native"
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    metadata = json.loads((runs_root / payload["run_id"] / "run_metadata.json").read_text())
    assert metadata["execution_backend"] == {"name": "native", "developer_unsafe": True}


def test_submit_candidate_cli_accepts_docker_backend_and_image(tmp_path: Path):
    completed = run_cli(
        "submit-candidate",
        "--candidate",
        str(tmp_path / "missing"),
        "--runs-root",
        str(tmp_path / "runs"),
        "--backend",
        "docker",
        "--docker-image",
        "custom:tag",
    )

    # Missing candidate rejects during validation before Docker is invoked, but the
    # Harness still records the selected Candidate Execution Boundary.
    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    metadata = json.loads((tmp_path / "runs" / payload["run_id"] / "run_metadata.json").read_text())
    assert metadata["execution_backend"] == {"name": "docker", "docker_image": "custom:tag"}
