import json
import subprocess
from pathlib import Path

import pytest

from ml_autoresearch.execution import DockerBackend, NativeBackend
from ml_autoresearch.runs import RunStatus, submit_candidate


VALID_MODEL = """
import torch
from torch import nn

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1)
    def forward(self, x):
        return {"mask_logits": self.conv(x)}

def build_model(input_spec, output_spec):
    return Tiny()
""".strip() + "\n"


def write_candidate(root: Path, model_py: str) -> Path:
    candidate = root / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        """
name: smoke_candidate
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
    (candidate / "model.py").write_text(model_py)
    return candidate


def test_native_backend_smoke_test_preserves_existing_outputs(tmp_path: Path):
    candidate = write_candidate(tmp_path, VALID_MODEL)

    run = submit_candidate(candidate, tmp_path / "runs", backend=NativeBackend())

    assert run.status == RunStatus.ACCEPTED
    metadata = json.loads((run.run_dir / "run_metadata.json").read_text())
    assert metadata["execution_backend"] == {"name": "native", "developer_unsafe": True}
    assert (run.run_dir / "outputs" / "model_summary.json").exists()
    assert "Smoke test accepted" in (run.run_dir / "outputs" / "logs" / "smoke_test.log").read_text()


def test_docker_backend_constructs_structurally_contained_smoke_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_dir = tmp_path / "runs" / "run_1"
    (run_dir / "candidate").mkdir(parents=True)
    (run_dir / "outputs" / "logs").mkdir(parents=True)
    (run_dir / "scratch").mkdir()
    (run_dir / "resolved_manifest.yaml").write_text("name: x\n")
    (run_dir / "run_metadata.json").write_text("{}\n")

    calls: list[list[str]] = []

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = DockerBackend("custom:tag").smoke_test(run_dir)

    assert result.backend == "docker"
    assert result.docker_image == "custom:tag"
    assert calls[0] == ["docker", "image", "inspect", "custom:tag"]
    docker_run = calls[1]
    assert docker_run[:5] == ["docker", "run", "--rm", "--network", "none"]
    assert "custom:tag" in docker_run
    assert "--entrypoint" in docker_run
    assert docker_run[docker_run.index("--entrypoint") + 1] == "python"
    assert docker_run[-2:] == ["-m", "ml_autoresearch.container_smoke"]
    joined = "\n".join(docker_run)
    assert f"type=bind,src={run_dir / 'candidate'},dst=/candidate,readonly" in joined
    assert f"type=bind,src={run_dir / 'resolved_manifest.yaml'},dst=/resolved_manifest.yaml,readonly" in joined
    assert f"type=bind,src={run_dir / 'run_metadata.json'},dst=/run_metadata.json,readonly" in joined
    assert f"type=bind,src={run_dir / 'outputs'},dst=/outputs" in joined
    assert f"type=bind,src={run_dir / 'scratch'},dst=/scratch" in joined
    assert "/var/run/docker.sock" not in joined


def test_docker_backend_missing_image_fails_with_manual_build_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fake_run(command, check, capture_output, text):
        raise subprocess.CalledProcessError(1, command, stderr="No such image")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="docker build -t ml-autoresearch-runner:local ."):
        DockerBackend().smoke_test(tmp_path)
