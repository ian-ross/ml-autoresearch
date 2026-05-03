import json
import os
import subprocess
from pathlib import Path

import pytest

from ml_autoresearch.execution import DockerBackend, NativeBackend, OperationResult
from ml_autoresearch.runs import RunStatus, run_candidate_with_synthetic_fixture


VALID_MODEL = """
from torch import nn
class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=1)
    def forward(self, x):
        return {'mask_logits': self.conv(x)}
def build_model(input_spec, output_spec):
    return Tiny()
""".strip() + "\n"


def write_candidate(root: Path) -> Path:
    candidate = root / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        """
name: synthetic_candidate
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
    (candidate / "model.py").write_text(VALID_MODEL)
    return candidate


def test_native_backend_synthetic_training_preserves_existing_outputs(tmp_path: Path):
    candidate = write_candidate(tmp_path)

    run = run_candidate_with_synthetic_fixture(candidate, tmp_path / "runs", backend=NativeBackend())

    assert run.status == RunStatus.COMPLETED
    assert (run.run_dir / "outputs" / "metrics.jsonl").exists()
    assert (run.run_dir / "outputs" / "final_metrics.json").exists()
    assert (run.run_dir / "outputs" / "logs" / "training.log").exists()
    assert (run.run_dir / "outputs" / "prediction_samples").is_dir()
    metadata = json.loads((run.run_dir / "run_metadata.json").read_text())
    assert metadata["execution_backend"] == {"name": "native", "developer_unsafe": True}


def test_docker_backend_constructs_structurally_contained_synthetic_training_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
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

    result = DockerBackend("custom:tag").train_synthetic(run_dir, max_prediction_samples=3)

    assert result.backend == "docker"
    assert result.operation == "train_synthetic"
    assert result.docker_image == "custom:tag"
    assert calls[0] == ["docker", "image", "inspect", "custom:tag"]
    docker_run = calls[1]
    assert docker_run[:5] == ["docker", "run", "--rm", "--network", "none"]
    assert "--user" in docker_run
    assert docker_run[docker_run.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert "custom:tag" in docker_run
    assert "TMPDIR=/scratch" in docker_run
    assert "TORCHINDUCTOR_CACHE_DIR=/scratch/torchinductor" in docker_run
    assert docker_run[-4:] == ["-m", "ml_autoresearch.container_runner", "train-synthetic", "--max-prediction-samples=3"]
    joined = "\n".join(docker_run)
    assert f"{run_dir / 'candidate'}:/candidate:ro,z" in joined
    assert f"{run_dir / 'resolved_manifest.yaml'}:/resolved_manifest.yaml:ro,z" in joined
    assert f"{run_dir / 'run_metadata.json'}:/run_metadata.json:ro,z" in joined
    assert f"{run_dir / 'outputs'}:/outputs:z" in joined
    assert f"{run_dir / 'scratch'}:/scratch:z" in joined
    assert "/var/run/docker.sock" not in joined


class NoArtifactBackend:
    name = "no_artifact"

    def smoke_test(self, run_dir: str | Path) -> OperationResult:
        (Path(run_dir) / "outputs" / "model_summary.json").write_text("{}\n")
        (Path(run_dir) / "outputs" / "logs" / "smoke_test.log").write_text("ok\n")
        return OperationResult(backend=self.name, operation="smoke_test")

    def train_synthetic(self, run_dir: str | Path, *, max_prediction_samples: int = 2) -> OperationResult:
        return OperationResult(backend=self.name, operation="train_synthetic")


def test_host_harness_marks_training_failed_when_backend_omits_required_synthetic_outputs(tmp_path: Path):
    candidate = write_candidate(tmp_path)

    run = run_candidate_with_synthetic_fixture(candidate, tmp_path / "runs", backend=NoArtifactBackend())

    assert run.status == RunStatus.FAILED
    assert "required synthetic training artifact is missing" in (run.rejection_reason or "")
    metadata = json.loads((run.run_dir / "run_metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert "required synthetic training artifact is missing" in metadata["training_failure_reason"]


def test_docker_backend_constructs_gvccs_training_command_with_read_only_data_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir = tmp_path / "runs" / "run_1"
    data_root = tmp_path / "gvccs"
    (run_dir / "candidate").mkdir(parents=True)
    (run_dir / "outputs" / "logs").mkdir(parents=True)
    (run_dir / "scratch").mkdir()
    (run_dir / "resolved_manifest.yaml").write_text("name: x\n")
    (run_dir / "run_metadata.json").write_text("{}\n")
    data_root.mkdir()

    calls: list[list[str]] = []

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = DockerBackend("custom:tag").train_gvccs(run_dir, data_root, max_samples=4, max_prediction_samples=1)

    assert result.backend == "docker"
    assert result.operation == "train_gvccs"
    assert calls[0] == ["docker", "image", "inspect", "custom:tag"]
    docker_run = calls[1]
    assert docker_run[-5:] == [
        "-m",
        "ml_autoresearch.container_runner",
        "train-gvccs",
        "--max-samples=4",
        "--max-prediction-samples=1",
    ]
    joined = "\n".join(docker_run)
    assert f"{data_root}:/data:ro,z" in joined
    assert f"{run_dir / 'candidate'}:/candidate:ro,z" in joined
    assert f"{run_dir / 'outputs'}:/outputs:z" in joined
    assert "/data" in joined
    assert f"{data_root}:/data:z" not in joined


def test_docker_backend_rejects_missing_or_file_gvccs_data_root_before_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[list[str]] = []

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    backend = DockerBackend("custom:tag")
    with pytest.raises(RuntimeError, match="GVCCS data root does not exist"):
        backend.train_gvccs(tmp_path / "run", tmp_path / "missing")

    file_root = tmp_path / "not-a-dir"
    file_root.write_text("x")
    with pytest.raises(RuntimeError, match="GVCCS data root is not a directory"):
        backend.train_gvccs(tmp_path / "run", file_root)

    assert calls == []
