import json
import os
import subprocess
from pathlib import Path

import pytest

from ml_autoresearch.execution import DockerBackend, NativeBackend, OperationResult
from ml_autoresearch.research_problems import ResearchProblemProviderConfig
from ml_autoresearch.runs import RunStatus
from research_problem_helpers import run_candidate_with_synthetic_fixture


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


def _provider_config_for_test(
    root: Path, *, data_root: Path | None = None
) -> tuple[ResearchProblemProviderConfig, Path]:
    package_root = root / "provider"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("")
    data_config: dict[str, object] = {}
    if data_root is not None:
        data_config["dataset_root"] = str(data_root)
    return (
        ResearchProblemProviderConfig(
            id="gvccs_like",
            package_root=package_root,
            provider_target="test_problem.provider:build_spec",
            expected_contract_version="v0",
            data_config=data_config,
        ),
        package_root,
    )


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

    provider_config, _package_root = _provider_config_for_test(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = DockerBackend("custom:tag").train_research_problem(run_dir, provider_config, max_prediction_samples=3)

    assert result.backend == "docker"
    assert result.operation == "train_research_problem"
    assert result.docker_image == "custom:tag"
    assert calls[0] == ["docker", "image", "inspect", "custom:tag"]
    assert calls[1] == ["docker", "info", "--format", "{{json .SecurityOptions}}"]
    docker_run = calls[2]
    assert docker_run[:3] == ["docker", "run", "--rm"]
    assert "--userns=host" in docker_run
    assert "--network" in docker_run
    assert docker_run[docker_run.index("--network") + 1] == "none"
    assert "--user" in docker_run
    assert docker_run[docker_run.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert "custom:tag" in docker_run
    assert "TMPDIR=/scratch" in docker_run
    assert "TORCHINDUCTOR_CACHE_DIR=/scratch/torchinductor" in docker_run
    assert "ML_AUTORESEARCH_TIMEOUT_SENTINEL=/scratch/ml_autoresearch_timeout_requested" in docker_run
    assert "--read-only" in docker_run
    assert docker_run[docker_run.index("--cap-drop") + 1] == "ALL"
    assert docker_run[docker_run.index("--security-opt") + 1] == "no-new-privileges"
    assert docker_run[docker_run.index("--pids-limit") + 1] == "512"
    assert docker_run[docker_run.index("--memory") + 1] == "4g"
    assert docker_run[docker_run.index("--cpus") + 1] == "2"
    assert "--privileged" not in docker_run
    assert "--gpus" not in docker_run
    provider_json_arg = next(arg for arg in docker_run if arg.startswith("--provider-config-json="))
    payload = json.loads(provider_json_arg.split("=", 1)[1])
    assert payload["id"] == "gvccs_like"
    assert docker_run[-6:] == [
        "-m",
        "ml_autoresearch.container_runner",
        "train-research-problem",
        provider_json_arg,
        "--max-prediction-samples=3",
        "--prediction-sample-policy=first_n",
    ]
    joined = "\n".join(docker_run)
    assert "--provider-config-json=" in joined
    assert f"{run_dir / 'candidate'}:/candidate:ro,z" in joined
    assert f"{run_dir / 'resolved_manifest.yaml'}:/resolved_manifest.yaml:ro,z" in joined
    assert f"{run_dir / 'run_metadata.json'}:/run_metadata.json:ro,z" in joined
    assert f"{run_dir / 'outputs'}:/outputs:rw,z" in joined
    assert f"{_package_root.resolve(strict=True)}:/research-problem:ro,z" in joined
    assert "type=tmpfs,destination=/scratch,tmpfs-size=2g,tmpfs-mode=1777" in joined
    assert "/var/run/docker.sock" not in joined


class NoArtifactBackend:
    name = "no_artifact"

    def smoke_test(self, run_dir: str | Path) -> OperationResult:
        (Path(run_dir) / "outputs" / "model_summary.json").write_text("{}\n")
        (Path(run_dir) / "outputs" / "logs" / "smoke_test.log").write_text("ok\n")
        return OperationResult(backend=self.name, operation="smoke_test")

    def train_research_problem(
        self,
        run_dir: str | Path,
        provider_config,
        *,
        max_samples: int | None = None,
        max_prediction_samples: int = 2,
        prediction_sample_policy: str = "first_n",
    ) -> OperationResult:
        return OperationResult(backend=self.name, operation="train_research_problem")

def test_host_harness_marks_training_failed_when_backend_omits_required_synthetic_outputs(tmp_path: Path):
    candidate = write_candidate(tmp_path)

    run = run_candidate_with_synthetic_fixture(candidate, tmp_path / "runs", backend=NoArtifactBackend())

    assert run.status == RunStatus.FAILED
    assert "required synthetic training artifact is missing" in (run.rejection_reason or "")
    metadata = json.loads((run.run_dir / "run_metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert "required synthetic training artifact is missing" in metadata["training_failure_reason"]

def test_docker_backend_constructs_research_problem_training_command_with_read_only_data_mount(
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

    provider_config, _package_root = _provider_config_for_test(tmp_path, data_root=data_root)

    calls: list[list[str]] = []

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = DockerBackend("custom:tag").train_research_problem(
        run_dir,
        provider_config,
        max_samples=4,
        max_prediction_samples=1,
        prediction_sample_policy="adjacent_and_scattered",
    )

    assert result.backend == "docker"
    assert result.operation == "train_research_problem"
    assert calls[0] == ["docker", "image", "inspect", "custom:tag"]
    assert calls[1] == ["docker", "info", "--format", "{{json .SecurityOptions}}"]
    docker_run = calls[2]
    assert "ml_autoresearch.container_runner" in docker_run
    assert "train-research-problem" in docker_run
    assert any(arg.startswith("--provider-config-json=") for arg in docker_run)
    assert "--max-samples=4" in docker_run
    assert "--max-prediction-samples=1" in docker_run
    assert "--prediction-sample-policy=adjacent_and_scattered" in docker_run
    joined = "\n".join(docker_run)
    assert f"{data_root.resolve() }:/data:ro,z" in joined
    assert f"{run_dir / 'candidate'}:/candidate:ro,z" in joined
    assert f"{run_dir / 'outputs'}:/outputs:rw,z" in joined
    assert f"{_package_root.resolve(strict=True)}:/research-problem:ro,z" in joined
    assert "/data" in joined
    assert f"{data_root}:/data:z" not in joined

def test_docker_backend_rejects_missing_or_file_research_problem_data_root_before_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls: list[list[str]] = []

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    backend = DockerBackend("custom:tag")
    provider_config, _ = _provider_config_for_test(tmp_path)

    with pytest.raises(RuntimeError, match="Research Problem data root does not exist"):
        invalid_config = provider_config.model_copy(update={"data_config": {"dataset_root": str(tmp_path / "missing")}})
        backend.train_research_problem(tmp_path / "run", invalid_config)

    file_root = tmp_path / "not-a-dir"
    file_root.write_text("x")
    with pytest.raises(RuntimeError, match="Research Problem data root is not a directory"):
        invalid_config = provider_config.model_copy(update={"data_config": {"dataset_root": str(file_root)}})
        backend.train_research_problem(tmp_path / "run", invalid_config)

    assert calls == []

def test_docker_training_timeout_requests_graceful_shutdown_and_records_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_dir = tmp_path / "runs" / "run_1"
    (run_dir / "candidate").mkdir(parents=True)
    (run_dir / "outputs" / "logs").mkdir(parents=True)
    (run_dir / "scratch").mkdir()
    (run_dir / "resolved_manifest.yaml").write_text("name: x\n")
    (run_dir / "run_metadata.json").write_text("{}\n")

    run_calls: list[list[str]] = []

    def fake_run(command, check, capture_output, text):
        run_calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    class FakeProcess:
        returncode = 0
        calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(["docker"], timeout)
            return "", ""

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    provider_config, _ = _provider_config_for_test(tmp_path)
    result = DockerBackend("custom:tag", wall_clock_timeout_seconds=0.01, timeout_grace_seconds=2).train_research_problem(
        run_dir, provider_config
    )

    assert result.lifecycle_status == "timeout_graceful"
    assert result.timeout == {
        "requested": True,
        "wall_clock_timeout_seconds": 0.01,
        "grace_seconds": 2,
        "forced_termination": False,
    }
    assert any(call[:2] == ["docker", "exec"] and call[2].startswith("ml-autoresearch-") for call in run_calls)
    assert "budget exhausted" in (run_dir / "outputs" / "logs" / "harness_timeout.log").read_text()

def test_docker_training_timeout_force_kills_after_grace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_dir = tmp_path / "runs" / "run_1"
    (run_dir / "candidate").mkdir(parents=True)
    (run_dir / "outputs" / "logs").mkdir(parents=True)
    (run_dir / "scratch").mkdir()
    (run_dir / "resolved_manifest.yaml").write_text("name: x\n")
    (run_dir / "run_metadata.json").write_text("{}\n")
    run_calls: list[list[str]] = []

    def fake_run(command, check, capture_output, text):
        run_calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    class FakeProcess:
        returncode = -9
        calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls <= 2:
                raise subprocess.TimeoutExpired(["docker"], timeout)
            return "", "killed"

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    provider_config, _ = _provider_config_for_test(tmp_path)
    with pytest.raises(Exception, match="grace period expired"):
        DockerBackend("custom:tag", wall_clock_timeout_seconds=0.01, timeout_grace_seconds=1).train_research_problem(
            run_dir, provider_config
        )

    assert any(call[:2] == ["docker", "kill"] for call in run_calls)
    assert "force-terminated" in (run_dir / "outputs" / "logs" / "harness_timeout.log").read_text()

def test_docker_backend_constructs_generic_research_problem_training_command_with_provider_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from ml_autoresearch.research_problems import ResearchProblemProviderConfig

    run_dir = tmp_path / "runs" / "run_1"
    data_root = tmp_path / "research-data"
    provider_root = tmp_path / "provider"
    (run_dir / "candidate").mkdir(parents=True)
    (run_dir / "outputs" / "logs").mkdir(parents=True)
    (run_dir / "scratch").mkdir()
    (run_dir / "resolved_manifest.yaml").write_text("name: x\n")
    (run_dir / "run_metadata.json").write_text("{}\n")
    data_root.mkdir()
    provider_root.mkdir()
    config = ResearchProblemProviderConfig(
        id="other_problem",
        package_root=provider_root,
        provider_target="problem_pkg.provider:build_spec",
        expected_contract_version="v0",
        data_config={"dataset_root": str(data_root)},
    )

    calls: list[list[str]] = []

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = DockerBackend("custom:tag").train_research_problem(
        run_dir,
        config,
        max_samples=4,
        max_prediction_samples=1,
        prediction_sample_policy="adjacent_and_scattered",
    )

    assert result.backend == "docker"
    assert result.operation == "train_research_problem"
    docker_run = calls[2]
    joined = "\n".join(docker_run)
    assert f"{data_root.resolve(strict=True)}:/data:ro,z" in joined
    assert f"{provider_root.resolve(strict=True)}:/research-problem:ro,z" in joined
    assert "train-gvccs" not in docker_run
    assert docker_run[-6] == "ml_autoresearch.container_runner"
    assert docker_run[-5] == "train-research-problem"
    provider_arg = docker_run[-4]
    assert provider_arg.startswith("--provider-config-json=")
    payload = json.loads(provider_arg.split("=", 1)[1])
    assert payload["id"] == "other_problem"
    assert payload["package_root"] == "/research-problem"
    assert payload["data_config"]["dataset_root"] == "/data"
    assert docker_run[-3:] == [
        "--max-samples=4",
        "--max-prediction-samples=1",
        "--prediction-sample-policy=adjacent_and_scattered",
    ]
