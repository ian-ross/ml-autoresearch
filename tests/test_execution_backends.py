import json
import os
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
        if command[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(command, 0, '["name=seccomp,profile=builtin"]', "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = DockerBackend("custom:tag").smoke_test(run_dir)

    assert result.backend == "docker"
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
    assert "--read-only" in docker_run
    assert docker_run[docker_run.index("--cap-drop") + 1] == "ALL"
    assert docker_run[docker_run.index("--security-opt") + 1] == "no-new-privileges"
    assert "--privileged" not in docker_run
    assert "--pids-limit" in docker_run
    assert "--memory" in docker_run
    assert "--cpus" in docker_run
    assert not any(arg.startswith("--env-file") for arg in docker_run)
    assert "--gpus" not in docker_run
    assert "--entrypoint" in docker_run
    assert docker_run[docker_run.index("--entrypoint") + 1] == "python"
    assert docker_run[-4:-1] == ["-m", "ml_autoresearch.container_runner", "run-operation"]
    request_payload = json.loads(docker_run[-1].split("=", 1)[1])
    assert request_payload == {"operation": "smoke_test", "run_dir": "/", "max_prediction_samples": 2, "prediction_sample_policy": "first_n", "max_artifact_samples": 12}
    joined = "\n".join(docker_run)
    assert f"{run_dir / 'candidate'}:/candidate:ro,z" in joined
    assert f"{run_dir / 'resolved_manifest.yaml'}:/resolved_manifest.yaml:ro,z" in joined
    assert f"{run_dir / 'run_metadata.json'}:/run_metadata.json:ro,z" in joined
    assert f"{run_dir / 'outputs'}:/outputs:rw,z" in joined
    assert "type=tmpfs,destination=/scratch,tmpfs-size=2g,tmpfs-mode=1777" in joined
    assert "/var/run/docker.sock" not in joined


def test_docker_backend_mounts_research_problem_package_at_manifest_path_for_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir = tmp_path / "runs" / "run_1"
    package_root = tmp_path / "research-problem-package"
    package_root.mkdir()
    (run_dir / "candidate").mkdir(parents=True)
    (run_dir / "outputs" / "logs").mkdir(parents=True)
    (run_dir / "scratch").mkdir()
    (run_dir / "resolved_manifest.yaml").write_text(
        f"""
name: x
research_problem:
  id: fake_problem
  contract_version: v0
  provider:
    target: fake_problem.research_problem:build_spec
    resolved_package_root: "{package_root}"
""".lstrip()
    )
    (run_dir / "run_metadata.json").write_text("{}\n")

    calls: list[list[str]] = []

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        if command[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(command, 0, '["name=seccomp,profile=builtin"]', "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    DockerBackend("custom:tag").smoke_test(run_dir)

    docker_run = calls[2]
    joined = "\n".join(docker_run)
    assert "ML_AUTORESEARCH_RESEARCH_PROBLEM_ROOT=/research-problem" in docker_run
    assert f"{package_root.resolve(strict=True)}:/research-problem:ro,z" in joined


def test_docker_backend_defaults_to_rootless_container_root_when_docker_is_rootless(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir = tmp_path / "runs" / "run_1"
    (run_dir / "candidate").mkdir(parents=True)
    (run_dir / "resolved_manifest.yaml").write_text("name: x\n")
    (run_dir / "run_metadata.json").write_text("{}\n")

    calls: list[list[str]] = []

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        if command[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(command, 0, '["name=rootless"]', "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    DockerBackend("custom:tag").smoke_test(run_dir)

    docker_run = calls[2]
    assert "--userns=host" not in docker_run
    assert docker_run[docker_run.index("--user") + 1] == "0:0"



def test_docker_backend_can_use_rootless_container_root_to_preserve_output_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir = tmp_path / "runs" / "run_1"
    (run_dir / "candidate").mkdir(parents=True)
    (run_dir / "resolved_manifest.yaml").write_text("name: x\n")
    (run_dir / "run_metadata.json").write_text("{}\n")

    calls: list[list[str]] = []

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    DockerBackend("custom:tag", rootless_container_root=True).smoke_test(run_dir)

    docker_run = calls[1]
    assert "--userns=host" not in docker_run
    assert docker_run[docker_run.index("--user") + 1] == "0:0"
    assert oct((run_dir / "outputs").stat().st_mode & 0o777) != "0o777"


def test_docker_backend_accepts_explicit_container_user_for_userns_remap_clusters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir = tmp_path / "runs" / "run_1"
    (run_dir / "candidate").mkdir(parents=True)
    (run_dir / "resolved_manifest.yaml").write_text("name: x\n")
    (run_dir / "run_metadata.json").write_text("{}\n")

    calls: list[list[str]] = []

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    DockerBackend("custom:tag", container_user="65534:65534").smoke_test(run_dir)

    docker_run = calls[1]
    assert docker_run[docker_run.index("--user") + 1] == "65534:65534"
    assert oct((run_dir / "outputs").stat().st_mode & 0o777) == "0o777"


def test_docker_backend_includes_gpus_all_only_when_explicitly_enabled(
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

    DockerBackend("custom:tag", enable_gpu=True).smoke_test(run_dir)

    docker_run = calls[-1]
    assert docker_run[docker_run.index("--gpus") + 1] == "all"


def test_docker_backend_missing_image_fails_with_manual_build_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fake_run(command, check, capture_output, text):
        raise subprocess.CalledProcessError(1, command, stderr="No such image")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="docker build -t ml-autoresearch-runner:local ."):
        DockerBackend().smoke_test(tmp_path)


def test_docker_backend_constructs_contained_evaluate_run_command_with_readonly_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir = tmp_path / "runs" / "run_1"
    data_root = tmp_path / "gvccs"
    data_root.mkdir(parents=True)
    (run_dir / "candidate").mkdir(parents=True)
    (run_dir / "outputs" / "models").mkdir(parents=True)
    (run_dir / "scratch").mkdir()
    (run_dir / "resolved_manifest.yaml").write_text("name: x\n")
    (run_dir / "outputs" / "best_metrics.json").write_text('{"model_artifact": "outputs/models/best_epoch_model.pt"}\n')
    (run_dir / "outputs" / "models" / "best_epoch_model.pt").write_text("checkpoint\n")
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"run_id": "run_1", "status": "completed", "dataset": {"host_data_path": str(data_root)}}) + "\n"
    )

    calls: list[list[str]] = []

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        if command[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(command, 0, '["name=seccomp,profile=builtin"]', "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = DockerBackend("custom:tag").evaluate_run(run_dir, max_artifact_samples=3)

    assert result.backend == "docker"
    assert result.operation == "evaluate_run"
    assert calls[0] == ["docker", "image", "inspect", "custom:tag"]
    docker_run = calls[2]
    assert docker_run[:3] == ["docker", "run", "--rm"]
    assert docker_run[docker_run.index("--network") + 1] == "none"
    assert "--read-only" in docker_run
    assert docker_run[docker_run.index("--cap-drop") + 1] == "ALL"
    assert docker_run[docker_run.index("--security-opt") + 1] == "no-new-privileges"
    assert "--privileged" not in docker_run
    joined = "\n".join(docker_run)
    assert f"{run_dir / 'candidate'}:/candidate:ro,z" in joined
    assert f"{run_dir / 'resolved_manifest.yaml'}:/resolved_manifest.yaml:ro,z" in joined
    assert f"{run_dir / 'run_metadata.json'}:/run_metadata.json:ro,z" in joined
    assert f"{run_dir / 'outputs'}:/outputs:ro,z" in joined
    assert f"{run_dir / 'outputs' / 'evaluations'}:/outputs/evaluations:rw,z" in joined
    assert f"{data_root.resolve(strict=True)}:/data:ro,z" in joined
    assert docker_run[-4:-1] == ["-m", "ml_autoresearch.container_runner", "run-operation"]
    request_payload = json.loads(docker_run[-1].split("=", 1)[1])
    assert request_payload["operation"] == "evaluate_run"
    assert request_payload["run_dir"] == "/"
    assert request_payload["data_root"] == "/data"
    assert request_payload["max_artifact_samples"] == 3


def test_docker_backend_constructs_request_gated_post_run_evaluation_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "run_1"
    data_root = tmp_path / "gvccs"
    data_root.mkdir(parents=True)
    (run_dir / "outputs" / "evaluations").mkdir(parents=True)
    (run_dir / "scratch").mkdir()
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_id": "run_1",
                "status": "completed",
                "dataset": {"host_data_path": str(data_root)},
                "research_problem": {
                    "id": "tiny_problem",
                    "contract_version": "v0",
                    "provider": {"target": "fake:build", "resolved_package_root": str(tmp_path)},
                },
            }
        )
        + "\n"
    )
    request_path = tmp_path / "request.yaml"
    request_path.write_text(
        "request_id: eval-threshold-sweep-run-1\n"
        "target_run_id: run_1\n"
        "evaluation_mode: threshold_sweep\n"
        "diagnostic_question: Which threshold is best?\n"
        "expected_decision_impact: Choose threshold.\n"
        "parameters:\n"
        "  threshold_sweep:\n"
        "    min: 0.1\n"
        "    max: 0.9\n"
        "    steps: 9\n"
        "artifact_budget:\n"
        "  max_artifacts: 2\n"
        "  max_runtime_seconds: 120\n"
    )
    ledger_path = tmp_path / "research-ledger.jsonl"

    calls: list[list[str]] = []

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        if command[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(command, 0, '["name=seccomp,profile=builtin"]', "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = DockerBackend("custom:tag").run_post_run_evaluation(request_path, runs_root=runs_root, ledger_path=ledger_path)

    assert result.backend == "docker"
    assert result.operation == "run_post_run_evaluation"
    docker_run = calls[2]
    joined = "\n".join(docker_run)
    assert f"{run_dir.resolve(strict=True)}:{run_dir.resolve(strict=True)}:ro,z" in joined
    assert f"{run_dir.resolve(strict=True) / 'outputs' / 'evaluations'}:{run_dir.resolve(strict=True) / 'outputs' / 'evaluations'}:rw,z" in joined
    assert f"{request_path.resolve(strict=True)}:{request_path.resolve(strict=True)}:ro,z" in joined
    assert f"{ledger_path.resolve()}:{ledger_path.resolve()}:rw,z" in joined
    assert f"{data_root.resolve(strict=True)}:{data_root.resolve(strict=True)}:ro,z" in joined
    assert docker_run[-4:-1] == ["-m", "ml_autoresearch.container_runner", "run-operation"]
    request_payload = json.loads(docker_run[-1].split("=", 1)[1])
    assert request_payload["operation"] == "run_post_run_evaluation"
    assert request_payload["request_path"] == str(request_path.resolve(strict=True))
    assert request_payload["runs_root"] == str(runs_root.resolve(strict=True))
    assert request_payload["ledger_path"] == str(ledger_path.resolve())


def test_docker_backend_evaluate_run_data_root_override_is_validated_and_mounted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir = tmp_path / "runs" / "run_1"
    metadata_root = tmp_path / "metadata_gvccs"
    override_root = tmp_path / "override_gvccs"
    metadata_root.mkdir(parents=True)
    override_root.mkdir(parents=True)
    (run_dir / "candidate").mkdir(parents=True)
    (run_dir / "outputs" / "models").mkdir(parents=True)
    (run_dir / "resolved_manifest.yaml").write_text("name: x\n")
    (run_dir / "outputs" / "best_metrics.json").write_text('{"model_artifact": "outputs/models/best_epoch_model.pt"}\n')
    (run_dir / "outputs" / "models" / "best_epoch_model.pt").write_text("checkpoint\n")
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"run_id": "run_1", "status": "completed", "dataset": {"host_data_path": str(metadata_root)}}) + "\n"
    )

    calls: list[list[str]] = []

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    DockerBackend("custom:tag", rootless_container_root=True).evaluate_run(run_dir, data_root=override_root)

    docker_run = calls[1]
    joined = "\n".join(docker_run)
    assert f"{override_root.resolve(strict=True)}:/data:ro,z" in joined
    assert f"{metadata_root.resolve(strict=True)}:/data:ro,z" not in joined


def test_container_runner_run_operation_uses_shared_operation_request(monkeypatch: pytest.MonkeyPatch):
    import sys
    import ml_autoresearch.container_runner as container_runner

    calls = []

    def fake_execute(request):
        calls.append(request)

    monkeypatch.setattr(container_runner, "execute_operation_request", fake_execute)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "container_runner",
            "run-operation",
            '--request-json={"operation":"train_synthetic","run_dir":"/","max_prediction_samples":1}',
        ],
    )

    container_runner.main()

    assert len(calls) == 1
    assert calls[0].operation == "train_synthetic"
    assert calls[0].run_dir == Path("/")
    assert calls[0].max_prediction_samples == 1


def test_container_smoke_uses_mounted_resolved_manifest_for_smoke_specs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import ml_autoresearch.container_smoke as container_smoke

    candidate_dir = tmp_path / "candidate"
    outputs_dir = tmp_path / "outputs"
    resolved_manifest = tmp_path / "resolved_manifest.yaml"
    candidate_dir.mkdir()
    outputs_dir.mkdir()
    resolved_manifest.write_text(
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
    calls = []

    def fake_smoke_test_candidate(candidate_arg, outputs_arg, *, input_spec=None, output_spec=None):
        calls.append((candidate_arg, outputs_arg, input_spec, output_spec))

    monkeypatch.setattr(container_smoke, "smoke_test_candidate", fake_smoke_test_candidate)

    container_smoke.smoke_test_container(candidate_dir, outputs_dir, resolved_manifest)

    assert calls == [
        (
            candidate_dir,
            outputs_dir,
            {"mode": "single_frame_rgb", "shape": [3, 128, 128]},
            {
                "form": "mask_logits",
                "shape": [1, 128, 128],
                "auxiliary_outputs": [{"target": "line", "name": "line_logits", "shape": [1, 128, 128]}],
            },
        )
    ]
