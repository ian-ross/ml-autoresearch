import subprocess
from pathlib import Path

from ml_autoresearch.execution import DEFAULT_DOCKER_IMAGE, docker_gpu_validation_command, validate_docker_gpu


def test_docker_gpu_validation_command_uses_runner_image_and_gpu_opt_in() -> None:
    command = docker_gpu_validation_command("custom:tag")

    assert command[:5] == ["docker", "run", "--rm", "--gpus", "all"]
    assert "--network" in command
    assert command[command.index("--network") + 1] == "none"
    assert "--entrypoint" in command
    assert command[command.index("--entrypoint") + 1] == "python"
    assert command[command.index("--entrypoint") + 2] == "custom:tag"
    assert command[-2] == "-c"
    probe = command[-1]
    assert "torch.__version__" in probe
    assert "torch.version.cuda" in probe
    assert "torch.cuda.is_available()" in probe
    assert "torch.cuda.get_device_name" in probe


def test_docker_gpu_validation_uses_default_runner_image(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "ok\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    completed = validate_docker_gpu()

    assert completed.stdout == "ok\n"
    assert calls[0][calls[0].index("--entrypoint") + 2] == DEFAULT_DOCKER_IMAGE


def test_readme_documents_container_gpu_validation_path() -> None:
    readme = Path("README.md").read_text()

    assert "validate-docker-gpu" in readme
    assert "torch.__version__" in readme
    assert "torch.version.cuda" in readme
    assert "torch.cuda.is_available()" in readme
    assert "host NVIDIA driver" in readme
    assert "container CUDA runtime" in readme
