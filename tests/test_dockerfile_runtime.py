from pathlib import Path


RUNNER_DOCKERFILE = Path("containers/Dockerfile.runner")
AGENT_DOCKERFILE = Path("containers/Dockerfile.agent")
ENCLAVE_DOCKERFILE = Path("containers/Dockerfile.enclave")


def test_runner_dockerfile_uses_cuda_121_base_and_installs_python_312() -> None:
    dockerfile = RUNNER_DOCKERFILE.read_text()

    assert dockerfile.startswith("FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04\n")
    assert "python3.12" in dockerfile
    assert "ln -sf /usr/bin/python3.12 /usr/local/bin/python" in dockerfile


def test_runner_dockerfile_installs_pinned_pytorch_cuda_121_for_python_312() -> None:
    dockerfile = RUNNER_DOCKERFILE.read_text()

    assert "https://download.pytorch.org/whl/cu121" in dockerfile
    assert "torch==2.5.1+cu121" in dockerfile


def test_runner_dockerfile_installs_package_without_replacing_image_torch() -> None:
    dockerfile = RUNNER_DOCKERFILE.read_text()

    assert "python -m pip install --no-cache-dir --no-deps ." in dockerfile
    assert "--ignore-requires-python" not in dockerfile


def test_runner_dockerfile_installs_runtime_dependencies_before_package_install() -> None:
    dockerfile = RUNNER_DOCKERFILE.read_text()

    assert "numpy>=2,<3" in dockerfile
    assert "pillow>=10,<12" in dockerfile
    assert "pydantic>=2,<3" in dockerfile
    assert "PyYAML>=6,<7" in dockerfile
    assert "typer>=0.12,<1" in dockerfile


def test_agent_dockerfile_installs_lightweight_agent_cli_without_torch() -> None:
    dockerfile = AGENT_DOCKERFILE.read_text()

    assert dockerfile.startswith("FROM python:3.12-slim-bookworm\n")
    assert "COPY pyproject.toml ./" in dockerfile
    assert "COPY src ./src" in dockerfile
    assert "python -m pip install --no-cache-dir --no-deps ." in dockerfile
    assert "ml-autoresearch-agent --help" in dockerfile
    assert "ml-autoresearch --help" not in dockerfile
    assert "torch" not in dockerfile.lower()
    assert "nvidia" not in dockerfile.lower()
    assert "cuda" not in dockerfile.lower()


def test_agent_dockerfile_smoke_checks_allowed_static_commands() -> None:
    dockerfile = AGENT_DOCKERFILE.read_text()

    assert "ml-autoresearch-agent list-runs" in dockerfile
    assert "ml-autoresearch-agent validate-candidate" in dockerfile
    assert "ml-autoresearch-agent prepare-candidate-submission" in dockerfile


def test_enclave_dockerfile_installs_python_312_and_harness_cli() -> None:
    dockerfile = ENCLAVE_DOCKERFILE.read_text()

    assert dockerfile.startswith("FROM python:3.12-slim-bookworm\n")
    assert "COPY pyproject.toml ./" in dockerfile
    assert "COPY src ./src" in dockerfile
    assert "https://download.pytorch.org/whl/cpu" in dockerfile
    assert "torch==2.5.1+cpu" in dockerfile
    assert "python -m pip install --no-cache-dir --no-deps ." in dockerfile
    assert "ml-autoresearch --help" in dockerfile


def test_enclave_dockerfile_leaves_command_available_for_agent_runtime() -> None:
    dockerfile = ENCLAVE_DOCKERFILE.read_text()

    assert "ENTRYPOINT" not in dockerfile
    assert 'CMD ["bash"]' in dockerfile
