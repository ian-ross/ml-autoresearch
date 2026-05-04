from pathlib import Path


DOCKERFILE = Path("Dockerfile")


def test_dockerfile_uses_pinned_pytorch_cuda_121_runtime_image() -> None:
    dockerfile = DOCKERFILE.read_text()

    assert dockerfile.startswith("FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime\n")


def test_dockerfile_installs_package_without_replacing_image_torch() -> None:
    dockerfile = DOCKERFILE.read_text()

    assert "pip install --no-cache-dir --no-deps --ignore-requires-python ." in dockerfile
    pip_lines = [line.strip() for line in dockerfile.splitlines() if line.startswith("RUN pip install")]
    assert not any("torch" in line.lower() for line in pip_lines)


def test_dockerfile_installs_runtime_dependencies_before_package_install() -> None:
    dockerfile = DOCKERFILE.read_text()

    assert "numpy>=2,<3" in dockerfile
    assert "pillow>=10,<12" in dockerfile
    assert "pydantic>=2,<3" in dockerfile
    assert "PyYAML>=6,<7" in dockerfile
    assert "typer>=0.12,<1" in dockerfile
