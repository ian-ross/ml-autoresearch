from pathlib import Path


DOCKERFILE = Path("Dockerfile")


def test_dockerfile_uses_cuda_121_base_and_installs_python_312() -> None:
    dockerfile = DOCKERFILE.read_text()

    assert dockerfile.startswith("FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04\n")
    assert "python3.12" in dockerfile
    assert "ln -sf /usr/bin/python3.12 /usr/local/bin/python" in dockerfile


def test_dockerfile_installs_pinned_pytorch_cuda_121_for_python_312() -> None:
    dockerfile = DOCKERFILE.read_text()

    assert "https://download.pytorch.org/whl/cu121" in dockerfile
    assert "torch==2.5.1+cu121" in dockerfile


def test_dockerfile_installs_package_without_replacing_image_torch() -> None:
    dockerfile = DOCKERFILE.read_text()

    assert "python -m pip install --no-cache-dir --no-deps ." in dockerfile
    assert "--ignore-requires-python" not in dockerfile


def test_dockerfile_installs_runtime_dependencies_before_package_install() -> None:
    dockerfile = DOCKERFILE.read_text()

    assert "numpy>=2,<3" in dockerfile
    assert "pillow>=10,<12" in dockerfile
    assert "pydantic>=2,<3" in dockerfile
    assert "PyYAML>=6,<7" in dockerfile
    assert "typer>=0.12,<1" in dockerfile
