from pathlib import Path


DOCKERFILE = Path("Dockerfile")


def test_dockerfile_uses_pinned_pytorch_cuda_121_runtime_image() -> None:
    dockerfile = DOCKERFILE.read_text()

    assert dockerfile.startswith("FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime\n")


def test_dockerfile_does_not_reinstall_or_upgrade_image_torch() -> None:
    dockerfile = DOCKERFILE.read_text()

    assert "pip install --no-cache-dir --no-deps ." in dockerfile
    assert "torch" not in [line.strip() for line in dockerfile.splitlines() if line.startswith("RUN pip install")]
