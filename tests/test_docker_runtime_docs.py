from pathlib import Path


def test_readme_documents_pinned_docker_cuda_runtime() -> None:
    readme = Path("README.md").read_text()

    assert "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime" in readme
    assert "Docker runtime issue is revisited with a Python 3.12-compatible CUDA 12.1 PyTorch image" in readme
    assert "docker build -t ml-autoresearch-runner:local ." in readme
