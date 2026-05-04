from pathlib import Path


def test_readme_documents_pinned_docker_cuda_runtime() -> None:
    readme = Path("README.md").read_text()

    assert "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04" in readme
    assert "PyTorch `2.5.1+cu121`" in readme
    assert "runner image and also use Python 3.12" in readme
    assert "docker build -t ml-autoresearch-runner:local ." in readme
