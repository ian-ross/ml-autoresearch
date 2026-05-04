from pathlib import Path


DOC = Path("docs/dependency-strategy.md")


def test_dependency_strategy_doc_covers_runtime_split() -> None:
    text = DOC.read_text()

    assert "Python 3.12" in text
    assert "host development" in text
    assert "CPU-only PyTorch" in text
    assert "base dependencies do not include PyTorch" in text
    assert "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime" in text
    assert "CUDA 12.1" in text


def test_dependency_strategy_doc_covers_cluster_gpu_workflow() -> None:
    text = DOC.read_text()

    assert "docker build -t ml-autoresearch-runner:local ." in text
    assert "validate-docker-gpu" in text
    assert "--docker-enable-gpu" in text
    assert "host NVIDIA driver" in text
    assert "container CUDA runtime" in text


def test_readme_links_dependency_strategy_doc() -> None:
    readme = Path("README.md").read_text()

    assert "docs/dependency-strategy.md" in readme
