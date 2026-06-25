from pathlib import Path


DOC = Path("docs/dependency-strategy.md")


def test_dependency_strategy_doc_covers_runtime_split() -> None:
    text = DOC.read_text()

    assert "Python 3.12" in text
    assert "host development" in text
    assert "CPU-only PyTorch" in text
    assert "base dependencies do not include PyTorch" in text
    assert "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04" in text
    assert "PyTorch `2.5.1+cu121`" in text
    assert "CUDA 12.1" in text


def test_dependency_strategy_doc_covers_workspace_runtime_image_workflow() -> None:
    text = DOC.read_text()

    assert "uv init --package --python 3.12" in text
    assert "uv add ml-autoresearch" in text
    assert "uv run ml-autoresearch setup" in text
    assert "build-runtime-images" in text
    assert "validate-runtime-images" in text
    assert "Gondolin Agent Runtime Image assets" in text
    assert "Docker runner image tag" in text
    assert "dev_source_path" in text
    assert ".ml-autoresearch/runtime-images.validated.json" in text


def test_dependency_strategy_doc_covers_cluster_gpu_workflow() -> None:
    text = DOC.read_text()

    assert "validate-docker-gpu" in text
    assert "--docker-enable-gpu" in text
    assert "host NVIDIA driver" in text
    assert "container CUDA runtime" in text


def test_readme_links_dependency_strategy_doc() -> None:
    readme = Path("README.md").read_text()

    assert "docs/dependency-strategy.md" in readme
