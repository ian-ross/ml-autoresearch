from pathlib import Path
import tomllib


def test_base_dependencies_do_not_include_torch() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    dependencies = pyproject["project"]["dependencies"]

    assert not any(dependency.lower().startswith("torch") for dependency in dependencies)


def test_dev_extra_uses_pinned_cpu_only_torch_source() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    dev_dependencies = pyproject["project"]["optional-dependencies"]["dev"]
    indexes = {index["name"]: index for index in pyproject["tool"]["uv"]["index"]}

    assert "torch==2.5.1+cpu" in dev_dependencies
    assert pyproject["tool"]["uv"]["sources"]["torch"] == {
        "index": "pytorch-cpu",
        "extra": "dev",
    }
    assert indexes["pytorch-cpu"] == {
        "name": "pytorch-cpu",
        "url": "https://download.pytorch.org/whl/cpu",
        "explicit": True,
    }


def test_readme_documents_separated_torch_runtime_strategy() -> None:
    readme = Path("README.md").read_text()

    assert "pinned CPU-only PyTorch build" in readme
    assert "Base `ml-autoresearch` installs do not include PyTorch" in readme
    assert "runtime PyTorch/CUDA stack from the workspace-specific runner image" in readme
