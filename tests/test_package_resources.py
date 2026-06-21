from __future__ import annotations

import tomllib
from pathlib import Path

from conftest import invoke_typer_cli
from ml_autoresearch.cli import app
from ml_autoresearch.package_resources import (
    copy_autoresearch_skills,
    packaged_container_build_recipe_names,
    stage_workspace_container_build_recipes,
)


def test_packaged_autoresearch_skills_copy_without_workspace_docs(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    skills_dir = workspace_root / "agent-work" / ".pi" / "skills"
    stale = skills_dir / "campaign-manager" / "stale.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale\n")
    local = skills_dir / "local-helper" / "SKILL.md"
    local.parent.mkdir(parents=True)
    local.write_text("# Local Helper\n")

    result = copy_autoresearch_skills(skills_dir)

    assert result.destination == skills_dir
    assert not (workspace_root / "docs").exists()
    assert not stale.exists()
    assert local.read_text() == "# Local Helper\n"
    campaign_manager = skills_dir / "campaign-manager" / "SKILL.md"
    assert "name: campaign-manager" in campaign_manager.read_text()
    assert (skills_dir / "candidate-implementer" / "SKILL.md").is_file()


def test_packaged_container_build_recipes_stage_to_hidden_workspace_state(tmp_path: Path) -> None:
    result = stage_workspace_container_build_recipes(tmp_path)

    staged = tmp_path / ".ml-autoresearch" / "container-build-recipes"
    assert result.destination == staged
    for name in ["Dockerfile.runner", "Dockerfile.agent", "Dockerfile.enclave", "gondolin-build-config.json", "Makefile"]:
        assert (staged / name).is_file(), name
    assert "Dockerfile.runner" in (staged / "Makefile").read_text()
    staged_names = {path.name for path in staged.iterdir()}
    assert "ml-autoresearch-agent" not in staged_names
    assert not any(path.suffix in {".tar", ".img"} for path in staged.iterdir())
    assert packaged_container_build_recipe_names() == tuple(sorted(staged_names))


def test_stage_runtime_build_recipes_cli_uses_packaged_resources_not_workspace_containers(tmp_path: Path) -> None:
    (tmp_path / "containers").mkdir()
    (tmp_path / "containers" / "Dockerfile.runner").write_text("FROM wrong\n")

    completed = invoke_typer_cli(app, ["stage-runtime-build-recipes"], cwd=tmp_path)

    assert completed.returncode == 0, completed.stderr
    staged_runner = tmp_path / ".ml-autoresearch" / "container-build-recipes" / "Dockerfile.runner"
    assert staged_runner.is_file()
    assert staged_runner.read_text() != "FROM wrong\n"


def test_packaged_resource_files_are_included_in_wheel(tmp_path: Path) -> None:
    completed = invoke_typer_cli(app, ["stage-runtime-build-recipes"], cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    # If these files are accessible through importlib.resources in this checkout,
    # Hatch should package them because they live under the ml_autoresearch package.
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    assert pyproject["build-system"]["build-backend"] == "hatchling.build"
    assert Path("src/ml_autoresearch/resources/container-build-recipes/Dockerfile.runner").is_file()
