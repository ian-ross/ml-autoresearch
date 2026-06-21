"""Packaged Harness-owned resource access."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Iterable


RESOURCE_PACKAGE = "ml_autoresearch.resources"
AUTORESEARCH_SKILLS_RESOURCE = "autoresearch-skills"
CONTAINER_BUILD_RECIPES_RESOURCE = "container-build-recipes"


class PackageResourceError(ValueError):
    """Raised when packaged Harness resources cannot be staged."""


@dataclass(frozen=True)
class ResourceCopyResult:
    """Summary of a packaged resource copy operation."""

    destination: Path
    copied: tuple[Path, ...]


def copy_autoresearch_skills(destination: str | Path) -> ResourceCopyResult:
    """Copy the packaged Autoresearch Skill Set into ``destination``.

    Existing entries with the same names as packaged skills are replaced. Other
    local skills in the destination are preserved.
    """

    destination_path = Path(destination)
    copied = _copy_resource_children(AUTORESEARCH_SKILLS_RESOURCE, destination_path, replace_existing=True)
    return ResourceCopyResult(destination=destination_path, copied=copied)


def stage_container_build_recipes(destination: str | Path) -> ResourceCopyResult:
    """Copy packaged runtime/container build recipes into hidden workspace state."""

    destination_path = Path(destination)
    copied = _copy_resource_children(CONTAINER_BUILD_RECIPES_RESOURCE, destination_path, replace_existing=True)
    return ResourceCopyResult(destination=destination_path, copied=copied)


def stage_workspace_container_build_recipes(workspace_root: str | Path) -> ResourceCopyResult:
    """Stage runtime build recipes under hidden Research Workspace operational state."""

    return stage_container_build_recipes(Path(workspace_root) / ".ml-autoresearch" / "container-build-recipes")


def packaged_container_build_recipe_names() -> tuple[str, ...]:
    """Return top-level packaged runtime/container build recipe names."""

    return tuple(sorted(_resource_child_names(CONTAINER_BUILD_RECIPES_RESOURCE)))


def _copy_resource_children(resource_name: str, destination: Path, *, replace_existing: bool) -> tuple[Path, ...]:
    root = resources.files(RESOURCE_PACKAGE).joinpath(resource_name)
    if not root.is_dir():
        raise PackageResourceError(f"packaged resource directory is missing: {resource_name}")
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for entry in root.iterdir():
        target = destination / entry.name
        if target.exists() and replace_existing:
            _remove_existing(target)
        if entry.is_dir():
            shutil.copytree(entry, target)
        elif entry.is_file():
            shutil.copy2(entry, target)
        else:
            continue
        copied.append(target)
    return tuple(copied)


def _resource_child_names(resource_name: str) -> Iterable[str]:
    root = resources.files(RESOURCE_PACKAGE).joinpath(resource_name)
    if not root.is_dir():
        raise PackageResourceError(f"packaged resource directory is missing: {resource_name}")
    for entry in root.iterdir():
        if entry.is_file() or entry.is_dir():
            yield entry.name


def _remove_existing(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()
