from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from ml_autoresearch.research_problems import ResearchProblemProviderConfig, ResearchProblemSpecRegistry, load_research_problem_provider
from ml_autoresearch.runs import run_candidate_with_research_problem

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PACKAGE_ROOT = REPO_ROOT / "src" / "ml_autoresearch"

TEST_PROBLEM_ROOT_ENV = "ML_AUTORESEARCH_TEST_PROBLEM_ROOT"
GVCCS_PROBLEM_ROOT_ENV = "ML_AUTORESEARCH_GVCCS_PROBLEM_ROOT"
GVCCS_RESEARCH_PROBLEM_ID = "ground_camera_contrail_detection"
GVCCS_PROVIDER_TARGET = "gvccs.research_problem:build_spec"


def _ensure_external_package_root(env_var: str, label: str) -> Path:
    root_value = os.environ.get(env_var)
    if not root_value:
        raise AssertionError(
            f"{env_var} must be set to an external {label} Research Problem package root (for example, a local checkout path)."
        )

    root = Path(root_value).expanduser()
    if not root.exists():
        raise AssertionError(f"{env_var} points to a missing path: {root}")
    if not root.is_dir():
        raise AssertionError(f"{env_var} must be a directory: {root}")

    resolved = root.resolve()
    if resolved == HARNESS_PACKAGE_ROOT or resolved.is_relative_to(HARNESS_PACKAGE_ROOT):
        raise AssertionError(
            f"{env_var} must point outside the reusable Harness package ({HARNESS_PACKAGE_ROOT}); got: {resolved}"
        )

    return resolved


def require_test_research_problem_root() -> Path:
    return _ensure_external_package_root(TEST_PROBLEM_ROOT_ENV, "test")


def gvccs_research_problem_root() -> Path:
    return _ensure_external_package_root(GVCCS_PROBLEM_ROOT_ENV, "GVCCS")


def gvccs_module():
    """Import the external GVCCS package using the configured env-root path."""

    root = gvccs_research_problem_root()
    root_path = str(root)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    return importlib.import_module("gvccs")


def gvccs_provider_config(data_root: str | Path | None = None, *, package_root: Path | str | None = None) -> ResearchProblemProviderConfig:
    data_config = {}
    if data_root is not None:
        data_config["dataset_root"] = str(data_root)
    package_root_path = Path(package_root) if package_root is not None else gvccs_research_problem_root()
    return ResearchProblemProviderConfig(
        id=GVCCS_RESEARCH_PROBLEM_ID,
        package_root=package_root_path,
        provider_target=GVCCS_PROVIDER_TARGET,
        expected_contract_version="v0",
        data_config=data_config,
    )


def gvccs_registry(data_root: str | Path | None = None, *, package_root: Path | str | None = None) -> ResearchProblemSpecRegistry:
    config = gvccs_provider_config(data_root, package_root=package_root)
    registry = ResearchProblemSpecRegistry(active_id=config.id)
    load_research_problem_provider(config, registry=registry)
    return registry


def run_candidate_with_gvccs_data(candidate_dir, runs_root, data_root, *, package_root: Path | str | None = None, **kwargs):
    return run_candidate_with_research_problem(candidate_dir, runs_root, gvccs_provider_config(data_root, package_root=package_root), **kwargs)
