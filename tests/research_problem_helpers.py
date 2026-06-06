from __future__ import annotations

from pathlib import Path

from ml_autoresearch.research_problems import ResearchProblemProviderConfig, ResearchProblemSpecRegistry, load_research_problem_provider
from ml_autoresearch.runs import run_candidate_with_research_problem

GVCCS_RESEARCH_PROBLEM_ROOT = Path('/home/iross/code/gvccs-research-problem')
GVCCS_RESEARCH_PROBLEM_ID = 'ground_camera_contrail_detection'
GVCCS_PROVIDER_TARGET = 'gvccs.research_problem:build_spec'


def gvccs_provider_config(data_root: str | Path | None = None) -> ResearchProblemProviderConfig:
    data_config = {}
    if data_root is not None:
        data_config['dataset_root'] = str(data_root)
    return ResearchProblemProviderConfig(
        id=GVCCS_RESEARCH_PROBLEM_ID,
        package_root=GVCCS_RESEARCH_PROBLEM_ROOT,
        provider_target=GVCCS_PROVIDER_TARGET,
        expected_contract_version='v0',
        data_config=data_config,
    )


def gvccs_registry(data_root: str | Path | None = None) -> ResearchProblemSpecRegistry:
    config = gvccs_provider_config(data_root)
    registry = ResearchProblemSpecRegistry(active_id=config.id)
    load_research_problem_provider(config, registry=registry)
    return registry


def run_candidate_with_gvccs_data(candidate_dir, runs_root, data_root, **kwargs):
    return run_candidate_with_research_problem(candidate_dir, runs_root, gvccs_provider_config(data_root), **kwargs)
