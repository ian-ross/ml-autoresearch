from pathlib import Path

import pytest

from ml_autoresearch.candidate_execution_config import (
    CandidateExecutionConfigError,
    execution_backend_from_config,
    load_candidate_execution_config,
    load_configured_research_problem_registry,
)
from ml_autoresearch.execution import DockerBackend, NativeBackend


def test_candidate_execution_config_selects_docker_gpu_policy_and_data_root(tmp_path: Path) -> None:
    data_root = tmp_path / "gvccs"
    data_root.mkdir()
    (tmp_path / "candidate-execution.toml").write_text(
        f'''
[candidate_execution]
backend = "docker"
docker_image = "custom:tag"
docker_enable_gpu = true
docker_rootless_container_root = true
data_root = "{data_root}"
max_samples = 8
max_prediction_samples = 4
prediction_sample_policy = "adjacent_and_scattered"
'''.lstrip()
    )

    config = load_candidate_execution_config(tmp_path)
    backend = execution_backend_from_config(config)

    assert config.backend == "docker"
    assert config.data_root == data_root
    assert config.max_samples == 8
    assert config.max_prediction_samples == 4
    assert config.prediction_sample_policy == "adjacent_and_scattered"
    assert isinstance(backend, DockerBackend)
    assert backend.docker_image == "custom:tag"
    assert backend.enable_gpu is True
    assert backend.rootless_container_root is True


def test_candidate_execution_config_loads_research_problem_provider_registry(tmp_path: Path) -> None:
    package = tmp_path / "tiny_problem"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "research_problem.py").write_text(
        "from ml_autoresearch.research_problems import ResearchProblemSpec\n"
        "def build_spec(data_config=None):\n"
        "    return ResearchProblemSpec(\n"
        "        id='tiny_problem', version='test-v0', contract_version='v0',\n"
        "        input_modes=('tiny_rgb',), input_specs={'tiny_rgb': {'mode': 'tiny_rgb', 'shape': [3, 8, 8]}},\n"
        "        output_forms=('tiny_mask_logits',), output_specs={'tiny_mask_logits': {'form': 'tiny_mask_logits', 'shape': [1, 8, 8]}},\n"
        "        losses=('tiny_loss',), optimizers=('sgd',),\n"
        "        sampling_policies=('sequential',), augmentation_policies=('none',), primary_metric='val/tiny_score')\n"
    )
    (tmp_path / "candidate-execution.toml").write_text(
        '''
[research_problem]
id = "tiny_problem"
package_root = "."
provider_target = "tiny_problem.research_problem:build_spec"
expected_contract_version = "v0"
'''.lstrip()
    )

    registry = load_configured_research_problem_registry(tmp_path)

    assert registry is not None
    assert registry.get("tiny_problem").losses == ("tiny_loss",)


def test_candidate_execution_config_allows_research_problem_values_from_config(tmp_path: Path) -> None:
    package_root = tmp_path / "configured-problem"
    dataset_root = tmp_path / "configured-data"
    package_root.mkdir()
    dataset_root.mkdir()
    (tmp_path / "candidate-execution.toml").write_text(
        f'''
[research_problem]
id = "ground_camera_contrail_detection"
package_root = "{package_root}"
provider_target = "gvccs.research_problem:build_spec"
expected_contract_version = "v0"
data_config = {{ dataset_root = "{dataset_root}" }}
'''.lstrip()
    )

    config = load_candidate_execution_config(tmp_path)

    assert config.research_problem_provider is not None
    assert config.research_problem_provider.id == "ground_camera_contrail_detection"
    assert config.research_problem_provider.provider_target == "gvccs.research_problem:build_spec"
    assert config.research_problem_provider.data_config == {"dataset_root": str(dataset_root)}


def test_candidate_execution_config_defaults_to_native_when_absent(tmp_path: Path) -> None:
    config = load_candidate_execution_config(tmp_path)

    assert config.backend == "native"
    assert config.runs_root == tmp_path / "runs"
    assert isinstance(execution_backend_from_config(config), NativeBackend)


def test_candidate_execution_config_resolves_configured_runs_root_and_ledger_path(tmp_path: Path) -> None:
    external_runs = tmp_path / "scratch" / "runs"
    (tmp_path / "candidate-execution.toml").write_text(
        f'''
[candidate_execution]
runs_root = "{external_runs}"
ledger_path = "research-ledger.jsonl"
'''.lstrip()
    )

    config = load_candidate_execution_config(tmp_path)

    assert config.runs_root == external_runs
    assert config.ledger_path == tmp_path / "research-ledger.jsonl"


@pytest.mark.parametrize(
    "body, match",
    [
        ('[candidate_execution]\nbackend = "native"\ndocker_enable_gpu = true\n', "requires backend"),
        (
            '[candidate_execution]\nbackend = "docker"\ndocker_user = "1000:1000"\ndocker_rootless_container_root = true\n',
            "choose either",
        ),
    ],
)
def test_candidate_execution_config_rejects_incoherent_docker_policy(tmp_path: Path, body: str, match: str) -> None:
    (tmp_path / "candidate-execution.toml").write_text(body)

    with pytest.raises(CandidateExecutionConfigError, match=match):
        load_candidate_execution_config(tmp_path)
