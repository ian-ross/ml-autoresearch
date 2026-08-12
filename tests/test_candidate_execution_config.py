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
    (tmp_path / "ml-autoresearch.toml").write_text(
        f'''
[candidate_execution]
backend = "docker"
docker_image = "custom:tag"
docker_enable_gpu = true
docker_gpu_device = "0"
docker_rootless_container_root = true
data_root = "{data_root}"
max_samples = 8
max_parameters = 25000000
max_epochs = 3
training_wall_clock_timeout_seconds = 1800
max_prediction_samples = 4
max_parallel_runs = 2
prediction_sample_policy = "adjacent_and_scattered"
'''.lstrip()
    )

    config = load_candidate_execution_config(tmp_path)
    backend = execution_backend_from_config(config)

    assert config.backend == "docker"
    assert config.data_root == data_root
    assert config.max_samples == 8
    assert config.max_parameters == 25_000_000
    assert config.max_epochs == 3
    assert config.training_wall_clock_timeout_seconds == 1800
    assert config.max_prediction_samples == 4
    assert config.max_parallel_runs == 2
    assert config.prediction_sample_policy == "adjacent_and_scattered"
    assert isinstance(backend, DockerBackend)
    assert backend.docker_image == "custom:tag"
    assert backend.enable_gpu is True
    assert backend.gpu_device == "0"
    assert backend.rootless_container_root is True
    assert backend.wall_clock_timeout_seconds == 1800


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
    (tmp_path / "ml-autoresearch.toml").write_text(
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
    (tmp_path / "ml-autoresearch.toml").write_text(
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


def test_candidate_execution_config_loads_named_research_problem_data_roots(tmp_path: Path) -> None:
    package_root = tmp_path / "configured-problem"
    training_root = tmp_path / "training"
    ancillary_root = tmp_path / "ancillary"
    package_root.mkdir()
    training_root.mkdir()
    ancillary_root.mkdir()
    (tmp_path / "ml-autoresearch.toml").write_text(
        f'''
[research_problem]
id = "multi_root_problem"
package_root = "{package_root}"
provider_target = "problem:build_spec"
expected_contract_version = "v0"
data_config = {{ manifest = "natural-earth/manifest.json" }}

[research_problem.data_roots]
training = "training"
ancillary = "{ancillary_root}"
'''.lstrip()
    )

    config = load_candidate_execution_config(tmp_path)

    assert config.research_problem_provider is not None
    assert config.research_problem_provider.data_roots == {
        "ancillary": ancillary_root.resolve(),
        "training": training_root.resolve(),
    }
    assert config.research_problem_provider.effective_data_config() == {
        "manifest": "natural-earth/manifest.json",
        "data_roots": {
            "ancillary": str(ancillary_root.resolve()),
            "training": str(training_root.resolve()),
        },
    }


@pytest.mark.parametrize(
    "data_roots, match",
    [
        ('"bad/name" = "training"', "names must match"),
        ('training = "missing"', "does not exist"),
    ],
)
def test_candidate_execution_config_rejects_invalid_named_data_roots(
    tmp_path: Path, data_roots: str, match: str
) -> None:
    package_root = tmp_path / "problem"
    training_root = tmp_path / "training"
    package_root.mkdir()
    training_root.mkdir()
    (tmp_path / "ml-autoresearch.toml").write_text(
        f'''
[research_problem]
id = "multi_root_problem"
package_root = "{package_root}"
provider_target = "problem:build_spec"
expected_contract_version = "v0"

[research_problem.data_roots]
{data_roots}
'''.lstrip()
    )

    with pytest.raises(CandidateExecutionConfigError, match=match):
        load_candidate_execution_config(tmp_path)


def test_candidate_execution_config_rejects_named_root_aliases_and_legacy_override(tmp_path: Path) -> None:
    package_root = tmp_path / "problem"
    training_root = tmp_path / "training"
    package_root.mkdir()
    training_root.mkdir()
    alias = tmp_path / "training-alias"
    alias.symlink_to(training_root, target_is_directory=True)
    (tmp_path / "ml-autoresearch.toml").write_text(
        f'''
[candidate_execution]
data_root = "{training_root}"

[research_problem]
id = "multi_root_problem"
package_root = "{package_root}"
provider_target = "problem:build_spec"
expected_contract_version = "v0"

[research_problem.data_roots]
training = "{training_root}"
ancillary = "{alias}"
'''.lstrip()
    )

    with pytest.raises(CandidateExecutionConfigError, match="resolve to the same directory|cannot be combined"):
        load_candidate_execution_config(tmp_path)


def test_candidate_execution_config_requires_workspace_config(tmp_path: Path) -> None:
    with pytest.raises(CandidateExecutionConfigError, match="missing Workspace Configuration"):
        load_candidate_execution_config(tmp_path)


def test_candidate_execution_config_ignores_legacy_split_config_files(tmp_path: Path) -> None:
    (tmp_path / "candidate-execution.toml").write_text('[candidate_execution]\nbackend = "native"\n')
    (tmp_path / "agent-boundary.toml").write_text('[agent_control_boundary]\n')
    (tmp_path / "notification.toml").write_text('[mailjet]\n')

    with pytest.raises(CandidateExecutionConfigError, match="missing Workspace Configuration"):
        load_candidate_execution_config(tmp_path)


def test_candidate_execution_config_reports_invalid_workspace_config(tmp_path: Path) -> None:
    (tmp_path / "ml-autoresearch.toml").write_text("[candidate_execution\n")

    with pytest.raises(CandidateExecutionConfigError, match="invalid Workspace Configuration"):
        load_candidate_execution_config(tmp_path)


def test_candidate_execution_config_resolves_configured_runs_root_and_ledger_path(tmp_path: Path) -> None:
    external_runs = tmp_path / "scratch" / "runs"
    (tmp_path / "ml-autoresearch.toml").write_text(
        f'''
[candidate_execution]
runs_root = "{external_runs}"
ledger_path = "research-ledger.jsonl"
'''.lstrip()
    )

    config = load_candidate_execution_config(tmp_path)

    assert config.runs_root == external_runs
    assert config.ledger_path == tmp_path / "research-ledger.jsonl"


def test_candidate_execution_config_rejects_external_ledger_path(tmp_path: Path) -> None:
    external_ledger = tmp_path.parent / "external-ledger.jsonl"
    (tmp_path / "ml-autoresearch.toml").write_text(
        f'''
[candidate_execution]
runs_root = "{tmp_path.parent / 'external-runs'}"
ledger_path = "{external_ledger}"
'''.lstrip()
    )

    with pytest.raises(CandidateExecutionConfigError, match="ledger_path.*Research Workspace Root"):
        load_candidate_execution_config(tmp_path)


@pytest.mark.parametrize(
    "body, match",
    [
        ('[candidate_execution]\nbackend = "native"\ndocker_enable_gpu = true\n', "requires backend"),
        (
            '[candidate_execution]\nbackend = "docker"\ndocker_user = "1000:1000"\ndocker_rootless_container_root = true\n',
            "choose either",
        ),
        (
            '[candidate_execution]\nbackend = "docker"\ndocker_gpu_device = "0"\n',
            "docker_gpu_device requires docker_enable_gpu",
        ),
        (
            '[candidate_execution]\nmax_parallel_runs = 5\n',
            "max_parallel_runs must be at most 4",
        ),
        (
            '[candidate_execution]\nmax_parameters = 100000001\n',
            "max_parameters must be at most 100000000",
        ),
        (
            '[candidate_execution]\nmax_epochs = 101\n',
            "max_epochs must be at most 100",
        ),
        (
            '[candidate_execution]\ntraining_wall_clock_timeout_seconds = 1800\n',
            "training_wall_clock_timeout_seconds requires backend",
        ),
        (
            '[candidate_execution]\nbackend = "docker"\ndocker_enable_gpu = true\ndocker_gpu_device = "0,1"\n',
            "docker_gpu_device must identify one GPU",
        ),
    ],
)
def test_candidate_execution_config_rejects_incoherent_docker_policy(tmp_path: Path, body: str, match: str) -> None:
    (tmp_path / "ml-autoresearch.toml").write_text(body)

    with pytest.raises(CandidateExecutionConfigError, match=match):
        load_candidate_execution_config(tmp_path)
