from pathlib import Path

import pytest

from ml_autoresearch.candidate_execution_config import (
    CandidateExecutionConfigError,
    execution_backend_from_config,
    load_candidate_execution_config,
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


def test_candidate_execution_config_defaults_to_native_when_absent(tmp_path: Path) -> None:
    config = load_candidate_execution_config(tmp_path)

    assert config.backend == "native"
    assert isinstance(execution_backend_from_config(config), NativeBackend)


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
