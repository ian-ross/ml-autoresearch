import json
from pathlib import Path

from ml_autoresearch.cli import app
from conftest import invoke_typer_cli
from research_problem_helpers import write_fake_research_problem_package, write_fake_candidate_execution_config


def run_cli(*args: str):
    return invoke_typer_cli(app, args)


def test_run_candidate_defaults_to_docker_backend(tmp_path: Path):
    write_fake_research_problem_package(tmp_path)
    write_fake_candidate_execution_config(tmp_path, backend="docker")
    completed = run_cli(
        "run-candidate",
        "--candidate",
        str(tmp_path / "missing"),
        "--runs-root",
        str(tmp_path / "runs"),
        "--workspace-root",
        str(tmp_path),
        "--skip-runtime-image-validation",
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    metadata = json.loads((tmp_path / "runs" / payload["run_id"] / "run_metadata.json").read_text())
    assert metadata["execution_backend"]["name"] == "docker"
    assert metadata["execution_backend"]["docker_image"] == "ml-autoresearch-runner:local"
    assert metadata["execution_backend"]["gpu_policy"] == "disabled_by_default"


def test_run_candidate_uses_configured_docker_defaults_when_options_are_omitted(tmp_path: Path):
    write_fake_research_problem_package(tmp_path)
    config_path = write_fake_candidate_execution_config(tmp_path, backend="docker")
    config_path.write_text(
        config_path.read_text().replace(
            'backend = "docker"\n',
            'backend = "docker"\n'
            'docker_image = "configured:runner"\n'
            'docker_enable_gpu = true\n'
            'docker_gpu_device = "2"\n'
            'docker_rootless_container_root = true\n',
        )
    )

    completed = run_cli(
        "run-candidate",
        "--candidate",
        str(tmp_path / "missing"),
        "--runs-root",
        str(tmp_path / "runs"),
        "--workspace-root",
        str(tmp_path),
        "--skip-runtime-image-validation",
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    metadata = json.loads((tmp_path / "runs" / payload["run_id"] / "run_metadata.json").read_text())
    assert metadata["execution_backend"]["docker_image"] == "configured:runner"
    assert metadata["execution_backend"]["gpu_policy"] == "enabled_by_harness_configuration"
    assert metadata["execution_backend"]["gpu_device"] == "2"
    assert metadata["execution_backend"]["rootless_container_root"] is True


def test_run_candidate_explicit_docker_options_override_configured_policy(tmp_path: Path):
    write_fake_research_problem_package(tmp_path)
    config_path = write_fake_candidate_execution_config(tmp_path, backend="docker")
    config_path.write_text(
        config_path.read_text().replace(
            'backend = "docker"\n',
            'backend = "docker"\n'
            'docker_image = "configured:runner"\n'
            'docker_enable_gpu = true\n'
            'docker_gpu_device = "2"\n'
            'docker_rootless_container_root = true\n',
        )
    )

    completed = run_cli(
        "run-candidate",
        "--candidate",
        str(tmp_path / "missing"),
        "--runs-root",
        str(tmp_path / "runs"),
        "--workspace-root",
        str(tmp_path),
        "--docker-image",
        "explicit:runner",
        "--no-docker-enable-gpu",
        "--docker-user",
        "123:456",
        "--skip-runtime-image-validation",
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    metadata = json.loads((tmp_path / "runs" / payload["run_id"] / "run_metadata.json").read_text())
    assert metadata["execution_backend"]["docker_image"] == "explicit:runner"
    assert metadata["execution_backend"]["gpu_policy"] == "disabled_by_default"
    assert "gpu_device" not in metadata["execution_backend"]
    assert metadata["execution_backend"]["docker_user"] == "123:456"
    assert metadata["execution_backend"]["rootless_container_root"] is False


def test_run_candidate_native_escape_hatch_records_developer_unsafe_backend(tmp_path: Path):
    write_fake_research_problem_package(tmp_path)
    write_fake_candidate_execution_config(tmp_path)
    completed = run_cli(
        "run-candidate",
        "--candidate",
        str(tmp_path / "missing"),
        "--runs-root",
        str(tmp_path / "runs"),
        "--workspace-root",
        str(tmp_path),
        "--backend",
        "native",
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    metadata = json.loads((tmp_path / "runs" / payload["run_id"] / "run_metadata.json").read_text())
    assert metadata["execution_backend"] == {"name": "native", "developer_unsafe": True}
