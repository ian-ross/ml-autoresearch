from __future__ import annotations

from pathlib import Path

import pytest

from research_problem_helpers import write_fake_candidate_execution_config, write_fake_research_problem_package
from test_cli_submission import write_valid_candidate_with_proposal
from test_cli_experiment_batch import write_batch

from ml_autoresearch.candidate_execution_config import CandidateExecutionConfig
from ml_autoresearch.research_loop_operations import (
    _run_ingested_experiment_batch,
    effective_execution_options,
    run_candidate_from_workspace,
    run_experiment_batch_from_workspace,
    select_execution_backend,
)


def write_fake_execution_config(root: Path) -> None:
    write_fake_research_problem_package(root)
    write_fake_candidate_execution_config(root)


def test_select_execution_backend_pins_one_harness_gpu() -> None:
    backend = select_execution_backend("docker", docker_enable_gpu=True, docker_gpu_device="0")

    assert backend.enable_gpu is True
    assert backend.gpu_device == "0"
    timed_backend = select_execution_backend(
        "docker",
        training_wall_clock_timeout_seconds=1800,
    )
    assert timed_backend.wall_clock_timeout_seconds == 1800
    with pytest.raises(ValueError, match="requires --docker-enable-gpu"):
        select_execution_backend("docker", docker_gpu_device="0")
    with pytest.raises(ValueError, match="must identify one GPU"):
        select_execution_backend("docker", docker_enable_gpu=True, docker_gpu_device="0,1")


def test_effective_execution_options_cannot_raise_workspace_sample_ceiling() -> None:
    config = CandidateExecutionConfig(max_samples=128, max_prediction_samples=2)

    assert effective_execution_options(
        config,
        max_samples=1024,
        max_prediction_samples=9,
        prediction_sample_policy=None,
    ) == (128, 9, "first_n")


def test_run_candidate_from_workspace_uses_research_workspace_configuration(tmp_path: Path):
    candidate = write_valid_candidate_with_proposal(tmp_path)
    runs_root = tmp_path / "configured-runs"
    write_fake_execution_config(tmp_path)
    config_path = tmp_path / "ml-autoresearch.toml"
    config_path.write_text(config_path.read_text().replace('backend = "native"\n', f'backend = "native"\nruns_root = "{runs_root}"\n'))

    result = run_candidate_from_workspace(candidate, workspace_root=tmp_path, backend_name="native")

    assert result["status"] == "completed"
    assert (runs_root / str(result["run_id"]) / "outputs" / "final_metrics.json").exists()


def test_run_candidate_from_workspace_rejects_manifest_above_workspace_epoch_ceiling(tmp_path: Path):
    candidate = write_valid_candidate_with_proposal(tmp_path)
    manifest_path = candidate / "manifest.yaml"
    manifest_path.write_text(manifest_path.read_text().replace("max_epochs: 1", "max_epochs: 2"))
    write_fake_execution_config(tmp_path)
    config_path = tmp_path / "ml-autoresearch.toml"
    config_path.write_text(config_path.read_text().replace('backend = "native"\n', 'backend = "native"\nmax_epochs = 1\n'))

    result = run_candidate_from_workspace(candidate, workspace_root=tmp_path, backend_name="native")

    assert result["status"] == "rejected"
    assert "Workspace ceiling 1" in str(result["rejection_reason"])


def test_run_experiment_batch_from_workspace_returns_serializable_summary(tmp_path: Path):
    batch = write_batch(tmp_path)
    write_fake_execution_config(tmp_path)

    result = run_experiment_batch_from_workspace(
        batch,
        batches_root=tmp_path / "batches",
        runs_root=tmp_path / "runs",
        workspace_root=tmp_path,
        backend_name="native",
        max_samples=2,
        max_prediction_samples=1,
    )

    assert result["status"] == "completed"
    assert str(result["batch_id"]).startswith("batch_")
    assert len(result["runs"]) == 1


def test_ingested_experiment_batch_uses_configured_parallel_run_cap(tmp_path: Path, monkeypatch):
    write_fake_execution_config(tmp_path)
    config_path = tmp_path / "ml-autoresearch.toml"
    config_path.write_text(
        config_path.read_text().replace(
            'backend = "native"\n',
            'backend = "native"\nmax_parallel_runs = 2\nmax_epochs = 3\n',
        )
    )
    captured: dict[str, object] = {}

    def fake_run(batch_path, **kwargs):
        captured.update(kwargs)
        return {"status": "completed", "batch_path": str(batch_path)}

    monkeypatch.setattr("ml_autoresearch.batches.run_experiment_batch_with_research_problem", fake_run)

    result = _run_ingested_experiment_batch(tmp_path, tmp_path / "experiment-batches" / "batch_a")

    assert result["status"] == "completed"
    assert captured["max_parallel_runs"] == 2
    assert captured["max_epochs"] == 3
