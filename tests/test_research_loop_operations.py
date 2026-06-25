from __future__ import annotations

from pathlib import Path

from research_problem_helpers import write_fake_candidate_execution_config, write_fake_research_problem_package
from test_cli_submission import write_valid_candidate_with_proposal
from test_cli_experiment_batch import write_batch

from ml_autoresearch.research_loop_operations import (
    run_candidate_from_workspace,
    run_experiment_batch_from_workspace,
)


def write_fake_execution_config(root: Path) -> None:
    write_fake_research_problem_package(root)
    write_fake_candidate_execution_config(root)


def test_run_candidate_from_workspace_uses_research_workspace_configuration(tmp_path: Path):
    candidate = write_valid_candidate_with_proposal(tmp_path)
    runs_root = tmp_path / "configured-runs"
    write_fake_execution_config(tmp_path)
    config_path = tmp_path / "ml-autoresearch.toml"
    config_path.write_text(config_path.read_text().replace('backend = "native"\n', f'backend = "native"\nruns_root = "{runs_root}"\n'))

    result = run_candidate_from_workspace(candidate, workspace_root=tmp_path, backend_name="native")

    assert result["status"] == "completed"
    assert (runs_root / str(result["run_id"]) / "outputs" / "final_metrics.json").exists()


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
