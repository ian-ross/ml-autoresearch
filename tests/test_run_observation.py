import json
import tomllib
from pathlib import Path

from ml_autoresearch.agent_cli import app as agent_app
from ml_autoresearch.cli import app
from ml_autoresearch.runs import get_best_runs, get_run_summary, list_runs
from conftest import invoke_typer_cli
from research_problem_helpers import write_static_candidate_execution_config


def write_run(
    runs_root: Path,
    run_id: str,
    status: str,
    dice: float | None = None,
    *,
    reason: str | None = None,
    best_dice: float | None = None,
    failure_classification: str | None = None,
    research_problem: dict[str, str] | None = None,
) -> Path:
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True)
    metadata = {
        "run_id": run_id,
        "status": status,
        "created_at": f"2026-05-02T00:00:0{len(list(runs_root.iterdir()))}Z",
        "updated_at": f"2026-05-02T00:00:1{len(list(runs_root.iterdir()))}Z",
        "candidate_source": {"path": "/tmp/candidate"},
        "rejection_reason": reason if status == "rejected" else None,
        "smoke_failure_reason": reason if status == "smoke_failed" else None,
        "training_failure_reason": reason if status == "failed" else None,
        "failure_classification": failure_classification,
    }
    if research_problem is not None:
        metadata["research_problem"] = research_problem
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata) + "\n")
    if dice is not None:
        outputs_dir = run_dir / "outputs"
        outputs_dir.mkdir()
        (outputs_dir / "final_metrics.json").write_text(json.dumps({"val/dice": dice, "val/iou": dice / 2}) + "\n")
        if best_dice is not None:
            (outputs_dir / "best_metrics.json").write_text(
                json.dumps(
                    {
                        "epoch": 1,
                        "selection_metric": "val/dice",
                        "selection_mode": "max",
                        "selection_value": best_dice,
                        "metrics": {"epoch": 1, "val/dice": best_dice, "val/iou": best_dice / 2},
                    }
                )
                + "\n"
            )
    return run_dir


def run_cli(*args: str):
    return invoke_typer_cli(app, args)


def run_agent_cli(*args: str):
    return invoke_typer_cli(agent_app, args)


def test_list_runs_summarizes_local_run_artifacts_and_reports_corrupt_runs(tmp_path: Path):
    runs_root = tmp_path / "runs"
    write_run(runs_root, "run_b", "completed", 0.7)
    write_run(runs_root, "run_a", "rejected", reason="forbidden file")
    corrupt = runs_root / "run_c"
    corrupt.mkdir()
    (corrupt / "run_metadata.json").write_text("not json")

    summaries = list_runs(runs_root)

    assert [summary["run_id"] for summary in summaries] == ["run_a", "run_b", "run_c"]
    assert summaries[0]["status"] == "rejected"
    assert summaries[0]["reason"] == "forbidden file"
    assert summaries[1]["metrics"]["val/dice"] == 0.7
    assert summaries[2]["status"] == "corrupt"
    assert "error" in summaries[2]


def test_list_runs_table_prints_empty_reason_when_reason_is_missing(tmp_path: Path):
    runs_root = tmp_path / "runs"
    write_run(runs_root, "run_missing", "rejected")

    completed = run_cli("list-runs", "--runs-root", str(runs_root))

    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert len(lines) == 2
    assert "\tNone\t" not in lines[1]
    parts = lines[1].split("\t")
    assert len(parts) == 4
    assert parts[3] == ""


def test_agent_cli_list_runs_table_prints_empty_reason_when_reason_is_missing(tmp_path: Path):
    runs_root = tmp_path / "runs"
    write_run(runs_root, "run_missing", "rejected")

    completed = run_agent_cli("list-runs", "--runs-root", str(runs_root))

    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert len(lines) == 2
    assert "\tNone\t" not in lines[1]
    parts = lines[1].split("\t")
    assert len(parts) == 4
    assert parts[3] == ""


def test_get_run_summary_reads_one_run_without_mlflow(tmp_path: Path):
    runs_root = tmp_path / "runs"
    write_run(
        runs_root,
        "run_done",
        "completed",
        0.82,
        best_dice=0.9,
        research_problem={"id": "ground_camera_contrail_detection", "version": "v0", "contract_version": "v0"},
    )

    summary = get_run_summary(runs_root, "run_done")

    assert summary["run_id"] == "run_done"
    assert summary["status"] == "completed"
    assert summary["research_problem"] == {"id": "ground_camera_contrail_detection", "version": "v0", "contract_version": "v0"}
    assert summary["metrics"]["val/dice"] == 0.82
    assert summary["best_metrics"]["selection_metric"] == "val/dice"
    assert summary["best_metrics"]["selection_value"] == 0.9
    assert summary["run_dir"] == str(runs_root / "run_done")
    assert summary["evaluations"] == []


def test_get_best_runs_sorts_completed_runs_by_best_val_dice_when_available(tmp_path: Path):
    runs_root = tmp_path / "runs"
    write_run(runs_root, "run_high_final_low_best", "completed", 0.9, best_dice=0.3)
    write_run(runs_root, "run_low_final_high_best", "completed", 0.2, best_dice=0.8)

    best = get_best_runs(runs_root)

    assert [summary["run_id"] for summary in best] == ["run_low_final_high_best", "run_high_final_low_best"]
    assert [summary["rank_metric"] for summary in best] == [0.8, 0.3]


def test_get_best_runs_sorts_completed_runs_by_val_dice_descending_and_skips_unrankable(tmp_path: Path):
    runs_root = tmp_path / "runs"
    write_run(runs_root, "run_low", "completed", 0.2)
    write_run(runs_root, "run_failed", "failed", reason="boom")
    write_run(runs_root, "run_high", "completed", 0.9)
    write_run(runs_root, "run_no_metric", "completed")

    best = get_best_runs(runs_root)

    assert [summary["run_id"] for summary in best] == ["run_high", "run_low"]
    assert [summary["rank_metric"] for summary in best] == [0.9, 0.2]


def test_observation_cli_json_commands_are_thin_over_local_artifacts(tmp_path: Path):
    runs_root = tmp_path / "runs"
    write_run(runs_root, "run_low", "completed", 0.2)
    write_run(runs_root, "run_high", "completed", 0.9)

    listed = run_cli("list-runs", "--runs-root", str(runs_root), "--json")
    summary = run_cli("run-summary", "--runs-root", str(runs_root), "--run-id", "run_high", "--json")
    best = run_cli("get-best-runs", "--runs-root", str(runs_root), "--json")

    assert listed.returncode == 0, listed.stderr
    assert summary.returncode == 0, summary.stderr
    assert best.returncode == 0, best.stderr
    assert [item["run_id"] for item in json.loads(listed.stdout)] == ["run_high", "run_low"]
    assert json.loads(summary.stdout)["metrics"]["val/dice"] == 0.9
    assert [item["run_id"] for item in json.loads(best.stdout)] == ["run_high", "run_low"]


def test_agent_cli_is_installed_as_separate_console_script() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert pyproject["project"]["scripts"]["ml-autoresearch-agent"] == "ml_autoresearch.agent_cli:main"


def test_agent_cli_help_is_agent_safe_and_excludes_execution_authority() -> None:
    completed = run_agent_cli("--help")

    assert completed.returncode == 0, completed.stderr
    help_text = completed.stdout
    assert "Agent-safe" in help_text
    assert "cannot run Candidate Experiments" in help_text
    for allowed in ["list-runs", "run-summary", "get-best-runs", "validate-candidate", "prepare-candidate-submission"]:
        assert allowed in help_text
    for disallowed in [
        "run-candidate",
        "submit-candidate",
        "evaluate-run",
        "run-post-run-evaluation",
        "validate-docker-gpu",
        "record-research-event",
        "pause-campaign",
        "record-campaign-report",
    ]:
        assert disallowed not in help_text


def test_agent_cli_observation_commands_work_against_fixture_runs(tmp_path: Path):
    runs_root = tmp_path / "runs"
    write_run(runs_root, "run_low", "completed", 0.2)
    write_run(runs_root, "run_high", "completed", 0.9)

    listed = run_agent_cli("list-runs", "--runs-root", str(runs_root), "--json")
    summary = run_agent_cli("run-summary", "--runs-root", str(runs_root), "--run-id", "run_high", "--json")
    best = run_agent_cli("get-best-runs", "--runs-root", str(runs_root), "--json")

    assert listed.returncode == 0, listed.stderr
    assert summary.returncode == 0, summary.stderr
    assert best.returncode == 0, best.stderr
    assert [item["run_id"] for item in json.loads(listed.stdout)] == ["run_high", "run_low"]
    assert json.loads(summary.stdout)["metrics"]["val/dice"] == 0.9
    assert [item["run_id"] for item in json.loads(best.stdout)] == ["run_high", "run_low"]


def test_agent_cli_validate_candidate_can_require_submission_readme(tmp_path: Path):
    write_static_candidate_execution_config(tmp_path)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        """
name: static_candidate
input_mode: single_frame_rgb
output_form: mask_logits
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )
    (candidate / "model.py").write_text("raise RuntimeError('model.py should not be imported during static validation')\n")

    missing = run_agent_cli(
        "validate-candidate",
        "--candidate",
        str(candidate),
        "--no-require-proposal",
        "--require-readme",
        "--project-root",
        str(tmp_path),
    )
    (candidate / "README.md").write_text("# Static candidate\n")
    valid = run_agent_cli(
        "validate-candidate",
        "--candidate",
        str(candidate),
        "--no-require-proposal",
        "--require-readme",
        "--project-root",
        str(tmp_path),
    )

    assert missing.returncode == 1
    assert "README.md" in json.loads(missing.stdout)["reason"]
    assert valid.returncode == 0, valid.stderr
    assert json.loads(valid.stdout)["status"] == "valid"


def test_agent_cli_validate_candidate_accepts_boundary_auxiliary_target(tmp_path: Path):
    write_static_candidate_execution_config(tmp_path)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        """
name: boundary_aux_candidate
input_mode: single_frame_rgb
output_form: mask_logits
auxiliary_targets:
  - name: boundary
    output: boundary_logits
    loss: weighted_bce
    weight: 0.05
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )
    (candidate / "model.py").write_text("raise RuntimeError('model.py should not be imported during static validation')\n")

    completed = run_agent_cli(
        "validate-candidate", "--candidate", str(candidate), "--no-require-proposal", "--project-root", str(tmp_path)
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "valid"
    assert payload["manifest"]["auxiliary_targets"] == [
        {"name": "boundary", "output": "boundary_logits", "loss": "weighted_bce", "weight": 0.05}
    ]


def test_agent_cli_validate_candidate_is_static_and_does_not_import_model_code(tmp_path: Path):
    write_static_candidate_execution_config(tmp_path)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        """
name: static_candidate
input_mode: single_frame_rgb
output_form: mask_logits
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )
    (candidate / "model.py").write_text("raise RuntimeError('model.py should not be imported during static validation')\n")

    completed = run_agent_cli(
        "validate-candidate", "--candidate", str(candidate), "--no-require-proposal", "--project-root", str(tmp_path)
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "valid"


def write_evaluation(run_dir: Path, evaluation_id: str, metadata: dict[str, object] | str | None) -> Path:
    evaluation_dir = run_dir / "outputs" / "evaluations" / evaluation_id
    evaluation_dir.mkdir(parents=True)
    if isinstance(metadata, dict):
        (evaluation_dir / "evaluation_metadata.json").write_text(json.dumps(metadata) + "\n")
    elif isinstance(metadata, str):
        (evaluation_dir / "evaluation_metadata.json").write_text(metadata)
    return evaluation_dir


def test_get_run_summary_lists_completed_running_and_failed_evaluations(tmp_path: Path):
    runs_root = tmp_path / "runs"
    run_dir = write_run(runs_root, "run_done", "completed", 0.82)
    write_evaluation(
        run_dir,
        "eval_20260506_010000_done",
        {
            "evaluation_id": "eval_20260506_010000_done",
            "status": "completed",
            "mode": "whole_validation_failure_analysis",
            "started_at": "2026-05-06T01:00:00Z",
            "completed_at": "2026-05-06T01:01:00Z",
        },
    )
    write_evaluation(
        run_dir,
        "eval_20260506_020000_running",
        {
            "evaluation_id": "eval_20260506_020000_running",
            "status": "running",
            "mode": "whole_validation_failure_analysis",
            "created_at": "2026-05-06T02:00:00Z",
        },
    )
    write_evaluation(
        run_dir,
        "eval_20260506_030000_failed",
        {
            "evaluation_id": "eval_20260506_030000_failed",
            "status": "failed",
            "mode": "whole_validation_failure_analysis",
            "started_at": "2026-05-06T03:00:00Z",
            "failed_at": "2026-05-06T03:00:30Z",
            "failure_reason": "model artifact is missing",
        },
    )

    summary = get_run_summary(runs_root, "run_done")

    evaluations = summary["evaluations"]
    assert [item["evaluation_id"] for item in evaluations] == [
        "eval_20260506_010000_done",
        "eval_20260506_020000_running",
        "eval_20260506_030000_failed",
    ]
    completed, running, failed = evaluations
    assert completed == {
        "evaluation_id": "eval_20260506_010000_done",
        "status": "completed",
        "mode": "whole_validation_failure_analysis",
        "path": str(run_dir / "outputs" / "evaluations" / "eval_20260506_010000_done"),
        "created_at": "2026-05-06T01:00:00Z",
        "completed_at": "2026-05-06T01:01:00Z",
    }
    assert running["status"] == "running"
    assert running["created_at"] == "2026-05-06T02:00:00Z"
    assert "completed_at" not in running
    assert failed["status"] == "failed"
    assert failed["failure_reason"] == "model artifact is missing"
    assert failed["completed_at"] == "2026-05-06T03:00:30Z"


def test_get_run_summary_reports_missing_and_corrupt_evaluation_metadata(tmp_path: Path):
    runs_root = tmp_path / "runs"
    run_dir = write_run(runs_root, "run_done", "completed", 0.82)
    write_evaluation(run_dir, "eval_missing_metadata", None)
    write_evaluation(run_dir, "eval_corrupt_metadata", "not json")

    summary = get_run_summary(runs_root, "run_done")

    evaluations = {item["evaluation_id"]: item for item in summary["evaluations"]}
    assert evaluations["eval_missing_metadata"]["status"] == "missing_metadata"
    assert "evaluation_metadata.json is missing" in evaluations["eval_missing_metadata"]["error"]
    assert evaluations["eval_corrupt_metadata"]["status"] == "corrupt"
    assert "cannot read evaluation_metadata.json" in evaluations["eval_corrupt_metadata"]["error"]


def test_get_best_runs_ignores_post_run_evaluation_metrics(tmp_path: Path):
    runs_root = tmp_path / "runs"
    low = write_run(runs_root, "run_low", "completed", 0.2)
    write_run(runs_root, "run_high", "completed", 0.9)
    eval_dir = write_evaluation(
        low,
        "eval_20260506_010000_done",
        {"evaluation_id": "eval_20260506_010000_done", "status": "completed", "mode": "whole_validation_failure_analysis"},
    )
    (eval_dir / "aggregate_metrics.json").write_text(json.dumps({"metrics": {"dice": 0.99}}) + "\n")

    best = get_best_runs(runs_root)

    assert [summary["run_id"] for summary in best] == ["run_high", "run_low"]
    assert [summary["rank_metric"] for summary in best] == [0.9, 0.2]


def test_run_summary_surfaces_failure_classification_without_losing_reason(tmp_path: Path):
    runs_root = tmp_path / "runs"
    write_run(
        runs_root,
        "run_timeout",
        "failed",
        reason="wall-clock budget exhausted",
        failure_classification="resource_failure",
    )

    summary = get_run_summary(runs_root, "run_timeout")

    assert summary["status"] == "failed"
    assert summary["reason"] == "wall-clock budget exhausted"
    assert summary["failure_classification"] == "resource_failure"
