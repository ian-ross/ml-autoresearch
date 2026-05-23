import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from ml_autoresearch.evaluation_requests import (
    EvaluationRequestError,
    run_post_run_evaluation,
    validate_evaluation_request_file,
)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_run(runs_root: Path, run_id: str = "run_123") -> Path:
    run_dir = runs_root / run_id
    (run_dir / "outputs").mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(json.dumps({"run_id": run_id, "status": "completed"}))
    (run_dir / "outputs" / "final_metrics.json").write_text(json.dumps({"val/dice": 0.42}))
    return run_dir


def write_request(path: Path, **overrides: object) -> Path:
    data = {
        "request_id": "eval-threshold-sweep-run-123",
        "target_run_id": "run_123",
        "evaluation_mode": "threshold_sweep",
        "diagnostic_question": "Which probability threshold best separates thin contrail masks?",
        "expected_decision_impact": "Decide whether low Dice is thresholding or representation failure.",
        "parameters": {
            "primary_threshold": 0.5,
            "threshold_sweep": {"min": 0.1, "max": 0.9, "steps": 9},
            "batch_size": 8,
            "artifact_count": 4,
            "failure_bucket_count": 3,
        },
        "artifact_budget": {"max_artifacts": 6, "max_runtime_seconds": 120},
    }
    data.update(overrides)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def test_validate_evaluation_request_file_accepts_valid_request(tmp_path: Path) -> None:
    request = validate_evaluation_request_file(write_request(tmp_path / "request.yaml"))

    assert request.request_id == "eval-threshold-sweep-run-123"
    assert request.target_run_id == "run_123"
    assert request.evaluation_mode == "threshold_sweep"
    assert request.parameters.threshold_sweep.min == 0.1
    assert request.artifact_budget.max_artifacts == 6


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("evaluation_mode", "custom_python", "evaluation_mode"),
        ("parameters", {"primary_threshold": 1.5}, "primary_threshold"),
        ("parameters", {"threshold_sweep": {"min": 0.8, "max": 0.2, "steps": 4}}, "threshold_sweep"),
        ("parameters", {"batch_size": 0}, "batch_size"),
        ("parameters", {"artifact_count": 101}, "artifact_count"),
        ("parameters", {"failure_bucket_count": 0}, "failure_bucket_count"),
        ("artifact_budget", {"max_artifacts": 101, "max_runtime_seconds": 60}, "max_artifacts"),
    ],
)
def test_validate_evaluation_request_file_rejects_unapproved_modes_and_unbounded_parameters(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    with pytest.raises(EvaluationRequestError, match=match):
        validate_evaluation_request_file(write_request(tmp_path / "bad.yaml", **{field: value}))


@pytest.mark.parametrize("target_run_id", ["../run_123", "/etc/passwd", "run_123/../evil", "..\\run_123", "run..run", "run 123"])
def test_validate_evaluation_request_file_rejects_unsafe_target_run_id(tmp_path: Path, target_run_id: str) -> None:
    with pytest.raises(EvaluationRequestError, match="target_run_id"):
        validate_evaluation_request_file(write_request(tmp_path / "bad.yaml", target_run_id=target_run_id))


def test_post_run_evaluation_requires_valid_request_and_links_artifacts_to_request_and_parent_run(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    write_run(runs_root)
    request_path = write_request(tmp_path / "request.yaml")
    ledger = tmp_path / "research-ledger.jsonl"

    result = run_post_run_evaluation(request_path, runs_root=runs_root, ledger_path=ledger)

    assert result["evaluation_id"] == "eval_eval-threshold-sweep-run-123"
    evaluation_dir = runs_root / "run_123" / "outputs" / "evaluations" / "eval_eval-threshold-sweep-run-123"
    metadata = json.loads((evaluation_dir / "evaluation_metadata.json").read_text())
    summary = json.loads((evaluation_dir / "summary.json").read_text())
    assert metadata["request_id"] == "eval-threshold-sweep-run-123"
    assert metadata["parent_run_id"] == "run_123"
    assert metadata["artifacts"]["summary"] == "outputs/evaluations/eval_eval-threshold-sweep-run-123/summary.json"
    assert summary["request_id"] == "eval-threshold-sweep-run-123"
    assert summary["parent_run_id"] == "run_123"

    rows = read_jsonl(ledger)
    assert [row["event_type"] for row in rows] == ["evaluation_requested", "evaluation_completed"]
    assert rows[0]["evaluation_request_id"] == "eval-threshold-sweep-run-123"
    assert rows[0]["run_id"] == "run_123"
    assert rows[1]["evaluation_id"] == "eval_eval-threshold-sweep-run-123"
    assert rows[1]["artifact_metadata_path"] == str((runs_root / "run_123" / "outputs" / "evaluations" / "eval_eval-threshold-sweep-run-123" / "evaluation_metadata.json").resolve())


def test_failed_evaluation_request_validation_does_not_create_artifacts_or_success_ledger_event(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    write_run(runs_root)
    ledger = tmp_path / "research-ledger.jsonl"
    request_path = write_request(tmp_path / "bad.yaml", evaluation_mode="arbitrary")

    with pytest.raises(EvaluationRequestError):
        run_post_run_evaluation(request_path, runs_root=runs_root, ledger_path=ledger)

    assert not (runs_root / "run_123" / "outputs" / "evaluations").exists()
    assert not ledger.exists()


def test_run_post_run_evaluation_cli_requires_request(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    write_run(runs_root)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_autoresearch.cli",
            "run-post-run-evaluation",
            "--runs-root",
            str(runs_root),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode != 0
    assert "request" in completed.stderr


def test_run_post_run_evaluation_cli_validates_request_and_records_linkage(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    write_run(runs_root)
    request_path = write_request(tmp_path / "request.yaml")
    ledger = tmp_path / "research-ledger.jsonl"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_autoresearch.cli",
            "run-post-run-evaluation",
            "--request",
            str(request_path),
            "--runs-root",
            str(runs_root),
            "--ledger-path",
            str(ledger),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["request"]["target_run_id"] == "run_123"
    assert payload["evaluation"]["parent_run_id"] == "run_123"
    assert read_jsonl(ledger)[1]["evaluation_id"] == payload["evaluation"]["evaluation_id"]


def test_run_post_run_evaluation_cli_reports_ledger_write_failure_without_traceback(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    write_run(runs_root)
    request_path = write_request(tmp_path / "request.yaml")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_autoresearch.cli",
            "run-post-run-evaluation",
            "--request",
            str(request_path),
            "--runs-root",
            str(runs_root),
            "--ledger-path",
            str(tmp_path),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 1
    assert "Traceback" not in completed.stderr
    assert str(tmp_path) in completed.stderr
