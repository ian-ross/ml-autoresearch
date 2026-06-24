import json
from pathlib import Path

import pytest
import yaml

from ml_autoresearch.cli import app
from ml_autoresearch.execution import OperationResult
from ml_autoresearch.evaluation_requests import (
    EvaluationRequestError,
    run_post_run_evaluation,
    validate_evaluation_request_file,
)
from conftest import invoke_typer_cli


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_ground_camera_problem_provider(root: Path) -> None:
    package = root / "ground_camera_problem"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text("")
    (package / "research_problem.py").write_text(
        "from ml_autoresearch.research_problems import ResearchProblemSpec\n"
        "def build_spec(data_config=None):\n"
        "    return ResearchProblemSpec(\n"
        "        id='ground_camera_contrail_detection', version='v1', contract_version='v0',\n"
        "        input_modes=('single_frame_rgb',), input_specs={'single_frame_rgb': {'mode': 'single_frame_rgb', 'shape': [3, 64, 64]}},\n"
        "        output_forms=('mask_logits',), output_specs={'mask_logits': {'form': 'mask_logits', 'shape': [1, 64, 64]}},\n"
        "        losses=('bce_dice',), optimizers=('adamw',), sampling_policies=('sequential',),\n"
        "        augmentation_policies=('none',), primary_metric='val/dice',\n"
        "    )\n"
    )


def write_run(runs_root: Path, run_id: str = "run_123") -> Path:
    package_root = runs_root.parent / "research-problem-package"
    package_root.mkdir(exist_ok=True)
    write_ground_camera_problem_provider(package_root)
    run_dir = runs_root / run_id
    (run_dir / "outputs").mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "completed",
                "research_problem": {
                    "id": "ground_camera_contrail_detection",
                    "version": "v1",
                    "contract_version": "v0",
                    "provider": {
                        "target": "ground_camera_problem.research_problem:build_spec",
                        "resolved_package_root": str(package_root),
                    },
                },
            }
        )
    )
    (run_dir / "outputs" / "final_metrics.json").write_text(json.dumps({"val/dice": 0.42}))
    return run_dir


def write_legacy_run_without_research_problem(runs_root: Path, run_id: str = "run_123") -> Path:
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
    assert metadata["research_problem"]["id"] == "ground_camera_contrail_detection"
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


def test_post_run_evaluation_rejects_legacy_run_without_research_problem_provider_metadata(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    write_legacy_run_without_research_problem(runs_root)
    request_path = write_request(tmp_path / "request.yaml")

    with pytest.raises(EvaluationRequestError, match="research_problem provider metadata is required"):
        run_post_run_evaluation(request_path, runs_root=runs_root, ledger_path=tmp_path / "ledger.jsonl")

    assert not (runs_root / "run_123" / "outputs" / "evaluations").exists()


def write_fake_problem_provider_without_evaluation_adapter(root: Path) -> None:
    package = root / "fake_problem"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "research_problem.py").write_text(
        "from ml_autoresearch.research_problems import ResearchProblemSpec\n"
        "def build_spec(data_config=None):\n"
        "    return ResearchProblemSpec(\n"
        "        id='fake_problem', version='v1', contract_version='v0',\n"
        "        input_modes=('fake_rgb',), input_specs={'fake_rgb': {'mode': 'fake_rgb', 'shape': [3, 8, 8]}},\n"
        "        output_forms=('mask_logits',), output_specs={'mask_logits': {'form': 'mask_logits', 'shape': [1, 8, 8]}},\n"
        "        losses=('bce_dice',), optimizers=('adamw',), sampling_policies=('sequential',),\n"
        "        augmentation_policies=('none',), primary_metric='val/dice',\n"
        "    )\n"
    )


def write_fake_provider_run(runs_root: Path, package_root: Path, run_id: str = "run_123") -> Path:
    run_dir = runs_root / run_id
    (run_dir / "outputs").mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "completed",
                "research_problem": {
                    "id": "fake_problem",
                    "version": "v1",
                    "contract_version": "v0",
                    "provider": {
                        "target": "fake_problem.research_problem:build_spec",
                        "resolved_package_root": str(package_root),
                    },
                },
            }
        )
    )
    return run_dir


def test_threshold_sweep_metadata_records_research_problem_provider_provenance(tmp_path: Path) -> None:
    write_fake_problem_provider_without_evaluation_adapter(tmp_path)
    runs_root = tmp_path / "runs"
    write_fake_provider_run(runs_root, tmp_path)
    result = run_post_run_evaluation(write_request(tmp_path / "request.yaml"), runs_root=runs_root, ledger_path=tmp_path / "ledger.jsonl")

    metadata = result["evaluation"]
    assert metadata["research_problem"]["id"] == "fake_problem"
    assert metadata["research_problem"]["provider"]["target"] == "fake_problem.research_problem:build_spec"
    assert metadata["research_problem"]["provider"]["resolved_package_root"] == str(tmp_path.resolve())


def test_failure_bucket_review_without_research_problem_evaluation_adapter_does_not_create_artifacts(tmp_path: Path) -> None:
    write_fake_problem_provider_without_evaluation_adapter(tmp_path)
    runs_root = tmp_path / "runs"
    run_dir = write_fake_provider_run(runs_root, tmp_path)
    request_path = write_request(
        tmp_path / "request.yaml",
        evaluation_mode="failure_bucket_review",
        parameters={"failure_bucket_count": 3},
    )

    with pytest.raises(EvaluationRequestError, match="does not support evaluation mode"):
        run_post_run_evaluation(request_path, runs_root=runs_root, ledger_path=tmp_path / "ledger.jsonl")

    assert not (run_dir / "outputs" / "evaluations").exists()


def test_run_post_run_evaluation_cli_requires_request(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    write_run(runs_root)

    completed = invoke_typer_cli(
        app,
        [
            "run-post-run-evaluation",
            "--runs-root",
            str(runs_root),
        ],
    )

    assert completed.returncode != 0
    assert "request" in completed.stderr


class FakeDockerBackend:
    name = "docker"
    calls: list[dict[str, Path]] = []

    def __init__(self, *args, **kwargs):
        pass

    def run_post_run_evaluation(self, request_path: str | Path, *, runs_root: str | Path, ledger_path: str | Path) -> OperationResult:
        from ml_autoresearch.evaluation_requests import _evaluation_id, validate_evaluation_request_file
        from ml_autoresearch.research_ledger import record_research_event

        request = validate_evaluation_request_file(request_path)
        assert request.request_id is not None
        evaluation_id = _evaluation_id(request.request_id)
        evaluation_dir = Path(runs_root) / request.target_run_id / "outputs" / "evaluations" / evaluation_id
        evaluation_dir.mkdir(parents=True)
        metadata = {"evaluation_id": evaluation_id, "request_id": request.request_id, "parent_run_id": request.target_run_id}
        (evaluation_dir / "evaluation_metadata.json").write_text(json.dumps(metadata) + "\n")
        record_research_event(
            "evaluation_requested",
            {
                "evaluation_request_id": request.request_id,
                "request_path": str(request_path),
                "run_id": request.target_run_id,
                "evaluation_mode": request.evaluation_mode,
            },
            ledger_path=ledger_path,
        )
        record_research_event(
            "evaluation_completed",
            {
                "evaluation_id": evaluation_id,
                "evaluation_request_id": request.request_id,
                "run_id": request.target_run_id,
                "evaluation_mode": request.evaluation_mode,
                "artifact_metadata_path": str((evaluation_dir / "evaluation_metadata.json").resolve()),
            },
            ledger_path=ledger_path,
        )
        self.calls.append({"request_path": Path(request_path), "runs_root": Path(runs_root), "ledger_path": Path(ledger_path)})
        return OperationResult(backend=self.name, operation="run_post_run_evaluation")


def test_run_post_run_evaluation_cli_validates_request_and_records_linkage(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    write_run(runs_root)
    request_path = write_request(tmp_path / "request.yaml")
    ledger = tmp_path / "research-ledger.jsonl"

    completed = invoke_typer_cli(
        app,
        [
            "run-post-run-evaluation",
            "--request",
            str(request_path),
            "--runs-root",
            str(runs_root),
            "--ledger-path",
            str(ledger),
        ],
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["request"]["target_run_id"] == "run_123"
    assert payload["evaluation"]["parent_run_id"] == "run_123"
    assert read_jsonl(ledger)[1]["evaluation_id"] == payload["evaluation"]["evaluation_id"]


def test_run_post_run_evaluation_cli_uses_configured_docker_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ml_autoresearch.cli as cli
    import ml_autoresearch.execution as execution

    runs_root = tmp_path / "runs"
    write_run(runs_root)
    request_path = write_request(tmp_path / "request.yaml")
    ledger = tmp_path / "research-ledger.jsonl"
    (tmp_path / "ml-autoresearch.toml").write_text(
        "[candidate_execution]\n"
        "backend = \"docker\"\n"
        "docker_image = \"custom:tag\"\n"
        f"runs_root = \"{runs_root}\"\n"
        f"ledger_path = \"{ledger}\"\n"
    )
    FakeDockerBackend.calls = []
    monkeypatch.setattr(execution, "DockerBackend", FakeDockerBackend)
    monkeypatch.setattr(
        cli,
        "run_post_run_evaluation",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("host native evaluation must not run")),
    )

    completed = invoke_typer_cli(
        app,
        [
            "run-post-run-evaluation",
            "--request",
            str(request_path),
            "--runs-root",
            str(runs_root),
            "--workspace-root",
            str(tmp_path),
            "--skip-runtime-image-validation",
        ],
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "completed"
    assert payload["backend"] == "docker"
    assert payload["evaluation_id"] == "eval_eval-threshold-sweep-run-123"
    assert FakeDockerBackend.calls == [{"request_path": request_path, "runs_root": runs_root, "ledger_path": ledger}]
    assert [row["event_type"] for row in read_jsonl(ledger)] == ["evaluation_requested", "evaluation_completed"]


def test_run_post_run_evaluation_cli_reports_ledger_write_failure_without_traceback(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    write_run(runs_root)
    request_path = write_request(tmp_path / "request.yaml")

    completed = invoke_typer_cli(
        app,
        [
            "run-post-run-evaluation",
            "--request",
            str(request_path),
            "--runs-root",
            str(runs_root),
            "--ledger-path",
            str(tmp_path),
        ],
    )

    assert completed.returncode == 1
    assert "Traceback" not in completed.stderr
    assert str(tmp_path) in completed.stderr
