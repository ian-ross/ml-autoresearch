import json
from pathlib import Path

import torch

from ml_autoresearch.cli import _daemonize_current_evaluate_run, app
from ml_autoresearch.evaluation_requests import run_post_run_evaluation
from ml_autoresearch.evaluations import _select_failure_bucket_indices, _threshold_sweep_summary, evaluate_run
from ml_autoresearch.runs import RunStatus, run_candidate_with_gvccs_data
from conftest import invoke_typer_cli


def write_valid_candidate(root: Path) -> Path:
    candidate = root / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        """
name: cli_candidate
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
    (candidate / "model.py").write_text(
        "from torch import nn\n"
        "class Tiny(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.conv = nn.Conv2d(3, 1, kernel_size=1)\n"
        "    def forward(self, x):\n"
        "        return {'mask_logits': self.conv(x)}\n"
        "def build_model(input_spec, output_spec):\n"
        "    return Tiny()\n"
    )
    return candidate


def run_cli(*args: str):
    return invoke_typer_cli(app, args)



def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_evaluate_run_api_writes_run_scoped_validation_artifacts(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    run = run_candidate_with_gvccs_data(candidate, tmp_path / "runs", "tests/fixtures/gvccs_like", max_samples=4)
    assert run.status == RunStatus.COMPLETED

    ledger = tmp_path / "evaluation-ledger.jsonl"
    result = evaluate_run(
        run.run_dir,
        split="val",
        backend="native",
        data_root="tests/fixtures/gvccs_like",
        max_artifact_samples=1,
        ledger_path=ledger,
    )

    evaluation_dir = result.evaluation_dir
    assert evaluation_dir.parent == run.run_dir / "outputs" / "evaluations"
    assert evaluation_dir.name.startswith("eval_")
    metadata = json.loads((evaluation_dir / "evaluation_metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["evaluation_id"] == evaluation_dir.name
    assert metadata["source_run"]["run_id"] == run.run_id
    assert metadata["split"] == "val"
    assert metadata["backend"] == "native"
    assert metadata["threshold"] == 0.5
    assert metadata["model_artifact"] == "outputs/models/best_epoch_model.pt"
    assert metadata["artifacts"]["threshold_sweep"] == "threshold_sweep.json"
    assert metadata["artifacts"]["diagnostic_samples"] == "diagnostic_samples/samples.json"

    aggregate = json.loads((evaluation_dir / "aggregate_metrics.json").read_text())
    assert aggregate["split"] == "val"
    assert aggregate["threshold"] == 0.5
    assert aggregate["sample_count"] == 1
    assert set(aggregate["metrics"]) == {"dice", "iou", "precision", "recall"}

    threshold_sweep = json.loads((evaluation_dir / "threshold_sweep.json").read_text())
    assert threshold_sweep["thresholds"] == [round(index * 0.05, 2) for index in range(1, 20)]
    assert threshold_sweep["default_threshold"] == 0.5
    assert set(threshold_sweep["groups"]) == {"all_samples", "positive_mask_samples", "empty_mask_samples"}
    assert len(threshold_sweep["groups"]["all_samples"]) == 19
    assert set(threshold_sweep["groups"]["all_samples"][0]["metrics"]) == {"dice", "iou", "precision", "recall"}
    assert set(threshold_sweep["best_threshold_by_dice"]) >= {"threshold", "dice"}

    records = [json.loads(line) for line in (evaluation_dir / "per_sample_metrics.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["sample_id"].startswith("val/")
    assert set(records[0]) >= {"dice", "iou", "precision", "recall", "image_path"}

    diagnostics_dir = evaluation_dir / "diagnostic_samples"
    diagnostic_manifest = json.loads((diagnostics_dir / "samples.json").read_text())
    assert diagnostic_manifest["sample_count"] == 1
    assert diagnostic_manifest["max_artifact_samples"] == 1
    diagnostic = diagnostic_manifest["samples"][0]
    assert set(diagnostic) >= {
        "source_image_path",
        "dataset_index",
        "threshold",
        "metrics",
        "positive_pixel_count",
        "predicted_positive_pixel_count",
        "true_positive_pixels",
        "false_positive_pixels",
        "false_negative_pixels",
        "bucket_memberships",
        "paths",
    }
    assert diagnostic["bucket_memberships"]
    assert diagnostic["paths"] == {
        "input": "sample_000_input.png",
        "ground_truth": "sample_000_ground_truth.png",
        "prediction": "sample_000_prediction.png",
        "overlay": "sample_000_overlay.png",
        "probability_heatmap": "sample_000_probability_heatmap.png",
    }
    for relative_path in diagnostic["paths"].values():
        assert (diagnostics_dir / relative_path).is_file()

    request = json.loads((evaluation_dir / "evaluation_request.json").read_text())
    assert request["request_id"] == result.evaluation_request_id
    events = read_jsonl(ledger)
    assert [event["event_type"] for event in events] == ["evaluation_requested", "evaluation_completed"]
    assert events[0]["evaluation_request_id"] == result.evaluation_request_id
    assert events[0]["request_path"] == str(result.request_path)
    assert events[1]["evaluation_id"] == result.evaluation_id


def test_request_gated_failure_bucket_review_writes_metrics_and_diagnostic_artifacts(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    run = run_candidate_with_gvccs_data(candidate, tmp_path / "runs", "tests/fixtures/gvccs_like", max_samples=4)
    assert run.status == RunStatus.COMPLETED
    request_path = tmp_path / "evaluation-request.yaml"
    request_path.write_text(
        f"""
request_id: eval-failure-buckets
 target_run_id: {run.run_id}
evaluation_mode: failure_bucket_review
diagnostic_question: Which validation samples explain the precision-recall tradeoff?
expected_decision_impact: Decide whether to tune recall or accept the current model.
parameters:
  primary_threshold: 0.4
  artifact_count: 1
  failure_bucket_count: 3
artifact_budget:
  max_artifacts: 2
  max_runtime_seconds: 120
""".replace("\n target_run_id", "\ntarget_run_id")
    )
    ledger = tmp_path / "ledger.jsonl"

    result = run_post_run_evaluation(request_path, runs_root=tmp_path / "runs", ledger_path=ledger)

    evaluation_dir = run.run_dir / "outputs" / "evaluations" / "eval_eval-failure-buckets"
    assert result["evaluation_id"] == "eval_eval-failure-buckets"
    assert (evaluation_dir / "aggregate_metrics.json").is_file()
    assert (evaluation_dir / "per_sample_metrics.jsonl").is_file()
    assert (evaluation_dir / "threshold_sweep.json").is_file()
    diagnostic_manifest = json.loads((evaluation_dir / "diagnostic_samples" / "samples.json").read_text())
    assert diagnostic_manifest["threshold"] == 0.4
    assert diagnostic_manifest["max_artifact_samples"] == 1
    assert diagnostic_manifest["samples"]
    diagnostic = diagnostic_manifest["samples"][0]
    assert "bucket_memberships" in diagnostic
    assert (evaluation_dir / "diagnostic_samples" / diagnostic["paths"]["overlay"]).is_file()
    assert (evaluation_dir / "diagnostic_samples" / diagnostic["paths"]["probability_heatmap"]).is_file()
    metadata = json.loads((evaluation_dir / "evaluation_metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["evaluation_mode"] == "failure_bucket_review"
    assert metadata["artifacts"]["aggregate_metrics"] == "aggregate_metrics.json"
    events = read_jsonl(ledger)
    assert [event["event_type"] for event in events] == ["evaluation_requested", "evaluation_completed"]
    assert events[1]["artifact_metadata_path"] == str((evaluation_dir / "evaluation_metadata.json").resolve())


def test_evaluate_run_api_records_failed_metadata_for_missing_model_artifact(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    run = run_candidate_with_gvccs_data(candidate, tmp_path / "runs", "tests/fixtures/gvccs_like", max_samples=4)
    assert run.status == RunStatus.COMPLETED
    (run.run_dir / "outputs" / "models" / "best_epoch_model.pt").unlink()

    try:
        evaluate_run(run.run_dir, split="val", backend="native", data_root="tests/fixtures/gvccs_like")
    except RuntimeError as exc:
        assert "model artifact is missing" in str(exc)
    else:
        raise AssertionError("expected missing model artifact failure")

    evaluation_dirs = sorted((run.run_dir / "outputs" / "evaluations").glob("eval_*"))
    assert len(evaluation_dirs) == 1
    metadata = json.loads((evaluation_dirs[0] / "evaluation_metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert "model artifact is missing" in metadata["failure_reason"]
    assert not (evaluation_dirs[0] / "aggregate_metrics.json").exists()


def test_evaluate_run_cli_uses_run_metadata_data_root_by_default(tmp_path: Path):
    candidate = write_valid_candidate(tmp_path)
    runs_root = tmp_path / "runs"
    run = run_candidate_with_gvccs_data(candidate, runs_root, "tests/fixtures/gvccs_like", max_samples=4)
    assert run.status == RunStatus.COMPLETED

    completed = run_cli(
        "evaluate-run", "--run", str(run.run_dir), "--split", "val", "--backend", "native", "--max-artifact-samples", "1"
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "completed"
    assert payload["evaluation_id"].startswith("eval_")
    evaluation_dir = Path(payload["evaluation_dir"])
    assert (evaluation_dir / "evaluation_metadata.json").exists()
    assert (evaluation_dir / "aggregate_metrics.json").exists()
    assert (evaluation_dir / "per_sample_metrics.jsonl").exists()
    assert (evaluation_dir / "threshold_sweep.json").exists()
    assert (evaluation_dir / "diagnostic_samples" / "samples.json").exists()

    request = json.loads((evaluation_dir / "evaluation_request.json").read_text())
    assert request["request_id"] == f"manual_{payload['evaluation_id']}"
    assert request["target_run_id"] == run.run_id
    assert request["evaluation_mode"] == "whole_validation_failure_analysis"

    rows = read_jsonl(tmp_path / "research-ledger.jsonl")
    evaluation_events = [row for row in rows if row["event_type"].startswith("evaluation_")]
    assert [row["event_type"] for row in evaluation_events] == ["evaluation_requested", "evaluation_completed"]
    assert evaluation_events[0]["evaluation_request_id"] == request["request_id"]
    assert evaluation_events[0]["request_path"] == str(evaluation_dir / "evaluation_request.json")
    assert evaluation_events[0]["run_id"] == run.run_id
    assert evaluation_events[0]["evaluation_mode"] == "whole_validation_failure_analysis"
    assert evaluation_events[1]["evaluation_id"] == payload["evaluation_id"]
    assert evaluation_events[1]["evaluation_request_id"] == request["request_id"]
    assert evaluation_events[1]["artifact_metadata_path"] == str(evaluation_dir / "evaluation_metadata.json")


def test_failure_bucket_selection_is_bounded_and_deduplicates_memberships():
    records = [
        {
            "dataset_index": 0,
            "dice": 0.1,
            "positive_pixel_count": 10,
            "predicted_positive_pixel_count": 0,
            "false_positive_pixels": 0,
            "false_negative_pixels": 10,
        },
        {
            "dataset_index": 1,
            "dice": 0.9,
            "positive_pixel_count": 10,
            "predicted_positive_pixel_count": 10,
            "false_positive_pixels": 0,
            "false_negative_pixels": 1,
        },
        {
            "dataset_index": 2,
            "dice": 0.2,
            "positive_pixel_count": 0,
            "predicted_positive_pixel_count": 7,
            "false_positive_pixels": 7,
            "false_negative_pixels": 0,
        },
        {
            "dataset_index": 3,
            "dice": 0.3,
            "positive_pixel_count": 8,
            "predicted_positive_pixel_count": 12,
            "false_positive_pixels": 6,
            "false_negative_pixels": 2,
        },
    ]

    selected = _select_failure_bucket_indices(records, max_artifact_samples=3)

    assert len(selected) == 3
    assert len({item["dataset_index"] for item in selected}) == 3
    memberships = {item["dataset_index"]: item["bucket_memberships"] for item in selected}
    assert "worst_by_dice" in memberships[0]
    assert "missed_positive_masks" in memberships[0]
    assert "empty_mask_false_positives" in memberships[2]
    assert "false_positive_heavy" in memberships[2] or "false_positive_heavy" in memberships[3]


def test_threshold_sweep_summary_selects_best_threshold_and_separates_sample_groups():
    probabilities = torch.tensor(
        [
            [[0.90, 0.80, 0.40, 0.10]],
            [[0.60, 0.20, 0.04, 0.01]],
        ]
    ).unsqueeze(1)
    targets = torch.tensor(
        [
            [[1.0, 1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0]],
        ]
    ).unsqueeze(1)

    summary = _threshold_sweep_summary(probabilities, targets)

    assert summary["thresholds"] == [round(index * 0.05, 2) for index in range(1, 20)]
    assert summary["default_threshold"] == 0.5
    assert summary["best_threshold_by_dice"]["threshold"] == 0.65
    assert summary["groups"]["all_samples"][0]["sample_count"] == 2
    assert summary["groups"]["positive_mask_samples"][0]["sample_count"] == 1
    assert summary["groups"]["empty_mask_samples"][0]["sample_count"] == 1

    empty_by_threshold = {item["threshold"]: item for item in summary["groups"]["empty_mask_samples"]}
    assert empty_by_threshold[0.05]["false_positive_pixels"] == 2
    assert empty_by_threshold[0.5]["false_positive_pixels"] == 1
    assert empty_by_threshold[0.65]["false_positive_pixels"] == 0
    assert empty_by_threshold[0.05]["samples_with_false_positives"] == 1
    assert empty_by_threshold[0.65]["samples_with_false_positives"] == 0


def test_evaluate_run_cli_can_daemonize_native_evaluation(tmp_path: Path, monkeypatch, capsys):
    run_dir = tmp_path / "runs" / "run_dummy"
    run_dir.mkdir(parents=True)
    popen_calls: list[dict[str, object]] = []

    class FakeProcess:
        pid = 12345

    def fake_popen(command, **kwargs):
        popen_calls.append({"command": command, **kwargs})
        return FakeProcess()

    monkeypatch.setattr("ml_autoresearch.cli.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "sys.argv",
        [
            "ml-autoresearch",
            "evaluate-run",
            "--run",
            str(run_dir),
            "--split",
            "val",
            "--backend",
            "native",
            "--max-artifact-samples",
            "1",
            "--daemonize",
        ],
    )

    _daemonize_current_evaluate_run(run_dir)

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "daemonized"
    assert payload["pid"] == 12345
    log_path = Path(payload["log_path"])
    assert log_path.parent == run_dir / "outputs" / "evaluation_daemon_logs"
    assert log_path.name.startswith("evaluate_run_")
    assert log_path.exists()
    assert "--daemonize" not in payload["command"]
    assert "--backend" in payload["command"]
    assert "native" in payload["command"]
    assert "--max-artifact-samples" in payload["command"]
    assert "1" in payload["command"]
    assert popen_calls[0]["start_new_session"] is True


def write_line_aux_candidate(root: Path) -> Path:
    candidate = root / "line_aux_candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        """
name: line_aux_candidate
input_mode: single_frame_rgb
output_form: mask_logits
auxiliary_targets:
  - name: line
    output: line_logits
    loss: weighted_bce
    weight: 0.25
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )
    (candidate / "model.py").write_text(
        "from torch import nn\n"
        "class Tiny(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.encoder = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())\n"
        "        self.mask = nn.Conv2d(4, 1, 1)\n"
        "        self.line = nn.Conv2d(4, 1, 1)\n"
        "    def forward(self, x):\n"
        "        features = self.encoder(x)\n"
        "        return {'mask_logits': self.mask(features), 'line_logits': self.line(features)}\n"
        "def build_model(input_spec, output_spec):\n"
        "    assert output_spec['auxiliary_outputs'][0]['name'] == 'line_logits'\n"
        "    return Tiny()\n"
    )
    return candidate


def test_evaluate_run_tolerates_auxiliary_outputs_and_reports_primary_metrics(tmp_path: Path):
    candidate = write_line_aux_candidate(tmp_path)
    run = run_candidate_with_gvccs_data(candidate, tmp_path / "runs", "tests/fixtures/gvccs_like", max_samples=4)
    assert run.status == RunStatus.COMPLETED

    result = evaluate_run(
        run.run_dir, split="val", backend="native", data_root="tests/fixtures/gvccs_like", max_artifact_samples=1
    )

    aggregate = json.loads((result.evaluation_dir / "aggregate_metrics.json").read_text())
    assert aggregate["split"] == "val"
    assert set(aggregate["metrics"]) == {"dice", "iou", "precision", "recall"}
    assert (result.evaluation_dir / "diagnostic_samples" / "samples.json").exists()
