from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from conftest import invoke_typer_cli
from ml_autoresearch.cli import app
from ml_autoresearch.managed_execution import start_run_supervisor, update_execution_record
from ml_autoresearch.research_problems import ResearchProblemProviderConfig
from ml_autoresearch.runs import (
    RunStatus,
    prepare_candidate_submission,
    reconcile_run,
    run_candidate_with_research_problem,
    submit_candidate,
)
from research_problem_helpers import write_fake_candidate_execution_config, write_fake_research_problem_package


def _write_candidate(root: Path) -> Path:
    candidate = root / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        """
name: reconciliation_candidate
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
        """
from torch import nn

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(3, 1, 1)

    def forward(self, inputs):
        return {"mask_logits": self.layer(inputs)}

def build_model(input_spec, output_spec):
    return Tiny()
""".strip()
        + "\n"
    )
    return candidate


def _provider(root: Path) -> ResearchProblemProviderConfig:
    return ResearchProblemProviderConfig(
        id="fake_problem",
        package_root=root,
        provider_target="fake_problem.research_problem:build_spec",
        expected_contract_version="v0",
        data_config={"sample_count": 4},
    )


def _events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_reconcile_pre_training_smoke_phase_is_idempotent_for_same_run(tmp_path: Path) -> None:
    ledger = tmp_path / "research-ledger.jsonl"
    run = prepare_candidate_submission(_write_candidate(tmp_path), tmp_path / "runs", ledger_path=ledger)
    command = [sys.executable, "-c", "import time; time.sleep(10)"]
    execution = start_run_supervisor(
        run.run_dir,
        command=command,
        log_path=run.run_dir / "outputs" / "logs" / "supervisor.log",
        backend="docker",
    )

    first = reconcile_run(run.run_dir, ledger_path=ledger)
    second = reconcile_run(run.run_dir, ledger_path=ledger)

    assert first.status == second.status == RunStatus.SMOKE_TESTING
    assert first.run_id == second.run_id == run.run_id
    assert len(list((tmp_path / "runs").glob("run_*"))) == 1
    assert all(
        event["event_type"] not in {"run_completed", "run_failed"}
        for event in _events(ledger)
    )

    os.kill(int(execution["supervisor"]["pid"]), signal.SIGTERM)
    os.waitpid(int(execution["supervisor"]["pid"]), 0)
    failed = reconcile_run(run.run_dir, ledger_path=ledger)
    repeated = reconcile_run(run.run_dir, ledger_path=ledger)

    assert failed.status == repeated.status == RunStatus.FAILED
    assert failed.failure_classification == repeated.failure_classification == "harness_failure"
    terminal = [
        event for event in _events(ledger) if event["event_type"] in {"run_completed", "run_failed"}
    ]
    assert len(terminal) == 1
    assert terminal[0]["run_id"] == run.run_id
    assert terminal[0]["failure_classification"] == "harness_failure"


def test_starting_managed_supervisor_twice_reuses_same_active_run_process(tmp_path: Path) -> None:
    run = submit_candidate(_write_candidate(tmp_path), tmp_path / "runs")
    command = [sys.executable, "-c", "import time; time.sleep(10)"]

    first = start_run_supervisor(
        run.run_dir,
        command=command,
        log_path=run.run_dir / "outputs" / "logs" / "supervisor.log",
        backend="native",
    )
    second = start_run_supervisor(
        run.run_dir,
        command=command,
        log_path=run.run_dir / "outputs" / "logs" / "supervisor.log",
        backend="native",
    )

    assert second["already_running"] is True
    assert second["supervisor"]["pid"] == first["supervisor"]["pid"]
    os.kill(int(first["supervisor"]["pid"]), signal.SIGTERM)
    os.waitpid(int(first["supervisor"]["pid"]), 0)


def test_reconcile_observes_running_container_after_supervisor_disappears(
    tmp_path: Path, monkeypatch
) -> None:
    run = submit_candidate(_write_candidate(tmp_path), tmp_path / "runs")
    metadata_path = run.run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["status"] = "training"
    metadata_path.write_text(json.dumps(metadata) + "\n")
    update_execution_record(
        run.run_dir,
        backend="docker",
        state="training",
        supervisor={"pid": 999_999_999},
        active_container={"name": "stable-container", "state": "starting"},
        containers=[{"attempt": 1, "name": "stable-container", "state": "starting"}],
    )

    def inspect_running(command, check, capture_output, text):
        assert command == ["docker", "inspect", "--format", "{{json .State}}", "stable-container"]
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "Status": "running",
                    "Running": True,
                    "ExitCode": 0,
                    "StartedAt": "2026-08-11T00:00:00Z",
                    "FinishedAt": "",
                    "Error": "",
                    "OOMKilled": False,
                }
            ),
            "",
        )

    monkeypatch.setattr("ml_autoresearch.managed_execution.subprocess.run", inspect_running)
    observed = reconcile_run(run.run_dir, ledger_path=tmp_path / "research-ledger.jsonl")

    assert observed.status == RunStatus.TRAINING
    assert json.loads(metadata_path.read_text())["status"] == "training"
    assert all(
        event["event_type"] not in {"run_completed", "run_failed"}
        for event in _events(tmp_path / "research-ledger.jsonl")
    )


def test_reconcile_exited_oom_container_is_resource_failure_without_relaunch(
    tmp_path: Path, monkeypatch
) -> None:
    run = submit_candidate(_write_candidate(tmp_path), tmp_path / "runs")
    metadata_path = run.run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["status"] = "training"
    metadata_path.write_text(json.dumps(metadata) + "\n")
    update_execution_record(
        run.run_dir,
        backend="docker",
        state="container_exited_failure",
        supervisor={"pid": 999_999_999},
        active_container={"name": "oom-container", "state": "exited_failure", "exit_code": 137},
        containers=[{"attempt": 1, "name": "oom-container", "state": "exited_failure", "exit_code": 137}],
    )

    def docker_state(command, check, capture_output, text):
        if command[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "Status": "exited",
                        "Running": False,
                        "ExitCode": 137,
                        "StartedAt": "2026-08-11T00:00:00Z",
                        "FinishedAt": "2026-08-11T00:00:01Z",
                        "Error": "",
                        "OOMKilled": True,
                    }
                ),
                "",
            )
        assert command == ["docker", "rm", "-f", "oom-container"]
        return subprocess.CompletedProcess(command, 0, "oom-container\n", "")

    monkeypatch.setattr("ml_autoresearch.managed_execution.subprocess.run", docker_state)
    reconciled = reconcile_run(run.run_dir, ledger_path=tmp_path / "research-ledger.jsonl")

    assert reconciled.status == RunStatus.FAILED
    assert reconciled.failure_classification == "resource_failure"
    assert len(list((tmp_path / "runs").glob("run_*"))) == 1
    assert not (run.run_dir / "outputs" / "logs" / "resource_retry.log").exists()
    terminal = [
        event
        for event in _events(tmp_path / "research-ledger.jsonl")
        if event["event_type"] in {"run_completed", "run_failed"}
    ]
    assert len(terminal) == 1
    assert terminal[0]["failure_classification"] == "resource_failure"


def test_reconcile_completed_artifacts_finalizes_stale_run_exactly_once(tmp_path: Path) -> None:
    write_fake_research_problem_package(tmp_path)
    ledger = tmp_path / "research-ledger.jsonl"
    run = run_candidate_with_research_problem(
        _write_candidate(tmp_path),
        tmp_path / "runs",
        _provider(tmp_path),
        max_samples=4,
        max_prediction_samples=1,
        ledger_path=ledger,
    )
    assert run.status == RunStatus.COMPLETED

    metadata_path = run.run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["status"] = "training"
    metadata["updated_at"] = metadata["created_at"]
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    ledger.write_text(
        "".join(
            json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
            for event in _events(ledger)
            if event["event_type"] != "run_completed"
        )
    )

    first = reconcile_run(run.run_dir, ledger_path=ledger)
    second = reconcile_run(run.run_dir, ledger_path=ledger)

    assert first.status == RunStatus.COMPLETED
    assert second.status == RunStatus.COMPLETED
    assert json.loads(metadata_path.read_text())["status"] == "completed"
    terminal = [
        event for event in _events(ledger) if event["event_type"] in {"run_completed", "run_failed"}
    ]
    assert len(terminal) == 1
    assert terminal[0]["event_type"] == "run_completed"
    assert terminal[0]["run_id"] == run.run_id


def test_reconcile_run_cli_reports_same_terminal_run_on_repeated_calls(tmp_path: Path) -> None:
    write_fake_research_problem_package(tmp_path)
    write_fake_candidate_execution_config(tmp_path)
    ledger = tmp_path / "research-ledger.jsonl"
    run = run_candidate_with_research_problem(
        _write_candidate(tmp_path),
        tmp_path / "runs",
        _provider(tmp_path),
        max_samples=4,
        max_prediction_samples=1,
        ledger_path=ledger,
    )

    first = invoke_typer_cli(
        app,
        ["reconcile-run", "--run-id", run.run_id, "--workspace-root", str(tmp_path)],
    )
    second = invoke_typer_cli(
        app,
        ["reconcile-run", "--run-id", run.run_id, "--workspace-root", str(tmp_path)],
    )

    assert first.returncode == second.returncode == 0
    assert json.loads(first.stdout)["run_id"] == run.run_id
    assert json.loads(second.stdout)["status"] == "completed"
    terminal = [
        event for event in _events(ledger) if event["event_type"] in {"run_completed", "run_failed"}
    ]
    assert len(terminal) == 1


def test_reconcile_nonfinite_completed_artifacts_fails_without_resource_retry(tmp_path: Path) -> None:
    write_fake_research_problem_package(tmp_path)
    ledger = tmp_path / "research-ledger.jsonl"
    run = run_candidate_with_research_problem(
        _write_candidate(tmp_path),
        tmp_path / "runs",
        _provider(tmp_path),
        max_samples=4,
        max_prediction_samples=1,
        ledger_path=ledger,
    )
    metadata_path = run.run_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["status"] = "training"
    metadata_path.write_text(json.dumps(metadata) + "\n")
    ledger.write_text(
        "".join(
            json.dumps(event) + "\n"
            for event in _events(ledger)
            if event["event_type"] != "run_completed"
        )
    )
    final_path = run.run_dir / "outputs" / "final_metrics.json"
    final = json.loads(final_path.read_text())
    final["val/dice"] = float("nan")
    final_path.write_text(json.dumps(final, allow_nan=True) + "\n")

    first = reconcile_run(run.run_dir, ledger_path=ledger)
    second = reconcile_run(run.run_dir, ledger_path=ledger)

    assert first.status == RunStatus.FAILED
    assert second.status == RunStatus.FAILED
    assert first.failure_classification == "candidate_bug"
    diagnostic = json.loads((run.run_dir / "outputs" / "nonfinite_diagnostic.json").read_text())
    assert diagnostic["phase"] == "terminal_validation"
    assert diagnostic["failing_quantity"] == "metric.final_metrics.val/dice"
    assert not (run.run_dir / "outputs" / "logs" / "resource_retry.log").exists()
    terminal = [
        event for event in _events(ledger) if event["event_type"] in {"run_completed", "run_failed"}
    ]
    assert len(terminal) == 1
    assert terminal[0]["event_type"] == "run_failed"
    assert terminal[0]["failure_classification"] == "candidate_bug"
