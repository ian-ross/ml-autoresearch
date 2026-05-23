import json
import subprocess
import sys
from pathlib import Path

from ml_autoresearch.autonomy_step import run_autonomy_step
from ml_autoresearch.execution import OperationResult


def run_cli(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ml_autoresearch.cli", *args],
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def write_project(root: Path, extra_config: str = "") -> None:
    (root / "CONTEXT.md").write_text("context v1\n")
    (root / "EXPERIMENT_INDEX.md").write_text(
        "# Experiment Index\n"
        "\n"
        "## Candidate Experiments and notes\n"
        "\n"
        "| Candidate Experiment | Description | Related Research Notes | Key Runs / Evaluations | Status |\n"
        "| --- | --- | --- | --- | --- |\n"
        "\n"
        "## Chronological Research Notes\n"
    )
    (root / "research-ledger.jsonl").write_text("")
    (root / "docs").mkdir()
    (root / "agent-boundary.toml").write_text(
        """
[agent_control_boundary]
distro = "debian"
image = "../../containers/ml-autoresearch-agent"
allow_egress = true
""".lstrip()
        + extra_config
    )


def write_fake_agent(path: Path, body: str) -> str:
    path.write_text(
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        + body
    )
    return f"{sys.executable} {path}"


def test_autonomy_step_writes_prompt_invokes_fake_agent_ingests_one_handoff_and_prints_summary(tmp_path: Path):
    write_project(tmp_path)
    fake_command = write_fake_agent(
        tmp_path / "fake_agent.py",
        """
assert sys.argv[1:] == ['-p', '@prompt.txt']
prompt = Path('prompt.txt').read_text()
assert 'Read AGENTS.md first' in prompt
assert 'campaign-manager skill' in prompt
assert 'Multiple primary handoff outcomes are forbidden' in prompt
assert 'exactly one primary research handoff outcome' in prompt
assert Path('AGENTS.md').is_file()
Path('research-notes').mkdir(exist_ok=True)
Path('research-notes/2026-05-22-agent-note.md').write_text('# Agent Note\\n\\n## Summary\\nOne note.\\n\\n## Decision\\nContinue.\\n')
Path('scratch/invocation.json').write_text(json.dumps({'cwd': os.getcwd(), 'reference': Path('../agent-reference/CONTEXT.md').read_text()}))
""",
    )

    completed = run_cli(tmp_path, "autonomy-step", "--project-root", str(tmp_path), "--agent-command", fake_command)

    assert completed.returncode == 0, completed.stderr
    assert "Autonomy Step complete" in completed.stdout
    assert "Status: ingested" in completed.stdout
    assert "Handoff ingestion: ingested" in completed.stdout
    assert not completed.stdout.lstrip().startswith("{")

    invocation = json.loads((tmp_path / "agent-work" / "scratch" / "invocation.json").read_text())
    assert invocation["cwd"] == str(tmp_path / "agent-work")
    assert invocation["reference"] == "context v1\n"
    assert (tmp_path / "research-notes" / "2026-05-22-agent-note.md").is_file()

    result = json.loads((tmp_path / "agent-work" / "autonomy-step-result.json").read_text())
    assert result["status"] == "ingested"
    assert result["agent_command"][-2:] == ["-p", "@prompt.txt"]
    assert result["agent_returncode"] == 0
    assert result["ingestion"]["handoff_type"] == "research_note"


def test_autonomy_step_refreshes_agent_control_boundary_before_invocation(tmp_path: Path):
    write_project(tmp_path)
    run_cli(tmp_path, "prepare-agent-boundary", "--project-root", str(tmp_path))
    (tmp_path / "CONTEXT.md").write_text("context v2\n")
    fake_command = write_fake_agent(
        tmp_path / "fake_agent.py",
        "Path('scratch/reference.txt').write_text(Path('../agent-reference/CONTEXT.md').read_text())\n",
    )

    completed = run_cli(tmp_path, "autonomy-step", "--project-root", str(tmp_path), "--agent-command", fake_command)

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "agent-work" / "scratch" / "reference.txt").read_text() == "context v2\n"
    result = json.loads((tmp_path / "agent-work" / "autonomy-step-result.json").read_text())
    assert result["status"] == "no_handoff"
    assert result["ingestion"]["next_action"] == "stop_for_human"


def test_autonomy_step_nonzero_agent_exit_writes_failure_result_and_does_not_ingest(tmp_path: Path):
    write_project(tmp_path)
    fake_command = write_fake_agent(
        tmp_path / "fake_agent.py",
        """
Path('research-notes').mkdir(exist_ok=True)
Path('research-notes/2026-05-22-untrusted-note.md').write_text('# Should Not Ingest\\n')
sys.exit(7)
""",
    )

    completed = run_cli(tmp_path, "autonomy-step", "--project-root", str(tmp_path), "--agent-command", fake_command)

    assert completed.returncode != 0
    assert "Status: agent_failed" in completed.stdout
    assert "handoff ingestion skipped" in completed.stdout
    assert not (tmp_path / "research-notes" / "2026-05-22-untrusted-note.md").exists()
    result = json.loads((tmp_path / "agent-work" / "autonomy-step-result.json").read_text())
    assert result["status"] == "agent_failed"
    assert result["agent_returncode"] == 7
    assert result["ingestion"] is None


def test_autonomy_step_uses_configured_agent_command_when_cli_option_is_absent(tmp_path: Path):
    fake_path = tmp_path / "fake_agent.py"
    write_project(tmp_path, extra_config=f'\n[autonomy_step]\nagent_command = "{sys.executable} {fake_path}"\n')
    write_fake_agent(fake_path, "Path('scratch/config-agent-used.txt').write_text('yes')\n")

    completed = run_cli(tmp_path, "autonomy-step", "--project-root", str(tmp_path))

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "agent-work" / "scratch" / "config-agent-used.txt").read_text() == "yes"
    result = json.loads((tmp_path / "agent-work" / "autonomy-step-result.json").read_text())
    assert result["agent_command"][:2] == [sys.executable, str(fake_path)]
    assert result["status"] == "no_handoff"


def test_autonomy_step_default_pi_command_records_sessions_beside_agent_workspace(tmp_path: Path, monkeypatch):
    import ml_autoresearch.autonomy_step as autonomy_step

    write_project(tmp_path)
    captured = {}

    def fake_run(command: list[str], *, cwd: Path, check: bool):
        captured["command"] = command
        captured["cwd"] = cwd

        class Completed:
            returncode = 0

        return Completed()

    monkeypatch.setattr(autonomy_step.subprocess, "run", fake_run)

    result = run_autonomy_step(tmp_path)

    assert result.status == "no_handoff"
    assert captured["cwd"] == tmp_path / "agent-work"
    assert captured["command"][:3] == ["pi", "--session-dir", "../agent-sessions"]
    assert (tmp_path / "agent-sessions").is_dir()


class FakeNativeBackend:
    name = "fake-native"

    def smoke_test(self, run_dir: str | Path) -> OperationResult:
        Path(run_dir, "outputs", "model_summary.json").write_text('{"ok": true}\n')
        return OperationResult(backend=self.name, operation="smoke_test")

    def train_synthetic(self, run_dir: str | Path, *, max_prediction_samples: int = 2, prediction_sample_policy: str = "first_n") -> OperationResult:
        raise AssertionError("candidate submission next-action execution must submit exactly one Run, not train it")

    def train_gvccs(self, run_dir: str | Path, data_root: str | Path, *, max_samples: int | None = None, max_prediction_samples: int = 2, prediction_sample_policy: str = "first_n") -> OperationResult:
        raise AssertionError("candidate submission next-action execution must not train GVCCS data")

    def evaluate_run(self, run_dir: str | Path, *, data_root: str | Path | None = None, max_artifact_samples: int = 12) -> OperationResult:
        raise AssertionError("candidate submission next-action execution must not evaluate")


def test_autonomy_step_dry_run_candidate_submission_reports_next_action_without_run(tmp_path: Path):
    write_project(tmp_path)
    fake_command = write_fake_agent(tmp_path / "fake_agent.py", _candidate_submission_agent_body())

    result = run_autonomy_step(tmp_path, agent_command=fake_command)

    assert result.status == "ingested"
    assert result.ingestion is not None
    assert result.ingestion["next_action"] == "run_candidate"
    assert result.ingestion["executed_next_action"] is False
    assert result.execution is None
    assert not (tmp_path / "runs").exists()


def test_autonomy_step_execute_candidate_submission_submits_one_run_after_ingestion(tmp_path: Path, monkeypatch):
    import ml_autoresearch.execution as execution

    write_project(tmp_path)
    monkeypatch.setattr(execution, "NativeBackend", FakeNativeBackend)
    fake_command = write_fake_agent(tmp_path / "fake_agent.py", _candidate_submission_agent_body())

    result = run_autonomy_step(tmp_path, agent_command=fake_command, execute_next_action=True)

    assert result.status == "ingested"
    assert result.ingestion is not None
    assert result.ingestion["executed_next_action"] is True
    execution_result = result.ingestion["next_action_result"]
    assert execution_result["action"] == "run_candidate"
    assert execution_result["run_status"] == "accepted"
    runs = sorted((tmp_path / "runs").glob("run_*"))
    assert len(runs) == 1
    events = [json.loads(line) for line in (tmp_path / "research-ledger.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events] == [
        "agent_handoff_ingested",
        "proposal_created",
        "candidate_created",
        "candidate_submitted",
    ]


def test_autonomy_step_dry_run_evaluation_request_does_not_record_evaluation_requested(tmp_path: Path):
    write_project(tmp_path)
    _write_completed_run(tmp_path, "run_123")
    fake_command = write_fake_agent(tmp_path / "fake_agent.py", _evaluation_request_agent_body())

    result = run_autonomy_step(tmp_path, agent_command=fake_command)

    assert result.status == "ingested"
    assert result.ingestion is not None
    assert result.ingestion["next_action"] == "run_post_run_evaluation"
    assert result.ingestion["executed_next_action"] is False
    assert "evaluation_requested" not in (tmp_path / "research-ledger.jsonl").read_text()
    assert not (tmp_path / "runs" / "run_123" / "outputs" / "evaluations").exists()


def test_autonomy_step_execute_evaluation_request_runs_post_run_evaluation_once(tmp_path: Path):
    write_project(tmp_path)
    _write_completed_run(tmp_path, "run_123")
    fake_command = write_fake_agent(tmp_path / "fake_agent.py", _evaluation_request_agent_body())

    result = run_autonomy_step(tmp_path, agent_command=fake_command, execute_next_action=True)

    assert result.status == "ingested"
    assert result.ingestion is not None
    assert result.ingestion["executed_next_action"] is True
    assert result.ingestion["next_action_result"]["action"] == "run_post_run_evaluation"
    assert (tmp_path / "runs" / "run_123" / "outputs" / "evaluations" / "eval_eval-threshold-sweep-run-123" / "summary.json").is_file()
    events = [json.loads(line) for line in (tmp_path / "research-ledger.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events] == [
        "agent_handoff_ingested",
        "evaluation_requested",
        "evaluation_completed",
    ]


def test_execute_next_action_cli_runs_outstanding_action_from_previous_autonomy_step(tmp_path: Path):
    write_project(tmp_path)
    _write_completed_run(tmp_path, "run_123")
    fake_command = write_fake_agent(tmp_path / "fake_agent.py", _evaluation_request_agent_body())
    result = run_autonomy_step(tmp_path, agent_command=fake_command)
    assert result.ingestion is not None
    assert result.ingestion["executed_next_action"] is False

    completed = run_cli(tmp_path, "execute-next-action", "--project-root", str(tmp_path))

    assert completed.returncode == 0, completed.stderr
    assert "Next action execution: completed" in completed.stdout
    assert (tmp_path / "runs" / "run_123" / "outputs" / "evaluations" / "eval_eval-threshold-sweep-run-123" / "summary.json").is_file()
    result_file = json.loads((tmp_path / "agent-work" / "autonomy-step-result.json").read_text())
    assert result_file["ingestion"]["executed_next_action"] is True
    assert result_file["ingestion"]["next_action_result"]["action"] == "run_post_run_evaluation"
    assert result_file["execution"]["executed"] is True


def test_execute_next_action_cli_reruns_legacy_evaluation_result_missing_outputs_artifact(tmp_path: Path):
    write_project(tmp_path)
    _write_completed_run(tmp_path, "run_123")
    fake_command = write_fake_agent(tmp_path / "fake_agent.py", _evaluation_request_agent_body())
    result = run_autonomy_step(tmp_path, agent_command=fake_command)
    assert result.ingestion is not None
    result_payload = result.to_json()
    result_payload["ingestion"]["executed_next_action"] = True
    result_payload["ingestion"]["next_action_result"] = {
        "action": "run_post_run_evaluation",
        "executed": True,
        "status": "completed",
        "evaluation_id": "eval_eval-threshold-sweep-run-123",
    }
    (tmp_path / "agent-work" / "autonomy-step-result.json").write_text(json.dumps(result_payload))

    completed = run_cli(tmp_path, "execute-next-action", "--project-root", str(tmp_path))

    assert completed.returncode == 0, completed.stderr
    assert "Next action execution: completed" in completed.stdout
    assert (tmp_path / "runs" / "run_123" / "outputs" / "evaluations" / "eval_eval-threshold-sweep-run-123" / "evaluation_metadata.json").is_file()



def test_autonomy_step_execute_next_action_skips_non_executable_handoff_types(tmp_path: Path):
    write_project(tmp_path)
    fake_command = write_fake_agent(
        tmp_path / "fake_agent.py",
        "Path('research-notes/2026-05-22-agent-note.md').write_text('# Agent Note\\n\\n## Summary\\nOne note.\\n\\n## Decision\\nContinue.\\n')\n",
    )

    result = run_autonomy_step(tmp_path, agent_command=fake_command, execute_next_action=True)

    assert result.status == "ingested"
    assert result.ingestion is not None
    assert result.ingestion["handoff_type"] == "research_note"
    assert result.ingestion["executed_next_action"] is False
    assert result.execution == {
        "status": "skipped",
        "executed": False,
        "reason": "no executable Harness action for 'research_note'",
    }
    assert not (tmp_path / "runs").exists()


def test_autonomy_step_execution_failure_preserves_ingestion_and_writes_failure_result(tmp_path: Path):
    write_project(tmp_path)
    fake_command = write_fake_agent(tmp_path / "fake_agent.py", _evaluation_request_agent_body(target_run_id="missing_run"))

    result = run_autonomy_step(tmp_path, agent_command=fake_command, execute_next_action=True)

    assert result.status == "execution_failed"
    assert result.ingestion is not None
    assert result.ingestion["status"] == "ingested"
    assert result.ingestion["executed_next_action"] is False
    assert (tmp_path / "evaluation-requests" / "eval-threshold-sweep-run-123.yaml").is_file()
    assert "target Run does not exist" in (result.reason or "")
    result_file = json.loads((tmp_path / "agent-work" / "autonomy-step-result.json").read_text())
    assert result_file["status"] == "execution_failed"
    assert result_file["ingestion"]["status"] == "ingested"
    assert result_file["execution"]["status"] == "failed"


def test_autonomy_step_cli_execute_next_action_flag_exits_nonzero_when_execution_fails(tmp_path: Path):
    write_project(tmp_path)
    fake_command = write_fake_agent(tmp_path / "fake_agent.py", _evaluation_request_agent_body(target_run_id="missing_run"))

    completed = run_cli(
        tmp_path,
        "autonomy-step",
        "--project-root",
        str(tmp_path),
        "--agent-command",
        fake_command,
        "--execute-next-action",
    )

    assert completed.returncode == 1
    assert "Status: execution_failed" in completed.stdout
    assert "Next action execution: failed" in completed.stdout
    result_file = json.loads((tmp_path / "agent-work" / "autonomy-step-result.json").read_text())
    assert result_file["status"] == "execution_failed"


def _candidate_submission_agent_body() -> str:
    return r'''
submission = Path('submissions/agent_candidate')
candidate = submission / 'candidate'
candidate.mkdir(parents=True)
(candidate / 'manifest.yaml').write_text(''' + repr("""
name: agent_candidate
description: Agent-submitted candidate.
input_mode: single_frame_rgb
output_form: mask_logits
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".lstrip()) + r''')
(candidate / 'model.py').write_text("raise RuntimeError('fake backend must not import model code')\n")
(candidate / 'README.md').write_text('# Agent candidate\n')
(candidate / 'PROPOSAL.md').write_text(''' + repr("""## Hypothesis
Test execution.

## Comparison Target
Baseline.

## Expected Effect
Queue for a later run.

## Implementation Sketch
Static model source.

## Contract Features Used
single_frame_rgb.

## Budget Requested
One run.

## Success Criteria
Ingests.

## Fallback/Next Decision
Fix validation.
""") + r''')
(submission / 'submission.json').write_text(json.dumps({
    'schema_version': 'candidate_submission.v1',
    'submission_type': 'candidate_experiment',
    'candidate_id': 'agent_candidate',
    'candidate_path': 'candidate',
    'requested_action': 'validate_and_queue_for_harness_execution',
}, indent=2) + '\n')
'''


def _evaluation_request_agent_body(target_run_id: str = "run_123") -> str:
    return f"""
Path('evaluation-requests/eval-threshold-sweep-run-123.yaml').write_text('''request_id: eval-threshold-sweep-run-123
target_run_id: {target_run_id}
evaluation_mode: threshold_sweep
diagnostic_question: Which threshold best separates thin masks?
expected_decision_impact: Decide whether low Dice is thresholding or representation failure.
parameters:
  threshold_sweep:
    min: 0.1
    max: 0.9
    steps: 9
  artifact_count: 4
artifact_budget:
  max_artifacts: 6
  max_runtime_seconds: 120
''')
"""


def _write_completed_run(root: Path, run_id: str) -> None:
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(json.dumps({"run_id": run_id, "status": "completed"}) + "\n")
