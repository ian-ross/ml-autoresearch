import json
import sys
from pathlib import Path

import pytest

from ml_autoresearch.autonomy_step import find_open_executable_actions, render_autonomy_step_prompt, run_autonomy_step
from ml_autoresearch.cli import app
from ml_autoresearch.execution import OperationResult
from conftest import invoke_typer_cli


def run_cli(cwd: Path, *args: str):
    return invoke_typer_cli(app, args, cwd=cwd)


@pytest.fixture(autouse=True)
def skip_pi_fort_install(monkeypatch):
    monkeypatch.setattr("ml_autoresearch.agent_boundary._install_pi_fort_extension", lambda workspace_dir: None)


def write_fake_research_problem_provider(root: Path) -> None:
    package = root / "fake_research_problem"
    (package / "brief").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "brief" / "overview.md").write_text("# Tiny overview\n")
    (package / "brief" / "baselines.md").write_text("# Tiny baselines\n")
    (package / "research_problem.py").write_text(
        "from ml_autoresearch.research_problems import ResearchProblemSpec\n"
        "class Adapter:\n"
        "    def validate_data_root(self, data_config): pass\n"
        "    def dataset_metadata(self, data_config): return {'kind': 'tiny_problem'}\n"
        "    def build_datasets(self, **kwargs): raise NotImplementedError\n"
        "    def apply_augmentation_policy(self, dataset, augmentation_policy): return dataset\n"
        "    def primary_output_name(self, output_spec): return 'mask_logits'\n"
        "    def compute_primary_loss(self, loss_name, logits, target_mask): raise NotImplementedError\n"
        "    def compute_auxiliary_losses(self, outputs, target_mask, auxiliary_targets): return {}\n"
        "    def compute_validation_metrics(self, logits, target_mask): return {'val/dice': 0.0}\n"
        "    def selection_policy(self): return 'val/dice', 'max'\n"
        "    def build_evaluation_dataset(self, **kwargs): raise NotImplementedError\n"
        "def build_spec(data_config=None):\n"
        "    return ResearchProblemSpec(\n"
        "        id='tiny_problem', version='test-spec-v0', contract_version='v0',\n"
        "        input_modes=('single_frame_rgb',), input_specs={'single_frame_rgb': {'mode': 'single_frame_rgb', 'shape': [3, 16, 16]}},\n"
        "        output_forms=('binary_mask', 'mask_logits'), output_specs={'binary_mask': {'form': 'binary_mask', 'shape': [1, 16, 16]}, 'mask_logits': {'form': 'mask_logits', 'shape': [1, 16, 16]}},\n"
        "        losses=('bce', 'dice_bce', 'bce_dice'), optimizers=('adamw',), sampling_policies=('sequential',),\n"
        "        augmentation_policies=('none',), primary_metric='val/dice',\n"
        "        operation_capabilities={'training': True},\n"
        "        training_adapter=Adapter(),\n"
        "        brief_documents=(\n"
        "            {'name': 'overview', 'role': 'problem_overview', 'path': 'fake_research_problem/brief/overview.md', 'summary': 'Tiny problem overview.'},\n"
        "            {'name': 'baselines', 'role': 'baseline_description', 'path': 'fake_research_problem/brief/baselines.md', 'summary': 'Tiny baseline notes.', 'required': True},\n"
        "        ),\n"
        "    )\n"
    )
    (root / "ml-autoresearch.toml").write_text(
        "[candidate_execution]\n"
        "ledger_path = \"research-ledger.jsonl\"\n"
        "\n"
        "[research_problem]\n"
        "id = \"tiny_problem\"\n"
        f"package_root = \"{root}\"\n"
        "provider_target = \"fake_research_problem.research_problem:build_spec\"\n"
        "expected_contract_version = \"v0\"\n"
    )

def write_project(root: Path, extra_config: str = "") -> None:
    write_fake_research_problem_provider(root)
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
    config_path = root / "ml-autoresearch.toml"
    config_path.write_text(
        config_path.read_text()
        + "\n"
        + """
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
Path('scratch/invocation.json').write_text(json.dumps({'cwd': os.getcwd(), 'reference': Path('../agent-reference/WORKSPACE_CONTEXT.md').read_text()}))
""",
    )

    completed = run_cli(tmp_path, "autonomy-step", "--workspace-root", str(tmp_path), "--agent-command", fake_command)

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
    run_cli(tmp_path, "prepare-agent-boundary", "--workspace-root", str(tmp_path))
    (tmp_path / "CONTEXT.md").write_text("context v2\n")
    fake_command = write_fake_agent(
        tmp_path / "fake_agent.py",
        "Path('scratch/reference.txt').write_text(Path('../agent-reference/WORKSPACE_CONTEXT.md').read_text())\n",
    )

    completed = run_cli(tmp_path, "autonomy-step", "--workspace-root", str(tmp_path), "--agent-command", fake_command)

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "agent-work" / "scratch" / "reference.txt").read_text() == "context v2\n"
    result = json.loads((tmp_path / "agent-work" / "autonomy-step-result.json").read_text())
    assert result["status"] == "no_handoff"
    assert result["ingestion"]["next_action"] == "stop_for_human"


def test_autonomy_step_prompt_does_not_treat_punctuated_none_pause_report_as_active_pause(tmp_path: Path):
    from ml_autoresearch.campaign_controls import record_campaign_report_written

    write_project(tmp_path)
    report = tmp_path / "campaign-reports" / "2026-06-24-status.md"
    report.parent.mkdir()
    report.write_text(
        """# Campaign Report: Test

## Pause recommendation
- Pause condition: none.
- Human decision needed: no.
"""
    )
    record_campaign_report_written("campaign-reports/2026-06-24-status.md", ledger_path=tmp_path / "research-ledger.jsonl")

    prompt = render_autonomy_step_prompt(tmp_path)

    assert "Campaign pause state:" not in prompt
    assert "recommends pause for `none.`" not in prompt


def test_autonomy_step_prompt_includes_campaign_report_schema_when_reports_are_allowed() -> None:
    prompt = render_autonomy_step_prompt()

    assert "If you write a Campaign Report" in prompt
    for heading in [
        "## Current best Result",
        "## Recent Runs",
        "## Failures",
        "## Pending Capability Requests",
        "## Budget use",
        "## Next hypothesis",
        "## Pause recommendation",
    ]:
        assert heading in prompt
    assert "- Pause condition: none" in prompt
    assert "Do not add punctuation or prose" in prompt


def test_autonomy_step_prompt_tells_agent_when_human_resumed_after_pause_report(tmp_path: Path):
    from ml_autoresearch.campaign_controls import record_campaign_report_written, record_campaign_resume

    write_project(tmp_path)
    report = tmp_path / "campaign-reports" / "2026-05-31-status.md"
    report.parent.mkdir()
    report.write_text(
        """# Campaign Report: Test

## Pause recommendation
- Pause condition: `scheduled_check_in`
- Human decision needed: yes.
"""
    )
    ledger = tmp_path / "research-ledger.jsonl"
    record_campaign_report_written("campaign-reports/2026-05-31-status.md", ledger_path=ledger)
    record_campaign_resume("human_review_complete", report_path="campaign-reports/2026-06-01-resume.md", ledger_path=ledger)
    fake_command = write_fake_agent(
        tmp_path / "fake_agent.py",
        """
prompt = Path('prompt.txt').read_text()
assert 'Human campaign review is complete' in prompt
assert 'Do not treat earlier scheduled_check_in' in prompt
Path('scratch/resume-prompt.txt').write_text(prompt)
""",
    )

    result = run_autonomy_step(tmp_path, agent_command=fake_command)

    assert result.status == "no_handoff"


def test_autonomy_step_removes_stale_loop_result_files_before_agent_invocation(tmp_path: Path):
    write_project(tmp_path)
    workspace = tmp_path / "agent-work"
    workspace.mkdir()
    (workspace / "autonomy-step-result.json").write_text('{"stale": "pause_campaign"}\n')
    (workspace / "autonomous-iteration-result.json").write_text('{"stop_reason": "campaign_paused"}\n')
    fake_command = write_fake_agent(
        tmp_path / "fake_agent.py",
        """
assert not Path('autonomy-step-result.json').exists()
assert not Path('autonomous-iteration-result.json').exists()
Path('scratch/stale-results-cleared.txt').write_text('yes')
""",
    )

    result = run_autonomy_step(tmp_path, agent_command=fake_command)

    assert result.status == "no_handoff"
    assert (workspace / "scratch" / "stale-results-cleared.txt").read_text() == "yes"
    assert (workspace / "autonomy-step-result.json").is_file()


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

    completed = run_cli(tmp_path, "autonomy-step", "--workspace-root", str(tmp_path), "--agent-command", fake_command)

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

    completed = run_cli(tmp_path, "autonomy-step", "--workspace-root", str(tmp_path))

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
        (evaluation_dir / "summary.json").write_text(json.dumps({"evaluation_id": evaluation_id}) + "\n")
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


class FakeNativeBackend:
    name = "fake-native"
    provider_ids: list[str] = []

    def smoke_test(self, run_dir: str | Path) -> OperationResult:
        Path(run_dir, "outputs", "model_summary.json").write_text('{"ok": true}\n')
        return OperationResult(backend=self.name, operation="smoke_test")

    def train_synthetic(self, run_dir: str | Path, *, max_prediction_samples: int = 2, prediction_sample_policy: str = "first_n") -> OperationResult:
        raise AssertionError("candidate submission next-action execution must not train synthetic data")

    def train_gvccs(self, run_dir: str | Path, data_root: str | Path, *, max_samples: int | None = None, max_prediction_samples: int = 2, prediction_sample_policy: str = "first_n") -> OperationResult:
        return self.train_research_problem(
            run_dir,
            provider_config=None,
            max_samples=max_samples,
            max_prediction_samples=max_prediction_samples,
            prediction_sample_policy=prediction_sample_policy,
        )

    def train_research_problem(self, run_dir: str | Path, provider_config, *, max_samples: int | None = None, max_prediction_samples: int = 2, prediction_sample_policy: str = "first_n") -> OperationResult:
        self.provider_ids.append(provider_config.id)
        outputs = Path(run_dir, "outputs")
        (outputs / "logs").mkdir(parents=True, exist_ok=True)
        (outputs / "logs" / "training.log").write_text("trained\n")
        (outputs / "metrics.jsonl").write_text('{"val/dice": 0.5}\n')
        (outputs / "best_metrics.json").write_text(
            json.dumps({"selection_metric": "val/dice", "selection_value": 0.5, "metrics": {"val/dice": 0.5}}) + "\n"
        )
        (outputs / "final_metrics.json").write_text('{"val/dice": 0.5}\n')
        return OperationResult(backend=self.name, operation="train_research_problem")

    def evaluate_run(self, run_dir: str | Path, *, data_root: str | Path | None = None, max_artifact_samples: int = 12) -> OperationResult:
        raise AssertionError("candidate submission next-action execution must not evaluate")


def test_find_open_executable_actions_reports_ingested_candidate_without_submission(tmp_path: Path):
    write_project(tmp_path)
    fake_command = write_fake_agent(tmp_path / "fake_agent.py", _candidate_submission_agent_body())
    run_autonomy_step(tmp_path, agent_command=fake_command)

    actions = find_open_executable_actions(tmp_path)

    assert [action["action"] for action in actions] == ["run_candidate"]
    assert actions[0]["handoff_type"] == "candidate_submission"
    assert actions[0]["candidate_id"] == "agent_candidate"
    assert actions[0]["canonical_path"] == "candidates/agent_candidate"


def test_execute_next_action_refuses_when_older_open_candidate_exists(tmp_path: Path):
    write_project(tmp_path)
    fake_command = write_fake_agent(tmp_path / "fake_agent.py", _candidate_submission_agent_body())
    run_autonomy_step(tmp_path, agent_command=fake_command)
    (tmp_path / "agent-work" / "autonomy-step-result.json").write_text(
        json.dumps(
            {
                "status": "ingested",
                "workspace_root": str(tmp_path),
                "agent_workspace": str(tmp_path / "agent-work"),
                "prompt_path": str(tmp_path / "agent-work" / "prompt.txt"),
                "agent_command": [],
                "agent_returncode": 0,
                "ingestion": {
                    "status": "ingested",
                    "handoff_type": "campaign_report",
                    "canonical_path": "campaign-reports/status.md",
                    "next_action": "stop_for_human",
                    "executed_next_action": False,
                },
            }
        )
        + "\n"
    )

    completed = run_cli(tmp_path, "execute-next-action", "--workspace-root", str(tmp_path))

    assert completed.returncode == 1
    assert "open Harness actions" in completed.stderr
    assert "run_candidate" in completed.stderr
    assert "agent_candidate" in completed.stderr


def test_execute_open_actions_runs_pending_candidate_in_ledger_order(tmp_path: Path, monkeypatch):
    import ml_autoresearch.execution as execution

    write_project(tmp_path)
    data_root = tmp_path / "gvccs"
    data_root.mkdir()
    _write_completed_run(tmp_path, "run_prior", data_root=data_root)
    FakeNativeBackend.provider_ids = []
    monkeypatch.setattr(execution, "NativeBackend", FakeNativeBackend)
    fake_command = write_fake_agent(tmp_path / "fake_agent.py", _candidate_submission_agent_body())
    run_autonomy_step(tmp_path, agent_command=fake_command)

    completed = run_cli(tmp_path, "execute-open-actions", "--workspace-root", str(tmp_path), "--max-actions", "1")

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["executed_count"] == 1
    assert payload["executions"][0]["action"]["action"] == "run_candidate"
    assert payload["executions"][0]["result"]["run_status"] == "completed"
    assert find_open_executable_actions(tmp_path) == []
    events = [json.loads(line) for line in (tmp_path / "research-ledger.jsonl").read_text().splitlines()]
    assert "candidate_submitted" in [event["event_type"] for event in events]


def test_find_open_executable_actions_reports_ingested_experiment_batch_without_completion(tmp_path: Path):
    write_project(tmp_path)
    event = {
        "event_type": "agent_handoff_ingested",
        "created_at": "2026-06-25T00:00:00Z",
        "handoff_type": "experiment_batch_submission",
        "artifact_id": "agent_batch",
        "source_path": "agent-work/batch-submissions/agent_batch",
        "canonical_path": "experiment-batches/agent_batch",
    }
    (tmp_path / "research-ledger.jsonl").write_text(json.dumps(event) + "\n")

    actions = find_open_executable_actions(tmp_path)

    assert [action["action"] for action in actions] == ["run_experiment_batch"]
    assert actions[0]["batch_id"] == "agent_batch"


def test_execute_open_actions_dry_run_lists_pending_evaluation_request(tmp_path: Path):
    write_project(tmp_path)
    _write_completed_run(tmp_path, "run_123")
    fake_command = write_fake_agent(tmp_path / "fake_agent.py", _evaluation_request_agent_body())
    run_autonomy_step(tmp_path, agent_command=fake_command)

    completed = run_cli(tmp_path, "execute-open-actions", "--workspace-root", str(tmp_path), "--dry-run")

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["dry_run"] is True
    assert payload["open_actions"][0]["action"] == "run_post_run_evaluation"
    assert payload["open_actions"][0]["request_id"] == "eval-threshold-sweep-run-123"
    assert not (tmp_path / "runs" / "run_123" / "outputs" / "evaluations").exists()


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


def test_autonomy_step_execute_candidate_submission_runs_one_run_after_ingestion(tmp_path: Path, monkeypatch):
    import ml_autoresearch.execution as execution

    write_project(tmp_path)
    data_root = tmp_path / "gvccs"
    data_root.mkdir()
    _write_completed_run(tmp_path, "run_prior", data_root=data_root)
    FakeNativeBackend.provider_ids = []
    monkeypatch.setattr(execution, "NativeBackend", FakeNativeBackend)
    fake_command = write_fake_agent(tmp_path / "fake_agent.py", _candidate_submission_agent_body())

    result = run_autonomy_step(tmp_path, agent_command=fake_command, execute_next_action=True)

    assert result.status == "ingested"
    assert result.ingestion is not None
    assert result.ingestion["executed_next_action"] is True
    execution_result = result.ingestion["next_action_result"]
    assert execution_result["action"] == "run_candidate"
    assert execution_result["run_status"] == "completed"
    assert FakeNativeBackend.provider_ids == ["tiny_problem"]
    runs = sorted((tmp_path / "runs").glob("run_*"))
    assert len(runs) == 2
    events = [json.loads(line) for line in (tmp_path / "research-ledger.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events] == [
        "agent_handoff_ingested",
        "proposal_created",
        "candidate_created",
        "candidate_submitted",
        "run_started",
        "run_completed",
    ]


def test_autonomy_step_execute_candidate_submission_writes_run_to_configured_external_runs_root(
    tmp_path: Path, monkeypatch
):
    import ml_autoresearch.execution as execution

    write_project(tmp_path)
    external_runs = tmp_path / "external-runs"
    config_path = tmp_path / "ml-autoresearch.toml"
    config_path.write_text(
        config_path.read_text().replace(
            'ledger_path = "research-ledger.jsonl"\n',
            f'ledger_path = "research-ledger.jsonl"\nruns_root = "{external_runs}"\n',
        )
    )
    FakeNativeBackend.provider_ids = []
    monkeypatch.setattr(execution, "NativeBackend", FakeNativeBackend)
    fake_command = write_fake_agent(tmp_path / "fake_agent.py", _candidate_submission_agent_body())

    result = run_autonomy_step(tmp_path, agent_command=fake_command, execute_next_action=True)

    assert result.status == "ingested"
    assert result.ingestion is not None
    assert result.ingestion["next_action_result"]["run_status"] == "completed"
    assert list(external_runs.glob("run_*/run_metadata.json"))
    assert not (tmp_path / "runs").exists()


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


def test_autonomy_step_execute_evaluation_request_uses_docker_backend_when_configured(tmp_path: Path, monkeypatch):
    import ml_autoresearch.execution as execution
    import ml_autoresearch.evaluation_requests as evaluation_requests

    write_project(tmp_path)
    config_path = tmp_path / "ml-autoresearch.toml"
    config_path.write_text(
        config_path.read_text().replace(
            '[candidate_execution]\nledger_path = "research-ledger.jsonl"\n',
            '[candidate_execution]\nledger_path = "research-ledger.jsonl"\nbackend = "docker"\ndocker_image = "custom:tag"\n',
        )
    )
    _write_completed_run(tmp_path, "run_123")
    FakeDockerBackend.calls = []
    monkeypatch.setattr(execution, "DockerBackend", FakeDockerBackend)
    monkeypatch.setattr(
        evaluation_requests,
        "run_post_run_evaluation",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("host native evaluation must not run")),
    )
    fake_command = write_fake_agent(tmp_path / "fake_agent.py", _evaluation_request_agent_body())

    result = run_autonomy_step(tmp_path, agent_command=fake_command, execute_next_action=True)

    assert result.ingestion is not None
    next_action = result.ingestion["next_action_result"]
    assert next_action["action"] == "run_post_run_evaluation"
    assert next_action["backend"] == "docker"
    assert next_action["evaluation_id"] == "eval_eval-threshold-sweep-run-123"
    assert FakeDockerBackend.calls == [
        {
            "request_path": tmp_path / "evaluation-requests" / "eval-threshold-sweep-run-123.yaml",
            "runs_root": tmp_path / "runs",
            "ledger_path": tmp_path / "research-ledger.jsonl",
        }
    ]
    events = [json.loads(line) for line in (tmp_path / "research-ledger.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events] == [
        "agent_handoff_ingested",
        "evaluation_requested",
        "evaluation_completed",
    ]


def test_execute_next_action_cli_continues_accepted_candidate_run_from_previous_action(tmp_path: Path, monkeypatch):
    import ml_autoresearch.execution as execution

    write_project(tmp_path)
    data_root = tmp_path / "gvccs"
    data_root.mkdir()
    _write_completed_run(tmp_path, "run_prior", data_root=data_root)
    monkeypatch.setattr(execution, "NativeBackend", FakeNativeBackend)
    fake_command = write_fake_agent(tmp_path / "fake_agent.py", _candidate_submission_agent_body())
    result = run_autonomy_step(tmp_path, agent_command=fake_command)
    assert result.ingestion is not None
    from ml_autoresearch.autonomy_step import execute_ingested_next_action
    execution_result = execute_ingested_next_action(tmp_path, result.ingestion)
    assert execution_result["run_status"] == "completed"

    # Simulate the pre-fix state: a candidate had been accepted, but not trained, and
    # execute-next-action had marked the accepted run as already executed.
    accepted_run = next(path for path in (tmp_path / "runs").glob("run_*") if path.name != "run_prior")
    metadata = json.loads((accepted_run / "run_metadata.json").read_text())
    metadata["status"] = "accepted"
    (accepted_run / "run_metadata.json").write_text(json.dumps(metadata) + "\n")
    result_payload = result.to_json()
    assert result_payload["ingestion"] is not None
    result_payload["ingestion"]["executed_next_action"] = True
    result_payload["ingestion"]["next_action_result"] = {
        "action": "run_candidate",
        "executed": True,
        "status": "completed",
        "run_id": accepted_run.name,
        "run_dir": str(accepted_run),
        "run_status": "accepted",
    }
    (tmp_path / "agent-work" / "autonomy-step-result.json").write_text(json.dumps(result_payload))

    from ml_autoresearch.autonomy_step import execute_outstanding_next_action

    continuation = execute_outstanding_next_action(tmp_path)

    assert continuation.execution is not None
    assert continuation.execution["status"] == "completed"
    result_file = json.loads((tmp_path / "agent-work" / "autonomy-step-result.json").read_text())
    assert result_file["ingestion"]["next_action_result"]["run_status"] == "completed"
    assert json.loads((accepted_run / "run_metadata.json").read_text())["status"] == "completed"



def test_execute_next_action_resubmits_candidate_when_recorded_accepted_run_was_deleted(tmp_path: Path, monkeypatch):
    import ml_autoresearch.execution as execution

    write_project(tmp_path)
    data_root = tmp_path / "gvccs"
    data_root.mkdir()
    _write_completed_run(tmp_path, "run_prior", data_root=data_root)
    monkeypatch.setattr(execution, "NativeBackend", FakeNativeBackend)
    fake_command = write_fake_agent(tmp_path / "fake_agent.py", _candidate_submission_agent_body())
    result = run_autonomy_step(tmp_path, agent_command=fake_command)
    assert result.ingestion is not None
    deleted_run = tmp_path / "runs" / "run_deleted"
    result_payload = result.to_json()
    assert result_payload["ingestion"] is not None
    result_payload["ingestion"]["executed_next_action"] = True
    result_payload["ingestion"]["next_action_result"] = {
        "action": "run_candidate",
        "executed": True,
        "status": "completed",
        "run_id": deleted_run.name,
        "run_dir": str(deleted_run),
        "run_status": "accepted",
    }
    (tmp_path / "agent-work" / "autonomy-step-result.json").write_text(json.dumps(result_payload))

    from ml_autoresearch.autonomy_step import execute_outstanding_next_action

    continuation = execute_outstanding_next_action(tmp_path)

    assert continuation.execution is not None
    assert continuation.execution["status"] == "completed"
    assert continuation.execution["run_status"] == "completed"
    completed_runs = [path for path in (tmp_path / "runs").glob("run_*") if path.name != "run_prior"]
    assert len(completed_runs) == 1
    assert json.loads((completed_runs[0] / "run_metadata.json").read_text())["status"] == "completed"



def test_execute_next_action_cli_runs_outstanding_action_from_previous_autonomy_step(tmp_path: Path):
    write_project(tmp_path)
    _write_completed_run(tmp_path, "run_123")
    fake_command = write_fake_agent(tmp_path / "fake_agent.py", _evaluation_request_agent_body())
    result = run_autonomy_step(tmp_path, agent_command=fake_command)
    assert result.ingestion is not None
    assert result.ingestion["executed_next_action"] is False

    completed = run_cli(tmp_path, "execute-next-action", "--workspace-root", str(tmp_path))

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

    completed = run_cli(tmp_path, "execute-next-action", "--workspace-root", str(tmp_path))

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
        "--workspace-root",
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
research_problem: tiny_problem
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


def _write_completed_run(root: Path, run_id: str, *, data_root: Path | None = None) -> None:
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    metadata = {
        "run_id": run_id,
        "status": "completed",
        "research_problem": {
            "id": "tiny_problem",
            "version": "test-spec-v0",
            "contract_version": "v0",
            "provider": {
                "target": "fake_research_problem.research_problem:build_spec",
                "resolved_package_root": str(root),
            },
        },
    }
    if data_root is not None:
        metadata["dataset"] = {"host_data_path": str(data_root)}
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata) + "\n")
