import json
import sys
from pathlib import Path

import pytest

from ml_autoresearch.autonomous_iteration import (
    AutonomousIterationError,
    parse_duration_seconds,
    run_autonomous_iteration,
)



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
        "def build_spec(data_config=None):\n"
        "    return ResearchProblemSpec(\n"
        "        id='tiny_problem', version='test-spec-v0', contract_version='v0',\n"
        "        input_modes=('single_frame_rgb',), input_specs={'single_frame_rgb': {'mode': 'single_frame_rgb', 'shape': [3, 16, 16]}},\n"
        "        output_forms=('binary_mask', 'mask_logits'), output_specs={'binary_mask': {'form': 'binary_mask', 'shape': [1, 16, 16]}, 'mask_logits': {'form': 'mask_logits', 'shape': [1, 16, 16]}},\n"
        "        losses=('bce', 'dice_bce', 'bce_dice'), optimizers=('adamw',), sampling_policies=('sequential',),\n"
        "        augmentation_policies=('none',), primary_metric='val/dice',\n"
        "        training_adapter=Adapter(),\n"
        "        brief_documents=(\n"
        "            {'name': 'overview', 'role': 'problem_overview', 'path': 'fake_research_problem/brief/overview.md', 'summary': 'Tiny problem overview.'},\n"
        "            {'name': 'baselines', 'role': 'baseline_description', 'path': 'fake_research_problem/brief/baselines.md', 'summary': 'Tiny baseline notes.', 'required': True},\n"
        "        ),\n"
        "    )\n"
    )
    (root / "candidate-execution.toml").write_text(
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


def write_notification_config(root: Path) -> None:
    (root / "notification.toml").write_text(
        """
[mailjet]
api_key = "key"
api_secret = "secret"
from_email = "autoresearch@example.com"
from_name = "ML Autoresearch"
""".lstrip()
    )


def test_duration_parser_accepts_seconds_and_simple_units() -> None:
    assert parse_duration_seconds("3600") == 3600
    assert parse_duration_seconds("30s") == 30
    assert parse_duration_seconds("10m") == 600
    assert parse_duration_seconds("2h") == 7200


@pytest.mark.parametrize("value", ["0", "0s", "1h30m", "soon", "-1", ""])
def test_duration_parser_rejects_invalid_or_non_positive_values(value: str) -> None:
    with pytest.raises(AutonomousIterationError):
        parse_duration_seconds(value)


def test_autonomous_iteration_runs_until_step_limit_and_sends_notification(tmp_path: Path) -> None:
    write_project(tmp_path)
    write_notification_config(tmp_path)
    fake_command = write_fake_agent(
        tmp_path / "fake_agent.py",
        """
count_path = Path('scratch/count.txt')
count = int(count_path.read_text()) if count_path.is_file() else 0
count += 1
count_path.write_text(str(count))
Path('research-notes').mkdir(exist_ok=True)
Path(f'research-notes/2026-05-22-note-{count}.md').write_text(f'# Note {count}\\n\\n## Summary\\nStep {count}.\\n')
""",
    )
    sent = []

    result = run_autonomous_iteration(
        tmp_path,
        agent_command=fake_command,
        max_steps=2,
        notify_email="user@example.com",
        send_mailjet=lambda message: sent.append(message),
    )

    assert result.stop_reason == "step_limit_reached"
    assert result.steps_completed == 2
    assert result.notification == {"status": "sent", "recipient": "user@example.com"}
    assert len(sent) == 1
    assert sent[0]["to_email"] == "user@example.com"
    assert "step_limit_reached" in sent[0]["subject"]
    payload = json.loads((tmp_path / "agent-work" / "autonomous-iteration-result.json").read_text())
    assert payload["stop_reason"] == "step_limit_reached"
    assert payload["steps_completed"] == 2
    assert len(payload["step_summaries"]) == 2


def test_autonomous_iteration_stops_for_capability_request_before_step_limit(tmp_path: Path) -> None:
    write_project(tmp_path)
    write_notification_config(tmp_path)
    fake_command = write_fake_agent(
        tmp_path / "fake_agent.py",
        """
Path('capability-requests').mkdir(exist_ok=True)
Path('capability-requests/need-capability.yaml').write_text('''request_id: need-capability
capability_type: approved_resource
blocked_hypothesis: Larger batches may stabilize validation loss.
current_contract_insufficiency: Current approved batch size is too small to test this safely.
expected_research_value: Better signal for comparing candidate architectures.
safety_reproducibility_risks: Higher resource use must stay harness bounded.
minimal_harness_change: Raise the approved batch-size bound for candidate execution.
candidate_authority_requested: none
example_follow_up_experiments:
  - Re-run the best recent architecture with the larger approved batch.
priority: medium
''')
""",
    )

    result = run_autonomous_iteration(
        tmp_path,
        agent_command=fake_command,
        max_steps=10,
        notify_email="user@example.com",
        send_mailjet=lambda message: None,
    )

    assert result.stop_reason == "capability_request"
    assert result.steps_completed == 1


def test_autonomous_iteration_requires_a_limit(tmp_path: Path) -> None:
    write_project(tmp_path)
    write_notification_config(tmp_path)

    with pytest.raises(AutonomousIterationError, match="at least one"):
        run_autonomous_iteration(tmp_path, max_steps=None, max_duration_seconds=None, notify_email="user@example.com")


def test_autonomous_iteration_rejects_dirty_workspace_before_start(tmp_path: Path) -> None:
    write_project(tmp_path)
    write_notification_config(tmp_path)
    dirty_dir = tmp_path / "agent-work" / "research-notes"
    dirty_dir.mkdir(parents=True)
    (dirty_dir / "stale.md").write_text("# Stale\n")

    with pytest.raises(AutonomousIterationError, match="un-ingested primary handoff"):
        run_autonomous_iteration(
            tmp_path,
            max_steps=1,
            notify_email="user@example.com",
            send_mailjet=lambda message: None,
        )


def test_autonomous_iteration_records_notification_failure(tmp_path: Path) -> None:
    write_project(tmp_path)
    write_notification_config(tmp_path)
    fake_command = write_fake_agent(tmp_path / "fake_agent.py", "")

    with pytest.raises(AutonomousIterationError, match="notification failed"):
        run_autonomous_iteration(
            tmp_path,
            agent_command=fake_command,
            max_steps=1,
            notify_email="user@example.com",
            send_mailjet=lambda message: (_ for _ in ()).throw(RuntimeError("mailjet down")),
        )

    payload = json.loads((tmp_path / "agent-work" / "autonomous-iteration-result.json").read_text())
    assert payload["stop_reason"] == "no_handoff"
    assert payload["notification"]["status"] == "failed"
    assert "mailjet down" in payload["notification"]["reason"]
