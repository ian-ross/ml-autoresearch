import json
from pathlib import Path

import pytest

from ml_autoresearch.cli import app
from ml_autoresearch.agent_handoffs import (
    AgentHandoffIngestionError,
    collect_agent_handoff,
    ingest_campaign_report,
    ingest_candidate_submission,
    ingest_capability_request,
    ingest_experiment_batch_submission,
    ingest_evaluation_request,
    ingest_research_note,
)
from conftest import invoke_typer_cli


def run_cli(cwd: Path, *args: str):
    return invoke_typer_cli(app, args, cwd=cwd)



def write_fake_research_problem_provider(root: Path) -> None:
    package = root / "fake_research_problem"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "research_problem.py").write_text(
        "from ml_autoresearch.research_problems import ResearchProblemSpec\n"
        "def build_spec(data_config=None):\n"
        "    return ResearchProblemSpec(\n"
        "        id='tiny_problem', version='test-spec-v0', contract_version='v0',\n"
        "        input_modes=('single_frame_rgb',), input_specs={'single_frame_rgb': {'mode': 'single_frame_rgb', 'shape': [3, 16, 16]}},\n"
        "        output_forms=('mask_logits',), output_specs={'mask_logits': {'form': 'mask_logits', 'shape': [1, 16, 16]}},\n"
        "        losses=('bce_dice',), optimizers=('adamw',), sampling_policies=('sequential',),\n"
        "        augmentation_policies=('none',), primary_metric='val/dice',\n"
        "    )\n"
    )
    (root / "candidate-execution.toml").write_text(
        "[candidate_execution]\n"
        "ledger_path = \"research-ledger.jsonl\"\n"
        "\n"
        "[research_problem]\n"
        "id = \"tiny_problem\"\n"
        f"package_root = \"{root}\"\n"
        "provider_target = \"fake_research_problem.research_problem:build_spec\"\n"
        "expected_contract_version = \"v0\"\n"
    )

def write_project(root: Path) -> None:
    write_fake_research_problem_provider(root)
    (root / "candidates").mkdir()
    (root / "agent-work" / "submissions").mkdir(parents=True)
    (root / "agent-work" / "research-notes").mkdir(parents=True)
    (root / "agent-work" / "capability-requests").mkdir(parents=True)
    (root / "agent-work" / "evaluation-requests").mkdir(parents=True)
    (root / "agent-work" / "campaign-reports").mkdir(parents=True)
    (root / "research-notes").mkdir()
    (root / "research-ledger.jsonl").write_text("")
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


def write_research_note(root: Path, filename: str = "2026-05-09-agent-note.md", title: str | None = "Agent Note") -> Path:
    note = root / "agent-work" / "research-notes" / filename
    heading = f"# {title}\n\n" if title is not None else ""
    note.write_text(
        heading
        + "## Summary\n"
        + "Agent-authored Research Note for ingestion.\n\n"
        + "## Decision\n"
        + "Continue the Autonomous Research Iteration.\n"
    )
    return note


def test_ingest_one_research_note_copies_to_canonical_updates_index_and_records_events(tmp_path: Path):
    write_project(tmp_path)
    source = write_research_note(tmp_path)

    result = ingest_research_note(tmp_path)

    assert result["status"] == "ingested"
    assert result["handoff_type"] == "research_note"
    assert result["note_path"] == "research-notes/2026-05-09-agent-note.md"
    assert result["next_action"] == "continue_autonomy"
    assert result["executed_next_action"] is False
    canonical = tmp_path / "research-notes" / "2026-05-09-agent-note.md"
    assert canonical.read_text() == source.read_text()
    assert source.is_file()

    index = (tmp_path / "EXPERIMENT_INDEX.md").read_text()
    assert "[`Agent Note`](research-notes/2026-05-09-agent-note.md)" in index

    events = [json.loads(line) for line in (tmp_path / "research-ledger.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events] == ["agent_handoff_ingested", "research_note_written"]
    assert events[0]["handoff_type"] == "research_note"
    assert events[0]["artifact_id"] == "2026-05-09-agent-note.md"
    assert events[0]["source_path"] == "agent-work/research-notes/2026-05-09-agent-note.md"
    assert events[0]["canonical_path"] == "research-notes/2026-05-09-agent-note.md"
    assert events[0]["note_path"] == "research-notes/2026-05-09-agent-note.md"
    assert events[1]["note_path"] == "research-notes/2026-05-09-agent-note.md"
    marker = json.loads((source.parent / "2026-05-09-agent-note.md.INGESTED.json").read_text())
    assert marker["handoff_type"] == "research_note"
    assert marker["canonical_path"] == "research-notes/2026-05-09-agent-note.md"


def test_ingest_research_note_uses_filename_title_fallback(tmp_path: Path):
    write_project(tmp_path)
    write_research_note(tmp_path, filename="2026-05-10-no-heading-note.md", title=None)

    ingest_research_note(tmp_path)

    index = (tmp_path / "EXPERIMENT_INDEX.md").read_text()
    assert "[`2026 05 10 No Heading Note`](research-notes/2026-05-10-no-heading-note.md)" in index


def test_ingest_research_note_fails_before_side_effects_for_empty_or_malformed_notes(tmp_path: Path):
    write_project(tmp_path)
    source = tmp_path / "agent-work" / "research-notes" / "empty.md"
    source.write_text("\n")

    with pytest.raises(AgentHandoffIngestionError, match="empty"):
        ingest_research_note(tmp_path)

    assert not (tmp_path / "research-notes" / "empty.md").exists()
    assert not (source.parent / "empty.md.INGESTED.json").exists()
    assert (tmp_path / "research-ledger.jsonl").read_text() == ""
    assert "empty.md" not in (tmp_path / "EXPERIMENT_INDEX.md").read_text()

    source.write_text(
        "# Bad figures\n\n"
        "```research-figures\n"
        "figures:\n"
        "  - figure_id: missing-source\n"
        "    source_artifact_path: runs/x/figure.png\n"
        "    reason: Demonstrate validation.\n"
        "```\n"
    )
    with pytest.raises(AgentHandoffIngestionError, match="Research Figure"):
        ingest_research_note(tmp_path)
    assert not (tmp_path / "research-notes" / "empty.md").exists()
    assert not (source.parent / "empty.md.INGESTED.json").exists()
    assert (tmp_path / "research-ledger.jsonl").read_text() == ""


def test_ingest_research_note_fails_closed_for_duplicate_destination_and_multiple_notes(tmp_path: Path):
    write_project(tmp_path)
    source = write_research_note(tmp_path)
    existing = tmp_path / "research-notes" / source.name
    existing.write_text("existing\n")

    with pytest.raises(AgentHandoffIngestionError, match="already exists"):
        ingest_research_note(tmp_path)

    assert existing.read_text() == "existing\n"
    assert not (source.parent / f"{source.name}.INGESTED.json").exists()
    assert (tmp_path / "research-ledger.jsonl").read_text() == ""
    assert source.name not in (tmp_path / "EXPERIMENT_INDEX.md").read_text()

    existing.unlink()
    write_research_note(tmp_path, filename="2026-05-10-other-note.md")
    with pytest.raises(AgentHandoffIngestionError, match="expected exactly one"):
        ingest_research_note(tmp_path)


def test_ingest_research_note_cli_reports_next_action_without_infrastructure_work(tmp_path: Path):
    write_project(tmp_path)
    write_research_note(tmp_path)

    completed = run_cli(tmp_path, "ingest-research-note")

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ingested"
    assert payload["next_action"] == "continue_autonomy"
    assert payload["executed_next_action"] is False
    assert (tmp_path / "research-notes" / "2026-05-09-agent-note.md").is_file()
    assert not (tmp_path / "runs").exists()


def write_batch_submission(root: Path, batch_id: str = "agent_batch") -> Path:
    submission = root / "agent-work" / "batch-submissions" / batch_id
    batch = submission / "experiment_batch"
    candidates = batch / "candidates"
    candidates.mkdir(parents=True)
    (batch / "BATCH_PROPOSAL.md").write_text(
        """# Batch Proposal

## Shared hypothesis
Small variants should train.

## Shared comparison target
Baseline run.

## Per-candidate variant rationale
Each candidate isolates one small variant.

## Decision criteria
Compare validation Dice.

## Success criteria
At least one variant completes.

## Requested budget/concurrency
Request two Runs with Harness-owned concurrency.
"""
    )
    for candidate_id in ["batch_a", "batch_b"]:
        candidate = candidates / candidate_id
        candidate.mkdir()
        (candidate / "manifest.yaml").write_text(
            f"""
name: {candidate_id}
description: Batch candidate.
research_problem: tiny_problem
input_mode: single_frame_rgb
output_form: mask_logits
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".lstrip()
        )
        (candidate / "model.py").write_text("raise RuntimeError('must not execute during ingestion')\n")
        (candidate / "README.md").write_text(f"# {candidate_id}\n")
    (submission / "submission.json").write_text(
        json.dumps(
            {
                "schema_version": "experiment_batch_submission.v1",
                "submission_type": "experiment_batch",
                "batch_submission_id": batch_id,
                "batch_path": "experiment_batch",
                "requested_action": "validate_and_queue_batch_for_harness_execution",
            },
            indent=2,
        )
        + "\n"
    )
    return submission


def write_submission(root: Path, candidate_id: str = "agent_candidate") -> Path:
    submission = root / "agent-work" / "submissions" / candidate_id
    candidate = submission / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "manifest.yaml").write_text(
        f"""
name: {candidate_id}
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
""".lstrip()
    )
    (candidate / "model.py").write_text("raise RuntimeError('must not execute during ingestion')\n")
    (candidate / "README.md").write_text("# Agent candidate\n")
    (candidate / "PROPOSAL.md").write_text(
        """## Hypothesis
Test ingestion.

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
"""
    )
    (submission / "submission.json").write_text(
        json.dumps(
            {
                "schema_version": "candidate_submission.v1",
                "submission_type": "candidate_experiment",
                "candidate_id": candidate_id,
                "candidate_path": "candidate",
                "requested_action": "validate_and_queue_for_harness_execution",
            },
            indent=2,
        )
        + "\n"
    )
    return submission


def test_ingest_one_candidate_submission_copies_to_canonical_updates_index_and_ledger(tmp_path: Path):
    write_project(tmp_path)
    source = write_submission(tmp_path)

    result = ingest_candidate_submission(tmp_path)

    assert result["status"] == "ingested"
    assert result["candidate_id"] == "agent_candidate"
    assert result["next_action"] == "run_candidate"
    assert result["executed_next_action"] is False
    canonical = tmp_path / "candidates" / "agent_candidate"
    assert (canonical / "manifest.yaml").read_text() == (source / "candidate" / "manifest.yaml").read_text()
    assert (canonical / "README.md").read_text() == "# Agent candidate\n"
    assert (canonical / "PROPOSAL.md").is_file()
    assert (source / "candidate" / "manifest.yaml").is_file()

    index = (tmp_path / "EXPERIMENT_INDEX.md").read_text()
    assert "`candidates/agent_candidate`" in index
    assert "[`README.md`](candidates/agent_candidate/README.md)" in index
    assert "Pending Harness Run" in index

    events = [json.loads(line) for line in (tmp_path / "research-ledger.jsonl").read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["event_type"] == "agent_handoff_ingested"
    assert events[0]["handoff_type"] == "candidate_submission"
    assert events[0]["candidate_id"] == "agent_candidate"
    assert events[0]["artifact_id"] == "agent_candidate"
    assert events[0]["canonical_path"] == "candidates/agent_candidate"
    marker = json.loads((source / "INGESTED.json").read_text())
    assert marker["candidate_id"] == "agent_candidate"
    assert marker["canonical_path"] == "candidates/agent_candidate"


def test_ingest_experiment_batch_submission_copies_to_canonical_batch_and_records_next_action(tmp_path: Path):
    write_project(tmp_path)
    source = write_batch_submission(tmp_path)

    result = ingest_experiment_batch_submission(tmp_path)

    assert result["status"] == "ingested"
    assert result["handoff_type"] == "experiment_batch_submission"
    assert result["next_action"] == "run_experiment_batch"
    assert result["candidate_count"] == 2
    assert (tmp_path / "experiment-batches" / "agent_batch" / "BATCH_PROPOSAL.md").is_file()
    assert (source / "experiment_batch" / "BATCH_PROPOSAL.md").is_file()
    marker = json.loads((source / "INGESTED.json").read_text())
    assert marker["handoff_type"] == "experiment_batch_submission"
    assert marker["canonical_path"] == "experiment-batches/agent_batch"


def test_ingest_candidate_submission_cli_reports_next_action_without_running_candidate(tmp_path: Path):
    write_project(tmp_path)
    write_submission(tmp_path)

    completed = run_cli(tmp_path, "ingest-candidate-submission")

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ingested"
    assert payload["next_action"] == "run_candidate"
    assert payload["executed_next_action"] is False
    assert (tmp_path / "candidates" / "agent_candidate").is_dir()
    assert not (tmp_path / "runs").exists()


def test_ingest_candidate_submission_fails_closed_when_canonical_candidate_exists(tmp_path: Path):
    write_project(tmp_path)
    source = write_submission(tmp_path)
    existing = tmp_path / "candidates" / "agent_candidate"
    existing.mkdir()
    (existing / "README.md").write_text("existing\n")

    with pytest.raises(AgentHandoffIngestionError, match="already exists"):
        ingest_candidate_submission(tmp_path)

    assert (existing / "README.md").read_text() == "existing\n"
    assert not (source / "INGESTED.json").exists()
    assert (tmp_path / "research-ledger.jsonl").read_text() == ""
    assert "agent_candidate" not in (tmp_path / "EXPERIMENT_INDEX.md").read_text()


def test_ingest_candidate_submission_requires_exactly_one_uningested_submission(tmp_path: Path):
    write_project(tmp_path)
    with pytest.raises(AgentHandoffIngestionError, match="expected exactly one"):
        ingest_candidate_submission(tmp_path)

    first = write_submission(tmp_path, "first_candidate")
    write_submission(tmp_path, "second_candidate")
    with pytest.raises(AgentHandoffIngestionError, match="expected exactly one"):
        ingest_candidate_submission(tmp_path)

    (first / "INGESTED.json").write_text(
        json.dumps(
            {
                "status": "ingested",
                "handoff_type": "candidate_submission",
                "canonical_path": "candidates/first_candidate",
            }
        )
        + "\n"
    )
    result = ingest_candidate_submission(tmp_path)
    assert result["candidate_id"] == "second_candidate"


def test_ingest_candidate_submission_validates_metadata_and_candidate_before_copy(tmp_path: Path):
    write_project(tmp_path)
    source = write_submission(tmp_path)
    (source / "submission.json").write_text('{"candidate_id": "agent_candidate"}\n')

    with pytest.raises(AgentHandoffIngestionError, match="submission.json"):
        ingest_candidate_submission(tmp_path)

    assert not (tmp_path / "candidates" / "agent_candidate").exists()
    assert not (source / "INGESTED.json").exists()
    assert (tmp_path / "research-ledger.jsonl").read_text() == ""


def write_capability_request(root: Path, filename: str = "capability-temporal-inputs.yaml", **overrides: object) -> Path:
    import yaml

    data = {
        "request_id": "capability-temporal-inputs",
        "capability_type": "contract_surface",
        "blocked_hypothesis": "Temporal context could improve thin contrail segmentation.",
        "current_contract_insufficiency": "The current contract only exposes single-frame RGB inputs.",
        "expected_research_value": "Test whether adjacent frames reduce false negatives.",
        "safety_reproducibility_risks": "Temporal grouping must remain Harness-owned.",
        "minimal_harness_change": "Add an allowlisted temporal input mode.",
        "candidate_authority_requested": "none",
        "example_follow_up_experiments": ["Compare single-frame RGB against temporal RGB clips."],
        "priority": "medium",
    }
    data.update(overrides)
    path = root / "agent-work" / "capability-requests" / filename
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def write_evaluation_request(root: Path, filename: str = "eval-threshold-sweep-run-123.yaml", **overrides: object) -> Path:
    import yaml

    data = {
        "request_id": "eval-threshold-sweep-run-123",
        "target_run_id": "run_123",
        "evaluation_mode": "threshold_sweep",
        "diagnostic_question": "Which threshold best separates thin masks?",
        "expected_decision_impact": "Decide whether low Dice is thresholding or representation failure.",
        "parameters": {"threshold_sweep": {"min": 0.1, "max": 0.9, "steps": 9}, "artifact_count": 4},
        "artifact_budget": {"max_artifacts": 6, "max_runtime_seconds": 120},
    }
    data.update(overrides)
    path = root / "agent-work" / "evaluation-requests" / filename
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def write_campaign_report(
    root: Path,
    filename: str = "2026-05-10-status.md",
    *,
    pause_condition: str = "none",
) -> Path:
    path = root / "agent-work" / "campaign-reports" / filename
    path.write_text(
        "# Campaign Report: GVCCS\n\n"
        "## Summary\nAutonomy smoke loop status.\n\n"
        "## Current best Result\n- Run: none\n\n"
        "## Recent Runs\n| Run | Candidate Experiment | Status | Key Result | Note |\n| --- | --- | --- | --- | --- |\n\n"
        "## Failures\n| Run | Failure classification | Symptom | Follow-up |\n| --- | --- | --- | --- |\n\n"
        "## Pending Capability Requests\n| Request | Status | Why it matters | Blocking? |\n| --- | --- | --- | --- |\n\n"
        "## Budget use\n- Wall-clock budget used: unknown\n\n"
        "## Next hypothesis\nContinue with the next bounded Candidate Experiment.\n\n"
        "## Pause recommendation\n"
        f"- Pause condition: {pause_condition}\n"
        "- Human decision needed: no\n"
    )
    return path


def test_ingest_capability_request_copies_to_canonical_records_events_and_stops(tmp_path: Path) -> None:
    write_project(tmp_path)
    source = write_capability_request(tmp_path)

    result = ingest_capability_request(tmp_path)

    assert result["status"] == "ingested"
    assert result["handoff_type"] == "capability_request"
    assert result["request_id"] == "capability-temporal-inputs"
    assert result["canonical_path"] == "capability-requests/capability-temporal-inputs.yaml"
    assert result["next_action"] == "stop_for_human"
    assert result["executed_next_action"] is False
    assert (tmp_path / "capability-requests" / source.name).read_text() == source.read_text()
    events = [json.loads(line) for line in (tmp_path / "research-ledger.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events] == ["agent_handoff_ingested", "capability_request_created"]
    assert events[0]["handoff_type"] == "capability_request"
    assert events[0]["request_id"] == "capability-temporal-inputs"
    assert events[1]["request_path"] == "capability-requests/capability-temporal-inputs.yaml"
    assert json.loads((source.parent / f"{source.name}.INGESTED.json").read_text())["canonical_path"] == result["canonical_path"]


def test_ingest_evaluation_request_copies_without_recording_execution_event(tmp_path: Path) -> None:
    write_project(tmp_path)
    source = write_evaluation_request(tmp_path)

    result = ingest_evaluation_request(tmp_path)

    assert result["handoff_type"] == "evaluation_request"
    assert result["request_id"] == "eval-threshold-sweep-run-123"
    assert result["run_id"] == "run_123"
    assert result["next_action"] == "run_post_run_evaluation"
    assert result["executed_next_action"] is False
    assert (tmp_path / "evaluation-requests" / source.name).read_text() == source.read_text()
    events = [json.loads(line) for line in (tmp_path / "research-ledger.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events] == ["agent_handoff_ingested"]
    assert events[0]["handoff_type"] == "evaluation_request"
    assert events[0]["run_id"] == "run_123"
    assert "evaluation_requested" not in (tmp_path / "research-ledger.jsonl").read_text()


def test_ingest_campaign_report_copies_records_events_and_selects_next_action(tmp_path: Path) -> None:
    write_project(tmp_path)
    source = write_campaign_report(tmp_path, pause_condition="`scheduled_check_in`")

    result = ingest_campaign_report(tmp_path)

    assert result["handoff_type"] == "campaign_report"
    assert result["report_path"] == "campaign-reports/2026-05-10-status.md"
    assert result["next_action"] == "pause_campaign"
    assert result["executed_next_action"] is False
    assert (tmp_path / "campaign-reports" / source.name).read_text() == source.read_text()
    events = [json.loads(line) for line in (tmp_path / "research-ledger.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events] == ["agent_handoff_ingested", "campaign_report_written"]
    assert events[0]["handoff_type"] == "campaign_report"
    assert events[1]["report_path"] == "campaign-reports/2026-05-10-status.md"


def test_request_and_report_ingestion_create_canonical_directories_lazily(tmp_path: Path) -> None:
    write_project(tmp_path)
    (tmp_path / "capability-requests").rmdir() if (tmp_path / "capability-requests").exists() else None
    source = write_capability_request(tmp_path)

    ingest_capability_request(tmp_path)

    assert (tmp_path / "capability-requests" / source.name).is_file()


def test_request_and_report_ingestion_fail_closed_for_duplicate_and_invalid_artifacts(tmp_path: Path) -> None:
    write_project(tmp_path)
    capability = write_capability_request(tmp_path, priority="invalid")

    with pytest.raises(AgentHandoffIngestionError, match="priority"):
        ingest_capability_request(tmp_path)

    assert not (tmp_path / "capability-requests" / capability.name).exists()
    assert not (capability.parent / f"{capability.name}.INGESTED.json").exists()
    assert (tmp_path / "research-ledger.jsonl").read_text() == ""

    capability.unlink()
    evaluation = write_evaluation_request(tmp_path)
    (tmp_path / "evaluation-requests").mkdir()
    (tmp_path / "evaluation-requests" / evaluation.name).write_text("existing\n")
    with pytest.raises(AgentHandoffIngestionError, match="already exists"):
        ingest_evaluation_request(tmp_path)
    assert (tmp_path / "evaluation-requests" / evaluation.name).read_text() == "existing\n"
    assert not (evaluation.parent / f"{evaluation.name}.INGESTED.json").exists()
    assert (tmp_path / "research-ledger.jsonl").read_text() == ""

    evaluation.unlink()
    report = write_campaign_report(tmp_path, pause_condition="agent whim")
    with pytest.raises(AgentHandoffIngestionError, match="Pause condition"):
        ingest_campaign_report(tmp_path)
    assert not (tmp_path / "campaign-reports" / report.name).exists()
    assert not (report.parent / f"{report.name}.INGESTED.json").exists()
    assert (tmp_path / "research-ledger.jsonl").read_text() == ""


def test_non_executing_handoff_cli_commands_report_next_actions(tmp_path: Path) -> None:
    write_project(tmp_path)
    write_evaluation_request(tmp_path)

    completed = run_cli(tmp_path, "ingest-evaluation-request")

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["next_action"] == "run_post_run_evaluation"
    assert payload["executed_next_action"] is False
    assert not (tmp_path / "runs").exists()


def test_collect_agent_handoff_cli_reports_invariant_failure_without_side_effects(tmp_path: Path) -> None:
    write_project(tmp_path)
    write_submission(tmp_path)
    write_capability_request(tmp_path)

    completed = run_cli(tmp_path, "ingest-agent-handoff")

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ingestion_failed"
    assert "multiple primary handoff categories" in payload["reason"]
    assert not (tmp_path / "candidates" / "agent_candidate").exists()
    assert not (tmp_path / "capability-requests" / "capability-temporal-inputs.yaml").exists()
    assert (tmp_path / "research-ledger.jsonl").read_text() == ""


def test_capability_request_ingestion_fails_closed_for_duplicate_destination(tmp_path: Path) -> None:
    write_project(tmp_path)
    source = write_capability_request(tmp_path)
    (tmp_path / "capability-requests").mkdir()
    (tmp_path / "capability-requests" / source.name).write_text("existing\n")

    with pytest.raises(AgentHandoffIngestionError, match="already exists"):
        ingest_capability_request(tmp_path)

    assert (tmp_path / "capability-requests" / source.name).read_text() == "existing\n"
    assert not (source.parent / f"{source.name}.INGESTED.json").exists()
    assert (tmp_path / "research-ledger.jsonl").read_text() == ""


def test_evaluation_request_ingestion_fails_before_side_effects_for_invalid_request(tmp_path: Path) -> None:
    write_project(tmp_path)
    source = write_evaluation_request(tmp_path, target_run_id="../unsafe")

    with pytest.raises(AgentHandoffIngestionError, match="target_run_id"):
        ingest_evaluation_request(tmp_path)

    assert not (tmp_path / "evaluation-requests" / source.name).exists()
    assert not (source.parent / f"{source.name}.INGESTED.json").exists()
    assert (tmp_path / "research-ledger.jsonl").read_text() == ""


def test_campaign_report_ingestion_fails_closed_for_duplicate_destination(tmp_path: Path) -> None:
    write_project(tmp_path)
    source = write_campaign_report(tmp_path)
    (tmp_path / "campaign-reports").mkdir()
    (tmp_path / "campaign-reports" / source.name).write_text("existing\n")

    with pytest.raises(AgentHandoffIngestionError, match="already exists"):
        ingest_campaign_report(tmp_path)

    assert (tmp_path / "campaign-reports" / source.name).read_text() == "existing\n"
    assert not (source.parent / f"{source.name}.INGESTED.json").exists()
    assert (tmp_path / "research-ledger.jsonl").read_text() == ""


def test_collect_agent_handoff_returns_no_handoff_for_empty_or_scratch_only_workspace(tmp_path: Path) -> None:
    write_project(tmp_path)

    result = collect_agent_handoff(tmp_path)

    assert result == {
        "status": "no_handoff",
        "next_action": "stop_for_human",
        "executed_next_action": False,
        "ledger_events": [],
    }
    assert (tmp_path / "research-ledger.jsonl").read_text() == ""

    (tmp_path / "agent-work" / "scratch").mkdir()
    (tmp_path / "agent-work" / "scratch" / "notes.md").write_text("# scratch\n")
    assert collect_agent_handoff(tmp_path)["status"] == "no_handoff"
    assert not (tmp_path / "research-notes" / "notes.md").exists()
    assert (tmp_path / "research-ledger.jsonl").read_text() == ""


def test_collect_agent_handoff_ingests_one_primary_handoff(tmp_path: Path) -> None:
    write_project(tmp_path)
    write_evaluation_request(tmp_path)

    result = collect_agent_handoff(tmp_path)

    assert result["status"] == "ingested"
    assert result["handoff_type"] == "evaluation_request"
    assert result["next_action"] == "run_post_run_evaluation"
    assert (tmp_path / "evaluation-requests" / "eval-threshold-sweep-run-123.yaml").is_file()
    events = [json.loads(line) for line in (tmp_path / "research-ledger.jsonl").read_text().splitlines()]
    assert [event["event_type"] for event in events] == ["agent_handoff_ingested"]


def test_collect_agent_handoff_fails_closed_for_multiple_primary_categories(tmp_path: Path) -> None:
    write_project(tmp_path)
    write_submission(tmp_path)
    write_research_note(tmp_path)

    result = collect_agent_handoff(tmp_path)

    assert result["status"] == "ingestion_failed"
    assert "multiple primary handoff categories" in result["reason"]
    assert result["handoff_types"] == ["candidate_submission", "research_note"]
    assert result["next_action"] == "stop_for_human"
    assert result["ledger_events"] == []
    assert not (tmp_path / "candidates" / "agent_candidate").exists()
    assert not (tmp_path / "research-notes" / "2026-05-09-agent-note.md").exists()
    assert not (tmp_path / "agent-work" / "submissions" / "agent_candidate" / "INGESTED.json").exists()
    assert not (tmp_path / "agent-work" / "research-notes" / "2026-05-09-agent-note.md.INGESTED.json").exists()
    assert (tmp_path / "research-ledger.jsonl").read_text() == ""


def test_collect_agent_handoff_fails_closed_for_multiple_artifacts_in_same_primary_category(tmp_path: Path) -> None:
    write_project(tmp_path)
    write_capability_request(tmp_path)
    write_capability_request(tmp_path, filename="capability-second.yaml", request_id="capability-second")

    result = collect_agent_handoff(tmp_path)

    assert result["status"] == "ingestion_failed"
    assert "multiple un-ingested capability_request artifacts" in result["reason"]
    assert result["handoff_types"] == ["capability_request"]
    assert not (tmp_path / "capability-requests" / "capability-temporal-inputs.yaml").exists()
    assert not (tmp_path / "capability-requests" / "capability-second.yaml").exists()
    assert not (tmp_path / "agent-work" / "capability-requests" / "capability-temporal-inputs.yaml.INGESTED.json").exists()
    assert (tmp_path / "research-ledger.jsonl").read_text() == ""


def test_collect_agent_handoff_ignores_already_ingested_artifacts_with_valid_source_markers(tmp_path: Path) -> None:
    write_project(tmp_path)
    source = write_campaign_report(tmp_path)

    ingest_campaign_report(tmp_path)
    first_ledger = (tmp_path / "research-ledger.jsonl").read_text()

    result = collect_agent_handoff(tmp_path)

    assert result["status"] == "no_handoff"
    assert (tmp_path / "research-ledger.jsonl").read_text() == first_ledger
    assert json.loads((source.parent / f"{source.name}.INGESTED.json").read_text())["handoff_type"] == "campaign_report"
