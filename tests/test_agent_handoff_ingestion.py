import json
import subprocess
import sys
from pathlib import Path

import pytest

from ml_autoresearch.agent_handoffs import (
    AgentHandoffIngestionError,
    ingest_candidate_submission,
    ingest_research_note,
)


def run_cli(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ml_autoresearch.cli", *args],
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def write_project(root: Path) -> None:
    (root / "candidates").mkdir()
    (root / "agent-work" / "submissions").mkdir(parents=True)
    (root / "agent-work" / "research-notes").mkdir(parents=True)
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


def write_submission(root: Path, candidate_id: str = "agent_candidate") -> Path:
    submission = root / "agent-work" / "submissions" / candidate_id
    candidate = submission / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "manifest.yaml").write_text(
        f"""
name: {candidate_id}
description: Agent-submitted candidate.
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

    (first / "INGESTED.json").write_text("{}\n")
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
