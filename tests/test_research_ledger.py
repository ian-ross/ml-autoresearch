import json
import subprocess
import sys
from pathlib import Path

import pytest

from ml_autoresearch.research_ledger import ResearchLedgerError, record_research_event


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_record_research_event_appends_valid_event_to_canonical_ledger(tmp_path: Path):
    ledger = tmp_path / "research-ledger.jsonl"

    event = record_research_event(
        "proposal_created",
        {"proposal_id": "proposal-001", "proposal_path": "proposals/proposal-001.md"},
        ledger_path=ledger,
    )

    rows = read_jsonl(ledger)
    assert rows == [event]
    assert event["event_type"] == "proposal_created"
    assert event["proposal_id"] == "proposal-001"
    assert event["proposal_path"] == "proposals/proposal-001.md"
    assert event["created_at"].endswith("Z")


def test_record_research_event_supports_candidate_linkage_for_proposal_created(tmp_path: Path):
    ledger = tmp_path / "research-ledger.jsonl"

    event = record_research_event(
        "proposal_created",
        {
            "proposal_id": "proposal-001",
            "proposal_path": "proposals/proposal-001.md",
            "candidate_id": "candidate-xyz",
        },
        ledger_path=ledger,
    )

    assert event["candidate_id"] == "candidate-xyz"
    rows = read_jsonl(ledger)
    assert rows[0]["candidate_id"] == "candidate-xyz"


def test_record_research_event_supports_candidate_linkage_for_candidate_created(tmp_path: Path):
    ledger = tmp_path / "research-ledger.jsonl"

    event = record_research_event(
        "candidate_created",
        {
            "candidate_id": "candidate-xyz",
            "candidate_path": "candidates/ledger_lifecycle_candidate",
            "proposal_id": "proposal-001",
        },
        ledger_path=ledger,
    )

    assert event["candidate_id"] == "candidate-xyz"
    assert event["candidate_path"] == "candidates/ledger_lifecycle_candidate"
    assert event["proposal_id"] == "proposal-001"
    rows = read_jsonl(ledger)
    assert rows[0]["candidate_path"] == "candidates/ledger_lifecycle_candidate"


def test_record_research_event_rejects_invalid_schema_without_corrupting_ledger(tmp_path: Path):
    ledger = tmp_path / "research-ledger.jsonl"
    record_research_event(
        "proposal_created",
        {"proposal_id": "proposal-001", "proposal_path": "proposals/proposal-001.md"},
        ledger_path=ledger,
    )
    original = ledger.read_text()

    with pytest.raises(ResearchLedgerError, match="proposal_path"):
        record_research_event("proposal_created", {"proposal_id": "proposal-002"}, ledger_path=ledger)
    with pytest.raises(ResearchLedgerError, match="unsupported event_type"):
        record_research_event("unknown", {"id": "x"}, ledger_path=ledger)

    assert ledger.read_text() == original


def test_record_research_event_preserves_append_only_history(tmp_path: Path):
    ledger = tmp_path / "research-ledger.jsonl"

    first = record_research_event("campaign_paused", {"reason": "waiting for review"}, ledger_path=ledger)
    second = record_research_event(
        "research_note_written",
        {"note_path": "research-notes/2026-05-09-note.md", "run_id": "run_abc"},
        ledger_path=ledger,
    )

    assert read_jsonl(ledger) == [first, second]


def test_research_note_written_event_can_reference_figure_provenance_path(tmp_path: Path):
    ledger = tmp_path / "research-ledger.jsonl"

    event = record_research_event(
        "research_note_written",
        {
            "note_path": "research-notes/2026-05-09-note.md",
            "figure_provenance_path": "research-notes/2026-05-09-note.md#research-figures",
        },
        ledger_path=ledger,
    )

    assert event["figure_provenance_path"] == "research-notes/2026-05-09-note.md#research-figures"


def test_research_note_written_event_can_embed_figure_provenance_metadata(tmp_path: Path):
    ledger = tmp_path / "research-ledger.jsonl"

    event = record_research_event(
        "research_note_written",
        {
            "note_path": "research-notes/2026-05-09-note.md",
            "figure_provenance": [
                {
                    "figure_id": "fig-overlay-001",
                    "source_run_id": "run_abc",
                    "source_artifact_path": "outputs/prediction_samples/sample_000_overlay.png",
                    "reason": "Shows the clearest false negative in the saved prediction samples.",
                }
            ],
        },
        ledger_path=ledger,
    )

    assert event["figure_provenance"][0]["source_run_id"] == "run_abc"


def test_record_research_event_cli_validates_and_appends_json(tmp_path: Path):
    ledger = tmp_path / "research-ledger.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_autoresearch.cli",
            "record-research-event",
            "--ledger-path",
            str(ledger),
            "--event-type",
            "run_started",
            "--field",
            "run_id=run_123",
            "--field",
            "candidate_id=candidate_abc",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["event_type"] == "run_started"
    assert payload["run_id"] == "run_123"
    assert read_jsonl(ledger) == [payload]


def test_record_research_event_cli_rejects_missing_required_fields(tmp_path: Path):
    ledger = tmp_path / "research-ledger.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_autoresearch.cli",
            "record-research-event",
            "--ledger-path",
            str(ledger),
            "--event-type",
            "run_completed",
            "--field",
            "run_id=run_123",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 1
    assert "metrics_path" in completed.stderr
    assert not ledger.exists()


def test_run_failed_event_persists_and_validates_failure_classification(tmp_path: Path):
    ledger = tmp_path / "research-ledger.jsonl"

    event = record_research_event(
        "run_failed",
        {"run_id": "run_123", "error": "out of memory", "failure_classification": "resource_failure"},
        ledger_path=ledger,
    )

    assert event["failure_classification"] == "resource_failure"
    assert read_jsonl(ledger) == [event]

    with pytest.raises(ResearchLedgerError, match="failure_classification"):
        record_research_event(
            "run_failed",
            {"run_id": "run_124", "error": "bad", "failure_classification": "invalid"},
            ledger_path=ledger,
        )

    assert read_jsonl(ledger) == [event]


def test_candidate_created_event_persists_repair_lineage(tmp_path: Path):
    ledger = tmp_path / "research-ledger.jsonl"

    event = record_research_event(
        "candidate_created",
        {
            "candidate_id": "candidate-repair-1",
            "candidate_path": "candidates/candidate-repair-1",
            "proposal_id": "candidate-repair-1",
            "repair_lineage": {
                "original_proposal_id": "proposal-original",
                "original_candidate_id": "candidate-original",
                "motivating_run_id": "run_20260501_120000_abcdef",
                "failure_classification": "candidate_bug",
                "preserves_original_hypothesis": True,
                "preserves_comparison_target": True,
            },
        },
        ledger_path=ledger,
    )

    assert event["repair_lineage"]["original_proposal_id"] == "proposal-original"
    assert read_jsonl(ledger) == [event]
