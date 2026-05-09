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
