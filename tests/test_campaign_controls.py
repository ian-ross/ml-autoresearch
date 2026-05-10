import json
import subprocess
import sys
from pathlib import Path

import pytest

from ml_autoresearch.campaign_controls import (
    PAUSE_CONDITIONS,
    CampaignControlError,
    record_campaign_pause,
    record_campaign_report_written,
)
from ml_autoresearch.research_ledger import ResearchLedgerError, record_research_event


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_campaign_report_written_event_persists_valid_report_path(tmp_path: Path) -> None:
    ledger = tmp_path / "research-ledger.jsonl"
    report = tmp_path / "campaign-reports" / "2026-05-10-status.md"
    report.parent.mkdir()
    report.write_text("# Campaign Report\n")

    result = record_campaign_report_written(report, ledger_path=ledger)

    assert result["ledger_event"]["event_type"] == "campaign_report_written"
    assert result["ledger_event"]["report_path"] == str(report)
    assert read_jsonl(ledger) == [result["ledger_event"]]


def test_campaign_pause_event_persists_valid_condition_and_optional_report_path(tmp_path: Path) -> None:
    ledger = tmp_path / "research-ledger.jsonl"
    report = tmp_path / "campaign-reports" / "2026-05-10-status.md"

    result = record_campaign_pause("scheduled_check_in", report_path=report, ledger_path=ledger)

    event = result["ledger_event"]
    assert event["event_type"] == "campaign_paused"
    assert event["reason"] == "scheduled_check_in"
    assert event["report_path"] == str(report)
    assert read_jsonl(ledger) == [event]


def test_campaign_pause_rejects_unapproved_condition_without_corrupting_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "research-ledger.jsonl"
    valid = record_campaign_pause("budget_exhausted", ledger_path=ledger)["ledger_event"]

    with pytest.raises(CampaignControlError, match="unsupported Campaign Pause Condition"):
        record_campaign_pause("waiting for review", ledger_path=ledger)
    with pytest.raises(ResearchLedgerError, match="reason"):
        record_research_event("campaign_paused", {"reason": "waiting for review"}, ledger_path=ledger)

    assert read_jsonl(ledger) == [valid]


def test_pause_condition_vocabulary_covers_required_conditions() -> None:
    assert set(PAUSE_CONDITIONS) >= {
        "budget_exhausted",
        "repeated_failures",
        "repeated_resource_failures",
        "stalled_research_progress",
        "too_many_pending_capability_requests",
        "storage_risk",
        "scheduled_check_in",
    }


def test_campaign_control_cli_records_report_and_pause_events(tmp_path: Path) -> None:
    ledger = tmp_path / "research-ledger.jsonl"
    report = tmp_path / "campaign-reports" / "status.md"
    report.parent.mkdir()
    report.write_text("# Campaign Report\n")

    report_completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_autoresearch.cli",
            "record-campaign-report",
            "--report-path",
            str(report),
            "--ledger-path",
            str(ledger),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert report_completed.returncode == 0, report_completed.stderr
    report_payload = json.loads(report_completed.stdout)

    pause_completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_autoresearch.cli",
            "pause-campaign",
            "--reason",
            "storage_risk",
            "--report-path",
            str(report),
            "--ledger-path",
            str(ledger),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert pause_completed.returncode == 0, pause_completed.stderr
    pause_payload = json.loads(pause_completed.stdout)

    assert read_jsonl(ledger) == [report_payload["ledger_event"], pause_payload["ledger_event"]]
    assert pause_payload["ledger_event"]["reason"] == "storage_risk"


def test_campaign_report_format_doc_covers_required_summary_sections() -> None:
    text = Path("docs/campaign-report-format.md").read_text()

    for required in [
        "current best Result",
        "recent Runs",
        "failures",
        "pending Capability Requests",
        "budget use",
        "next hypothesis",
    ]:
        assert required in text
    for condition in PAUSE_CONDITIONS:
        assert condition in text
