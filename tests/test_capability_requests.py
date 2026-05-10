import json
import subprocess
import sys
from pathlib import Path

import pytest

from ml_autoresearch.capability_requests import CapabilityRequestError, create_capability_request, validate_capability_request_file


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_request(path: Path, **overrides: object) -> Path:
    data = {
        "request_id": "capability-temporal-inputs",
        "capability_type": "contract_surface",
        "blocked_hypothesis": "Temporal context could improve thin contrail segmentation.",
        "current_contract_insufficiency": "The current Candidate Experiment Contract only exposes single-frame RGB inputs.",
        "expected_research_value": "This would test whether adjacent frames reduce false negatives.",
        "safety_reproducibility_risks": "Temporal grouping must remain Harness-owned and deterministic.",
        "minimal_harness_change": "Add an allowlisted centered temporal clip Input Mode.",
        "candidate_authority_requested": "none",
        "example_follow_up_experiments": ["Compare single-frame RGB against centered temporal RGB clip."],
        "priority": "medium",
    }
    data.update(overrides)
    import yaml

    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def test_validate_capability_request_file_accepts_valid_request(tmp_path: Path) -> None:
    request_path = write_request(tmp_path / "request.yaml")

    request = validate_capability_request_file(request_path)

    assert request.request_id == "capability-temporal-inputs"
    assert request.capability_type == "contract_surface"
    assert request.candidate_authority_requested == "none"
    assert request.example_follow_up_experiments == ["Compare single-frame RGB against centered temporal RGB clip."]


@pytest.mark.parametrize("field", [
    "capability_type",
    "blocked_hypothesis",
    "current_contract_insufficiency",
    "expected_research_value",
    "safety_reproducibility_risks",
    "minimal_harness_change",
    "candidate_authority_requested",
    "example_follow_up_experiments",
    "priority",
])
def test_validate_capability_request_file_rejects_missing_required_fields(tmp_path: Path, field: str) -> None:
    request_path = write_request(tmp_path / "request.yaml")
    import yaml

    data = yaml.safe_load(request_path.read_text())
    data.pop(field)
    request_path.write_text(yaml.safe_dump(data))

    with pytest.raises(CapabilityRequestError, match=field):
        validate_capability_request_file(request_path)


def test_validate_capability_request_file_rejects_invalid_priority_and_type(tmp_path: Path) -> None:
    with pytest.raises(CapabilityRequestError, match="priority"):
        validate_capability_request_file(write_request(tmp_path / "bad-priority.yaml", priority="someday"))

    with pytest.raises(CapabilityRequestError, match="capability_type"):
        validate_capability_request_file(write_request(tmp_path / "bad-type.yaml", capability_type="candidate_filesystem_access"))


def test_create_capability_request_records_ledger_event(tmp_path: Path) -> None:
    request_path = write_request(tmp_path / "request.yaml")
    ledger = tmp_path / "research-ledger.jsonl"

    result = create_capability_request(request_path, ledger_path=ledger)

    rows = read_jsonl(ledger)
    assert rows == [result["ledger_event"]]
    assert result["request"]["request_id"] == "capability-temporal-inputs"
    assert result["ledger_event"]["event_type"] == "capability_request_created"
    assert result["ledger_event"]["request_id"] == "capability-temporal-inputs"
    assert result["ledger_event"]["request_path"] == str(request_path)


def test_failed_capability_request_validation_does_not_append_ledger_event(tmp_path: Path) -> None:
    request_path = write_request(tmp_path / "request.yaml", priority="invalid")
    ledger = tmp_path / "research-ledger.jsonl"

    with pytest.raises(CapabilityRequestError):
        create_capability_request(request_path, ledger_path=ledger)

    assert not ledger.exists()


def test_create_capability_request_cli_validates_and_records_event(tmp_path: Path) -> None:
    request_path = write_request(tmp_path / "request.yaml")
    ledger = tmp_path / "research-ledger.jsonl"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_autoresearch.cli",
            "create-capability-request",
            "--request",
            str(request_path),
            "--ledger-path",
            str(ledger),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["request"]["candidate_authority_requested"] == "none"
    assert payload["ledger_event"]["request_id"] == "capability-temporal-inputs"
    assert read_jsonl(ledger) == [payload["ledger_event"]]


def test_create_capability_request_cli_rejects_invalid_request_without_event(tmp_path: Path) -> None:
    request_path = write_request(tmp_path / "request.yaml", capability_type="unsafe")
    ledger = tmp_path / "research-ledger.jsonl"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_autoresearch.cli",
            "create-capability-request",
            "--request",
            str(request_path),
            "--ledger-path",
            str(ledger),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 1
    assert "capability_type" in completed.stderr
    assert not ledger.exists()


def test_create_capability_request_cli_reports_ledger_write_failure_without_traceback(tmp_path: Path) -> None:
    request_path = write_request(tmp_path / "request.yaml")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_autoresearch.cli",
            "create-capability-request",
            "--request",
            str(request_path),
            "--ledger-path",
            str(tmp_path),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 1
    assert "Traceback" not in completed.stderr
    assert str(tmp_path) in completed.stderr
