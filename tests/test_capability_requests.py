import json
from pathlib import Path

import pytest

from ml_autoresearch.capability_requests import CapabilityRequestError, create_capability_request, validate_capability_request_file
from ml_autoresearch.cli import app
from conftest import invoke_typer_cli


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


def test_validate_capability_request_file_accepts_dataset_profile_artifact_request(tmp_path: Path) -> None:
    request_path = write_request(
        tmp_path / "dataset-profile.yaml",
        request_id="capability-gvccs-mask-area-profile",
        capability_type="dataset_profile_artifact",
        blocked_hypothesis="Tiny positive masks may need a recall-oriented architecture change.",
        current_contract_insufficiency="Existing dataset profile artifacts do not summarize positive-mask area by split.",
        expected_research_value="This would show whether missed positives are dominated by small masks before proposing a new Candidate Experiment.",
        safety_reproducibility_risks="The summary must be generated deterministically without exposing raw training images.",
        minimal_harness_change="Generate a durable dataset profile artifact with mask-area histograms for train and validation splits.",
        diagnostic_question="Are positive Contrail Masks concentrated in a small-area tail that explains recent false negatives?",
        expected_research_decision_impact="Decide whether the next Candidate Experiment should prioritize thin-structure recall or a different error mode.",
        scope_split="GVCCS Working Validation Split and training split; aggregate counts only.",
        bounded_computation_artifact_budget="One offline scan producing one YAML/Markdown summary and up to four provenance-linked plots.",
        provenance_requirements="Record dataset version, split definition, generation command, code version, and source mask identifiers or hashes.",
    )

    request = validate_capability_request_file(request_path)

    assert request.capability_type == "dataset_profile_artifact"
    assert request.diagnostic_question.startswith("Are positive Contrail Masks")
    assert request.candidate_authority_requested == "none"


def test_validate_capability_request_file_rejects_malformed_dataset_profile_artifact_request(tmp_path: Path) -> None:
    request_path = write_request(
        tmp_path / "dataset-profile.yaml",
        capability_type="dataset_profile_artifact",
        diagnostic_question="Which camera sources contain small masks?",
        expected_research_decision_impact="Choose the next architecture family.",
        scope_split="Working Validation Split.",
        bounded_computation_artifact_budget="One summary table.",
    )

    with pytest.raises(CapabilityRequestError, match="dataset_profile_artifact requests require provenance_requirements"):
        validate_capability_request_file(request_path)


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

    completed = invoke_typer_cli(
        app,
        [
            "create-capability-request",
            "--request",
            str(request_path),
            "--ledger-path",
            str(ledger),
        ],
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["request"]["candidate_authority_requested"] == "none"
    assert payload["ledger_event"]["request_id"] == "capability-temporal-inputs"
    assert read_jsonl(ledger) == [payload["ledger_event"]]


def test_create_capability_request_cli_rejects_invalid_request_without_event(tmp_path: Path) -> None:
    request_path = write_request(tmp_path / "request.yaml", capability_type="unsafe")
    ledger = tmp_path / "research-ledger.jsonl"

    completed = invoke_typer_cli(
        app,
        [
            "create-capability-request",
            "--request",
            str(request_path),
            "--ledger-path",
            str(ledger),
        ],
    )

    assert completed.returncode == 1
    assert "capability_type" in completed.stderr
    assert not ledger.exists()


def test_create_capability_request_cli_reports_ledger_write_failure_without_traceback(tmp_path: Path) -> None:
    request_path = write_request(tmp_path / "request.yaml")

    completed = invoke_typer_cli(
        app,
        [
            "create-capability-request",
            "--request",
            str(request_path),
            "--ledger-path",
            str(tmp_path),
        ],
    )

    assert completed.returncode == 1
    assert "Traceback" not in completed.stderr
    assert str(tmp_path) in completed.stderr
