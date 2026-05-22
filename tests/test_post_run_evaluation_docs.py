from pathlib import Path


def test_evaluation_request_docs_distinguish_request_modes_from_whole_validation_analysis() -> None:
    text = Path("docs/evaluation-request-format.md").read_text()

    assert "Whole-Validation Failure Analysis" in text
    assert "umbrella diagnostic" in text
    assert "`evaluate-run`" in text
    assert "manual/operator convenience command" in text
    assert "`threshold_sweep`" in text
    assert "`failure_bucket_review`" in text
    assert "select bounded parts of Whole-Validation Failure Analysis" in text


def test_run_lifecycle_docs_define_whole_validation_failure_analysis_outputs() -> None:
    text = Path("docs/run-lifecycle.md").read_text()

    assert "Whole-Validation Failure Analysis" in text
    assert "full Working Validation Split" in text
    assert "aggregate_metrics.json" in text
    assert "per_sample_metrics.jsonl" in text
    assert "threshold_sweep.json" in text
    assert "diagnostic_samples/" in text
    assert "not a new Run" in text
