from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_run_lifecycle_docs_define_fail_fast_and_recoverable_long_run_semantics() -> None:
    lifecycle = (ROOT / "docs" / "run-lifecycle.md").read_text()
    contract = (ROOT / "docs" / "candidate-experiment-contract.md").read_text()
    adr = (ROOT / "docs" / "adr" / "0011-managed-run-execution-and-idempotent-terminalization.md").read_text()

    assert "non_finite_training_state" in lifecycle
    assert "not a Resource Failure" in lifecycle
    assert "never retrains" in lifecycle
    assert "run-status" in lifecycle
    assert "reconcile-run" in lifecycle
    assert "execution.json" in lifecycle
    assert "created before smoke" in lifecycle
    assert "--no-docker-enable-gpu" in lifecycle
    assert "cannot be disabled" in contract
    assert "stable Run identity" in adr
    assert "Caller interruption during smoke or training" in adr
    assert "idempotent" in adr


def test_agent_skills_forbid_relaunch_after_disconnection_and_nonfinite_retry() -> None:
    skills = ROOT / "src" / "ml_autoresearch" / "resources" / "autoresearch-skills"
    observer = (skills / "run-observer" / "SKILL.md").read_text()
    classifier = (skills / "failure-classifier" / "SKILL.md").read_text()
    manager = (skills / "campaign-manager" / "SKILL.md").read_text()

    assert "never submit or launch the Candidate again" in observer
    assert "never a Resource Failure" in classifier
    assert "must not trigger batch-size retry" in classifier
    assert "never relaunch the Candidate" in manager
