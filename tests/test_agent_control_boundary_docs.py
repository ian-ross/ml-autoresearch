from pathlib import Path


DOC_PATH = Path("docs/agent-control-boundary.md")


def test_agent_control_boundary_docs_describe_workspace_and_reference_layout() -> None:
    text = DOC_PATH.read_text()

    assert "Agent Workspace is the current writable directory" in text
    for path in [
        "drafts/candidates/",
        "submissions/",
        "research-notes/",
        "capability-requests/",
        "evaluation-requests/",
        "campaign-reports/",
        "scratch/",
    ]:
        assert path in text

    for path in [
        "/reference",
        "/history",
        "/docs",
        "/data",
        "/reference/CONTEXT.md",
        "/reference/EXPERIMENT_INDEX.md",
        "/history/research-ledger.jsonl",
    ]:
        assert path in text

    assert "`submissions/` entries are immutable once created" in text
    assert "Submissions are ingested by the Harness outside the Agent Control Boundary" in text


def test_agent_control_boundary_docs_define_inner_agent_commands_and_data_policy() -> None:
    text = DOC_PATH.read_text()

    assert "Use `ml-autoresearch-agent`, not `ml-autoresearch`, inside the VM." in text
    assert "must not run Candidate Experiments" in text
    for forbidden in [
        "Docker",
        "GPU tools",
        "cluster scheduler commands",
        "direct training/evaluation scripts",
    ]:
        assert forbidden in text

    assert "Read-only `/data` inspection is allowed for hypothesis formation" in text
    assert "Candidate Experiment code must remain data-path agnostic" in text
    assert "Authoritative Results come from the Harness" in text
    assert "protects infrastructure authority" in text
    assert "dataset hiding" in text
