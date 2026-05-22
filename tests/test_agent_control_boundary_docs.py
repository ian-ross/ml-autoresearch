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
    assert "Autoresearch Skill Set" in text
    assert "`agent-work/.pi/skills/`" in text
    assert "`agent-work/AGENTS.md`" in text
    assert "path map" in text


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


def test_agent_control_boundary_docs_describe_lightweight_image_dependency_boundary() -> None:
    text = DOC_PATH.read_text()

    assert "Agent Control Boundary image installs the base `ml-autoresearch` package" in text
    assert "`ml-autoresearch-agent` console script" in text
    assert "does not install PyTorch" in text
    assert "NVIDIA/CUDA" in text
    assert "heavy ML/runtime dependencies belong to the outer Harness and Candidate Execution Boundary" in text


def test_agent_control_boundary_docs_define_autonomy_step_operator_workflow() -> None:
    text = DOC_PATH.read_text()

    assert "`ml-autoresearch autonomy-step`" in text
    assert "ADR 0007" in text
    assert "generated `prompt.txt`" in text
    assert "one-primary-handoff rule" in text
    assert "one primary research handoff outcome" in text
    assert "`.INGESTED.json` marker" in text
    assert "`agent-work/autonomy-step-result.json`" in text
    assert "--execute-next-action" in text
    assert "Agent Handoff Ingestion is not Candidate Experiment Run execution" in text
    assert "is not\nPost-Run Evaluation execution" in text


def test_agent_control_boundary_docs_cover_manual_autonomy_step_checklist() -> None:
    text = DOC_PATH.read_text()

    assert "### Manual test checklist" in text
    for case in [
        "Candidate Submission",
        "Research Note",
        "Capability Request",
        "Evaluation Request",
        "Campaign Report",
        "No handoff",
        "Duplicate artifact",
        "Multi-handoff failure",
    ]:
        assert case in text
    assert "next_action: run_candidate" in text
    assert "next_action:\n  run_post_run_evaluation" in text
    assert "fails before copying, ledger updates, index updates, or marker creation" in text


def test_agent_control_boundary_docs_define_root_config_schema() -> None:
    text = DOC_PATH.read_text()

    assert "## `agent-boundary.toml` schema" in text
    assert "[agent_control_boundary]" in text
    for field in ["distro", "image", "allow_egress", "[[data_mounts]]", "name", "path", "target", "readonly"]:
        assert field in text
    assert "direct child of `/data`" in text
    assert "relative to the project root" in text
    assert "image paths are interpreted by pi-fort" in text
    assert "`agent-work/.pi/fort.toml` file" in text
