import json
import subprocess
import sys
from pathlib import Path


def run_cli(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ml_autoresearch.cli", *args],
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def write_project(root: Path, extra_config: str = "") -> None:
    (root / "CONTEXT.md").write_text("context v1\n")
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
    (root / "research-ledger.jsonl").write_text("")
    (root / "docs").mkdir()
    (root / "agent-boundary.toml").write_text(
        """
[agent_control_boundary]
distro = "debian"
image = "../../containers/ml-autoresearch-agent"
allow_egress = true
""".lstrip()
        + extra_config
    )


def write_fake_agent(path: Path, body: str) -> str:
    path.write_text(
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        + body
    )
    return f"{sys.executable} {path}"


def test_autonomy_step_writes_prompt_invokes_fake_agent_ingests_one_handoff_and_prints_summary(tmp_path: Path):
    write_project(tmp_path)
    fake_command = write_fake_agent(
        tmp_path / "fake_agent.py",
        """
assert sys.argv[1:] == ['-p', '@prompt.txt']
prompt = Path('prompt.txt').read_text()
assert 'Read AGENTS.md first' in prompt
assert 'campaign-manager skill' in prompt
assert 'Multiple primary handoff outcomes are forbidden' in prompt
assert 'exactly one primary research handoff outcome' in prompt
assert Path('AGENTS.md').is_file()
Path('research-notes').mkdir(exist_ok=True)
Path('research-notes/2026-05-22-agent-note.md').write_text('# Agent Note\\n\\n## Summary\\nOne note.\\n\\n## Decision\\nContinue.\\n')
Path('scratch/invocation.json').write_text(json.dumps({'cwd': os.getcwd(), 'reference': Path('../agent-reference/CONTEXT.md').read_text()}))
""",
    )

    completed = run_cli(tmp_path, "autonomy-step", "--project-root", str(tmp_path), "--agent-command", fake_command)

    assert completed.returncode == 0, completed.stderr
    assert "Autonomy Step complete" in completed.stdout
    assert "Status: ingested" in completed.stdout
    assert "Handoff ingestion: ingested" in completed.stdout
    assert not completed.stdout.lstrip().startswith("{")

    invocation = json.loads((tmp_path / "agent-work" / "scratch" / "invocation.json").read_text())
    assert invocation["cwd"] == str(tmp_path / "agent-work")
    assert invocation["reference"] == "context v1\n"
    assert (tmp_path / "research-notes" / "2026-05-22-agent-note.md").is_file()

    result = json.loads((tmp_path / "agent-work" / "autonomy-step-result.json").read_text())
    assert result["status"] == "ingested"
    assert result["agent_command"][-2:] == ["-p", "@prompt.txt"]
    assert result["agent_returncode"] == 0
    assert result["ingestion"]["handoff_type"] == "research_note"


def test_autonomy_step_refreshes_agent_control_boundary_before_invocation(tmp_path: Path):
    write_project(tmp_path)
    run_cli(tmp_path, "prepare-agent-boundary", "--project-root", str(tmp_path))
    (tmp_path / "CONTEXT.md").write_text("context v2\n")
    fake_command = write_fake_agent(
        tmp_path / "fake_agent.py",
        "Path('scratch/reference.txt').write_text(Path('../agent-reference/CONTEXT.md').read_text())\n",
    )

    completed = run_cli(tmp_path, "autonomy-step", "--project-root", str(tmp_path), "--agent-command", fake_command)

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "agent-work" / "scratch" / "reference.txt").read_text() == "context v2\n"
    result = json.loads((tmp_path / "agent-work" / "autonomy-step-result.json").read_text())
    assert result["status"] == "no_handoff"
    assert result["ingestion"]["next_action"] == "stop_for_human"


def test_autonomy_step_nonzero_agent_exit_writes_failure_result_and_does_not_ingest(tmp_path: Path):
    write_project(tmp_path)
    fake_command = write_fake_agent(
        tmp_path / "fake_agent.py",
        """
Path('research-notes').mkdir(exist_ok=True)
Path('research-notes/2026-05-22-untrusted-note.md').write_text('# Should Not Ingest\\n')
sys.exit(7)
""",
    )

    completed = run_cli(tmp_path, "autonomy-step", "--project-root", str(tmp_path), "--agent-command", fake_command)

    assert completed.returncode != 0
    assert "Status: agent_failed" in completed.stdout
    assert "handoff ingestion skipped" in completed.stdout
    assert not (tmp_path / "research-notes" / "2026-05-22-untrusted-note.md").exists()
    result = json.loads((tmp_path / "agent-work" / "autonomy-step-result.json").read_text())
    assert result["status"] == "agent_failed"
    assert result["agent_returncode"] == 7
    assert result["ingestion"] is None


def test_autonomy_step_uses_configured_agent_command_when_cli_option_is_absent(tmp_path: Path):
    fake_path = tmp_path / "fake_agent.py"
    write_project(tmp_path, extra_config=f'\n[autonomy_step]\nagent_command = "{sys.executable} {fake_path}"\n')
    write_fake_agent(fake_path, "Path('scratch/config-agent-used.txt').write_text('yes')\n")

    completed = run_cli(tmp_path, "autonomy-step", "--project-root", str(tmp_path))

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "agent-work" / "scratch" / "config-agent-used.txt").read_text() == "yes"
    result = json.loads((tmp_path / "agent-work" / "autonomy-step-result.json").read_text())
    assert result["agent_command"][:2] == [sys.executable, str(fake_path)]
    assert result["status"] == "no_handoff"
