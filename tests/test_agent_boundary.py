from pathlib import Path

from ml_autoresearch.cli import app
from conftest import invoke_typer_cli


def run_cli(cwd: Path, *args: str):
    return invoke_typer_cli(app, args, cwd=cwd)



def write_fake_research_problem_provider(root: Path) -> None:
    package = root / "fake_research_problem"
    (package / "brief").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "brief" / "overview.md").write_text("# Tiny overview\n")
    (package / "brief" / "baselines.md").write_text("# Tiny baselines\n")
    (package / "research_problem.py").write_text(
        "from ml_autoresearch.research_problems import ResearchProblemSpec\n"
        "class Adapter:\n"
        "    def validate_data_root(self, data_config): pass\n"
        "    def dataset_metadata(self, data_config): return {'kind': 'tiny_problem'}\n"
        "def build_spec(data_config=None):\n"
        "    return ResearchProblemSpec(\n"
        "        id='tiny_problem', version='test-spec-v0', contract_version='v0',\n"
        "        input_modes=('single_frame_rgb',), input_specs={'single_frame_rgb': {'mode': 'single_frame_rgb', 'shape': [3, 16, 16]}},\n"
        "        output_forms=('binary_mask', 'mask_logits'), output_specs={'binary_mask': {'form': 'binary_mask', 'shape': [1, 16, 16]}, 'mask_logits': {'form': 'mask_logits', 'shape': [1, 16, 16]}},\n"
        "        losses=('bce', 'dice_bce', 'bce_dice'), optimizers=('adamw',), sampling_policies=('sequential',),\n"
        "        augmentation_policies=('none',), primary_metric='val/dice',\n"
        "        training_adapter=Adapter(),\n"
        "        brief_documents=(\n"
        "            {'name': 'overview', 'role': 'problem_overview', 'path': 'fake_research_problem/brief/overview.md', 'summary': 'Tiny problem overview.'},\n"
        "            {'name': 'baselines', 'role': 'baseline_description', 'path': 'fake_research_problem/brief/baselines.md', 'summary': 'Tiny baseline notes.', 'required': True},\n"
        "        ),\n"
        "    )\n"
    )
    (root / "candidate-execution.toml").write_text(
        "[research_problem]\n"
        "id = \"tiny_problem\"\n"
        f"package_root = \"{root}\"\n"
        "provider_target = \"fake_research_problem.research_problem:build_spec\"\n"
        "expected_contract_version = \"v0\"\n"
    )

def write_project(root: Path, data_mounts: str = "") -> None:
    write_fake_research_problem_provider(root)
    (root / "CONTEXT.md").write_text("context v1\n")
    (root / "EXPERIMENT_INDEX.md").write_text("index v1\n")
    (root / "research-ledger.jsonl").write_text('{"event_type":"proposal_created"}\n')
    (root / "docs").mkdir()
    (root / "docs" / "guide.md").write_text("docs\n")
    (root / "src" / "ml_autoresearch").mkdir(parents=True)
    (root / "src" / "ml_autoresearch" / "__init__.py").write_text("\n")
    (root / "agent-boundary.toml").write_text(
        """
[agent_control_boundary]
distro = "debian"
image = "../../containers/ml-autoresearch-agent"
allow_egress = true
""".lstrip()
        + data_mounts
    )


def test_prepare_agent_boundary_generates_workspace_snapshots_skills_and_fort_config(tmp_path: Path):
    data_root = tmp_path / "gvccs-data"
    data_root.mkdir()
    write_project(
        tmp_path,
        f'''
[[data_mounts]]
name = "gvccs"
path = "{data_root}"
''',
    )
    (tmp_path / "runs" / "run_123" / "outputs" / "evaluations" / "eval_abc").mkdir(parents=True)
    (tmp_path / "runs" / "run_123" / "outputs" / "evaluations" / "eval_abc" / "summary.json").write_text('{"ok": true}\n')
    (tmp_path / "candidates" / "candidate_123").mkdir(parents=True)
    (tmp_path / "candidates" / "candidate_123" / "manifest.yaml").write_text("name: candidate_123\n")
    (tmp_path / "research-notes").mkdir()
    (tmp_path / "research-notes" / "note.md").write_text("# Note\n")
    (tmp_path / "docs" / "autoresearch-skills" / "campaign-manager").mkdir(parents=True)
    (tmp_path / "docs" / "autoresearch-skills" / "campaign-manager" / "SKILL.md").write_text("# Campaign Manager\n")

    preserved = tmp_path / "agent-work" / "scratch" / "keep.txt"
    preserved.parent.mkdir(parents=True)
    preserved.write_text("do not delete\n")
    settings = tmp_path / "agent-work" / ".pi" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text("{}\n")
    stale_skill_file = tmp_path / "agent-work" / ".pi" / "skills" / "campaign-manager" / "stale.txt"
    stale_skill_file.parent.mkdir(parents=True)
    stale_skill_file.write_text("stale\n")
    unrelated_skill_file = tmp_path / "agent-work" / ".pi" / "skills" / "local-helper" / "SKILL.md"
    unrelated_skill_file.parent.mkdir(parents=True)
    unrelated_skill_file.write_text("# Local Helper\n")
    old_dropin = tmp_path / "agent-work" / ".pi" / "fort.d" / "old.toml"
    old_dropin.parent.mkdir(parents=True)
    old_dropin.write_text("old\n")

    completed = run_cli(tmp_path, "prepare-agent-boundary")

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "agent-reference" / "CONTEXT.md").read_text() == "context v1\n"
    assert (tmp_path / "agent-reference" / "EXPERIMENT_INDEX.md").read_text() == "index v1\n"
    assert (tmp_path / "agent-history" / "research-ledger.jsonl").read_text() == '{"event_type":"proposal_created"}\n'
    assert (tmp_path / "agent-history" / "runs" / "run_123" / "outputs" / "evaluations" / "eval_abc" / "summary.json").read_text() == '{"ok": true}\n'
    assert (tmp_path / "agent-history" / "candidates" / "candidate_123" / "manifest.yaml").read_text() == "name: candidate_123\n"
    assert (tmp_path / "agent-history" / "research-notes" / "note.md").read_text() == "# Note\n"
    for relative in [
        "drafts/candidates",
        "submissions",
        "research-notes",
        "capability-requests",
        "evaluation-requests",
        "campaign-reports",
        "scratch",
    ]:
        assert (tmp_path / "agent-work" / relative).is_dir()
    assert preserved.read_text() == "do not delete\n"
    assert settings.read_text() == "{}\n"

    instructions = (tmp_path / "agent-work" / "AGENTS.md").read_text()
    assert "Agent Control Boundary path map" in instructions
    assert "## Active Research Problem Brief" in instructions
    assert "Active Research Problem: `tiny_problem`" in instructions
    assert "**overview** (`problem_overview`): Tiny problem overview." in instructions
    assert "cat /research-problem/fake_research_problem/brief/overview.md" in instructions
    assert "# Tiny overview" not in instructions
    brief_index = (tmp_path / "agent-work" / "RESEARCH_PROBLEM_BRIEF_INDEX.md").read_text()
    assert "# Research Problem Brief index" in brief_index
    assert "**baselines** (`baseline_description`): Tiny baseline notes. Required." in brief_index
    assert "# Tiny baselines" not in brief_index
    assert "`CONTEXT.md` -> `/reference/CONTEXT.md`" in instructions
    assert "`docs/` -> `/docs/`" in instructions
    assert "`research-notes/` -> `/history/research-notes/` for prior notes" in instructions
    assert "write new draft Research Notes under `research-notes/`" in instructions
    assert "One Autonomy Step means one primary handoff outcome, then stop." in instructions
    assert "Do not\nproduce a second Candidate Submission" in instructions
    assert "Use `ml-autoresearch-agent`, not `ml-autoresearch`" in instructions

    skill_path = tmp_path / "agent-work" / ".pi" / "skills" / "campaign-manager" / "SKILL.md"
    assert skill_path.read_text() == "# Campaign Manager\n"
    assert not stale_skill_file.exists()
    assert unrelated_skill_file.read_text() == "# Local Helper\n"

    fort_toml = (tmp_path / "agent-work" / ".pi" / "fort.toml").read_text()
    assert "# Generated by ml-autoresearch prepare-agent-boundary" in fort_toml
    for target in [
        'target="/reference"',
        'target="/history"',
        'target="/history/candidates"',
        'target="/history/runs"',
        'target="/history/research-notes"',
        'target="/docs"',
        'target="/research-problem"',
        'target="/usr/local/lib/python3.12/site-packages/ml_autoresearch"',
        'target="/usr/local/lib/python3.12/dist-packages/ml_autoresearch"',
        'target="/data/gvccs"',
    ]:
        assert target in fort_toml
    assert f'path="{tmp_path / "src" / "ml_autoresearch"}"' in fort_toml
    assert f'path="{data_root}"' in fort_toml
    assert 'target="/data/gvccs", readonly=true' in fort_toml
    assert not old_dropin.exists()
    assert (tmp_path / "agent-work" / ".pi" / "fort.d" / "README.md").is_file()



def test_prepare_agent_boundary_requires_explicit_research_problem_provider(tmp_path: Path):
    write_project(tmp_path)
    (tmp_path / "candidate-execution.toml").unlink()

    completed = run_cli(tmp_path, "prepare-agent-boundary")

    assert completed.returncode == 1
    assert "require an explicit [research_problem] provider" in completed.stderr
    assert "built-in/default Research Problem fallback is not allowed" in completed.stderr

def test_prepare_agent_boundary_fails_for_missing_data_mount_path(tmp_path: Path):
    write_project(
        tmp_path,
        '''
[[data_mounts]]
name = "missing"
path = "missing-data"
''',
    )

    completed = run_cli(tmp_path, "prepare-agent-boundary")

    assert completed.returncode == 1
    assert "data mount path does not exist" in completed.stderr


def test_prepare_agent_boundary_rejects_invalid_and_overlapping_data_targets(tmp_path: Path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    write_project(
        tmp_path,
        f'''
[[data_mounts]]
name = "one"
path = "{data_root}"
target = "/data/nested/one"

[[data_mounts]]
name = "two"
path = "{data_root}"
target = "/data/one"
''',
    )

    completed = run_cli(tmp_path, "prepare-agent-boundary")

    assert completed.returncode == 1
    assert "direct children of /data" in completed.stderr

    (tmp_path / "agent-boundary.toml").write_text(
        f'''
[agent_control_boundary]
distro = "debian"
image = "../../containers/ml-autoresearch-agent"
allow_egress = true

[[data_mounts]]
name = "one"
path = "{data_root}"
target = "/data/shared"

[[data_mounts]]
name = "two"
path = "{data_root}"
target = "/data/shared"
'''.lstrip()
    )
    completed = run_cli(tmp_path, "prepare-agent-boundary")

    assert completed.returncode == 1
    assert "overlapping data mount target" in completed.stderr


def test_prepare_agent_boundary_exposes_configured_external_runs_root_without_copying_artifacts(tmp_path: Path):
    write_project(tmp_path)
    external_runs = tmp_path / "scratch-runs"
    (external_runs / "run_external" / "outputs").mkdir(parents=True)
    (external_runs / "run_external" / "run_metadata.json").write_text('{"run_id":"run_external","status":"completed"}\n')
    (external_runs / "run_external" / "outputs" / "large-checkpoint.bin").write_text("do not copy\n")
    (tmp_path / "runs").symlink_to(external_runs, target_is_directory=True)
    (tmp_path / "candidate-execution.toml").write_text(
        (tmp_path / "candidate-execution.toml").read_text()
        + f'\n[candidate_execution]\nruns_root = "{external_runs}"\n'
    )

    completed = run_cli(tmp_path, "prepare-agent-boundary")

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "agent-history" / "runs").is_dir()
    assert not (tmp_path / "agent-history" / "runs").is_symlink()
    assert not (tmp_path / "agent-history" / "runs" / "run_external").exists()
    fort_toml = (tmp_path / "agent-work" / ".pi" / "fort.toml").read_text()
    assert f'path="{external_runs}"' in fort_toml
    assert 'target="/history/runs", readonly=true' in fort_toml
    assert f'target="{tmp_path / "runs"}"' not in fort_toml
    assert f'target="{tmp_path / "agent-history" / "runs"}"' not in fort_toml


def test_prepare_agent_boundary_replaces_existing_snapshot_contents_instead_of_appending(tmp_path: Path):
    write_project(tmp_path)
    run_cli(tmp_path, "prepare-agent-boundary")

    stale_reference_root = tmp_path / "agent-reference"
    stale_history_root = tmp_path / "agent-history"
    (stale_reference_root / "stale.txt").write_text("stale")
    (stale_reference_root / "old-dir").mkdir()
    (stale_reference_root / "old-dir" / "keep.txt").write_text("keep")
    (stale_history_root / "stale.txt").write_text("stale")
    (stale_history_root / "old-dir").mkdir()
    (stale_history_root / "old-dir" / "keep.txt").write_text("keep")
    (tmp_path / "runs" / "run_new").mkdir(parents=True)
    (tmp_path / "runs" / "run_new" / "run_metadata.json").write_text("{}\n")

    completed = run_cli(tmp_path, "prepare-agent-boundary")

    assert completed.returncode == 0, completed.stderr
    assert not (stale_reference_root / "stale.txt").exists()
    assert not (stale_reference_root / "old-dir").exists()
    assert not (stale_history_root / "stale.txt").exists()
    assert not (stale_history_root / "old-dir").exists()
    assert (tmp_path / "agent-history" / "runs" / "run_new" / "run_metadata.json").read_text() == "{}\n"
    for relative in ["candidates", "runs", "research-notes"]:
        assert (tmp_path / "agent-history" / relative).is_dir()


def test_prepare_agent_boundary_escapes_control_characters_in_generated_fort_toml(tmp_path: Path):
    write_project(tmp_path)
    (tmp_path / "agent-boundary.toml").write_text(
        '''
[agent_control_boundary]
distro = "debian\\nedge"
image = "agent\\\"studio\\nedge"
allow_egress = true
'''.lstrip()
    )
    completed = run_cli(tmp_path, "prepare-agent-boundary")

    assert completed.returncode == 0, completed.stderr
    fort_toml = (tmp_path / "agent-work" / ".pi" / "fort.toml").read_text()
    assert r'distro = "debian\nedge"' in fort_toml
    assert r'image = "agent\"studio\nedge"' in fort_toml
