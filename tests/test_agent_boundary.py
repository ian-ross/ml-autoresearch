import subprocess
from pathlib import Path

from ml_autoresearch.cli import app
from conftest import invoke_typer_cli


def run_cli(cwd: Path, *args: str):
    arg_list = list(args)
    if arg_list and arg_list[0] == "prepare-agent-boundary" and "--skip-runtime-image-validation" not in arg_list:
        arg_list.append("--skip-runtime-image-validation")
    return invoke_typer_cli(app, arg_list, cwd=cwd)



def write_fake_research_problem_provider(root: Path) -> None:
    package = root / "fake_research_problem"
    (package / "brief").mkdir(parents=True)
    (package / "profile").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "brief" / "overview.md").write_text("# Tiny overview\n")
    (package / "brief" / "baselines.md").write_text("# Tiny baselines\n")
    (package / "brief" / "undeclared.md").write_text("# Do not expose\n")
    (package / "profile" / "tiny-dataset-profile.json").write_text('{"provenance": {"research_problem_id": "tiny_problem"}}\n')
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
        "        dataset_profile_artifacts=(\n"
        "            {'name': 'tiny_profile', 'role': 'deterministic_test_dataset_profile', 'path': 'fake_research_problem/profile/tiny-dataset-profile.json', 'summary': 'Tiny deterministic profile for Harness tests.', 'split_scope': 'train+validation', 'required': True},\n"
        "        ),\n"
        "    )\n"
    )
    (root / "ml-autoresearch.toml").write_text(
        "[candidate_execution]\n"
        "ledger_path = \"research-ledger.jsonl\"\n"
        "\n"
        "[research_problem]\n"
        "id = \"tiny_problem\"\n"
        f"package_root = \"{root}\"\n"
        "provider_target = \"fake_research_problem.research_problem:build_spec\"\n"
        "expected_contract_version = \"v0\"\n"
        f"data_config = {{ dataset_root = \"{root / 'gvccs-data'}\" }}\n"
    )

def configure_fake_pi_fort(root: Path, monkeypatch, *, returncode: int = 0, stdout: str = "", stderr: str = "") -> list[dict[str, object]]:
    pi_fort = root / "local-pi-fort"
    pi_fort.mkdir()
    monkeypatch.setenv("ML_AUTORESEARCH_PI_FORT", str(pi_fort))
    calls: list[dict[str, object]] = []

    def fake_run(command, *, cwd, capture_output, text, check):
        calls.append(
            {
                "command": command,
                "cwd": cwd,
                "capture_output": capture_output,
                "text": text,
                "check": check,
            }
        )
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr("ml_autoresearch.agent_boundary.subprocess.run", fake_run)
    return calls


def write_project(root: Path, data_mounts: str = "") -> None:
    write_fake_research_problem_provider(root)
    (root / "CONTEXT.md").write_text("context v1\n")
    (root / "EXPERIMENT_INDEX.md").write_text("index v1\n")
    (root / "research-ledger.jsonl").write_text('{"event_type":"proposal_created"}\n')
    (root / "docs").mkdir()
    (root / "docs" / "guide.md").write_text("docs\n")
    (root / "src" / "ml_autoresearch").mkdir(parents=True)
    (root / "src" / "ml_autoresearch" / "__init__.py").write_text("\n")
    config_path = root / "ml-autoresearch.toml"
    config_path.write_text(
        config_path.read_text()
        + "\n"
        + """
[agent_control_boundary]
distro = "debian"
image = "../../containers/ml-autoresearch-agent"
allow_egress = true
""".lstrip()
        + data_mounts
    )


def test_prepare_agent_boundary_generates_workspace_snapshots_skills_and_fort_config(tmp_path: Path, monkeypatch):
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
    pi_install_calls = configure_fake_pi_fort(tmp_path, monkeypatch)

    completed = run_cli(tmp_path, "prepare-agent-boundary")

    assert completed.returncode == 0, completed.stderr
    assert pi_install_calls == [
        {
            "command": ["pi", "install", "-l", str((tmp_path / "local-pi-fort").resolve(strict=True))],
            "cwd": tmp_path / "agent-work",
            "capture_output": True,
            "text": True,
            "check": False,
        }
    ]
    assert (tmp_path / "agent-reference" / "HARNESS_CONTEXT.md").read_text() == "context v1\n"
    assert not (tmp_path / "agent-reference" / "CONTEXT.md").exists()
    assert (tmp_path / "agent-reference" / "EXPERIMENT_INDEX.md").read_text() == "index v1\n"
    assert (tmp_path / "agent-research-problem" / "fake_research_problem" / "brief" / "overview.md").read_text() == "# Tiny overview\n"
    assert (tmp_path / "agent-research-problem" / "fake_research_problem" / "brief" / "baselines.md").read_text() == "# Tiny baselines\n"
    assert (tmp_path / "agent-research-problem" / "fake_research_problem" / "profile" / "tiny-dataset-profile.json").read_text() == '{"provenance": {"research_problem_id": "tiny_problem"}}\n'
    assert (tmp_path / "agent-research-problem" / "RESEARCH_PROBLEM_BRIEF_INDEX.md").is_file()
    assert not (tmp_path / "agent-research-problem" / "fake_research_problem" / "brief" / "undeclared.md").exists()
    assert not (tmp_path / "agent-research-problem" / "fake_research_problem" / "research_problem.py").exists()
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
    assert "Available Dataset Profile Artifacts:" in brief_index
    assert "**tiny_profile** (`deterministic_test_dataset_profile`)" in brief_index
    assert "# Tiny baselines" not in brief_index
    assert "`CONTEXT.md` -> `/reference/HARNESS_CONTEXT.md`" in instructions
    assert "`docs/` -> `/docs/`" in instructions
    assert "`research-notes/` -> `/history/research-notes/` for prior notes" in instructions
    assert "write new draft Research Notes under `research-notes/`" in instructions
    assert "One Autonomy Step means one primary handoff outcome, then stop." in instructions
    assert "Do not\nproduce a second Candidate Submission" in instructions
    assert "Use `ml-autoresearch-agent`, not `ml-autoresearch`" in instructions
    assert "## Dataset profile artifacts" in instructions
    assert "`/research-problem/profile/`" in instructions
    assert "not raw training data or authoritative Run Results" in instructions
    assert "**tiny_profile** (`deterministic_test_dataset_profile`): Tiny deterministic profile for Harness tests. Scope: train+validation. Required." in instructions
    assert "cat /research-problem/fake_research_problem/profile/tiny-dataset-profile.json" in instructions

    agent_candidate_config = (tmp_path / "agent-work" / "ml-autoresearch.toml").read_text()
    assert "# Generated by ml-autoresearch prepare-agent-boundary" in agent_candidate_config
    assert 'id = "tiny_problem"' in agent_candidate_config
    assert 'package_root = "/research-problem"' in agent_candidate_config
    assert 'provider_target = "fake_research_problem.research_problem:build_spec"' in agent_candidate_config
    assert 'expected_contract_version = "v0"' in agent_candidate_config
    assert 'data_config = { dataset_root = "/data/gvccs" }' in agent_candidate_config

    skill_path = tmp_path / "agent-work" / ".pi" / "skills" / "campaign-manager" / "SKILL.md"
    skill_text = skill_path.read_text()
    assert "name: campaign-manager" in skill_text
    assert "# Campaign Manager" in skill_text
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
        'target="/data/gvccs"',
    ]:
        assert target in fort_toml
    assert f'path="{tmp_path / "agent-research-problem"}"' in fort_toml
    assert f'path="{tmp_path}"' not in fort_toml
    assert 'target="/usr/local/lib/python3.12/dist-packages/ml_autoresearch"' not in fort_toml
    assert fort_toml.count('target="/usr/local/lib/python3.12/site-packages/ml_autoresearch"') == 1
    assert f'path="{tmp_path / "src" / "ml_autoresearch"}"' not in fort_toml
    assert 'path="' in fort_toml and 'src/ml_autoresearch' in fort_toml
    assert f'path="{data_root}"' in fort_toml
    assert 'target="/data/gvccs", readonly=true' in fort_toml
    assert not old_dropin.exists()
    assert (tmp_path / "agent-work" / ".pi" / "fort.d" / "README.md").is_file()



def test_prepare_agent_boundary_rejects_colliding_research_problem_snapshot_paths(tmp_path: Path, monkeypatch):
    write_project(tmp_path)
    configure_fake_pi_fort(tmp_path, monkeypatch)
    provider = tmp_path / "fake_research_problem" / "research_problem.py"
    provider.write_text(
        provider.read_text().replace(
            "        dataset_profile_artifacts=(\n"
            "            {'name': 'tiny_profile', 'role': 'deterministic_test_dataset_profile', 'path': 'fake_research_problem/profile/tiny-dataset-profile.json', 'summary': 'Tiny deterministic profile for Harness tests.', 'split_scope': 'train+validation', 'required': True},\n"
            "        ),\n",
            "        dataset_profile_artifacts=(\n"
            "            {'name': 'tiny_profile', 'role': 'deterministic_test_dataset_profile', 'path': 'fake_research_problem/brief/overview.md', 'summary': 'Tiny deterministic profile for Harness tests.', 'split_scope': 'train+validation', 'required': True},\n"
            "        ),\n",
        )
    )

    completed = run_cli(tmp_path, "prepare-agent-boundary")

    assert completed.returncode == 1
    assert "colliding Agent Research Problem Snapshot path" in completed.stderr


def test_prepare_agent_boundary_requires_explicit_research_problem_provider(tmp_path: Path):
    write_project(tmp_path)
    (tmp_path / "ml-autoresearch.toml").write_text(
        """
[candidate_execution]
ledger_path = "research-ledger.jsonl"

[agent_control_boundary]
distro = "debian"
image = "../../containers/ml-autoresearch-agent"
allow_egress = true
""".lstrip()
    )

    completed = run_cli(tmp_path, "prepare-agent-boundary")

    assert completed.returncode == 1
    assert "require an explicit [research_problem] provider" in completed.stderr
    assert "built-in/default Research Problem fallback is not allowed" in completed.stderr

def test_prepare_agent_boundary_default_does_not_expose_raw_dataset_mount_or_path(tmp_path: Path, monkeypatch):
    write_project(tmp_path)
    configure_fake_pi_fort(tmp_path, monkeypatch)

    completed = run_cli(tmp_path, "prepare-agent-boundary")

    assert completed.returncode == 0, completed.stderr
    instructions = (tmp_path / "agent-work" / "AGENTS.md").read_text()
    assert "Full training datasets are not part of the default Agent Control Boundary" in instructions
    assert "`/data/` contains approved read-only Research Problem data mounts when present" not in instructions
    agent_candidate_config = (tmp_path / "agent-work" / "ml-autoresearch.toml").read_text()
    assert "dataset_root" not in agent_candidate_config
    assert str(tmp_path / "gvccs-data") not in agent_candidate_config
    fort_toml = (tmp_path / "agent-work" / ".pi" / "fort.toml").read_text()
    assert 'target="/data/gvccs"' not in fort_toml
    assert str(tmp_path / "gvccs-data") not in fort_toml



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

    (tmp_path / "ml-autoresearch.toml").write_text(
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


def test_prepare_agent_boundary_requires_pi_fort_environment_variable(tmp_path: Path, monkeypatch):
    write_project(tmp_path)
    monkeypatch.delenv("ML_AUTORESEARCH_PI_FORT", raising=False)

    completed = run_cli(tmp_path, "prepare-agent-boundary")

    assert completed.returncode == 1
    assert "ML_AUTORESEARCH_PI_FORT must be set" in completed.stderr


def test_prepare_agent_boundary_rejects_relative_pi_fort_path(tmp_path: Path, monkeypatch):
    write_project(tmp_path)
    monkeypatch.setenv("ML_AUTORESEARCH_PI_FORT", "relative/pi-fort")

    completed = run_cli(tmp_path, "prepare-agent-boundary")

    assert completed.returncode == 1
    assert "ML_AUTORESEARCH_PI_FORT must be an absolute local path" in completed.stderr


def test_prepare_agent_boundary_rejects_missing_pi_fort_path(tmp_path: Path, monkeypatch):
    write_project(tmp_path)
    monkeypatch.setenv("ML_AUTORESEARCH_PI_FORT", str(tmp_path / "missing-pi-fort"))

    completed = run_cli(tmp_path, "prepare-agent-boundary")

    assert completed.returncode == 1
    assert "ML_AUTORESEARCH_PI_FORT path does not exist" in completed.stderr


def test_prepare_agent_boundary_surfaces_failed_pi_fort_install(tmp_path: Path, monkeypatch):
    write_project(tmp_path)
    configure_fake_pi_fort(tmp_path, monkeypatch, returncode=2, stdout="out text", stderr="err text")

    completed = run_cli(tmp_path, "prepare-agent-boundary")

    assert completed.returncode == 1
    assert "failed to install pi-fort into Agent Workspace" in completed.stderr
    assert "command: pi install -l" in completed.stderr
    assert f"cwd: {tmp_path / 'agent-work'}" in completed.stderr
    assert "stdout: out text" in completed.stderr
    assert "stderr: err text" in completed.stderr


def test_prepare_agent_boundary_surfaces_missing_pi_executable(tmp_path: Path, monkeypatch):
    write_project(tmp_path)
    pi_fort = tmp_path / "local-pi-fort"
    pi_fort.mkdir()
    monkeypatch.setenv("ML_AUTORESEARCH_PI_FORT", str(pi_fort))

    def missing_pi(command, *, cwd, capture_output, text, check):
        raise FileNotFoundError("pi")

    monkeypatch.setattr("ml_autoresearch.agent_boundary.subprocess.run", missing_pi)

    completed = run_cli(tmp_path, "prepare-agent-boundary")

    assert completed.returncode == 1
    assert "`pi` executable was not found" in completed.stderr
    assert "command: pi install -l" in completed.stderr
    assert f"cwd: {tmp_path / 'agent-work'}" in completed.stderr


def test_prepare_agent_boundary_exposes_configured_external_runs_root_without_copying_artifacts(tmp_path: Path, monkeypatch):
    write_project(tmp_path)
    external_runs = tmp_path / "scratch-runs"
    (external_runs / "run_external" / "outputs").mkdir(parents=True)
    (external_runs / "run_external" / "run_metadata.json").write_text('{"run_id":"run_external","status":"completed"}\n')
    (external_runs / "run_external" / "outputs" / "large-checkpoint.bin").write_text("do not copy\n")
    (tmp_path / "runs").symlink_to(external_runs, target_is_directory=True)
    config_path = tmp_path / "ml-autoresearch.toml"
    config_path.write_text(
        config_path.read_text().replace(
            'ledger_path = "research-ledger.jsonl"\n',
            f'ledger_path = "research-ledger.jsonl"\nruns_root = "{external_runs}"\n',
        )
    )
    configure_fake_pi_fort(tmp_path, monkeypatch)

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


def test_prepare_agent_boundary_replaces_existing_snapshot_contents_instead_of_appending(tmp_path: Path, monkeypatch):
    write_project(tmp_path)
    configure_fake_pi_fort(tmp_path, monkeypatch)
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


def test_prepare_agent_boundary_escapes_control_characters_in_generated_fort_toml(tmp_path: Path, monkeypatch):
    write_project(tmp_path)
    config_path = tmp_path / "ml-autoresearch.toml"
    config_path.write_text(
        config_path.read_text().replace(
            '''[agent_control_boundary]
distro = "debian"
image = "../../containers/ml-autoresearch-agent"
allow_egress = true
''',
            '''[agent_control_boundary]
distro = "debian\\nedge"
image = "agent\\\"studio\\nedge"
allow_egress = true
''',
        )
    )
    configure_fake_pi_fort(tmp_path, monkeypatch)
    completed = run_cli(tmp_path, "prepare-agent-boundary")

    assert completed.returncode == 0, completed.stderr
    fort_toml = (tmp_path / "agent-work" / ".pi" / "fort.toml").read_text()
    assert r'distro = "debian\nedge"' in fort_toml
    assert r'image = "agent\"studio\nedge"' in fort_toml
