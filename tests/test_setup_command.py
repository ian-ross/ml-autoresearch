from __future__ import annotations

import tomllib
from pathlib import Path

from conftest import invoke_typer_cli
from ml_autoresearch.cli import app
from ml_autoresearch.research_problems import ResearchProblemProviderConfig, load_research_problem_provider


def _invoke(args, cwd: Path):
    return invoke_typer_cli(app, ["setup", *args], cwd=cwd)


def test_setup_non_interactive_creates_workspace_from_pyproject_defaults(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "my-research-problem"\n')

    completed = _invoke(["--non-interactive", "--problem-id", "my_problem"], cwd=tmp_path)

    assert completed.returncode == 0, completed.stderr
    config = tomllib.loads((tmp_path / "ml-autoresearch.toml").read_text())
    assert config["research_problem"] == {
        "id": "my_problem",
        "package_root": ".",
        "provider_target": "my_research_problem.research_problem:build_spec",
        "expected_contract_version": "v0",
        "data_config": {"dataset_root": "data"},
    }
    assert config["candidate_execution"]["runs_root"] == "runs"
    assert config["candidate_execution"]["ledger_path"] == "research-ledger.jsonl"

    for path in [
        "CONTEXT.md",
        "AGENTS.md",
        "EXPERIMENT_INDEX.md",
        "research-ledger.jsonl",
        "brief/overview.md",
        "profile/dataset-profile.md",
        "my_research_problem/__init__.py",
        "my_research_problem/research_problem.py",
        ".ml-autoresearch/.gitkeep",
        "candidates/.gitkeep",
        "research-notes/.gitkeep",
        "capability-requests/.gitkeep",
        "evaluation-requests/.gitkeep",
        "campaign-reports/.gitkeep",
        "agent-work/submissions/.gitkeep",
    ]:
        assert (tmp_path / path).exists(), path

    loaded = load_research_problem_provider(
        ResearchProblemProviderConfig(
            id="my_problem",
            package_root=tmp_path,
            provider_target="my_research_problem.research_problem:build_spec",
            expected_contract_version="v0",
            data_config={"dataset_root": "data"},
        )
    )
    assert loaded.spec.id == "my_problem"
    assert loaded.spec.output_forms == ("mask_logits",)
    assert loaded.spec.primary_metric == "val/dice"


def test_setup_creates_selected_runs_root_and_gitignore(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "demo"\n')

    completed = _invoke(
        ["--non-interactive", "--problem-id", "demo", "--runs-root", "local-runs"],
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "local-runs").is_dir()
    assert "Run artifacts are normally not committed" in completed.stdout
    gitignore = (tmp_path / ".gitignore").read_text()
    assert "local-runs/" in gitignore
    assert ".ml-autoresearch/" in gitignore


def test_setup_is_idempotent_and_does_not_reset_research_memory(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "demo"\n')
    first = _invoke(["--non-interactive", "--problem-id", "demo"], cwd=tmp_path)
    assert first.returncode == 0, first.stderr
    (tmp_path / "research-ledger.jsonl").write_text('{"event":"keep"}\n')

    second = _invoke(["--non-interactive", "--problem-id", "demo"], cwd=tmp_path)

    assert second.returncode == 0, second.stderr
    assert (tmp_path / "research-ledger.jsonl").read_text() == '{"event":"keep"}\n'


def test_setup_explicit_reset_truncates_research_memory(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "demo"\n')
    first = _invoke(["--non-interactive", "--problem-id", "demo"], cwd=tmp_path)
    assert first.returncode == 0, first.stderr
    (tmp_path / "research-ledger.jsonl").write_text('{"event":"discard"}\n')

    reset = _invoke(["--non-interactive", "--problem-id", "demo", "--reset-research-memory"], cwd=tmp_path)

    assert reset.returncode == 0, reset.stderr
    assert (tmp_path / "research-ledger.jsonl").read_text() == ""



def test_setup_refuses_unsafe_existing_file_by_default(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "demo"\n')
    (tmp_path / "ml-autoresearch.toml").write_text("[unexpected]\nvalue = true\n")

    completed = _invoke(["--non-interactive", "--problem-id", "demo"], cwd=tmp_path)

    assert completed.returncode == 1
    assert "refusing to overwrite" in completed.stderr
    assert (tmp_path / "ml-autoresearch.toml").read_text() == "[unexpected]\nvalue = true\n"


def test_setup_unsupported_problem_type_generates_todo_provider(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "demo"\n')

    completed = _invoke(
        ["--non-interactive", "--problem-id", "demo", "--problem-type", "classification"],
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    provider = (tmp_path / "demo" / "research_problem.py").read_text()
    assert "Unsupported Research Problem type" in provider
    assert "TODO" in provider
