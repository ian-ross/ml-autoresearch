from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path

import pytest

from ml_autoresearch.research_problems import (
    ResearchProblemProviderConfig,
    ResearchProblemProviderLoadError,
    ResearchProblemSpecRegistry,
    load_research_problem_provider,
)


def _write_fake_package(root):
    package = root / "tiny_problem"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (root / ".gitignore").write_text("__pycache__/\n*.pyc\n")
    (package / "research_problem.py").write_text(
        "from ml_autoresearch.research_problems import ResearchProblemSpec\n"
        "\n"
        "def build_spec(data_config=None):\n"
        "    return ResearchProblemSpec(\n"
        "        id='tiny_problem',\n"
        "        version='test-spec-v0',\n"
        "        contract_version='v0',\n"
        "        input_modes=('tiny_rgb',),\n"
        "        input_specs={'tiny_rgb': {'mode': 'tiny_rgb', 'shape': [3, 8, 8]}},\n"
        "        output_forms=('tiny_mask_logits',),\n"
        "        output_specs={'tiny_mask_logits': {'form': 'tiny_mask_logits', 'shape': [1, 8, 8]}},\n"
        "        losses=('tiny_loss',),\n"
        "        optimizers=('sgd',),\n"
        "        sampling_policies=('sequential',),\n"
        "        augmentation_policies=('none',),\n"
        "        primary_metric='val/tiny_score',\n"
        "    )\n"
        "\n"
        "def build_spec_with_brief_documents(data_config=None):\n"
        "    return {**build_spec(data_config).model_dump(), 'brief_documents': [\n"
        "        {'name': 'overview', 'role': 'problem_overview', 'path': 'brief/overview.md', 'summary': 'Tiny overview'},\n"
        "        {'name': 'baselines', 'role': 'baseline_description', 'path': 'brief/baselines.md', 'required': True},\n"
        "        {'name': 'future_notes', 'role': 'modeling_suggestions', 'path': 'brief/future.md'},\n"
        "    ]}\n"
        "\n"
        "def build_spec_with_missing_required_brief(data_config=None):\n"
        "    return {**build_spec(data_config).model_dump(), 'brief_documents': [\n"
        "        {'name': 'required_missing', 'role': 'problem_overview', 'path': 'brief/missing.md', 'required': True},\n"
        "    ]}\n"
        "\n"
        "def build_spec_with_escaping_brief(data_config=None):\n"
        "    return {**build_spec(data_config).model_dump(), 'brief_documents': [{'name': 'bad', 'role': 'problem_overview', 'path': '../outside.md'}]}\n"
        "\n"
        "def build_spec_with_absolute_brief(data_config=None):\n"
        "    return {**build_spec(data_config).model_dump(), 'brief_documents': [{'name': 'bad', 'role': 'problem_overview', 'path': '/tmp/outside.md'}]}\n"
        "\n"
        "def build_spec_with_backslash_brief(data_config=None):\n"
        "    return {**build_spec(data_config).model_dump(), 'brief_documents': [{'name': 'bad', 'role': 'problem_overview', 'path': 'brief\\\\overview.md'}]}\n"
        "\n"
        "def build_dict_spec(data_config=None):\n"
        "    return build_spec(data_config).model_dump()\n"
        "\n"
        "def build_wrong_id(data_config=None):\n"
        "    spec = build_spec(data_config).model_copy(update={'id': 'other_problem'})\n"
        "    return spec\n"
        "\n"
        "def build_wrong_contract(data_config=None):\n"
        "    return build_spec(data_config).model_copy(update={'contract_version': 'v9'})\n"
        "\n"
        "def build_invalid_shape(data_config=None):\n"
        "    return {'id': 'tiny_problem', 'version': 'test-spec-v0', 'contract_version': 'v0'}\n"
        "\n"
        "NOT_CALLABLE = object()\n"
    )


def _config(root, target="tiny_problem.research_problem:build_spec"):
    return ResearchProblemProviderConfig(
        id="tiny_problem",
        package_root=root,
        provider_target=target,
        expected_contract_version="v0",
        data_config={"dataset_root": "/trusted/data"},
    )


def test_loads_filesystem_research_problem_provider_and_registers_spec(tmp_path) -> None:
    _write_fake_package(tmp_path)
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, stdout=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "fake problem"], cwd=tmp_path, check=True, stdout=subprocess.PIPE)
    registry = ResearchProblemSpecRegistry(active_id="tiny_problem")

    loaded = load_research_problem_provider(_config(tmp_path), registry=registry)

    assert loaded.spec.id == "tiny_problem"
    assert registry.get("tiny_problem") == loaded.spec
    assert registry.get_provenance("tiny_problem") == loaded.provenance
    assert loaded.provenance.provider_target == "tiny_problem.research_problem:build_spec"
    assert loaded.provenance.resolved_package_root == tmp_path.resolve()
    assert loaded.provenance.git is not None
    assert loaded.provenance.git["commit"]
    assert loaded.provenance.git["dirty"] is False
    assert loaded.run_metadata() == {
        "id": "tiny_problem",
        "version": "test-spec-v0",
        "contract_version": "v0",
        "provider": {
            "target": "tiny_problem.research_problem:build_spec",
            "resolved_package_root": str(tmp_path.resolve()),
            "git": loaded.provenance.git,
        },
    }


def test_provider_can_return_mapping_that_is_checked_before_registration(tmp_path) -> None:
    _write_fake_package(tmp_path)
    registry = ResearchProblemSpecRegistry(active_id="tiny_problem")

    loaded = load_research_problem_provider(_config(tmp_path, "tiny_problem.research_problem:build_dict_spec"), registry=registry)

    assert loaded.spec.id == "tiny_problem"
    assert registry.get("tiny_problem") == loaded.spec


def test_external_fake_provider_package_can_declare_brief_documents() -> None:
    package_root = Path("/home/iross/code/test-research-problem")
    package = package_root / "tiny_problem"
    module_name = f"brief_provider_{os.getpid()}_{uuid.uuid4().hex}"
    provider_path = package / f"{module_name}.py"
    brief_dir = package_root / "brief" / module_name
    overview = brief_dir / "overview.md"
    baselines = brief_dir / "baselines.md"
    try:
        brief_dir.mkdir(parents=True)
        overview.write_text("# External overview\n")
        baselines.write_text("# External baselines\n")
        provider_path.write_text(
            "from ml_autoresearch.research_problems import ResearchProblemSpec\n"
            "def build_spec(data_config=None):\n"
            "    return ResearchProblemSpec(\n"
            "        id='tiny_problem', version='test-spec-v0', contract_version='v0',\n"
            "        input_modes=('tiny_rgb',), input_specs={'tiny_rgb': {'mode': 'tiny_rgb', 'shape': [3, 8, 8]}},\n"
            "        output_forms=('tiny_mask_logits',), output_specs={'tiny_mask_logits': {'form': 'tiny_mask_logits', 'shape': [1, 8, 8]}},\n"
            "        losses=('tiny_bce',), optimizers=('adamw',), sampling_policies=('sequential',),\n"
            "        augmentation_policies=('none',), primary_metric='val/tiny_dice',\n"
            f"        brief_documents=(\n"
            f"            {{'name': 'overview', 'role': 'problem_overview', 'path': 'brief/{module_name}/overview.md'}},\n"
            f"            {{'name': 'baselines', 'role': 'baseline_description', 'path': 'brief/{module_name}/baselines.md', 'required': True}},\n"
            f"        ),\n"
            "    )\n"
        )

        loaded = load_research_problem_provider(_config(package_root, f"tiny_problem.{module_name}:build_spec"))

        assert [document.name for document in loaded.brief_documents] == ["overview", "baselines"]
        assert loaded.brief_documents[0].resolved_path == overview.resolve()
        assert loaded.brief_documents[1].resolved_path == baselines.resolve()
    finally:
        provider_path.unlink(missing_ok=True)
        overview.unlink(missing_ok=True)
        baselines.unlink(missing_ok=True)
        brief_dir.rmdir() if brief_dir.exists() else None


def test_provider_brief_documents_are_resolved_relative_to_package_root(tmp_path) -> None:
    _write_fake_package(tmp_path)
    brief_dir = tmp_path / "brief"
    brief_dir.mkdir()
    (brief_dir / "overview.md").write_text("# Tiny overview\n")
    (brief_dir / "baselines.md").write_text("# Baselines\n")

    loaded = load_research_problem_provider(_config(tmp_path, "tiny_problem.research_problem:build_spec_with_brief_documents"))

    assert [document.name for document in loaded.brief_documents] == ["overview", "baselines", "future_notes"]
    assert loaded.brief_documents[0].resolved_path == (tmp_path / "brief" / "overview.md").resolve()
    assert loaded.brief_documents[1].required is True
    assert loaded.brief_documents[2].resolved_path == (tmp_path / "brief" / "future.md").resolve()
    assert loaded.run_metadata()["brief_documents"][0] == {
        "name": "overview",
        "role": "problem_overview",
        "path": "brief/overview.md",
        "resolved_path": str((tmp_path / "brief" / "overview.md").resolve()),
        "required": False,
        "summary": "Tiny overview",
    }


def test_advisory_brief_documents_may_be_missing_but_required_documents_must_exist(tmp_path) -> None:
    _write_fake_package(tmp_path)

    with pytest.raises(ResearchProblemProviderLoadError, match="required Research Problem Brief document"):
        load_research_problem_provider(_config(tmp_path, "tiny_problem.research_problem:build_spec_with_missing_required_brief"))


@pytest.mark.parametrize(
    "provider_symbol",
    [
        "build_spec_with_escaping_brief",
        "build_spec_with_absolute_brief",
        "build_spec_with_backslash_brief",
    ],
)
def test_unsafe_brief_document_paths_are_rejected(tmp_path, provider_symbol) -> None:
    _write_fake_package(tmp_path)

    with pytest.raises(ResearchProblemProviderLoadError, match="invalid Research Problem Spec"):
        load_research_problem_provider(_config(tmp_path, f"tiny_problem.research_problem:{provider_symbol}"))


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (lambda root: _config(root / "missing"), "package root does not exist"),
        (lambda root: _config(root, "tiny_problem.research_problem"), "provider target must be"),
        (lambda root: _config(root, "tiny_problem.nope:build_spec"), "cannot import Research Problem provider module"),
        (lambda root: _config(root, "tiny_problem.research_problem:nope"), "does not define provider symbol"),
        (lambda root: _config(root, "tiny_problem.research_problem:NOT_CALLABLE"), "is not callable"),
        (lambda root: _config(root, "tiny_problem.research_problem:build_wrong_id"), "id mismatch"),
        (lambda root: _config(root, "tiny_problem.research_problem:build_wrong_contract"), "contract-version mismatch"),
        (lambda root: _config(root, "tiny_problem.research_problem:build_invalid_shape"), "invalid Research Problem Spec"),
    ],
)
def test_provider_loading_failures_are_clear(tmp_path, config, match) -> None:
    _write_fake_package(tmp_path)

    with pytest.raises(ResearchProblemProviderLoadError, match=match):
        load_research_problem_provider(config(tmp_path))
