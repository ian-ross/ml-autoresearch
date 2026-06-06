from __future__ import annotations

import subprocess

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
    registry = ResearchProblemSpecRegistry(default_id="tiny_problem")

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
    registry = ResearchProblemSpecRegistry(default_id="tiny_problem")

    loaded = load_research_problem_provider(_config(tmp_path, "tiny_problem.research_problem:build_dict_spec"), registry=registry)

    assert loaded.spec.id == "tiny_problem"
    assert registry.get("tiny_problem") == loaded.spec


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
