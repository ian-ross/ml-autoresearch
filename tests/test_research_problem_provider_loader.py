from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ml_autoresearch.research_problems import (
    ResearchProblemProviderConfig,
    ResearchProblemProviderLoadError,
    ResearchProblemSpec,
    ResearchProblemSpecRegistry,
    load_research_problem_provider,
)
from research_problem_helpers import gvccs_research_problem_root, require_test_research_problem_root


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REUSABLE_HARNESS_ROOT = PROJECT_ROOT / "src" / "ml_autoresearch"


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
        "def build_spec_with_dataset_profile_artifacts(data_config=None):\n"
        "    return {**build_spec(data_config).model_dump(), 'dataset_profile_artifacts': [\n"
        "        {'name': 'tiny_profile', 'role': 'deterministic_test_dataset_profile', 'path': 'profile/tiny-dataset-profile.json', 'summary': 'Tiny profile', 'split_scope': 'train+validation', 'required': True},\n"
        "        {'name': 'future_profile', 'role': 'optional_operator_generated_profile', 'path': 'profile/future.json'},\n"
        "    ]}\n"
        "\n"
        "def build_spec_with_missing_required_profile(data_config=None):\n"
        "    return {**build_spec(data_config).model_dump(), 'dataset_profile_artifacts': [\n"
        "        {'name': 'required_missing', 'role': 'deterministic_test_dataset_profile', 'path': 'profile/missing.json', 'required': True},\n"
        "    ]}\n"
        "\n"
        "def build_spec_with_escaping_profile(data_config=None):\n"
        "    return {**build_spec(data_config).model_dump(), 'dataset_profile_artifacts': [{'name': 'bad', 'role': 'profile', 'path': '../outside.json'}]}\n"
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
        "class IncompleteTrainingAdapter:\n"
        "    def validate_data_root(self, data_config):\n"
        "        return data_config.get('dataset_root')\n"
        "\n"
        "class EvaluationAdapter:\n"
        "    def run_evaluation_mode(self, **kwargs):\n"
        "        return {}, [], {}, {}\n"
        "\n"
        "def build_training_adapter_without_capability(data_config=None):\n"
        "    return build_spec(data_config).model_copy(update={'training_adapter': IncompleteTrainingAdapter()})\n"
        "\n"
        "def build_training_capability_without_adapter(data_config=None):\n"
        "    return build_spec(data_config).model_copy(update={'operation_capabilities': {'training': True}})\n"
        "\n"
        "def build_incomplete_training_capability(data_config=None):\n"
        "    return build_spec(data_config).model_copy(update={'operation_capabilities': {'training': True}, 'training_adapter': IncompleteTrainingAdapter()})\n"
        "\n"
        "def build_evaluation_modes_without_adapter(data_config=None):\n"
        "    return build_spec(data_config).model_copy(update={'operation_capabilities': {'evaluation_modes': ('whole_validation_failure_analysis',)}})\n"
        "\n"
        "def build_evaluation_adapter_without_modes(data_config=None):\n"
        "    return build_spec(data_config).model_copy(update={'evaluation_adapter': EvaluationAdapter()})\n"
        "\n"
        "def build_duplicate_evaluation_modes(data_config=None):\n"
        "    return build_spec(data_config).model_copy(update={'operation_capabilities': {'evaluation_modes': ('whole_validation_failure_analysis', 'whole_validation_failure_analysis')}, 'evaluation_adapter': EvaluationAdapter()})\n"
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
        "operation_capabilities": {"training": False, "evaluation_modes": []},
    }


def test_provider_can_return_mapping_that_is_checked_before_registration(tmp_path) -> None:
    _write_fake_package(tmp_path)
    registry = ResearchProblemSpecRegistry(active_id="tiny_problem")

    loaded = load_research_problem_provider(_config(tmp_path, "tiny_problem.research_problem:build_dict_spec"), registry=registry)

    assert loaded.spec.id == "tiny_problem"
    assert registry.get("tiny_problem") == loaded.spec


def test_provider_rejects_training_adapter_without_explicit_operation_capability(tmp_path) -> None:
    _write_fake_package(tmp_path)

    with pytest.raises(ResearchProblemProviderLoadError, match="training_adapter requires operation_capabilities.training"):
        load_research_problem_provider(_config(tmp_path, "tiny_problem.research_problem:build_training_adapter_without_capability"))


def test_provider_rejects_training_capability_without_complete_adapter(tmp_path) -> None:
    _write_fake_package(tmp_path)

    with pytest.raises(ResearchProblemProviderLoadError, match="operation_capabilities.training requires training_adapter"):
        load_research_problem_provider(_config(tmp_path, "tiny_problem.research_problem:build_training_capability_without_adapter"))

    with pytest.raises(ResearchProblemProviderLoadError, match="training_adapter is missing callable method: dataset_metadata"):
        load_research_problem_provider(_config(tmp_path, "tiny_problem.research_problem:build_incomplete_training_capability"))


def test_provider_rejects_evaluation_capability_without_declared_adapter(tmp_path) -> None:
    _write_fake_package(tmp_path)

    with pytest.raises(ResearchProblemProviderLoadError, match="operation_capabilities.evaluation_modes requires evaluation_adapter"):
        load_research_problem_provider(_config(tmp_path, "tiny_problem.research_problem:build_evaluation_modes_without_adapter"))

    with pytest.raises(ResearchProblemProviderLoadError, match="evaluation_adapter requires operation_capabilities.evaluation_modes"):
        load_research_problem_provider(_config(tmp_path, "tiny_problem.research_problem:build_evaluation_adapter_without_modes"))


def test_provider_rejects_duplicate_evaluation_modes(tmp_path) -> None:
    _write_fake_package(tmp_path)

    with pytest.raises(ResearchProblemProviderLoadError, match="evaluation_modes must not contain duplicates"):
        load_research_problem_provider(_config(tmp_path, "tiny_problem.research_problem:build_duplicate_evaluation_modes"))


def test_evaluation_dispatch_rejects_undeclared_mode(tmp_path) -> None:
    from ml_autoresearch.evaluations import EvaluationError, ResolvedEvaluationResearchProblem, dispatch_evaluation_mode

    class EvaluationAdapter:
        supported_evaluation_modes = ("declared",)

        def run_evaluation_mode(self, **kwargs):
            raise AssertionError("undeclared mode should not reach the evaluation adapter")

    spec = ResearchProblemSpec(
        id="tiny_problem",
        version="test-spec-v0",
        contract_version="v0",
        input_modes=("tiny_rgb",),
        input_specs={"tiny_rgb": {"mode": "tiny_rgb", "shape": [3, 8, 8]}},
        output_forms=("tiny_mask_logits",),
        output_specs={"tiny_mask_logits": {"form": "tiny_mask_logits", "shape": [1, 8, 8]}},
        losses=("tiny_loss",),
        optimizers=("sgd",),
        sampling_policies=("sequential",),
        augmentation_policies=("none",),
        primary_metric="val/tiny_score",
        operation_capabilities={"evaluation_modes": ("declared",)},
        evaluation_adapter=EvaluationAdapter(),
    )

    with pytest.raises(EvaluationError, match="does not declare Post-Run Evaluation mode"):
        dispatch_evaluation_mode(
            research_problem=ResolvedEvaluationResearchProblem(spec=spec, metadata={}),
            mode="undeclared",
            run_dir=tmp_path,
            data_root=tmp_path,
            model_artifact_path=tmp_path / "model.pt",
            threshold=0.5,
            evaluation_dir=tmp_path / "evaluation",
            max_artifact_samples=1,
        )


def test_provider_load_tolerates_missing_git_executable_for_best_effort_provenance(tmp_path, monkeypatch) -> None:
    import ml_autoresearch.research_problems as research_problems

    _write_fake_package(tmp_path)

    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(research_problems, "_SUBPROCESS_RUN", missing_git)

    loaded = load_research_problem_provider(_config(tmp_path))

    assert loaded.spec.id == "tiny_problem"
    assert loaded.provenance.git is None


def test_external_fake_provider_package_ships_brief_documents_used_by_harness_tests() -> None:
    package_root = require_test_research_problem_root()

    assert not package_root.is_relative_to(REUSABLE_HARNESS_ROOT)

    loaded = load_research_problem_provider(_config(package_root))

    assert [(document.name, document.role) for document in loaded.brief_documents] == [
        ("overview", "problem_overview"),
        ("baselines", "baseline_description"),
    ]
    assert [document.summary for document in loaded.brief_documents] == [
        "Tiny synthetic segmentation problem overview.",
        "Minimal deterministic baseline notes for Harness tests.",
    ]
    assert loaded.brief_documents[0].resolved_path == (package_root / "brief" / "overview.md").resolve()
    assert loaded.brief_documents[1].resolved_path == (package_root / "brief" / "baselines.md").resolve()
    assert loaded.brief_documents[1].required is True
    assert "Tiny Problem Overview" in loaded.brief_documents[0].resolved_path.read_text()
    assert loaded.spec.operation_capabilities.training is True
    assert loaded.spec.operation_capabilities.evaluation_modes == ()
    assert len(loaded.dataset_profile_artifacts) == 1
    profile = loaded.dataset_profile_artifacts[0]
    assert profile.name == "tiny_dataset_profile"
    assert profile.role == "deterministic_test_dataset_profile"
    assert profile.required is True
    assert profile.resolved_path == (package_root / "profile" / "tiny-dataset-profile.json").resolve()
    assert "positive_mask_prevalence" in profile.resolved_path.read_text()


def test_external_gvccs_provider_package_ships_ground_camera_contrail_detection_brief() -> None:
    package_root = gvccs_research_problem_root()

    assert not package_root.is_relative_to(REUSABLE_HARNESS_ROOT)
    loaded = load_research_problem_provider(
        ResearchProblemProviderConfig(
            id="ground_camera_contrail_detection",
            package_root=package_root,
            provider_target="gvccs.research_problem:build_spec",
            expected_contract_version="v0",
            data_config={"dataset_root": "/trusted/gvccs"},
        )
    )

    assert loaded.spec.operation_capabilities.training is True
    assert loaded.spec.operation_capabilities.evaluation_modes == ("whole_validation_failure_analysis",)
    assert len(loaded.brief_documents) == 1
    document = loaded.brief_documents[0]
    assert document.name == "ground_camera_contrail_detection"
    assert document.role == "problem_brief"
    assert document.required is True
    assert document.resolved_path == (package_root / "brief" / "ground-camera-contrail-detection.md").resolve()
    assert document.metadata()["resolved_path"].startswith(str(package_root.resolve()))
    text = document.resolved_path.read_text()
    for required_section in [
        "## Task context",
        "## Data notes",
        "## Baseline and prior-approach notes",
        "## Reference pointers",
        "## Architecture suggestions",
    ]:
        assert required_section in text
    assert "zenodo.org/records/16612390" in text
    assert len(loaded.dataset_profile_artifacts) == 1
    profile = loaded.dataset_profile_artifacts[0]
    assert profile.name == "gvccs_initial_dataset_profile"
    assert profile.role == "operator_generated_dataset_profile_or_generator"
    assert profile.required is True
    assert profile.resolved_path == (package_root / "profile" / "initial-dataset-profile.md").resolve()
    profile_text = profile.resolved_path.read_text()
    assert "positive-Contrail-Mask image counts" in profile_text
    assert "If `--data-root` is absent or malformed" in profile_text


def test_gvccs_profile_generator_documents_missing_optional_data(tmp_path) -> None:
    package_root = gvccs_research_problem_root()
    output = tmp_path / "missing-profile.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(package_root) + os.pathsep + env.get("PYTHONPATH", "")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gvccs.profile",
            "--data-root",
            str(tmp_path / "missing-gvccs"),
            "--output",
            str(output),
            "--allow-missing",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert '"status": "missing_optional_data"' in output.read_text()
    assert "Run the generator outside the Agent Control Boundary" in output.read_text()


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


def test_provider_dataset_profile_artifacts_are_resolved_relative_to_package_root(tmp_path) -> None:
    _write_fake_package(tmp_path)
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "tiny-dataset-profile.json").write_text('{"provenance": {"research_problem_id": "tiny_problem"}}\n')

    loaded = load_research_problem_provider(_config(tmp_path, "tiny_problem.research_problem:build_spec_with_dataset_profile_artifacts"))

    assert [artifact.name for artifact in loaded.dataset_profile_artifacts] == ["tiny_profile", "future_profile"]
    assert loaded.dataset_profile_artifacts[0].resolved_path == (profile_dir / "tiny-dataset-profile.json").resolve()
    assert loaded.dataset_profile_artifacts[0].required is True
    assert loaded.dataset_profile_artifacts[1].resolved_path == (profile_dir / "future.json").resolve()
    assert loaded.run_metadata()["dataset_profile_artifacts"][0] == {
        "name": "tiny_profile",
        "role": "deterministic_test_dataset_profile",
        "path": "profile/tiny-dataset-profile.json",
        "resolved_path": str((profile_dir / "tiny-dataset-profile.json").resolve()),
        "required": True,
        "summary": "Tiny profile",
        "split_scope": "train+validation",
    }


def test_advisory_brief_documents_may_be_missing_but_required_documents_must_exist(tmp_path) -> None:
    _write_fake_package(tmp_path)

    with pytest.raises(ResearchProblemProviderLoadError, match="required Research Problem Brief document"):
        load_research_problem_provider(_config(tmp_path, "tiny_problem.research_problem:build_spec_with_missing_required_brief"))


def test_optional_profile_artifacts_may_be_missing_but_required_artifacts_must_exist(tmp_path) -> None:
    _write_fake_package(tmp_path)

    with pytest.raises(ResearchProblemProviderLoadError, match="required Dataset Profile Artifact"):
        load_research_problem_provider(_config(tmp_path, "tiny_problem.research_problem:build_spec_with_missing_required_profile"))


@pytest.mark.parametrize(
    "provider_symbol",
    [
        "build_spec_with_escaping_brief",
        "build_spec_with_absolute_brief",
        "build_spec_with_backslash_brief",
        "build_spec_with_escaping_profile",
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
