from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from ml_autoresearch.research_problems import ResearchProblemProviderConfig, ResearchProblemSpecRegistry, load_research_problem_provider
from ml_autoresearch.runs import run_candidate_with_research_problem

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_PACKAGE_ROOT = REPO_ROOT / "src" / "ml_autoresearch"

TEST_PROBLEM_ROOT_ENV = "ML_AUTORESEARCH_TEST_PROBLEM_ROOT"
GVCCS_PROBLEM_ROOT_ENV = "ML_AUTORESEARCH_GVCCS_PROBLEM_ROOT"
GVCCS_RESEARCH_PROBLEM_ID = "ground_camera_contrail_detection"
GVCCS_PROVIDER_TARGET = "gvccs.research_problem:build_spec"


def _ensure_external_package_root(env_var: str, label: str) -> Path:
    root_value = os.environ.get(env_var)
    if not root_value:
        raise AssertionError(
            f"{env_var} must be set to an external {label} Research Problem package root (for example, a local checkout path)."
        )

    root = Path(root_value).expanduser()
    if not root.exists():
        raise AssertionError(f"{env_var} points to a missing path: {root}")
    if not root.is_dir():
        raise AssertionError(f"{env_var} must be a directory: {root}")

    resolved = root.resolve()
    if resolved == HARNESS_PACKAGE_ROOT or resolved.is_relative_to(HARNESS_PACKAGE_ROOT):
        raise AssertionError(
            f"{env_var} must point outside the reusable Harness package ({HARNESS_PACKAGE_ROOT}); got: {resolved}"
        )

    return resolved


def require_test_research_problem_root() -> Path:
    return _ensure_external_package_root(TEST_PROBLEM_ROOT_ENV, "test")


def gvccs_research_problem_root() -> Path:
    configured = os.environ.get(GVCCS_PROBLEM_ROOT_ENV)
    if configured:
        return _ensure_external_package_root(GVCCS_PROBLEM_ROOT_ENV, "GVCCS")
    local_checkout = Path("/home/iross/code/gvccs-research-problem")
    if local_checkout.is_dir():
        return local_checkout.resolve()
    return _ensure_external_package_root(GVCCS_PROBLEM_ROOT_ENV, "GVCCS")


def gvccs_module():
    """Import the external GVCCS package using the configured env-root path."""

    root = gvccs_research_problem_root()
    root_path = str(root)
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    return importlib.import_module("gvccs")


def gvccs_provider_config(data_root: str | Path | None = None, *, package_root: Path | str | None = None) -> ResearchProblemProviderConfig:
    data_config = {}
    if data_root is not None:
        data_config["dataset_root"] = str(data_root)
    package_root_path = Path(package_root) if package_root is not None else gvccs_research_problem_root()
    return ResearchProblemProviderConfig(
        id=GVCCS_RESEARCH_PROBLEM_ID,
        package_root=package_root_path,
        provider_target=GVCCS_PROVIDER_TARGET,
        expected_contract_version="v0",
        data_config=data_config,
    )


def gvccs_registry(data_root: str | Path | None = None, *, package_root: Path | str | None = None) -> ResearchProblemSpecRegistry:
    config = gvccs_provider_config(data_root, package_root=package_root)
    registry = ResearchProblemSpecRegistry(active_id=config.id)
    load_research_problem_provider(config, registry=registry)
    return registry


def run_candidate_with_gvccs_data(candidate_dir, runs_root, data_root, *, package_root: Path | str | None = None, **kwargs):
    return run_candidate_with_research_problem(candidate_dir, runs_root, gvccs_provider_config(data_root, package_root=package_root), **kwargs)


def run_candidate_with_synthetic_fixture(
    candidate_dir,
    runs_root,
    *,
    data_root: str | Path | None = None,
    package_root: Path | str | None = None,
    **kwargs,
):
    resolved_root = Path(data_root) if data_root is not None else REPO_ROOT / "tests" / "fixtures" / "gvccs_like"
    return run_candidate_with_research_problem(candidate_dir, runs_root, gvccs_provider_config(resolved_root, package_root=package_root), **kwargs)

def run_experiment_batch_with_synthetic_fixture(
    batch_dir,
    *,
    batches_root,
    runs_root,
    backend=None,
    max_parallel_runs: int = 4,
    max_samples: int | None = None,
    max_prediction_samples: int = 2,
    prediction_sample_policy: str = "first_n",
    ledger_path: Path | str | None = None,
    data_root: str | Path | None = None,
    package_root: Path | str | None = None,
    **kwargs,
):
    from ml_autoresearch.batches import run_experiment_batch_with_research_problem

    resolved_root = Path(data_root) if data_root is not None else REPO_ROOT / "tests" / "fixtures" / "gvccs_like"
    return run_experiment_batch_with_research_problem(
        batch_dir,
        batches_root=batches_root,
        runs_root=runs_root,
        provider_config=gvccs_provider_config(resolved_root, package_root=package_root),
        backend=backend,
        max_parallel_runs=max_parallel_runs,
        max_samples=max_samples,
        max_prediction_samples=max_prediction_samples,
        prediction_sample_policy=prediction_sample_policy,
        ledger_path=ledger_path,
        **kwargs,
    )


def write_fake_research_problem_package(root: Path, *, package_name: str = "fake_problem") -> Path:
    package = root / package_name
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text("")
    (package / "research_problem.py").write_text(
        "from pathlib import Path\n"
        "import torch\n"
        "from ml_autoresearch.metrics import binary_segmentation_metrics\n"
        "from ml_autoresearch.research_problems import ResearchProblemSpec\n"
        "from ml_autoresearch.synthetic import SyntheticContrailConfig, SyntheticContrailDataset\n"
        "from ml_autoresearch.training_adapters import ResearchProblemDatasets\n"
        "\n"
        "class FakeTrainingAdapter:\n"
        "    def validate_data_root(self, data_config):\n"
        "        dataset_root = data_config.get(\"dataset_root\")\n"
        "        if dataset_root is None:\n"
        "            return Path(\".\").resolve()\n"
        "        return Path(dataset_root).expanduser().resolve()\n"
        "\n"
        "    def dataset_metadata(self, data_config):\n"
        "        return {\"id\": \"fake_dataset\", \"fixture\": data_config.get(\"fixture\", \"tiny\")}\n"
        "\n"
        "    def build_datasets(self, *, data_config, resolved_manifest_path, max_samples=None):\n"
        "        sample_count = int(max_samples or data_config.get(\"sample_count\", 4))\n"
        "        config = SyntheticContrailConfig(image_size=8)\n"
        "        return ResearchProblemDatasets(\n"
        "            train_dataset=SyntheticContrailDataset(sample_count, seed=123, config=config),\n"
        "            validation_dataset=SyntheticContrailDataset(2, seed=456, config=config),\n"
        "            start_line=\"Starting fake Research Problem training.\",\n"
        "            success_line=\"Fake Research Problem training completed.\",\n"
        "            failure_prefix=\"Fake Research Problem training failed\",\n"
        "            data_policy_metadata={\"adapter\": \"fake\"},\n"
        "        )\n"
        "\n"
        "    def primary_output_name(self, output_spec):\n"
        "        return \"mask_logits\"\n"
        "\n"
        "    def compute_primary_loss(self, loss_name, logits, target_mask):\n"
        "        if loss_name != \"bce_dice\":\n"
        "            raise ValueError(f\"unsupported loss: {loss_name}\")\n"
        "        return torch.nn.functional.binary_cross_entropy_with_logits(logits, target_mask)\n"
        "\n"
        "    def compute_auxiliary_losses(self, outputs, target_mask, auxiliary_targets):\n"
        "        return {}\n"
        "\n"
        "    def compute_validation_metrics(self, logits, target_mask):\n"
        "        prediction = torch.sigmoid(logits) >= 0.5\n"
        "        result = binary_segmentation_metrics(prediction, target_mask)\n"
        "        return {\"val/dice\": float(result['dice'])}\n"
        "\n"
        "def build_spec(data_config=None):\n"
        "    return ResearchProblemSpec(\n"
        "        id=\"fake_problem\",\n"
        "        version=\"fake-spec-v1\",\n"
        "        contract_version=\"v0\",\n"
        "        input_modes=(\"single_frame_rgb\",),\n"
        "        input_specs={\"single_frame_rgb\": {\"mode\": \"single_frame_rgb\", \"shape\": [3, 8, 8]}},\n"
        "        output_forms=(\"mask_logits\",),\n"
        "        output_specs={\"mask_logits\": {\"form\": \"mask_logits\", \"shape\": [1, 8, 8]}},\n"
        "        losses=(\"bce_dice\",),\n"
        "        optimizers=(\"adamw\",),\n"
        "        sampling_policies=(\"sequential\",),\n"
        "        augmentation_policies=(\"none\",),\n"
        "        primary_metric=\"val/dice\",\n"
        "        training_adapter=FakeTrainingAdapter(),\n"
        "    )\n"
    )
    return package


def write_fake_candidate_execution_config(
    root: Path,
    *,
    package_name: str = "fake_problem",
    package_root: Path | str | None = None,
    data_root: str | Path | None = None,
    sample_count: int = 4,
    backend: str = "native",
    max_samples: int | None = None,
    max_prediction_samples: int = 2,
    prediction_sample_policy: str = "first_n",
    data_config: dict[str, object] | None = None,
) -> Path:
    config_root = root
    package_container = Path(package_root) if package_root is not None else root
    package_path = package_container / package_name
    config_path = config_root / "candidate-execution.toml"
    config_lines: list[str] = [
        "[candidate_execution]",
        f"backend = \"{backend}\"",
        f"max_prediction_samples = {max_prediction_samples}",
        f"prediction_sample_policy = \"{prediction_sample_policy}\"",
    ]
    if max_samples is not None:
        config_lines.append(f"max_samples = {max_samples}")
    config_lines.extend([
        "",
        "[research_problem]",
        "id = \"fake_problem\"",
        f"package_root = \"{package_container}\"",
        f"provider_target = \"{package_name}.research_problem:build_spec\"",
        "expected_contract_version = \"v0\"",
    ])
    merged_data_config = {
        "dataset_root": data_root,
        "sample_count": sample_count,
    }
    if data_config:
        merged_data_config.update(data_config)
    config_lines.append("[research_problem.data_config]")
    for key, value in merged_data_config.items():
        if value is None:
            continue
        if isinstance(value, bool):
            config_lines.append(f"{key} = {str(value).lower()}")
        elif isinstance(value, int | float):
            config_lines.append(f"{key} = {value}")
        else:
            config_lines.append(f"{key} = \"{value}\"")
    config_path.write_text("\n".join(config_lines) + "\n")
    return config_path
