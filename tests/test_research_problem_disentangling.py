from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from research_problem_helpers import require_test_research_problem_root


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REUSABLE_HARNESS_ROOT = PROJECT_ROOT / "src" / "ml_autoresearch"


FORBIDDEN_PRODUCTION_REFERENCES = (
    "gvccs",
    "ground_camera_contrail_detection",
    "ground-camera contrail detection",
    "research_problem_packages.gvccs",
    "train-gvccs",
    "train_gvccs",
    "with_gvccs",
)


def test_reusable_harness_modules_do_not_contain_gvccs_specific_production_references() -> None:
    """Guard the deletion seam: reusable Harness source must stay Research Problem-generic."""

    assert not (PROJECT_ROOT / "src/ml_autoresearch/research_problem_packages/gvccs").exists()
    assert not (PROJECT_ROOT / "src/ml_autoresearch/gvccs.py").exists()

    violations: list[str] = []
    tracked_source_paths = subprocess.run(
        ["git", "ls-files", "src/ml_autoresearch"],
        check=True,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    for tracked_path in tracked_source_paths:
        path = PROJECT_ROOT / tracked_path
        if not path.is_file():
            continue
        try:
            text = path.read_text().lower()
        except UnicodeDecodeError:
            continue
        for forbidden in FORBIDDEN_PRODUCTION_REFERENCES:
            if forbidden in text:
                violations.append(f"{tracked_path}: contains {forbidden!r}")

    assert violations == []


def test_reusable_harness_modules_do_not_directly_import_gvccs_package() -> None:
    """Guard the deletion seam: reusable Harness modules must not import GVCCS directly."""

    violations: list[str] = []
    for path in sorted(REUSABLE_HARNESS_ROOT.rglob("*.py")):
        relative = path.relative_to(PROJECT_ROOT)
        tree = ast.parse(path.read_text(), filename=str(relative))
        for node in ast.walk(tree):
            imported_modules: list[str] = []
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.append(node.module)
            for module in imported_modules:
                if "gvccs" in module.lower():
                    violations.append(f"{relative}:{node.lineno}: {module}")

    assert violations == []


def test_harness_imports_and_fake_research_problem_flow_work_when_gvccs_package_is_unavailable(tmp_path: Path) -> None:
    package_root = require_test_research_problem_root()

    assert not package_root.is_relative_to(REUSABLE_HARNESS_ROOT)
    candidate = tmp_path / "candidate"
    _write_fake_candidate(candidate)
    runs_root = tmp_path / "runs"

    script = f"""
from __future__ import annotations

import importlib.abc
import json
import sys
from pathlib import Path

sys.path.insert(0, {str(PROJECT_ROOT / 'src')!r})

class BlockGVCCS(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ('ml_autoresearch.gvccs', 'gvccs')
        if fullname in blocked or fullname.startswith('gvccs.'):
            raise ModuleNotFoundError(f'Simulated deleted GVCCS package: {{fullname}}')
        return None

sys.meta_path.insert(0, BlockGVCCS())

import ml_autoresearch
from ml_autoresearch.research_problems import ResearchProblemProviderConfig
from ml_autoresearch.runs import RunStatus, run_candidate_with_research_problem

config = ResearchProblemProviderConfig(
    id='tiny_problem',
    package_root=Path({str(package_root)!r}),
    provider_target='tiny_problem.research_problem:build_spec',
    expected_contract_version='v0',
    data_config={{'dataset_root': {str(tmp_path / 'fake-data')!r}}},
)
run = run_candidate_with_research_problem(
    {str(candidate)!r},
    {str(runs_root)!r},
    config,
    max_prediction_samples=1,
    require_proposal=False,
)
metadata = json.loads((run.run_dir / 'run_metadata.json').read_text())
assert run.status == RunStatus.COMPLETED
assert metadata['research_problem']['id'] == 'tiny_problem'
assert metadata['dataset']['id'] == 'tiny_dataset'
"""

    subprocess.run([sys.executable, "-c", script], check=True, cwd=PROJECT_ROOT)


def _write_fake_candidate(candidate: Path) -> None:
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        """
name: tiny_problem_candidate
research_problem: tiny_problem
input_mode: tiny_rgb
output_form: tiny_mask_logits
data:
  sampling_policy: sequential
  augmentation_policy: none
training:
  loss: tiny_bce
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )
    (candidate / "model.py").write_text(
        "from torch import nn\n"
        "class Tiny(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.net = nn.Conv2d(3, 1, 1)\n"
        "    def forward(self, x):\n"
        "        return {'tiny_mask_logits': self.net(x)}\n"
        "def build_model(input_spec, output_spec):\n"
        "    return Tiny()\n"
    )


def _write_fake_problem_package(root: Path) -> None:
    package = root / "tiny_problem"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "research_problem.py").write_text(
        "from __future__ import annotations\n"
        "\n"
        "from pathlib import Path\n"
        "import torch\n"
        "from torch.utils.data import Dataset\n"
        "\n"
        "from ml_autoresearch.metrics import binary_segmentation_metrics\n"
        "from ml_autoresearch.research_problems import ResearchProblemSpec\n"
        "from ml_autoresearch.training_adapters import ResearchProblemDatasets\n"
        "\n"
        "class TinyDataset(Dataset):\n"
        "    def __init__(self, size):\n"
        "        self.size = size\n"
        "    def __len__(self):\n"
        "        return self.size\n"
        "    def __getitem__(self, index):\n"
        "        image = torch.full((3, 8, 8), float(index % 3) / 2.0)\n"
        "        mask = torch.zeros((1, 8, 8))\n"
        "        mask[:, 2:6, 2:6] = float(index % 2)\n"
        "        return image, mask\n"
        "\n"
        "class TinyTrainingAdapter:\n"
        "    def validate_data_root(self, data_config):\n"
        "        return Path(data_config['dataset_root'])\n"
        "    def dataset_metadata(self, data_config):\n"
        "        return {'id': 'tiny_dataset', 'root': str(data_config['dataset_root'])}\n"
        "    def build_datasets(self, *, data_config, resolved_manifest_path, max_samples=None):\n"
        "        return ResearchProblemDatasets(\n"
        "            train_dataset=TinyDataset(max_samples or 4),\n"
        "            validation_dataset=TinyDataset(2),\n"
        "            start_line='Starting tiny fake Research Problem training.',\n"
        "            success_line='Tiny fake Research Problem training completed.',\n"
        "            failure_prefix='Tiny fake Research Problem training failed',\n"
        "            data_policy_metadata={'source': 'fake_research_problem'},\n"
        "        )\n"
        "    def apply_augmentation_policy(self, dataset, augmentation_policy):\n"
        "        if augmentation_policy != 'none':\n"
        "            raise ValueError(f'unsupported augmentation policy: {augmentation_policy}')\n"
        "        return dataset\n"
        "    def primary_output_name(self, output_spec):\n"
        "        return 'tiny_mask_logits'\n"
        "    def compute_primary_loss(self, loss_name, logits, target_mask):\n"
        "        if loss_name != 'tiny_bce':\n"
        "            raise ValueError(f'unsupported loss: {loss_name}')\n"
        "        return torch.nn.functional.binary_cross_entropy_with_logits(logits, target_mask)\n"
        "    def compute_auxiliary_losses(self, outputs, target_mask, auxiliary_targets):\n"
        "        return {}\n"
        "    def compute_validation_metrics(self, logits, target_mask):\n"
        "        metrics = binary_segmentation_metrics(torch.sigmoid(logits) >= 0.5, target_mask >= 0.5)\n"
        "        return {'val/tiny_dice': metrics['dice'], 'val/tiny_iou': metrics['iou']}\n"
        "    def selection_policy(self):\n"
        "        return 'val/tiny_dice', 'max'\n"
        "    def build_evaluation_dataset(self, *, data_config, resolved_manifest_path):\n"
        "        return TinyDataset(2)\n"
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
        "        losses=('tiny_bce',),\n"
        "        optimizers=('adamw',),\n"
        "        sampling_policies=('sequential',),\n"
        "        augmentation_policies=('none',),\n"
        "        primary_metric='val/tiny_dice',\n"
        "        training_adapter=TinyTrainingAdapter(),\n"
        "    )\n"
    )
