from __future__ import annotations

import json
from pathlib import Path

import yaml

from ml_autoresearch.research_problems import ResearchProblemProviderConfig
from ml_autoresearch.runs import RunStatus, run_candidate_with_research_problem


def _write_fake_problem_package(root: Path) -> None:
    package = root / "fake_problem"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "research_problem.py").write_text(
        "from ml_autoresearch.research_problems import ResearchProblemSpec\n"
        "from ml_autoresearch.synthetic import SyntheticContrailConfig, SyntheticContrailDataset\n"
        "from ml_autoresearch.training_adapters import ResearchProblemDatasets\n"
        "\n"
        "class FakeTrainingAdapter:\n"
        "    def dataset_metadata(self, data_config):\n"
        "        return {'id': 'fake_dataset', 'fixture': data_config.get('fixture', 'tiny')}\n"
        "\n"
        "    def build_datasets(self, *, data_config, resolved_manifest_path, max_samples=None):\n"
        "        count = int(max_samples or data_config.get('sample_count', 4))\n"
        "        return ResearchProblemDatasets(\n"
        "            train_dataset=SyntheticContrailDataset(count, seed=123, config=SyntheticContrailConfig(image_size=8)),\n"
        "            validation_dataset=SyntheticContrailDataset(2, seed=456, config=SyntheticContrailConfig(image_size=8)),\n"
        "            start_line='Starting fake Research Problem training.',\n"
        "            success_line='Fake Research Problem training completed.',\n"
        "            failure_prefix='Fake Research Problem training failed',\n"
        "            data_policy_metadata={'adapter': 'fake'},\n"
        "        )\n"
        "\n"
        "def build_spec(data_config=None):\n"
        "    return ResearchProblemSpec(\n"
        "        id='fake_problem',\n"
        "        version='fake-spec-v1',\n"
        "        contract_version='v0',\n"
        "        input_modes=('fake_rgb',),\n"
        "        input_specs={'fake_rgb': {'mode': 'fake_rgb', 'shape': [3, 8, 8]}},\n"
        "        output_forms=('mask_logits',),\n"
        "        output_specs={'mask_logits': {'form': 'mask_logits', 'shape': [1, 8, 8]}},\n"
        "        losses=('bce_dice',),\n"
        "        optimizers=('adamw',),\n"
        "        sampling_policies=('sequential',),\n"
        "        augmentation_policies=('none',),\n"
        "        primary_metric='val/dice',\n"
        "        training_adapter=FakeTrainingAdapter(),\n"
        "    )\n"
    )


def _write_fake_candidate(root: Path) -> Path:
    candidate = root / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "fake_problem_candidate",
                "research_problem": "fake_problem",
                "input_mode": "fake_rgb",
                "output_form": "mask_logits",
                "training": {
                    "loss": "bce_dice",
                    "optimizer": "adamw",
                    "learning_rate": 0.001,
                    "batch_size": 2,
                    "max_epochs": 1,
                },
            },
            sort_keys=False,
        )
    )
    (candidate / "model.py").write_text(
        "import torch\n"
        "\n"
        "class TinyMask(torch.nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.net = torch.nn.Conv2d(3, 1, kernel_size=1)\n"
        "\n"
        "    def forward(self, x):\n"
        "        return self.net(x)\n"
        "\n"
        "def build_model(input_spec, output_spec):\n"
        "    return TinyMask()\n"
    )
    return candidate


def test_run_candidate_trains_through_generic_research_problem_provider(tmp_path: Path) -> None:
    _write_fake_problem_package(tmp_path)
    candidate = _write_fake_candidate(tmp_path)
    config = ResearchProblemProviderConfig(
        id="fake_problem",
        package_root=tmp_path,
        provider_target="fake_problem.research_problem:build_spec",
        expected_contract_version="v0",
        data_config={"fixture": "tiny", "sample_count": 4},
    )

    run = run_candidate_with_research_problem(candidate, tmp_path / "runs", config, max_samples=4, max_prediction_samples=1)

    assert run.status == RunStatus.COMPLETED
    metadata = json.loads((run.run_dir / "run_metadata.json").read_text())
    assert metadata["research_problem"]["id"] == "fake_problem"
    assert metadata["research_problem"]["version"] == "fake-spec-v1"
    assert metadata["research_problem"]["contract_version"] == "v0"
    assert metadata["research_problem"]["provider"]["target"] == "fake_problem.research_problem:build_spec"
    assert metadata["dataset"] == {"id": "fake_dataset", "fixture": "tiny"}
    assert metadata["data_policy"] == {"adapter": "fake"}
    assert metadata["sample_counts"] == {"train": 4, "validation": 2}
    assert (run.run_dir / "outputs" / "best_metrics.json").is_file()
