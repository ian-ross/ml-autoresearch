from __future__ import annotations

import json
from pathlib import Path

from ml_autoresearch.cli import app
from conftest import invoke_typer_cli
from research_problem_helpers import write_fake_candidate_execution_config, write_fake_research_problem_package


def write_candidate(root: Path, name: str) -> Path:
    candidate = root / name
    candidate.mkdir(parents=True)
    (candidate / "manifest.yaml").write_text(
        f"""
name: {name}
input_mode: single_frame_rgb
output_form: mask_logits
training:
  loss: bce_dice
  optimizer: adamw
  learning_rate: 0.001
  batch_size: 2
  max_epochs: 1
""".strip()
        + "\n"
    )
    (candidate / "model.py").write_text(
        "import torch\n"
        "from torch import nn\n"
        "class Tiny(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.conv = nn.Conv2d(3, 1, kernel_size=1)\n"
        "    def forward(self, x):\n"
        "        return self.conv(x)\n"
        "def build_model(input_spec, output_spec):\n"
        "    return Tiny()\n"
    )
    (candidate / "README.md").write_text(f"# {name}\n")
    return candidate


def write_batch(root: Path) -> Path:
    batch = root / "experiment_batch"
    candidates = batch / "candidates"
    candidates.mkdir(parents=True)
    (batch / "BATCH_PROPOSAL.md").write_text(
        """# Batch Proposal\n\n## Shared Hypothesis\nSome hypothesis.\n\n## Shared Comparison Target\nA shared target.\n\n## Per-candidate Variant Rationale\nVariant rationale section.\n\n## Expected Ordering or Decision Criteria\nCompare A > B.\n\n## Batch-level Success Criteria\nPasses threshold.\n\n## Requested Budget/Concurrency\n1 hour, 1 concurrent.\n"""
    )
    write_candidate(candidates, "variant_a")
    return batch


def run_cli(*args: str):
    return invoke_typer_cli(app, args)


def test_run_experiment_batch_command_uses_configured_research_problem(tmp_path: Path):
    batch_dir = write_batch(tmp_path)
    write_fake_research_problem_package(tmp_path)
    write_fake_candidate_execution_config(tmp_path)

    completed = run_cli(
        "run-experiment-batch",
        "--batch",
        str(batch_dir),
        "--batches-root",
        str(tmp_path / "batches"),
        "--runs-root",
        str(tmp_path / "runs"),
        "--workspace-root",
        str(tmp_path),
        "--backend",
        "native",
        "--max-samples",
        "2",
        "--max-prediction-samples",
        "1",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "completed"
    assert payload["batch_id"].startswith("batch_")
    assert len(payload["runs"]) == 1


def test_run_experiment_batch_command_requires_configured_problem(tmp_path: Path):
    batch_dir = write_batch(tmp_path)

    completed = run_cli(
        "run-experiment-batch",
        "--batch",
        str(batch_dir),
        "--batches-root",
        str(tmp_path / "batches"),
        "--runs-root",
        str(tmp_path / "runs"),
        "--workspace-root",
        str(tmp_path),
        "--backend",
        "native",
    )

    assert completed.returncode != 0
    assert "ml-autoresearch.toml" in completed.stderr
