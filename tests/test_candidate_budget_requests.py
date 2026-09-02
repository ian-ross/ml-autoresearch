from __future__ import annotations

import json
from pathlib import Path

import yaml

from ml_autoresearch.research_loop_operations import prepare_candidate_run_from_workspace


def _write_fake_problem_package(root: Path) -> None:
    package = root / "fake_problem"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "research_problem.py").write_text(
        "from ml_autoresearch.research_problems import ResearchProblemSpec\n"
        "def build_spec(data_config=None):\n"
        "    return ResearchProblemSpec(\n"
        "        id='fake_problem', version='v1', contract_version='v0',\n"
        "        input_modes=('single_frame_rgb',),\n"
        "        input_specs={'single_frame_rgb': {'mode': 'single_frame_rgb', 'shape': [3, 8, 8]}},\n"
        "        output_forms=('mask_logits',),\n"
        "        output_specs={'mask_logits': {'form': 'mask_logits', 'shape': [1, 8, 8]}},\n"
        "        losses=('bce_dice',), optimizers=('adamw',),\n"
        "        sampling_policies=('sequential',), frame_selection_policies=('all_target_frames',),\n"
        "        augmentation_policies=('none',), primary_metric='val/dice')\n"
    )


def _write_workspace_config(root: Path) -> None:
    (root / "ml-autoresearch.toml").write_text(
        """
[candidate_execution]
backend = "docker"
docker_image = "test:configured"
runs_root = "runs"
ledger_path = "research-ledger.jsonl"
max_parameters = 1000000
max_epochs = 2
late_improving_max_epochs = 5
max_batch_size = 4
allowed_scheduler_policies = ["constant_lr"]
early_stopping_policy = "disabled"
training_wall_clock_timeout_seconds = 10
late_improving_training_wall_clock_timeout_seconds = 20
max_prediction_samples = 1
max_parallel_runs = 1
prediction_sample_policy = "first_n"

[research_problem]
id = "fake_problem"
package_root = "."
provider_target = "fake_problem.research_problem:build_spec"
expected_contract_version = "v0"
""".lstrip()
    )


def _proposal() -> str:
    return """
# Proposal

## Hypothesis
A bounded test.

## Comparison Target
Current baseline.

## Expected Effect
Learning may continue.

## Implementation Sketch
Tiny model.

## Contract Features Used
Allowed manifest fields only.

## Budget Requested
Exploratory unless a budget request is present.

## Success Criteria
Accepted by the Harness.

## Fallback Next Decision
Continue with exploratory evidence.
""".lstrip()


def _write_candidate(root: Path, *, manifest_epochs: int = 2, budget_request: dict[str, object] | str | None = None) -> Path:
    candidate = root / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "candidate",
                "research_problem": "fake_problem",
                "input_mode": "single_frame_rgb",
                "output_form": "mask_logits",
                "training": {
                    "loss": "bce_dice",
                    "optimizer": "adamw",
                    "learning_rate": 0.001,
                    "batch_size": 2,
                    "max_epochs": manifest_epochs,
                    "scheduler": {"policy": "constant_lr"},
                    "early_stopping": {"enabled": False},
                },
            },
            sort_keys=False,
        )
    )
    (candidate / "model.py").write_text("def build_model(input_spec, output_spec):\n    raise RuntimeError('not used')\n")
    (candidate / "PROPOSAL.md").write_text(_proposal())
    if isinstance(budget_request, dict):
        (candidate / "budget_request.yaml").write_text(yaml.safe_dump(budget_request, sort_keys=False))
    elif isinstance(budget_request, str):
        (candidate / "budget_request.yaml").write_text(budget_request)
    return candidate


def _workspace(root: Path) -> None:
    _write_fake_problem_package(root)
    _write_workspace_config(root)


def _write_completed_parent_run(root: Path, run_id: str = "run_parent") -> None:
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "completed",
                "failure_classification": None,
            }
        )
        + "\n"
    )


def _prepare(candidate: Path, root: Path) -> Path:
    payload = prepare_candidate_run_from_workspace(
        candidate,
        workspace_root=root,
        backend_name="docker",
        docker_image="test:override",
    )
    return Path(str(payload["run_dir"]))


def test_late_improving_budget_request_overrides_resolved_manifest_and_timeout(tmp_path: Path) -> None:
    _workspace(tmp_path)
    _write_completed_parent_run(tmp_path)
    candidate = _write_candidate(
        tmp_path,
        budget_request={
            "schema_version": "candidate_budget_request.v1",
            "budget_class": "late_improving",
            "requested_max_epochs": 5,
            "parent_run_ids": ["run_parent"],
            "justification": "Parent best checkpoint was near the exploratory ceiling.",
            "fallback_if_denied": "run_exploratory_budget",
        },
    )

    run_dir = _prepare(candidate, tmp_path)

    resolved = yaml.safe_load((run_dir / "resolved_manifest.yaml").read_text())
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    source_manifest = yaml.safe_load((run_dir / "candidate" / "manifest.yaml").read_text())
    assert source_manifest["training"]["max_epochs"] == 2
    assert resolved["training"]["max_epochs"] == 5
    assert resolved["budget"]["budget_request_status"] == "approved"
    assert metadata["budget_class_requested"] == "late_improving"
    assert metadata["budget_class_effective"] == "late_improving"
    assert metadata["effective_max_epochs"] == 5
    assert metadata["effective_training_wall_clock_timeout_seconds"] == 20
    assert metadata["budget_request_parent_run_ids"] == ["run_parent"]
    assert metadata["execution_backend"]["wall_clock_budget"]["timeout_seconds"] == 20


def test_denied_late_improving_request_falls_back_to_exploratory_budget(tmp_path: Path) -> None:
    _workspace(tmp_path)
    candidate = _write_candidate(
        tmp_path,
        budget_request={
            "schema_version": "candidate_budget_request.v1",
            "budget_class": "late_improving",
            "requested_max_epochs": 6,
            "parent_run_ids": ["run_parent"],
            "justification": "Too large.",
            "fallback_if_denied": "run_exploratory_budget",
        },
    )

    run_dir = _prepare(candidate, tmp_path)

    resolved = yaml.safe_load((run_dir / "resolved_manifest.yaml").read_text())
    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert resolved["training"]["max_epochs"] == 2
    assert metadata["budget_request_status"] == "denied"
    assert metadata["budget_class_effective"] == "exploratory"
    assert metadata["effective_max_epochs"] == 2
    assert metadata["effective_training_wall_clock_timeout_seconds"] == 10
    assert metadata["budget_request_denial_reason"] == "requested_max_epochs_exceeds_late_improving_limit"
    assert metadata["execution_backend"]["wall_clock_budget"]["timeout_seconds"] == 10


def test_malformed_budget_request_falls_back_without_rejecting_candidate(tmp_path: Path) -> None:
    _workspace(tmp_path)
    candidate = _write_candidate(tmp_path, budget_request="schema_version: wrong\n")

    run_dir = _prepare(candidate, tmp_path)

    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["status"] == "smoke_testing"
    assert metadata["budget_request_status"] == "denied"
    assert metadata["budget_class_effective"] == "exploratory"
    assert metadata["budget_request_denial_reason"].startswith("malformed_request:")


def test_manifest_cannot_directly_select_late_improving_epochs(tmp_path: Path) -> None:
    _workspace(tmp_path)
    candidate = _write_candidate(tmp_path, manifest_epochs=5)

    run_dir = _prepare(candidate, tmp_path)

    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    assert metadata["status"] == "rejected"
    assert "exceeds the Harness-owned Workspace ceiling 2" in metadata["rejection_reason"]
    assert metadata["budget_request_status"] == "not_requested"
    assert metadata["budget_class_effective"] == "exploratory"
