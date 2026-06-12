import json
from pathlib import Path

import pytest
import yaml

from ml_autoresearch.candidates import CandidateValidationError, validate_candidate_directory
from ml_autoresearch.research_problems import (
    ResearchProblemProviderConfig,
    ResearchProblemProviderProvenance,
    ResearchProblemSpec,
    ResearchProblemSpecRegistry,
)
from ml_autoresearch.runs import RunStatus, run_candidate_with_synthetic_fixture, submit_candidate
from ml_autoresearch.smoke import smoke_specs_from_resolved_manifest


CANDIDATES_ROOT = Path("candidates")


EXPECTED_GVCCS_CANDIDATE_CONTRACTS = {
    "single_frame_large_unet_line_auxiliary_w010": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "none", (("line", "line_logits", "weighted_bce", 0.10),)),
    "single_frame_medium_unet_line_auxiliary_w005": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "none", (("line", "line_logits", "weighted_bce", 0.05),)),
    "single_frame_medium_unet_line_auxiliary_w010": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "none", (("line", "line_logits", "weighted_bce", 0.10),)),
    "single_frame_medium_unet_line_auxiliary_w010_light_photometric": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "light_photometric", (("line", "line_logits", "weighted_bce", 0.10),)),
    "single_frame_small_unet": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "sequential", "none", ()),
    "single_frame_small_unet_line_auxiliary": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "none", (("line", "line_logits", "weighted_bce", 0.25),)),
    "single_frame_small_unet_line_auxiliary_w010": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "none", (("line", "line_logits", "weighted_bce", 0.10),)),
    "single_frame_small_unet_line_auxiliary_w010_light_combined": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "light_combined", (("line", "line_logits", "weighted_bce", 0.10),)),
    "single_frame_small_unet_line_auxiliary_w010_light_geometric": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "light_geometric", (("line", "line_logits", "weighted_bce", 0.10),)),
    "single_frame_small_unet_line_auxiliary_w010_light_photometric": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "light_photometric", (("line", "line_logits", "weighted_bce", 0.10),)),
    "single_frame_small_unet_realistic_training": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "none", ()),
    "single_frame_small_unet_realistic_training_shuffled": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "none", ()),
    "single_frame_wide_unet_line_auxiliary_w010": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "none", (("line", "line_logits", "weighted_bce", 0.10),)),
    "single_frame_xwide_unet_line_aux_w010_attention_dropout": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "none", (("line", "line_logits", "weighted_bce", 0.10),)),
    "single_frame_xwide_unet_line_aux_w010_dropout": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "none", (("line", "line_logits", "weighted_bce", 0.10),)),
    "single_frame_xwide_unet_line_aux_w010_dropout_p005_epoch40": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "none", (("line", "line_logits", "weighted_bce", 0.10),)),
    "single_frame_xwide_unet_line_aux_w010_dropout_p0075": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "none", (("line", "line_logits", "weighted_bce", 0.10),)),
    "single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch40": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "none", (("line", "line_logits", "weighted_bce", 0.10),)),
    "single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "none", (("line", "line_logits", "weighted_bce", 0.10),)),
    "single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60_detail_fuse": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "none", (("line", "line_logits", "weighted_bce", 0.10),)),
    "single_frame_xwide_unet_line_aux_w010_dropout_p0075_epoch60_head_dropout_p005": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "none", (("line", "line_logits", "weighted_bce", 0.10),)),
    "single_frame_xwide_unet_line_aux_w010_dropout_p0075_refine": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "none", (("line", "line_logits", "weighted_bce", 0.10),)),
    "single_frame_xwide_unet_line_aux_w010_dropout_p0075_rerun": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "none", (("line", "line_logits", "weighted_bce", 0.10),)),
    "single_frame_xwide_unet_line_aux_w010_dropout_p015": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "none", (("line", "line_logits", "weighted_bce", 0.10),)),
    "single_frame_xwide_unet_line_auxiliary_w010": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "none", (("line", "line_logits", "weighted_bce", 0.10),)),
    "single_frame_xwide_unet_line_boundary_aux_w010_w001_dropout_p0075_epoch40": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "none", (("line", "line_logits", "weighted_bce", 0.10), ("boundary", "boundary_logits", "weighted_bce", 0.01))),
    "single_frame_xwide_unet_line_boundary_aux_w010_w003_dropout_p0075_epoch40": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "none", (("line", "line_logits", "weighted_bce", 0.10), ("boundary", "boundary_logits", "weighted_bce", 0.03))),
    "single_frame_xwide_unet_line_boundary_aux_w010_w005_dropout_p0075": ("single_frame_rgb", "mask_logits", "bce_dice", "adamw", "deterministic_shuffle", "none", (("line", "line_logits", "weighted_bce", 0.10), ("boundary", "boundary_logits", "weighted_bce", 0.05))),
}


def _fake_gvccs_research_problem_spec() -> ResearchProblemSpec:
    return ResearchProblemSpec(
        id="ground_camera_contrail_detection",
        version="v0",
        input_modes=("single_frame_rgb", "centered_temporal_rgb_clip"),
        input_specs={
            "single_frame_rgb": {"mode": "single_frame_rgb", "shape": [3, 128, 128]},
            "centered_temporal_rgb_clip": {"mode": "single_frame_rgb", "shape": [3, 64, 64]},
        },
        output_forms=("mask_logits",),
        output_specs={"mask_logits": {"form": "mask_logits", "shape": [1, 128, 128]}},
        auxiliary_targets=("line", "boundary"),
        auxiliary_outputs={"line": "line_logits", "boundary": "boundary_logits"},
        losses=("bce_dice",),
        auxiliary_losses=("weighted_bce",),
        optimizers=("adamw",),
        sampling_policies=("sequential", "deterministic_shuffle"),
        frame_selection_policies=("all_target_frames", "temporal_eligible_center"),
        input_mode_frame_selection_defaults={"centered_temporal_rgb_clip": "temporal_eligible_center", "single_frame_rgb": "all_target_frames"},
        augmentation_policies=("none", "light_combined", "light_geometric", "light_photometric"),
        primary_metric="val/dice",
    )


def fake_research_problem_spec_provider(config: ResearchProblemProviderConfig | None = None) -> ResearchProblemSpec:
    return _fake_gvccs_research_problem_spec()


def _gvccs_fake_registry() -> ResearchProblemSpecRegistry:
    spec = _fake_gvccs_research_problem_spec()
    registry = ResearchProblemSpecRegistry(active_id=spec.id)
    registry.register(
        spec,
        provenance=ResearchProblemProviderProvenance(
            resolved_package_root=Path(__file__).resolve().parent.parent,
            provider_target="tests.test_gvccs_characterization:fake_research_problem_spec_provider",
        ),
    )
    return registry


def _contract_snapshot(candidate_dir: Path):
    manifest = validate_candidate_directory(candidate_dir, research_problem_registry=_gvccs_fake_registry())
    return (
        manifest.input_mode,
        manifest.output_form,
        manifest.training.loss,
        manifest.training.optimizer,
        manifest.data.sampling_policy,
        manifest.data.augmentation_policy,
        tuple((target.name, target.output, target.loss, target.weight) for target in manifest.auxiliary_targets),
    )


def test_existing_gvccs_candidate_manifests_validate_with_same_contract_choices() -> None:
    observed = {path.name: _contract_snapshot(path) for path in sorted(CANDIDATES_ROOT.iterdir()) if path.is_dir()}

    assert observed == EXPECTED_GVCCS_CANDIDATE_CONTRACTS


def test_resolved_gvccs_manifest_and_smoke_specs_stay_single_frame_mask_only(tmp_path: Path) -> None:
    run = submit_candidate(
        CANDIDATES_ROOT / "single_frame_small_unet",
        tmp_path / "runs",
        research_problem_registry=_gvccs_fake_registry(),
    )

    assert run.status == RunStatus.ACCEPTED
    resolved = yaml.safe_load((run.run_dir / "resolved_manifest.yaml").read_text())
    assert resolved["name"] == "single_frame_small_unet"
    assert resolved["description"] == "Small standard U-Net-style single-frame mask-only segmentation baseline."
    assert resolved["research_problem"]["id"] == "ground_camera_contrail_detection"
    assert resolved["research_problem"]["version"] == "v0"
    assert resolved["research_problem"]["contract_version"] == "v0"
    assert resolved["input_mode"] == "single_frame_rgb"
    assert resolved["output_form"] == "mask_logits"
    assert resolved["auxiliary_targets"] == []
    assert resolved["data"] == {
        "sampling_policy": "sequential",
        "frame_selection_policy": "all_target_frames",
        "frame_selection_policy_effective": "all_target_frames",
        "augmentation_policy": "none",
        "augmentation_policy_effective": "none",
    }
    assert resolved["training"] == {
        "loss": "bce_dice",
        "optimizer": "adamw",
        "learning_rate": 0.001,
        "batch_size": 2,
        "max_epochs": 1,
    }
    assert resolved["repair"] is None
    input_spec, output_spec = smoke_specs_from_resolved_manifest(run.run_dir / "resolved_manifest.yaml")
    assert input_spec == {"mode": "single_frame_rgb", "shape": [3, 128, 128]}
    assert output_spec == {"form": "mask_logits", "shape": [1, 128, 128]}


def test_gvccs_training_artifact_names_and_key_metrics_stay_stable(tmp_path: Path) -> None:
    run = run_candidate_with_synthetic_fixture(CANDIDATES_ROOT / "single_frame_small_unet", tmp_path / "runs")

    assert run.status == RunStatus.COMPLETED
    metadata = json.loads((run.run_dir / "run_metadata.json").read_text())
    assert metadata["research_problem"]["id"] == "ground_camera_contrail_detection"
    assert metadata["research_problem"]["version"] == "v0"
    assert metadata["research_problem"]["contract_version"] == "v0"
    assert metadata["artifacts"] == {
        "prediction_samples": "outputs/prediction_samples/samples.json",
        "best_metrics": "outputs/best_metrics.json",
        "best_epoch_model": "outputs/models/best_epoch_model.pt",
    }

    final = json.loads((run.run_dir / "outputs" / "final_metrics.json").read_text())
    assert set(final) >= {
        "epoch",
        "train/loss",
        "train/mask_loss",
        "val/dice",
        "val/iou",
        "val/precision",
        "val/recall",
        "val/loss",
        "hardware/device",
    }
    assert final["artifacts"] == metadata["artifacts"]

    best = json.loads((run.run_dir / "outputs" / "best_metrics.json").read_text())
    assert best["selection_metric"] == "val/dice"
    assert best["selection_mode"] == "max"
    assert best["model_artifact"] == "outputs/models/best_epoch_model.pt"
    assert set(best["metrics"]) >= {"epoch", "val/dice", "val/iou", "val/loss"}


@pytest.mark.parametrize(
    ("patch", "expected_terms"),
    [
        ({"input_mode": "arbitrary_loader"}, ("input_mode", "arbitrary_loader", "ground_camera_contrail_detection")),
        ({"output_form": "custom_logits"}, ("output_form", "custom_logits", "ground_camera_contrail_detection")),
        ({"training": {"loss": "custom_loss"}}, ("training.loss", "custom_loss", "ground_camera_contrail_detection")),
        ({"data": {"augmentation_policy": "candidate_custom_transform"}}, ("data.augmentation_policy", "candidate_custom_transform", "ground_camera_contrail_detection")),
    ],
)
def test_invalid_gvccs_manifest_values_still_fail_clearly(tmp_path: Path, patch: dict, expected_terms: tuple[str, ...]) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    manifest = yaml.safe_load((CANDIDATES_ROOT / "single_frame_small_unet" / "manifest.yaml").read_text())
    for key, value in patch.items():
        if isinstance(value, dict):
            manifest.setdefault(key, {}).update(value)
        else:
            manifest[key] = value
    (candidate / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    (candidate / "model.py").write_text((CANDIDATES_ROOT / "single_frame_small_unet" / "model.py").read_text())

    with pytest.raises(CandidateValidationError) as excinfo:
        validate_candidate_directory(candidate, research_problem_registry=_gvccs_fake_registry())

    message = str(excinfo.value)
    for term in expected_terms:
        assert term in message
