from pathlib import Path

from ml_autoresearch.candidates import validate_candidate_directory


def test_sample_candidate_fixture_satisfies_issue_1_contract():
    fixture = Path("tests/fixtures/candidates/single_frame_unet_baseline")

    manifest = validate_candidate_directory(fixture)

    assert manifest.name == "single_frame_unet_baseline"
    assert manifest.input_mode == "single_frame_rgb"
    assert manifest.output_form == "mask_logits"
