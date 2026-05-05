from pathlib import Path

from ml_autoresearch.artifacts import select_prediction_sample_indices
from ml_autoresearch.gvccs import GVCCSSample, infer_frame_sequences


def sample(name: str, *, image_id: int, positive: bool = False) -> GVCCSSample:
    return GVCCSSample(
        image_id=image_id,
        image_path=Path("/gvccs/train/images") / name,
        width=128,
        height=128,
        segmentations=((1.0, 1.0, 8.0, 1.0, 8.0, 8.0),) if positive else (),
    )


def test_infer_frame_sequences_splits_sorted_timestamp_filenames_on_gaps_greater_than_30_seconds():
    samples = [
        sample("image_20260502000130.png", image_id=4),
        sample("image_20260502000000.png", image_id=1),
        sample("image_20260502000030.png", image_id=2),
        sample("image_20260502000100.png", image_id=3),
        sample("image_20260502000300.png", image_id=5),
        sample("image_20260502000330.png", image_id=6),
    ]

    sequences = infer_frame_sequences(samples)

    assert [[item.image_path.name for item in sequence] for sequence in sequences] == [
        ["image_20260502000000.png", "image_20260502000030.png", "image_20260502000100.png", "image_20260502000130.png"],
        ["image_20260502000300.png", "image_20260502000330.png"],
    ]


def test_adjacent_and_scattered_selects_spaced_positive_sequences_and_mixed_scattered_samples():
    samples = [
        sample("image_20260502000000.png", image_id=1),
        sample("image_20260502000030.png", image_id=2, positive=True),
        sample("image_20260502000100.png", image_id=3),
        sample("image_20260502001000.png", image_id=4),
        sample("image_20260502001030.png", image_id=5, positive=True),
        sample("image_20260502001100.png", image_id=6),
        sample("image_20260502002000.png", image_id=7),
        sample("image_20260502002030.png", image_id=8, positive=True),
        sample("image_20260502002100.png", image_id=9),
        sample("image_20260502003000.png", image_id=10),
        sample("image_20260502004000.png", image_id=11, positive=True),
        sample("image_20260502005000.png", image_id=12, positive=True),
    ]

    first = select_prediction_sample_indices(samples, policy="adjacent_and_scattered", max_samples=6)
    second = select_prediction_sample_indices(samples, policy="adjacent_and_scattered", max_samples=6)

    assert first == second
    assert len(first) == 6
    adjacent = [item for item in first if item["selection_kind"] == "adjacent_window"]
    scattered = [item for item in first if item["selection_kind"] == "scattered_singleton"]
    assert len(adjacent) == 4
    assert len(scattered) == 2
    assert len({item["frame_sequence_id"] for item in adjacent}) == 2
    assert any(samples[item["dataset_index"]].segmentations for item in scattered)
    assert any(not samples[item["dataset_index"]].segmentations for item in scattered)

    by_window: dict[str, list[int]] = {}
    for item in adjacent:
        by_window.setdefault(str(item["adjacent_window_id"]), []).append(item["dataset_index"])
    assert all(indices[1] == indices[0] + 1 for indices in by_window.values())


def test_first_n_prediction_policy_preserves_current_order_metadata():
    samples = [sample(f"image_202605020000{i}0.png", image_id=i) for i in range(4)]

    selected = select_prediction_sample_indices(samples, policy="first_n", max_samples=2)

    assert [item["dataset_index"] for item in selected] == [0, 1]
    assert all(item["selection_kind"] == "first_n" for item in selected)
