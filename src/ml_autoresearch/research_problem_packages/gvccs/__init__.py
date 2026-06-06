"""Ground-Camera Contrail Detection Research Problem package."""

_ADAPTER_EXPORTS = {"GVCCSTrainingAdapter", "build_spec"}
_DATASET_EXPORTS = {
    "DEFAULT_SPLIT_SEED",
    "IMAGE_SIZE",
    "GVCCSDataError",
    "GVCCSDataset",
    "GVCCSSample",
    "GVCCSSplit",
    "GVCCSTemporalClip",
    "GVCCSTemporalClipDataset",
    "build_centered_temporal_clips",
    "deterministic_train_val_split",
    "discover_gvccs_samples",
    "infer_frame_sequences",
    "select_gvccs_frames",
}

__all__ = sorted(_ADAPTER_EXPORTS | _DATASET_EXPORTS)


def __getattr__(name: str) -> object:
    if name in _ADAPTER_EXPORTS:
        from ml_autoresearch.research_problem_packages.gvccs import adapters

        return getattr(adapters, name)
    if name in _DATASET_EXPORTS:
        from ml_autoresearch.research_problem_packages.gvccs import datasets

        return getattr(datasets, name)
    raise AttributeError(name)
