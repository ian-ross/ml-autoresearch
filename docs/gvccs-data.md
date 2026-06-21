# GVCCS Dataset local layout

The GVCCS example Research Problem package adapter expects a local download rooted at `ml-autoresearch.toml` `research_problem.data_config.dataset_root`.
Do not commit real GVCCS samples to this repository.

Expected layout:

```text
<data-root>/
  train/
    images/
      image_YYYYMMDDhhmmss.jpg  # `.jpg`, `.jpeg`, or `.png`
    annotations.json            # COCO-style images/annotations/categories
  test/                         # optional for current training path
    images/
    annotations.json
```

The adapter currently uses only Single-Frame RGB Input from `train/images` and binary Contrail Masks rasterized from polygon `segmentation` entries in `train/annotations.json`. Images and masks are resized to `[128, 128]`; masks use nearest-neighbor resizing and are thresholded to binary values.

Local real-data example, using the default Docker backend:

```bash
# Ensure ml-autoresearch.toml sets research_problem.data_config.dataset_root to your local GVCCS dataset path.

ml-autoresearch run-candidate \
  --candidate tests/fixtures/candidates/single_frame_unet_baseline \
  --runs-root /tmp/ml-autoresearch-runs \
  --workspace-root . \
  --max-samples 8
```

`--max-samples` bounds the discovered, sorted training sample list before the deterministic train/validation split.

## Docker data mount

For Docker-backed runs using the GVCCS example Research Problem package, the host Harness validates that configured `dataset_root` exists and is a directory before launching Docker. The Docker backend mounts that host path read-only at `/data` inside the Candidate Execution Boundary, and the in-container Research Problem adapter reads `/data`. Candidate Experiments cannot request mounts, choose data paths, or receive host dataset paths.

Synthetic fixture training and smoke testing continue to run without any data mount.

Completed runs using the GVCCS example package record dataset metadata in `run_metadata.json`:

```json
{
  "dataset": {
    "id": "gvccs",
    "host_data_path": "/real/host/path/to/GVCCS",
    "container_data_path": "/data"
  }
}
```

Malformed GVCCS roots fail the Run with a clear `training_failure_reason` and `outputs/logs/training.log`. Missing or non-directory configured data roots fail before a Run is created.
