# GVCCS Dataset local layout

The Harness-owned GVCCS Dataset adapter expects a local download rooted at `--data-root`.
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
ml-autoresearch run-candidate \
  --candidate tests/fixtures/candidates/single_frame_unet_baseline \
  --runs-root /tmp/ml-autoresearch-runs \
  --data-root /data2/GVCCS \
  --max-samples 8
```

`--max-samples` bounds the discovered, sorted training sample list before the deterministic train/validation split.

## Docker data mount

For Docker-backed GVCCS training, the host Harness validates that `--data-root` exists and is a directory before launching Docker. The Docker backend mounts that host path read-only at `/data` inside the Candidate Execution Boundary, and the in-container Harness-owned GVCCS adapter reads `/data`. Candidate Experiments cannot request mounts, choose data paths, or receive host dataset paths.

Synthetic fixture training and smoke testing continue to run without any data mount.

Completed GVCCS Runs record dataset metadata in `run_metadata.json`:

```json
{
  "dataset": {
    "id": "gvccs",
    "host_data_path": "/real/host/path/to/GVCCS",
    "container_data_path": "/data"
  }
}
```

Malformed GVCCS roots fail the Run with a clear `training_failure_reason` and `outputs/logs/training.log`. Missing or non-directory host data roots fail before a Run is created.
