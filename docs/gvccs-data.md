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

Local real-data example:

```bash
ml-autoresearch run-candidate \
  --candidate tests/fixtures/candidates/single_frame_unet_baseline \
  --runs-root /tmp/ml-autoresearch-runs \
  --data-root /data2/GVCCS \
  --max-samples 8
```

`--max-samples` bounds the discovered, sorted training sample list before the deterministic train/validation split.
