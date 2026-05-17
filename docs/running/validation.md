# Validation and Inspection

M7 is intentionally human-led. The scripts below make the inspection work
repeatable.

## Inspect Anchor Fits

After `fit`, inspect posterior quality before trusting `U.npy`:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --start-at inspect \
  --stop-after inspect \
  --inspect-top-k 10 \
  --use-service-labels
```

Inspect one anchor and export its assigned points:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --start-at inspect \
  --stop-after inspect \
  --inspect-anchor-id 12345 \
  --inspect-export-ply outputs/anchor_12345_points.ply \
  --use-service-labels
```

Look for:

- compact, well-sampled anchors with low entropy;
- sparse/noisy anchors with high entropy;
- coarse anchors that received points, not only finest-level anchors;
- fits that hit `K_MAX` while still improving.

## Export Map-Scale Anchor Uncertainty

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --start-at map-viz \
  --stop-after map-viz \
  --use-service-labels
```

Outputs are CloudCompare-friendly PLY files under:

```text
outputs/v1_0/<drive>/pointclouds/anchors/
```

Useful flags:

| Flag | Use |
| --- | --- |
| `--map-viz-observed-only` | Hide unobserved anchors from the exported PLY |
| `--map-viz-no-split-levels` | Write only the all-level combined PLY |
| `--map-viz-vmin`, `--map-viz-vmax` | Fix the color scale across runs |

## Render Uncertainty Views

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --start-at render \
  --stop-after render \
  --render-split both \
  --render-max-views 5 \
  --use-service-labels
```

The render stage writes RGB, uncertainty, and side-by-side diagnostics under:

```text
outputs/v1_0/<drive>/views/
```

## Query One View

`scripts/query_uncertainty_view.py` renders one camera pose, exports visible
anchors, and writes heatmaps. It is useful for debugging a surprising NBV pick.

Example using a held-out test camera:

```bash
python scripts/query_uncertainty_view.py \
  --drive 2013_05_28_drive_0007_sync \
  --camera-source test \
  --camera-index 0 \
  --selection-mode frustum-and-rendered
```

## Score and Visualize NBV Candidates

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --start-at nbv \
  --stop-after nbv-viz \
  --nbv-top-k 10 \
  --nbv-save-top-images 5 \
  --use-service-labels
```

The score is:

```text
sum(uncertainty_image) / (sum(alpha_image) + EPS)
```

This is alpha-normalized, so it favors the most uncertain visible content
rather than simply the most visible surface area.
