# Troubleshooting

## MkDocs Is Not Installed

Symptom:

```text
No module named mkdocs
```

Install the docs dependency:

```bash
python -m pip install -r docs/requirements.txt
```

Then serve:

```bash
python -m mkdocs serve
```

## Docker Cannot See the GPU

Check from the host:

```bash
nvidia-smi
```

Check from the pipeline container:

```bash
docker compose --project-directory . \
  -f docker/compose/compose.yml \
  -f docker/compose/dev.yml \
  exec vbogs-pipeline nvidia-smi
```

If the host works but the container does not, check NVIDIA Container Toolkit
installation and Docker runtime configuration.

## KITTI-360 Paths Are Not Found

Confirm the preferred layout:

```text
data/KITTI-360/calibration/perspective.txt
data/KITTI-360/data_poses/<drive>/cam0_to_world.txt
data/KITTI-360/images/<drive>/image_00/data_rect/
data/KITTI-360/images/<drive>/image_01/data_rect/
```

Or pass explicit overrides:

```bash
--raw-root <path> --poses-root <path> --calibration-dir <path>
```

## No Octree-AnyGS Model Is Found

`bucket`, `render`, and `nbv` default to the latest run under:

```text
/data/OCTREE-ANYGS/<drive>/
```

Use an explicit run directory when needed:

```bash
--model-path /data/OCTREE-ANYGS/<drive>/<timestamp>
```

## M4 or M4b Is Too Large

Use smaller smoke settings:

```bash
--frame-step 20
--max-frames 30
--max-points-per-frame 50000
--bucket-max-points 1000000
```

For fitting memory pressure:

```bash
--fit-mode batched
--batch-size 5000
--vmap-group-size 32
--max-padded-points-per-group <N>
```

## Coarse Anchors Look Falsely Uncertain

This usually means points were assigned only to the finest level. VBOGS must
bucket every point at every Octree-AnyGS level whose anchor cell contains it.
Use `scripts/bucket_points.py`; do not replace it with a finest-leaf-only
assignment.

## Uncertainty Values Look Incomparable

Check that:

- bucketing used world-frame coordinates from `points_world`;
- VBGS fitting used globally normalized coordinates from `points_norm`;
- normalization was global, not per-anchor.

Per-anchor normalization changes differential entropy scales and makes raw
`U` values incomparable unless a Jacobian correction is added.

## NBV Ignores Empty Space

This is expected in the current algorithm. `render_scalar` only renders
existing anchors. Empty unobserved volume has no Gaussian and contributes zero
to the score. Adding empty-space exploration requires a new occupancy prior or
unknown-ray penalty.

## Generated Outputs Are Missing from the Bundle

Run through `bundle`:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --drive <drive> \
  --start-at bundle \
  --stop-after bundle \
  --use-service-labels
```

If a stage output lives outside the derived defaults, pass the relevant output
directory override or check `run_manifest.json` for the path the bundler used.
