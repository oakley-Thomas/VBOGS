# Command Reference

This page lists the commands operators reach for most often. The exhaustive
pipeline argument reference is [Pipeline Arguments](../documentation/RUN_DRIVE_PIPELINE_ARGS.md).

## Documentation

```bash
python -m pip install -r docs/requirements.txt
python -m mkdocs serve
python -m mkdocs build
```

## Docker Stack

Build every image:

```bash
bash scripts/build_stack_serial.sh
```

Build one image:

```bash
bash scripts/build_stack_serial.sh vbogs-torch
bash scripts/build_stack_serial.sh vbogs-jax
bash scripts/build_stack_serial.sh vbogs-pipeline
```

Start the local dev stack:

```bash
docker compose --project-directory . \
  -f docker/compose/compose.yml \
  -f docker/compose/dev.yml \
  up -d --no-build
```

Enter the pipeline container:

```bash
docker compose --project-directory . \
  -f docker/compose/compose.yml \
  -f docker/compose/dev.yml \
  exec vbogs-pipeline bash
```

Check GPU visibility:

```bash
docker compose --project-directory . \
  -f docker/compose/compose.yml \
  -f docker/compose/dev.yml \
  exec vbogs-pipeline nvidia-smi
```

## Pipeline

Dry run:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --use-service-labels \
  --dry-run
```

Full run:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --gpu 0 \
  --jax-device 0 \
  --start-at prepare \
  --stop-after bundle \
  --use-service-labels
```

Small smoke run:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --gpu 0 \
  --jax-device 0 \
  --start-at prepare \
  --stop-after render \
  --frame-step 20 \
  --max-frames 30 \
  --resolution 4 \
  --iterations 7000 \
  --max-points-per-frame 50000 \
  --render-max-views 2 \
  --use-service-labels
```

## Direct Stage Entry Points

Prepare KITTI-360 into COLMAP layout:

```bash
python scripts/prepare_kitti360_colmap.py \
  --drive 2013_05_28_drive_0007_sync \
  --frame-step 10 \
  --max-frames 100
```

Train Octree-AnyGS:

```bash
python scripts/train_octree_anygs.py \
  --drive 2013_05_28_drive_0007_sync \
  --source-path /data/COLMAP/2013_05_28_drive_0007_sync \
  --gpu 0 \
  --resolution 4 \
  --iterations 30000
```

Export stereo points:

```bash
python scripts/stereo_to_pointcloud.py \
  --drive 2013_05_28_drive_0007_sync \
  --matcher sgbm \
  --pixel-step 1 \
  --max-points-per-frame 100000 \
  --write-ply
```

Bucket points:

```bash
python scripts/bucket_points.py \
  --drive 2013_05_28_drive_0007_sync \
  --iteration -1 \
  --point-chunk-size 1000000
```

Fit anchors:

```bash
python scripts/fit_anchors.py \
  --drive 2013_05_28_drive_0007_sync \
  --fit-mode batched \
  --batch-size 5000 \
  --vmap-group-size 64
```

Compute uncertainty:

```bash
python scripts/compute_uncertainty.py \
  --drive 2013_05_28_drive_0007_sync
```

Run the original global VBGS KITTI baseline from `vbogs-pipeline`:

```bash
python scripts/run_vbgs_kitti_baseline.py \
  --drive 2013_05_28_drive_0007_sync \
  --use-service-labels
```

`vbogs-pipeline` only orchestrates this command. The actual JAX/VBGS fit runs
inside the sibling `vbogs-jax` container and writes artifacts under
`outputs/vbgs_baseline/<drive>/` by default. Use `--input-mode bucket` to force the
same normalized points as VBOGS, or `--input-mode stereo` to train directly from
`data/points_world/<drive>/points_world.npz`.

Run the VBGS vs VBOGS uncertainty-quality comparison:

```bash
python scripts/run_vbgs_vbogs_comparison.py \
  --drive 2013_05_28_drive_0007_sync \
  --use-service-labels
```

The comparison writes split point clouds, train/eval anchor buckets, VBOGS
uncertainty, global VBGS K-sweep projections, metrics, maps, and view renders
under `outputs/vbgs_comparison/<drive>/`.

Render uncertainty diagnostics:

```bash
python scripts/render_uncertainty_views.py \
  --drive 2013_05_28_drive_0007_sync \
  --split both \
  --max-views 5
```

Score next-best views:

```bash
python scripts/score_nbv.py \
  --drive 2013_05_28_drive_0007_sync \
  --candidate-source test \
  --top-k 10 \
  --save-top-images 5
```

Bundle outputs:

```bash
python scripts/bundle_run_outputs.py \
  --drive 2013_05_28_drive_0007_sync \
  --run-output-dir outputs/v1_0/2013_05_28_drive_0007_sync
```

## Tests

```bash
pytest
```

Focused tests:

```bash
pytest tests/test_run_drive_pipeline.py
pytest tests/test_bucket_points.py
pytest tests/test_compute_uncertainty.py
pytest tests/test_render.py
```
