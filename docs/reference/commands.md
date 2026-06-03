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
bash scripts/build_stack_serial.sh vbogs-vbgs-render
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

Bootstrap a non-dev stack checkout from inside `vbogs-pipeline`:

```bash
vbogs-bootstrap-repo
```

Check GPU visibility:

```bash
docker compose --project-directory . \
  -f docker/compose/compose.yml \
  -f docker/compose/dev.yml \
  exec vbogs-pipeline nvidia-smi
```

Start the realtime Octree-AnyGS viewer from the published render container:

```bash
docker compose --project-directory . \
  -f docker/compose/compose.yml \
  -f docker/compose/dev.yml \
  exec vbogs-vbgs-render \
  python scripts/view_octree_anygs.py \
    --drive 2013_05_28_drive_0007_sync \
    --resolution 4
```

## Pipeline

Dry run:

```bash
python scripts/run_drive_pipeline.py \
  --config configs/pipeline/dev.yaml \
  --drive 2013_05_28_drive_0007_sync \
  --dry-run \
  --compose-file docker/compose/compose.yml \
  --compose-file docker/compose/dev.yml \
  --compose-project-directory .
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
  --compose-file docker/compose/compose.yml \
  --compose-file docker/compose/dev.yml \
  --compose-project-directory .
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
  --compose-file docker/compose/compose.yml \
  --compose-file docker/compose/dev.yml \
  --compose-project-directory .
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

Prepare NVIDIA NCore into COLMAP layout:

```bash
python scripts/prepare_nvidia_ncore_colmap.py \
  --scene-id <clip-id> \
  --camera-id camera_front_wide_120fov \
  --frame-step 2 \
  --max-frames 200
```

Export world points:

```bash
python scripts/export_points_world.py \
  --dataset-name kitti360 \
  --scene-id 2013_05_28_drive_0007_sync \
  --point-source stereo \
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

Run the original global VBGS KITTI baseline from the Docker host:

```bash
python scripts/run_vbgs_kitti_baseline.py \
  --drive 2013_05_28_drive_0007_sync \
  --use-service-labels
```

This wrapper uses Docker CLI access to run the actual JAX/VBGS fit inside the
`vbogs-jax` container and writes artifacts under
`outputs/vbgs_baseline/<drive>/` by default. Use `--input-mode bucket` to force the
same normalized points as VBOGS, or `--input-mode stereo` to train directly from
`data/points_world/<drive>/points_world.npz`.

Render the original global VBGS KITTI baseline from the dedicated render
container:

```bash
docker compose exec vbogs-vbgs-render \
  python scripts/render_vbgs_kitti_baseline.py \
    --drive 2013_05_28_drive_0007_sync \
    --max-views 5
```

This renders `outputs/vbgs_baseline/<drive>/model_final.json` through prepared
KITTI cameras under `/data/COLMAP/<drive>` and writes predicted and side-by-side
PNGs under `outputs/vbgs_baseline/<drive>/renders/`.

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

## Online ROS2 Loop

Build the online state bundle after M7 validation:

```bash
python scripts/build_online_state.py \
  --drive 2013_05_28_drive_0008_sync \
  --model-path /data/OCTREE-ANYGS/2013_05_28_drive_0008_sync/<run>
```

Run the updater process in the JAX environment:

```bash
python scripts/online_jax_updater.py \
  --state-root data/online/2013_05_28_drive_0008_sync
```

Run the ROS2 node in a ROS2 Humble environment with the Torch/Octree stack:

```bash
python scripts/ros2_online_nbv_node.py \
  --config configs/online/ros2_default.yaml
```

Replay existing `points_world.npz` artifacts through the online handoff and
latency logger:

```bash
python scripts/benchmark_online_loop.py \
  --state-root data/online/2013_05_28_drive_0008_sync \
  --points-world data/points_world/2013_05_28_drive_0008_sync/points_world.npz \
  --run-updater
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
pytest tests/test_online.py
```
